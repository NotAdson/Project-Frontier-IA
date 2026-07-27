"""
Generates a synthetic training dataset for the fused_features autoencoder (issue #10).

The autoencoder trains standalone, without running the real Meta-Planner. Each
training example is a synthetic stand-in for "fused_features" from
train_nn.py:429 (3074-dim), assembled as:

  1. inp_dense (NUM_DENSE_FEATURES=758) + the 4 blocks of categorical indices
     (species/moves/items/abilities, NUM_EMBEDDING_INDICES=84 total) come from
     REAL battle states: the first TOTAL_FEATURES=842 values of a real step's
     "features" vector, taken from data/genrandom_bootstrap/game_*.json.
  2. emb_species_main / emb_moves_main / emb_items_main / emb_abilities_main
     are produced by looking up the real categorical indices from step 1 into
     UNTRAINED embedding tables of the exact main-trunk dimensions
     (MAIN_EMB_SPECIES_DIM=32, MAIN_EMB_MOVES_DIM=32, MAIN_EMB_ITEMS_DIM=16,
     MAIN_EMB_ABILITY_DIM=16), initialized as Normal(mean=0, stddev=0.05).
     NOTE: this is mathematically identical to running an untrained
     keras.layers.Embedding / torch.nn.Embedding lookup (a lookup is just
     indexing into the weight matrix) — implemented here as a direct numpy
     matrix index for simplicity, with no Keras/PyTorch import needed.
     Also NOTE: Normal(0, 0.05) is the value the spec asked for, but it is NOT
     actually Keras's literal default — keras.layers.Embedding defaults to
     embeddings_initializer="uniform", i.e. RandomUniform(-0.05, 0.05), not a
     normal distribution. We implement exactly the numeric spec given
     (Normal(0, 0.05)), not the literal Keras default, and flag the
     discrepancy here for the record.
  3. meta_plan (12) is sampled independently per example: 6 values via
     np.random.dirichlet (own weights, sum to 1), 6 via np.random.uniform(0, 1)
     (opponent weights) — matching the own/opp split documented in
     train_nn.py's meta_plan (softmax own_weights + sigmoid opp_weights).
  4. Concatenated in the exact order of train_nn.py:424-429:
     [inp_dense, emb_species_main, emb_moves_main, emb_items_main,
      emb_abilities_main, meta_plan] -> 3074 dims.

Only 2,000,000 of the ~9.26M available steps are used (~24-25GB on disk
instead of ~115GB for the full set), chosen via reservoir sampling
(Algorithm R) over a single pass across ALL game_*.json files in
--data-dir. Reservoir sampling gives every step across the ENTIRE directory
equal probability of being picked, regardless of which file it's in or the
order files are visited — this is what "spread across the whole directory,
not concentrated in the first files" requires, and it does not need to know
the total step count in advance.

This script does NOT build or train the autoencoder itself.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from battle_agents.mcts_approximation.db.python.database import db
from battle_agents.mcts_approximation.state_encoder import (
    MAIN_EMB_ABILITY_DIM, MAIN_EMB_ITEMS_DIM, MAIN_EMB_MOVES_DIM,
    MAIN_EMB_SPECIES_DIM, NUM_ABILITY_INDICES, NUM_DENSE_FEATURES,
    NUM_ITEM_INDICES, NUM_MOVE_INDICES, NUM_SPECIES_INDICES, OFF_ABILITIES,
    OFF_ITEMS, OFF_MOVES, OFF_SPECIES, TOTAL_FEATURES)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "genrandom_bootstrap"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "autoencoder_bootstrap" / "fused_features_synthetic.npy"

N_SAMPLES = 2_000_000
SEED = 42  # fixed seed, documented: derives 3 independent RNGs below, one per random stage
SEED_RESERVOIR = SEED       # which steps get sampled (Algorithm R)
SEED_EMBEDDINGS = SEED + 1  # untrained embedding table weights
SEED_META_PLAN = SEED + 2   # per-example meta_plan draws
EMBEDDING_STD = 0.05

CHUNK_ROWS = 100_000  # rows per batch during the in-RAM assembly/write phase

# Pieces of fused_features, in train_nn.py:424-429 order, with their dims.
PIECES = [
    ("dense", NUM_DENSE_FEATURES),
    ("emb_species", NUM_SPECIES_INDICES * MAIN_EMB_SPECIES_DIM),
    ("emb_moves", NUM_MOVE_INDICES * MAIN_EMB_MOVES_DIM),
    ("emb_items", NUM_ITEM_INDICES * MAIN_EMB_ITEMS_DIM),
    ("emb_abilities", NUM_ABILITY_INDICES * MAIN_EMB_ABILITY_DIM),
    ("meta_plan", 12),
]
FUSED_DIM = sum(dim for _, dim in PIECES)
assert FUSED_DIM == 3074, FUSED_DIM


def piece_offsets():
    """[(name, start, end), ...] over the FUSED_DIM axis, from PIECES (dense,
    emb_species, emb_moves, emb_items, emb_abilities, meta_plan) in order.
    Single source of truth — shared by train_autoencoder.py (weighted loss)
    and test_reconstruction.py (per-piece MSE report), so both always slice
    fused_features the same way."""
    offsets = []
    start = 0
    for name, dim in PIECES:
        offsets.append((name, start, start + dim))
        start += dim
    assert start == FUSED_DIM, (start, FUSED_DIM)
    return offsets


def reservoir_sample_raw_features(files, n_samples, seed):
    """Single pass, Algorithm R reservoir sampling of raw TOTAL_FEATURES-long
    rows (the untouched "features" vector of each step), across every step of
    every file, no early exit — every file is visited so the sample is spread
    across the entire directory."""
    rng = np.random.default_rng(seed)
    reservoir = np.empty((n_samples, TOTAL_FEATURES), dtype=np.float32)
    seen = 0

    t0 = time.time()
    for fi, fpath in enumerate(files):
        with open(fpath, "r") as fh:
            game_data = json.load(fh)
        for step in game_data:
            feat = step["features"]
            if seen < n_samples:
                reservoir[seen] = feat
            else:
                j = rng.integers(0, seen + 1)
                if j < n_samples:
                    reservoir[j] = feat
            seen += 1
        if (fi + 1) % 5000 == 0 or (fi + 1) == len(files):
            elapsed = time.time() - t0
            print(f"  ... {fi + 1}/{len(files)} files scanned, "
                  f"{seen:,} steps seen, {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"Reservoir sampling finished in {elapsed:.1f}s. "
          f"Total steps in stream: {seen:,}  |  Reservoir size: {n_samples:,}")
    if seen < n_samples:
        raise SystemExit(
            f"Only {seen:,} steps available across all game files — "
            f"fewer than the requested {n_samples:,}. Lower --n-samples or add more games."
        )
    return reservoir


def build_embedding_tables(seed):
    rng = np.random.default_rng(seed)
    num_species = db.get_num_species()
    num_moves = db.get_num_moves()
    num_items = db.get_num_items()
    num_abilities = db.get_num_abilities()

    def table(vocab_size, dim):
        return rng.normal(0.0, EMBEDDING_STD, size=(vocab_size, dim)).astype(np.float32)

    print(f"Untrained embedding tables — Normal(0, {EMBEDDING_STD}) init, seed={seed}:")
    print(f"  species:   vocab={num_species:5d}  dim={MAIN_EMB_SPECIES_DIM}")
    print(f"  moves:     vocab={num_moves:5d}  dim={MAIN_EMB_MOVES_DIM}")
    print(f"  items:     vocab={num_items:5d}  dim={MAIN_EMB_ITEMS_DIM}")
    print(f"  abilities: vocab={num_abilities:5d}  dim={MAIN_EMB_ABILITY_DIM}")

    return {
        "species":   (table(num_species, MAIN_EMB_SPECIES_DIM), num_species),
        "moves":     (table(num_moves, MAIN_EMB_MOVES_DIM), num_moves),
        "items":     (table(num_items, MAIN_EMB_ITEMS_DIM), num_items),
        "abilities": (table(num_abilities, MAIN_EMB_ABILITY_DIM), num_abilities),
    }


def embed_lookup(table, vocab_size, idx):
    """idx: (batch, k) float32 indices (stored as floats in the source data).
    Rounds to int and clips out-of-range indices to 0, mirroring evaluator.py's
    _build_inputs clipping behavior for the same categorical blocks."""
    idx_int = np.rint(idx).astype(np.int64)
    idx_int = np.where((idx_int >= 0) & (idx_int < vocab_size), idx_int, 0)
    return table[idx_int]  # (batch, k, dim)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--max-games", type=int, default=None,
                         help="Optional cap on number of game files scanned (for a quick dry run).")
    args = parser.parse_args()

    files = sorted(glob.glob(str(args.data_dir / "game_*.json")))
    if not files:
        raise SystemExit(f"No game_*.json files found in {args.data_dir}")
    if args.max_games is not None:
        files = files[:args.max_games]

    print(f"Fixed seeds — reservoir sampling: {SEED_RESERVOIR}, "
          f"embedding init: {SEED_EMBEDDINGS}, meta_plan draws: {SEED_META_PLAN}")
    print(f"Target samples: {args.n_samples:,}  |  fused_features dim: {FUSED_DIM}")
    print(f"Scanning {len(files)} game files from {args.data_dir} ...")

    reservoir = reservoir_sample_raw_features(files, args.n_samples, SEED_RESERVOIR)
    tables = build_embedding_tables(SEED_EMBEDDINGS)
    meta_rng = np.random.default_rng(SEED_META_PLAN)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nAssembling and writing {args.output} (shape ({args.n_samples:,}, {FUSED_DIM})) ...")
    final = np.lib.format.open_memmap(
        str(args.output), mode="w+", dtype=np.float32, shape=(args.n_samples, FUSED_DIM)
    )

    # Running per-piece accumulators (flat mean/std across every value in the piece).
    sums = {name: 0.0 for name, _ in PIECES}
    sumsqs = {name: 0.0 for name, _ in PIECES}
    counts = {name: 0 for name, _ in PIECES}

    t0 = time.time()
    for start in range(0, args.n_samples, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, args.n_samples)
        n = end - start
        chunk = reservoir[start:end]

        dense = chunk[:, :NUM_DENSE_FEATURES]
        cat = chunk[:, NUM_DENSE_FEATURES:]

        species_idx = cat[:, OFF_SPECIES:OFF_SPECIES + NUM_SPECIES_INDICES]
        moves_idx = cat[:, OFF_MOVES:OFF_MOVES + NUM_MOVE_INDICES]
        items_idx = cat[:, OFF_ITEMS:OFF_ITEMS + NUM_ITEM_INDICES]
        abilities_idx = cat[:, OFF_ABILITIES:OFF_ABILITIES + NUM_ABILITY_INDICES]

        emb_species = embed_lookup(*tables["species"], species_idx).reshape(n, -1)
        emb_moves = embed_lookup(*tables["moves"], moves_idx).reshape(n, -1)
        emb_items = embed_lookup(*tables["items"], items_idx).reshape(n, -1)
        emb_abilities = embed_lookup(*tables["abilities"], abilities_idx).reshape(n, -1)

        own_weights = meta_rng.dirichlet(np.ones(6), size=n)
        opp_weights = meta_rng.uniform(0.0, 1.0, size=(n, 6))
        meta_plan = np.concatenate([own_weights, opp_weights], axis=1).astype(np.float32)

        row = np.concatenate(
            [dense, emb_species, emb_moves, emb_items, emb_abilities, meta_plan], axis=1
        ).astype(np.float32)
        final[start:end] = row

        for name, piece in [
            ("dense", dense), ("emb_species", emb_species), ("emb_moves", emb_moves),
            ("emb_items", emb_items), ("emb_abilities", emb_abilities), ("meta_plan", meta_plan),
        ]:
            sums[name] += piece.sum(dtype=np.float64)
            sumsqs[name] += np.square(piece, dtype=np.float64).sum()
            counts[name] += piece.size

    final.flush()
    del final
    elapsed = time.time() - t0
    print(f"Assembly + write finished in {elapsed:.1f}s.")

    out_size_bytes = os.path.getsize(args.output)
    print()
    print(f"Saved {args.output}")
    print(f"  shape: ({args.n_samples:,}, {FUSED_DIM})  dtype: float32  "
          f"size on disk: {out_size_bytes / 1e9:.2f} GB")
    print()
    print("Per-piece mean / std (flat across all values in the piece, not per-column):")
    for name, dim in PIECES:
        mean = sums[name] / counts[name]
        var = max(sumsqs[name] / counts[name] - mean ** 2, 0.0)
        std = var ** 0.5
        print(f"  {name:>14s} (dim={dim:5d}): mean={mean: .6f}  std={std: .6f}")

    total, used, free = shutil.disk_usage(str(args.output.parent))
    print()
    print(f"Disk space after saving — total: {total / 1e9:.1f} GB  "
          f"used: {used / 1e9:.1f} GB  free: {free / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
