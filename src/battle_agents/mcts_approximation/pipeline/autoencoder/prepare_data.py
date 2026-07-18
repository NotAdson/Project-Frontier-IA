"""
Pre-processing step for the state-autoencoder (issue #10).

Reads self-play game files produced by generate_data.py (each a JSON list of
steps with a "features" vector of TOTAL_FEATURES=842 floats), keeps only the
dense block [0:NUM_DENSE_FEATURES], and slices it into 12 per-Pokemon blocks
of PER_MON_DENSE floats each (6 own team + 6 opponent team). The result is a
single consolidated (N, PER_MON_DENSE) float32 .npy array, meant to become the
training set for the autoencoder. This script does NOT build or train the
autoencoder itself.

Usage:
    python src/battle_agents/mcts_approximation/pipeline/autoencoder/prepare_data.py \
        [--data-dir data/genrandom_bootstrap] \
        [--output data/autoencoder_bootstrap/dense_pokemon_slices.npy] \
        [--max-games N]
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from battle_agents.mcts_approximation.state_encoder import (
    NUM_BENCH, NUM_DENSE_FEATURES, NUM_MOVES, OPP_TEAM_DENSE, OPP_TEAM_START,
    OWN_TEAM_DENSE, PER_MON_DENSE, STATUSES, TYPES)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "genrandom_bootstrap"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "autoencoder_bootstrap" / "dense_pokemon_slices.npy"

# Human-readable label per column, in the same order _encode_own_team lays
# out a PER_MON_DENSE block (hp, fainted, statuses, is_active, level, stats,
# types, moves). Built from the real encoder constants, not re-typed by hand.
COLUMN_LABELS = (
    ["hp_ratio", "fainted"]
    + [f"status_{s}" for s in STATUSES]
    + ["is_active", "level"]
    + ["stat_atk", "stat_def", "stat_spa", "stat_spd", "stat_spe"]
    + [f"type_{t}" for t in TYPES]
)
for _m in range(NUM_MOVES):
    COLUMN_LABELS += [
        f"move{_m}_power", f"move{_m}_accuracy",
        f"move{_m}_is_physical", f"move{_m}_is_special", f"move{_m}_is_status",
    ]
assert len(COLUMN_LABELS) == PER_MON_DENSE, (len(COLUMN_LABELS), PER_MON_DENSE)


def extract_slices(dense: list) -> np.ndarray:
    """Splits a NUM_DENSE_FEATURES-long dense vector into 12 (PER_MON_DENSE,) blocks.

    Returns shape (12, PER_MON_DENSE): rows 0-5 are the own team
    (offset i * PER_MON_DENSE, i.e. OWN_TEAM_DENSE worth of data starting at 0),
    rows 6-11 are the opponent team (starting at OPP_TEAM_START).
    """
    arr = np.asarray(dense, dtype=np.float32)
    own = arr[:OWN_TEAM_DENSE].reshape(NUM_BENCH, PER_MON_DENSE)
    opp = arr[OPP_TEAM_START:OPP_TEAM_START + OPP_TEAM_DENSE].reshape(NUM_BENCH, PER_MON_DENSE)
    return np.concatenate([own, opp], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-games", type=int, default=None,
                         help="Optional cap on number of game files processed (for a quick dry run).")
    args = parser.parse_args()

    files = sorted(glob.glob(str(args.data_dir / "game_*.json")))
    if not files:
        raise SystemExit(f"No game_*.json files found in {args.data_dir}")
    if args.max_games is not None:
        files = files[:args.max_games]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".raw.tmp")

    total_rows = 0
    sum_ = np.zeros(PER_MON_DENSE, dtype=np.float64)
    sumsq = np.zeros(PER_MON_DENSE, dtype=np.float64)

    print(f"Scanning {len(files)} game files from {args.data_dir} ...")
    t0 = time.time()
    with open(tmp_path, "wb") as tmp_f:
        for fi, fpath in enumerate(files):
            with open(fpath, "r") as fh:
                game_data = json.load(fh)
            if not game_data:
                continue

            per_step_slices = []
            for step in game_data:
                dense = step["features"][:NUM_DENSE_FEATURES]
                per_step_slices.append(extract_slices(dense))
            batch = np.concatenate(per_step_slices, axis=0)  # (n_steps_in_file * 12, PER_MON_DENSE)

            tmp_f.write(batch.tobytes())
            sum_ += batch.sum(axis=0, dtype=np.float64)
            sumsq += np.square(batch, dtype=np.float64).sum(axis=0)
            total_rows += batch.shape[0]

            if (fi + 1) % 5000 == 0 or (fi + 1) == len(files):
                elapsed = time.time() - t0
                print(f"  ... {fi + 1}/{len(files)} files scanned, "
                      f"{total_rows:,} slices so far, {elapsed:.0f}s elapsed")

    scan_elapsed = time.time() - t0
    print(f"Finished scanning in {scan_elapsed:.1f}s -> {total_rows:,} total (Pokemon, dense) slices.")

    if total_rows == 0:
        os.remove(tmp_path)
        raise SystemExit("No steps found across the scanned game files — nothing to save.")

    # Consolidate the raw scratch file into a single proper .npy of known shape,
    # copying in bounded chunks so we never hold the full array in RAM.
    print(f"Consolidating into {args.output} (shape ({total_rows:,}, {PER_MON_DENSE})) ...")
    final = np.lib.format.open_memmap(
        str(args.output), mode="w+", dtype=np.float32, shape=(total_rows, PER_MON_DENSE)
    )
    chunk_rows = 2_000_000
    row_bytes = PER_MON_DENSE * 4
    with open(tmp_path, "rb") as tmp_f:
        offset = 0
        while offset < total_rows:
            n = min(chunk_rows, total_rows - offset)
            raw = tmp_f.read(n * row_bytes)
            chunk = np.frombuffer(raw, dtype=np.float32).reshape(n, PER_MON_DENSE)
            final[offset:offset + n] = chunk
            offset += n
    final.flush()
    del final
    os.remove(tmp_path)

    mean = sum_ / total_rows
    var = np.maximum(sumsq / total_rows - mean ** 2, 0.0)
    std = np.sqrt(var)

    out_size_bytes = os.path.getsize(args.output)
    print()
    print(f"Saved {args.output}")
    print(f"  shape: ({total_rows:,}, {PER_MON_DENSE})  dtype: float32  "
          f"size on disk: {out_size_bytes / 1e9:.2f} GB")
    print()
    print(f"Per-column mean / std ({PER_MON_DENSE} columns, PER_MON_DENSE layout):")
    for i in range(PER_MON_DENSE):
        print(f"  col {i:2d} [{COLUMN_LABELS[i]:>18s}]: mean={mean[i]: .6f}  std={std[i]: .6f}")


if __name__ == "__main__":
    main()
