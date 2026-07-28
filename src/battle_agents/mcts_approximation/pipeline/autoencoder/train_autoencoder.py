"""
Trains FusedFeaturesAutoencoder (model.py) on the synthetic fused_features
dataset produced by generate_synthetic_dataset.py (issue #10).

Loads data/autoencoder_bootstrap/fused_features_synthetic.npy via mmap
(np.load(..., mmap_mode='r')) so the 24.59GB array is never fully materialized
in RAM — a custom Dataset indexes into the memmap on demand, per batch.

The reconstruction loss is a weighted sum of per-piece error (dense / each
embedding), not a single flat nn.MSELoss() over all 3062 dims —
see compute_piece_weights() / SegmentedPieceLoss below for why and the exact
formula. A smoke test on plain flat MSE showed the dense block (24.7% of the
dims, where the real tactical info lives) can carry ~89% of the aggregate
squared error while the 4 embeddings (75% of the dims, but a narrow near-zero
synthetic target) mask it by being easy to fit — so training on flat MSE
mostly teaches the model to reconstruct embeddings well.

Within the dense piece specifically, the loss is further split by feature
type: continuous positions (hp_ratio, stats, level, move power/accuracy,
boosts, field, PP) use MSE; binary positions (fainted, statuses, is_active,
types, move category flags — see model.py's compute_dense_binary_mask())
use BCEWithLogitsLoss instead, since MSE is a poor fit for 0/1 targets. See
SegmentedPieceLoss for the exact combination.

Besides the per-epoch print, every epoch also appends a row to
--checkpoint-dir/training_metrics.csv (see METRICS_CSV_COLUMNS below) so a
training run's convergence can be plotted afterward without re-parsing
stdout. export_training_metrics.py packages this CSV (plus the checkpoint's
saved args) into a delivery folder once training finishes.

Usage:
    python src/battle_agents/mcts_approximation/pipeline/autoencoder/train_autoencoder.py \
        [--data-path data/autoencoder_bootstrap/fused_features_synthetic.npy] \
        [--checkpoint-dir data/autoencoder_bootstrap/checkpoints] \
        [--lr 1e-3] [--batch-size 4096] [--epochs 100] [--patience 5] \
        [--val-fraction 0.2] [--seed 123] [--num-workers 4] \
        [--dense-weight 70.0] [--species-weight 1.0] [--moves-weight 1.0] \
        [--items-weight 1.0] [--abilities-weight 1.0] \
        [--resume-from-checkpoint data/autoencoder_bootstrap/checkpoints/fused_autoencoder_best.pt]

Resuming: --resume-from-checkpoint loads model AND optimizer state (real Adam moment
estimates, not just weights) and continues epoch numbering (--epochs means "N more
epochs" on resume, not a total budget). Only works on checkpoints saved by a version of
this script that already includes "optimizer_state_dict" — older checkpoints (e.g. ones
saved before this flag existed) raise an explicit error instead of silently resuming with
a freshly-reset optimizer, which would be misleading. --seed/--val-fraction/--max-rows
must match the checkpoint's saved args, or the resume is refused (different args would
silently reproduce a different train/val split).
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from battle_agents.mcts_approximation.pipeline.autoencoder.generate_synthetic_dataset import (
    PIECES, piece_offsets)
from battle_agents.mcts_approximation.pipeline.autoencoder.model import (
    FUSED_DIM, LATENT_DIM, FusedFeaturesAutoencoder)
from battle_agents.mcts_approximation.state_encoder import NUM_DENSE_FEATURES

# Default multiplier per piece, applied ON TOP of the inverse-dim-normalized base weight
# (see compute_piece_weights()). Only dense is boosted by default — it's the block the
# real MCTS state actually needs; the others default to 1.0 (their plain base weight).
#
# dense=8.0 was the original default and a v1 smoke test (flat, unweighted MSELoss) showed
# dense carrying ~89% of the aggregate error while contributing only 24.7% of the dims. A v2
# The bootstrap keeps the validated v5 dense multiplier of 70.0.
DEFAULT_PIECE_MULTIPLIERS = {
    "dense": 70.0,
    "emb_species": 1.0,
    "emb_moves": 1.0,
    "emb_items": 1.0,
    "emb_abilities": 1.0,
}


def compute_piece_weights(multipliers: dict) -> dict:
    """
    weight[piece] = multiplier[piece] * (1 / dim[piece]) / sum_p(1 / dim[p])

    i.e. base weights are inversely proportional to each piece's own dimension
    count and normalized to sum to 1 — so, before any multiplier, every piece
    would contribute EQUALLY to the loss on average, regardless of how many
    raw numbers it has (a 1536-dim piece doesn't automatically dominate a
    12-dim piece just by having more numbers). Each piece's own configurable
    multiplier (--dense-weight etc.) then scales that equal share up or down;
    the CLI default only boosts "dense" (8x), everything else stays at its
    plain inverse-dim base weight.
    """
    inv_dim = {name: 1.0 / dim for name, dim in PIECES}
    total_inv_dim = sum(inv_dim.values())
    base = {name: inv_dim[name] / total_inv_dim for name in inv_dim}
    return {name: base[name] * multipliers[name] for name in base}


class WeightedPieceMSELoss(nn.Module):
    """loss = sum_piece( weight[piece] * mean((recon_piece - target_piece) ** 2) ),
    slicing fused_features with the same offsets used everywhere else (piece_offsets(),
    from generate_synthetic_dataset.py — single source of truth, not redefined here).

    Superseded as the default training loss by SegmentedPieceLoss (below), which treats
    binary dense features with BCE instead of MSE. Kept here for reference/comparison;
    --dense-only mode does NOT use this either (it uses plain nn.MSELoss(), see main()) —
    this class is currently unused by main() but left in place."""

    def __init__(self, weights: dict):
        super().__init__()
        offsets = piece_offsets()
        self.offsets = [(start, end) for _, start, end in offsets]
        self.weights = [weights[name] for name, _, _ in offsets]

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = recon.new_zeros(())
        for (start, end), w in zip(self.offsets, self.weights):
            piece_mse = torch.mean((recon[:, start:end] - target[:, start:end]) ** 2)
            total = total + w * piece_mse
        return total


class SegmentedPieceLoss(nn.Module):
    """Like WeightedPieceMSELoss, but the "dense" piece (the 758-dim block, see
    state_encoder.py) is itself split by feature type instead of treated as one
    uniform MSE target:
      - continuous dense positions (hp_ratio, level, stats, move power/accuracy,
        boosts, field, PP): plain MSE, as before.
      - binary dense positions (fainted, statuses, is_active, types, move
        category flags — see model.py's compute_dense_binary_mask()): BCE via
        nn.BCEWithLogitsLoss. The model's decoder output is RAW (unactivated)
        at these positions by design (see model.py's docstring) specifically so
        it can be fed to BCEWithLogitsLoss directly, without applying sigmoid
        here first (BCEWithLogitsLoss applies it internally, in a numerically
        stable way).
    The two dense sub-losses (continuous MSE, binary BCE) are combined as
    `cont_mse + bce_ratio * bin_bce` into a single "dense" term, and
    --dense-weight (or whichever multiplier compute_piece_weights() assigns
    to "dense") is then applied to that combined term, exactly like it was
    applied to the single dense MSE term before. bce_ratio exists because MSE
    and BCE are on different numeric scales (BCE's floor for a ~50/50 binary
    feature is around 0.69 "nats", vs. the MSE plateau already observed in
    smoke tests, ~0.03-0.08) — summed unweighted, BCE would dominate the
    combined dense term's magnitude and gradient just by having a higher
    floor, not because it's a harder target. --dense-bce-ratio's default
    (0.1) brings BCE's floor down to ~0.0693, the same order of magnitude as
    the MSE floor.
    Every other piece (the 4 embeddings) is unchanged: plain MSE.
    """

    def __init__(self, weights: dict, binary_mask: torch.Tensor, bce_ratio: float = 0.1):
        super().__init__()
        offsets = piece_offsets()
        self.names = [name for name, _, _ in offsets]
        self.offsets = [(start, end) for _, start, end in offsets]
        self.weights = [weights[name] for name in self.names]
        self.bce_ratio = bce_ratio
        self.register_buffer("binary_mask", binary_mask)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = recon.new_zeros(())
        for name, (start, end), w in zip(self.names, self.offsets, self.weights):
            recon_piece = recon[:, start:end]
            target_piece = target[:, start:end]
            if name == "dense":
                bin_mask = self.binary_mask[start:end]
                cont_mask = ~bin_mask
                cont_mse = torch.mean((recon_piece[:, cont_mask] - target_piece[:, cont_mask]) ** 2)
                bin_bce = self.bce(recon_piece[:, bin_mask], target_piece[:, bin_mask])
                piece_loss = cont_mse + self.bce_ratio * bin_bce
            else:
                piece_loss = torch.mean((recon_piece - target_piece) ** 2)
            total = total + w * piece_loss
        return total


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "autoencoder_bootstrap" / "fused_features_synthetic.npy"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "data" / "autoencoder_bootstrap" / "checkpoints"
DEFAULT_CHECKPOINT_NAME = "fused_autoencoder_best.pt"

# Per-epoch metrics, written incrementally (appended, not batched) to --checkpoint-dir
# so a crash mid-training still leaves every completed epoch on disk. export_training_metrics.py
# reads this back once training finishes -- single source of truth for the column order.
METRICS_CSV_FILENAME = "training_metrics.csv"
METRICS_CSV_COLUMNS = ["epoch", "epoch_total", "train_loss", "val_loss", "is_best", "time_seconds"]

# Written to --checkpoint-dir when main()'s epoch loop returns normally (full epoch
# budget or early stopping, not a crash/interrupt) -- see write_completion_marker().
# pipeline_bootstrap.py writes the SAME marker again, at the same path, right after
# it calls this script as one of its own stages (ensure_autoencoder_ready() needs its
# own record of completion even when it's driving several stages end to end, not just
# this one). Both call sites share this one function so there's a single place that
# defines what "training complete" means and how the marker is written.
TRAINING_COMPLETE_MARKER_NAME = "TRAINING_COMPLETE.marker"


def write_completion_marker(checkpoint_path: Path, marker_path: Path) -> None:
    """Reads the epoch/val_loss actually saved in the checkpoint (not a value carried
    over from the training loop in memory) so the marker reflects what's really on disk.
    Idempotent by construction: Path.write_text() always overwrites, so calling this a
    second time (e.g. once from here, once more from pipeline_bootstrap.py right after)
    never fails just because the marker already exists -- it's the same content anyway,
    since both calls read it from the same checkpoint file."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    marker_path.write_text(json.dumps({
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": ckpt["epoch"],
        "val_loss": ckpt["val_loss"],
    }, indent=2))


DEFAULT_SEED = 123  # controls the train/val split only (independent of the dataset-generation seeds)
DEFAULT_VAL_FRACTION = 0.2


def split_indices(n_total: int, val_fraction: float = DEFAULT_VAL_FRACTION, seed: int = DEFAULT_SEED):
    """Deterministic 80/20 (by default) train/val split over row indices.
    Shared by train_autoencoder.py and test_reconstruction.py so the acceptance
    test evaluates on exactly the rows the model never trained on."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_val = int(round(n_total * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


class FusedFeaturesMemmapDataset(Dataset):
    """Indexes a subset of rows of a (N, FUSED_DIM) float32 .npy file, opened
    lazily (per worker process) via mmap so nothing is loaded eagerly.
    n_cols: optionally truncate each row to its first n_cols columns (used by
    --dense-only to isolate the dense block without a separate dataset file)."""

    def __init__(self, npy_path, indices: np.ndarray, n_cols: int = None):
        self.npy_path = str(npy_path)
        self.indices = indices
        self.n_cols = n_cols
        self._data = None

    def _ensure_open(self):
        if self._data is None:
            self._data = np.load(self.npy_path, mmap_mode="r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._ensure_open()
        raw_row = self._data[self.indices[i]]
        if self.n_cols is not None:
            raw_row = raw_row[:self.n_cols]
        # np.array(..., copy=True) instead of asarray: the mmap slice is read-only,
        # and torch.from_numpy requires a writable buffer (a view would just warn).
        row = np.array(raw_row, dtype=np.float32, copy=True)
        return torch.from_numpy(row)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """optimizer=None -> eval mode, no gradient updates. Returns mean loss over all rows."""
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = 0.0
    n_seen = 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            recon = model(batch)
            loss = criterion(recon, batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * batch.size(0)
            n_seen += batch.size(0)

    return loss_sum / n_seen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=100,
                         help="Max epochs; early stopping usually ends training sooner. With "
                              "--resume-from-checkpoint, this is epochs to run ADDITIONALLY on top of "
                              "the checkpoint's saved epoch, not a total budget (checkpoint stopped at "
                              "epoch 33 + --epochs 20 -> runs epoch 34..53).")
    parser.add_argument("--patience", type=int, default=5,
                         help="Early-stopping patience. With --resume-from-checkpoint, the patience "
                              "counter always restarts at 0 (a fresh budget) — it is NOT a continuation "
                              "of however many non-improving epochs had already elapsed before the "
                              "original run stopped/was interrupted, since that count isn't saved.")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None,
                         help="Debug/smoke-test aid: only use the first N rows of the dataset "
                              "(the dataset rows are already reservoir-sampled, so the first N "
                              "are still a valid random subset, not just 'the earliest games').")
    parser.add_argument("--dense-weight", type=float, default=DEFAULT_PIECE_MULTIPLIERS["dense"],
                         help="Multiplier on top of dense's inverse-dim base weight (see compute_piece_weights).")
    parser.add_argument("--species-weight", type=float, default=DEFAULT_PIECE_MULTIPLIERS["emb_species"])
    parser.add_argument("--moves-weight", type=float, default=DEFAULT_PIECE_MULTIPLIERS["emb_moves"])
    parser.add_argument("--items-weight", type=float, default=DEFAULT_PIECE_MULTIPLIERS["emb_items"])
    parser.add_argument("--abilities-weight", type=float, default=DEFAULT_PIECE_MULTIPLIERS["emb_abilities"])
    parser.add_argument("--dense-bce-ratio", type=float, default=0.1,
                         help="Multiplier on the binary-features BCE term within the dense piece, "
                              "combined as cont_mse + dense_bce_ratio * bin_bce (see SegmentedPieceLoss). "
                              "Default 0.1 brings BCE's ~0.693 floor (ln 2, for a ~50/50 binary feature) "
                              "down to ~0.0693 — the same order of magnitude as the ~0.03-0.08 MSE floor "
                              "already observed in smoke tests, so BCE doesn't dominate the combined "
                              "dense term's gradient just by having a higher floor. Ignored when "
                              "--dense-only is set (that mode uses plain nn.MSELoss()).")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM,
                         help="Overrides model.py's default 64-dim bottleneck (e.g. to test "
                              "whether a dense-MSE plateau is a capacity limit, not a loss-weighting one).")
    parser.add_argument("--dense-only", action="store_true",
                         help="Diagnostic mode: train on ONLY the first NUM_DENSE_FEATURES (758) "
                              "columns of the dataset, with a 758-dim (not 3062-dim) autoencoder and "
                              "plain nn.MSELoss() — isolates whether the dense block's MSE plateau is "
                              "caused by competing for gradient with the embeddings, or is "
                              "inherent to the dense data itself. Ignores all --*-weight flags.")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None,
                         help="Path to a .pt checkpoint saved by THIS script to resume from — loads "
                              "model AND optimizer state (Adam moment estimates), continues epoch "
                              "numbering, and seeds best_val_loss from the checkpoint. Requires the "
                              "checkpoint to have been saved with optimizer_state_dict (i.e. saved by "
                              "a version of this script that already has this flag) — older checkpoints "
                              "saved without it cannot be resumed with real Adam state and will raise "
                              "an error rather than silently resuming with a freshly-reset optimizer. "
                              "Also requires --seed/--val-fraction/--max-rows to match the checkpoint's "
                              "saved args, so the train/val split is reproduced exactly — mismatches abort.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(str(args.data_path), mmap_mode="r")
    n_total, fused_dim = data.shape
    assert fused_dim == FUSED_DIM, f"Dataset dim {fused_dim} != model FUSED_DIM {FUSED_DIM}"
    del data  # just used to read shape; the Dataset re-opens its own handle lazily

    if args.max_rows is not None:
        n_total = min(n_total, args.max_rows)
        print(f"--max-rows {args.max_rows}: limiting to the first {n_total:,} rows.")

    train_idx, val_idx = split_indices(n_total, args.val_fraction, args.seed)
    print(f"Rows: {n_total:,}  ->  train: {len(train_idx):,}  val: {len(val_idx):,}  "
          f"(val_fraction={args.val_fraction}, split seed={args.seed})")

    effective_fused_dim = NUM_DENSE_FEATURES if args.dense_only else FUSED_DIM
    dataset_n_cols = NUM_DENSE_FEATURES if args.dense_only else None
    if args.dense_only:
        print(f"--dense-only: training on just the first {NUM_DENSE_FEATURES} columns "
              f"(no embeddings), plain nn.MSELoss().")

    train_ds = FusedFeaturesMemmapDataset(args.data_path, train_idx, n_cols=dataset_n_cols)
    val_ds = FusedFeaturesMemmapDataset(args.data_path, val_idx, n_cols=dataset_n_cols)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = FusedFeaturesAutoencoder(fused_dim=effective_fused_dim, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    resume_start_epoch = 1
    best_val_loss = float("inf")
    if args.resume_from_checkpoint is not None:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        resume_ckpt = torch.load(str(args.resume_from_checkpoint), map_location=device)
        resume_ckpt_args = resume_ckpt["args"]

        # Refuse to silently reproduce a DIFFERENT train/val split than the one this
        # checkpoint was actually trained on.
        mismatches = []
        if resume_ckpt_args["seed"] != args.seed:
            mismatches.append(f"seed: checkpoint={resume_ckpt_args['seed']!r} vs --seed={args.seed!r}")
        if resume_ckpt_args["val_fraction"] != args.val_fraction:
            mismatches.append(
                f"val_fraction: checkpoint={resume_ckpt_args['val_fraction']!r} vs --val-fraction={args.val_fraction!r}"
            )
        if resume_ckpt_args["max_rows"] != args.max_rows:
            mismatches.append(f"max_rows: checkpoint={resume_ckpt_args['max_rows']!r} vs --max-rows={args.max_rows!r}")
        if mismatches:
            raise SystemExit(
                "Refusing to resume: the following args differ from the checkpoint's, which would "
                "silently reproduce a DIFFERENT train/val split than the one this checkpoint was "
                "trained on:\n  " + "\n  ".join(mismatches) +
                "\nPass matching --seed/--val-fraction/--max-rows, or start a fresh run instead."
            )

        if "optimizer_state_dict" not in resume_ckpt:
            raise SystemExit(
                f"Refusing to resume: {args.resume_from_checkpoint} has no 'optimizer_state_dict' "
                "(it was saved by an older version of this script, before optimizer state was added "
                "to the checkpoint). Resuming without the real Adam state (moment estimates) would "
                "silently restart Adam from scratch while keeping only the model weights — likely "
                "WORSE than either accepting this checkpoint's current result as final, or starting "
                "a brand new run with more epochs. Not doing that silently — pick one of those instead."
            )

        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        best_val_loss = resume_ckpt["val_loss"]
        resume_start_epoch = resume_ckpt["epoch"] + 1
        print(f"Resumed from epoch {resume_ckpt['epoch']} (val_loss={best_val_loss:.6f}). "
              f"patience counter reset to 0 (fresh budget, see --patience help).")

    if args.dense_only:
        loss_weights = None
        criterion = nn.MSELoss()
    else:
        multipliers = {
            "dense": args.dense_weight,
            "emb_species": args.species_weight,
            "emb_moves": args.moves_weight,
            "emb_items": args.items_weight,
            "emb_abilities": args.abilities_weight,
        }
        loss_weights = compute_piece_weights(multipliers)
        criterion = SegmentedPieceLoss(loss_weights, model.binary_mask, bce_ratio=args.dense_bce_ratio).to(device)
        print("Per-piece loss weights (base 1/dim, normalized, times each --*-weight multiplier):")
        print(f"(dense piece = continuous MSE + {args.dense_bce_ratio} * binary BCEWithLogitsLoss — see SegmentedPieceLoss)")
        for name, dim in PIECES:
            print(f"  {name:>14s} (dim={dim:5d}): multiplier={multipliers[name]:.3f}  "
                  f"final_weight={loss_weights[name]:.6f}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / DEFAULT_CHECKPOINT_NAME

    metrics_csv_path = args.checkpoint_dir / METRICS_CSV_FILENAME
    if not metrics_csv_path.exists():
        with open(metrics_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(METRICS_CSV_COLUMNS)

    patience_counter = 0
    end_epoch = resume_start_epoch + args.epochs - 1

    print(f"Training: lr={args.lr}  batch_size={args.batch_size}  "
          f"epochs={resume_start_epoch}..{end_epoch}  patience={args.patience}")

    for epoch in range(resume_start_epoch, end_epoch + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        elapsed = time.time() - t0

        improved = val_loss < best_val_loss
        print(f"epoch {epoch:3d}/{end_epoch}  train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  {'(best)' if improved else ''}  {elapsed:.1f}s")

        # Appended immediately (not batched at the end) so a crash mid-training still
        # leaves every completed epoch's row on disk for export_training_metrics.py.
        with open(metrics_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, end_epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", improved, f"{elapsed:.3f}",
            ])

        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            # Path objects aren't allowed under torch>=2.6's default weights_only=True
            # load; stringify them so the checkpoint stays loadable without opting out
            # of that safety check.
            json_safe_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
            if loss_weights is not None:
                json_safe_args["loss_weights"] = loss_weights  # final per-piece weights actually used (see above)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "fused_dim": effective_fused_dim,
                "latent_dim": args.latent_dim,
                "epoch": epoch,
                "val_loss": val_loss,
                "n_total": n_total,  # exact row count the split was computed over (respects --max-rows)
                "args": json_safe_args,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val_loss improvement for {args.patience} epochs).")
                break

    print(f"Best val_loss: {best_val_loss:.6f}  |  checkpoint saved to {checkpoint_path}")

    # Written unconditionally on normal completion (full epoch budget or early stopping
    # above), whether this script is run standalone or as a stage inside
    # pipeline_bootstrap.py's ensure_autoencoder_ready() -- see write_completion_marker()'s
    # docstring for why a second write from there afterward is safe.
    marker_path = args.checkpoint_dir / TRAINING_COMPLETE_MARKER_NAME
    write_completion_marker(checkpoint_path, marker_path)
    print(f"Training-complete marker written to {marker_path}")


if __name__ == "__main__":
    main()
