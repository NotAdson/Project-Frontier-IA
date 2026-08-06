"""
Acceptance test for the fused_features autoencoder (issue #10).

Loads the best checkpoint saved by train_autoencoder.py, reconstructs the
EXACT validation split held out during training (same seed/fraction, read
back from the checkpoint's saved args — never re-derived by guessing), and
checks the formal acceptance criterion: aggregate reconstruction MSE < 0.01.
Also reports MSE broken down per feature-group (dense / each embedding),
reusing the same offsets already used to build the dataset in
generate_synthetic_dataset.py (PIECES), so this is purely a read-only check —
no new offset constants are introduced here.

Can run either as a pytest test:
    pytest src/battle_agents/mcts_approximation/pipeline/autoencoder/test_reconstruction.py
(skips cleanly if torch isn't installed or no checkpoint has been trained yet)

...or standalone, for an explicit non-zero exit code on failure:
    python src/battle_agents/mcts_approximation/pipeline/autoencoder/test_reconstruction.py
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from battle_agents.mcts_approximation.pipeline.autoencoder.generate_synthetic_dataset import (
    FUSED_DIM, PIECES, piece_offsets)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
# Mirrors train_autoencoder.py's DEFAULT_CHECKPOINT_DIR / DEFAULT_CHECKPOINT_NAME.
# Duplicated as plain path literals (not logic) so this file stays importable —
# and able to report a useful skip reason — even in environments without torch.
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "autoencoder_bootstrap" / "checkpoints" / "fused_autoencoder_best.pt"

ACCEPTANCE_THRESHOLD = 0.01  # formal issue #10 acceptance criterion: aggregate MSE < 0.01
# Second, independent criterion added after the smoke test showed the dense block (24.7% of
# dims) can carry ~89% of the aggregate squared error while the aggregate MSE still looks fine —
# the embeddings are an easy target and mask a bad dense reconstruction. Same numeric value as
# ACCEPTANCE_THRESHOLD by default, but checked and reported separately, and independently
# configurable via --dense-mse-threshold.
DEFAULT_DENSE_MSE_THRESHOLD = 0.01

try:
    import torch
    from torch.utils.data import DataLoader

    from battle_agents.mcts_approximation.pipeline.autoencoder.model import \
        FusedFeaturesAutoencoder
    from battle_agents.mcts_approximation.pipeline.autoencoder.train_autoencoder import (
        FusedFeaturesMemmapDataset, split_indices)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def evaluate(checkpoint_path, data_path=None, batch_size=4096, device=None) -> dict:
    """Runs the trained encoder+decoder over the held-out validation split and
    returns aggregate + per-piece MSE. Never touches the training split."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    ckpt_args = checkpoint["args"]
    data_path = Path(data_path) if data_path is not None else Path(ckpt_args["data_path"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FusedFeaturesAutoencoder(fused_dim=checkpoint["fused_dim"], latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    data = np.load(str(data_path), mmap_mode="r")
    file_rows = data.shape[0]
    del data

    # Use the exact row count the split was computed over at training time (respects
    # --max-rows), not the full file's row count — otherwise the permutation in
    # split_indices() would differ and we'd no longer be evaluating on the true held-out set.
    n_total = checkpoint["n_total"]
    assert n_total <= file_rows, (
        f"checkpoint expects {n_total} rows but {data_path} only has {file_rows}"
    )

    _, val_idx = split_indices(n_total, ckpt_args["val_fraction"], ckpt_args["seed"])
    val_ds = FusedFeaturesMemmapDataset(data_path, val_idx)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    offsets = piece_offsets()
    piece_sq_err = {name: 0.0 for name, _, _ in offsets}
    piece_count = {name: 0 for name, _, _ in offsets}
    agg_sq_err = 0.0
    agg_count = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            # model(batch) (== model.forward()/decode()) intentionally returns RAW
            # (unactivated) values at the binary dense positions — logits, not
            # probabilities in [0, 1] — because training feeds them straight into
            # nn.BCEWithLogitsLoss (see model.py's docstring and train_autoencoder.py's
            # SegmentedPieceLoss). The raw-MSE metric below is unchanged from before
            # (same ruler as v2/v3's 0.039518 / 0.037787), but to compare like-for-like
            # against the [0, 1]-scaled binary targets, sigmoid must be applied to those
            # positions here first — model.reconstruct() does exactly that.
            recon = model.reconstruct(batch)
            diff2 = (recon - batch) ** 2

            agg_sq_err += diff2.sum().item()
            agg_count += diff2.numel()

            for name, start, end in offsets:
                piece = diff2[:, start:end]
                piece_sq_err[name] += piece.sum().item()
                piece_count[name] += piece.numel()

    return {
        "aggregate_mse": agg_sq_err / agg_count,
        "piece_mse": {name: piece_sq_err[name] / piece_count[name] for name, _, _ in offsets},
        "n_val": len(val_idx),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_val_loss": checkpoint["val_loss"],
        # Weighted-loss config used at training time (train_autoencoder.py), if present —
        # older checkpoints saved before the weighted loss was added won't have this.
        "loss_weights": ckpt_args.get("loss_weights"),
    }


def check_acceptance(results: dict, aggregate_threshold: float, dense_mse_threshold: float) -> list:
    """Returns a list of human-readable failure messages (empty = both criteria passed)."""
    failures = []
    if results["aggregate_mse"] >= aggregate_threshold:
        failures.append(
            f"aggregate MSE {results['aggregate_mse']:.6f} >= acceptance threshold {aggregate_threshold} "
            f"(formal issue #10 criterion, raw/unweighted)"
        )
    dense_mse = results["piece_mse"]["dense"]
    if dense_mse >= dense_mse_threshold:
        failures.append(
            f"dense-block MSE {dense_mse:.6f} >= dense acceptance threshold {dense_mse_threshold} "
            f"(the tactical info block the MCTS actually consumes)"
        )
    return failures


def _print_report(results: dict):
    print(f"Checkpoint epoch: {results['checkpoint_epoch']}  "
          f"(saved val_loss: {results['checkpoint_val_loss']:.6f})")
    print(f"Validation rows evaluated: {results['n_val']:,}")
    if results["loss_weights"] is not None:
        print("Training loss weights used (see train_autoencoder.py --*-weight flags):")
        for name, w in results["loss_weights"].items():
            print(f"  {name:>14s}: {w:.6f}")
    else:
        print("Training loss weights: not recorded in this checkpoint (plain nn.MSELoss(), or an older checkpoint).")
    print(f"Aggregate MSE (raw, unweighted — formal issue #10 criterion): {results['aggregate_mse']:.6f}")
    print("Per-piece MSE (raw, unweighted):")
    for name, dim in PIECES:
        print(f"  {name:>14s} (dim={dim:5d}): mse={results['piece_mse'][name]:.6f}")


def test_reconstruction():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is not installed in this environment")
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip(f"no trained checkpoint at {DEFAULT_CHECKPOINT} yet — run train_autoencoder.py first")

    results = evaluate(DEFAULT_CHECKPOINT)
    _print_report(results)
    failures = check_acceptance(results, ACCEPTANCE_THRESHOLD, DEFAULT_DENSE_MSE_THRESHOLD)
    assert not failures, "; ".join(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-path", type=Path, default=None,
                         help="Defaults to the dataset path recorded in the checkpoint's args.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--aggregate-mse-threshold", type=float, default=ACCEPTANCE_THRESHOLD,
                         help="Formal issue #10 criterion: raw aggregate MSE must be below this.")
    parser.add_argument("--dense-mse-threshold", type=float, default=DEFAULT_DENSE_MSE_THRESHOLD,
                         help="Second, independent criterion: raw MSE of the dense block (0:758) alone "
                              "must also be below this — the tactical info the MCTS actually consumes.")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("[ERROR] torch is not installed in this environment.", file=sys.stderr)
        sys.exit(1)
    if not args.checkpoint.exists():
        print(f"[ERROR] no checkpoint found at {args.checkpoint}. Run train_autoencoder.py first.", file=sys.stderr)
        sys.exit(1)

    results = evaluate(args.checkpoint, data_path=args.data_path, batch_size=args.batch_size)
    _print_report(results)

    failures = check_acceptance(results, args.aggregate_mse_threshold, args.dense_mse_threshold)
    if failures:
        print("\nFAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print(f"\nPASS: aggregate MSE {results['aggregate_mse']:.6f} < {args.aggregate_mse_threshold}  "
          f"AND dense MSE {results['piece_mse']['dense']:.6f} < {args.dense_mse_threshold}")


if __name__ == "__main__":
    main()
