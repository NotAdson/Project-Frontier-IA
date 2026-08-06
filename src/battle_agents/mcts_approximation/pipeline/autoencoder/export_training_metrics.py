"""
Packages a finished autoencoder training run's per-epoch metrics for handoff
(issue #10; the deliverable filenames below use a "_v6" suffix for this
handoff round, referring to the no-meta_plan fused_features layout -- see
generate_synthetic_dataset.py's FUSED_DIM -- not to any checkpoint directory
name, which is unrelated).

Reads --checkpoint-dir/training_metrics.csv (written incrementally by
train_autoencoder.py, one row per epoch) plus the checkpoint's own saved args,
and writes three files into --output-dir:
  1. training_metrics_v6.csv  -- the same rows, re-serialized with a clean header.
  2. training_metrics_v6.json -- the same rows, typed (int/float/bool, not strings).
  3. summary_v6.txt           -- total epochs, best epoch/val_loss, whether the run
                                  stopped early or hit its epoch budget, total training
                                  time in hours, and the run's config (latent_dim,
                                  dense_weight, dense_bce_ratio).

Refuses to run until --checkpoint-dir/TRAINING_COMPLETE.marker exists. Both
train_autoencoder.py (at the end of its own run) and pipeline_bootstrap.py
(right after driving train_autoencoder.py as one of its stages) write this
marker -- either write is enough, both are idempotent -- see
train_autoencoder.write_completion_marker()'s docstring. A checkpoint
mid-training has a partial CSV and no final "best epoch" to report.

"Stopped early vs. hit its epoch budget" is read off the LAST row only (its epoch vs.
its epoch_total): train_autoencoder.py's --epochs is "N more epochs" on resume, so a
resumed run's earlier rows can carry a different epoch_total than its later ones --
only the final segment's outcome is the run's actual outcome.

Usage:
    python src/battle_agents/mcts_approximation/pipeline/autoencoder/export_training_metrics.py \\
        [--checkpoint-dir data/autoencoder_bootstrap/checkpoints_v5_fixed256] \\
        [--output-dir data/autoencoder_bootstrap/entrega_matheus]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from battle_agents.mcts_approximation.pipeline.autoencoder.train_autoencoder import (
    DEFAULT_CHECKPOINT_NAME, METRICS_CSV_COLUMNS, METRICS_CSV_FILENAME,
    TRAINING_COMPLETE_MARKER_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[5]

# Mirrors pipeline_bootstrap.py's DEFAULT_CHECKPOINT_PATH.parent (currently
# "checkpoints_v5_fixed256" -- that name predates the meta_plan removal but the
# directory itself is reused in place by the fused_dim compatibility check in
# ensure_autoencoder_ready(), not renamed), duplicated as a plain literal (not an
# import) so this script doesn't pull in generate_data.py's self-play/ShowdownClient
# dependency chain just to package already-finished metrics.
# TRAINING_COMPLETE_MARKER_NAME (imported above) doesn't need the same treatment -- it
# already comes from train_autoencoder.py, which this script imports anyway.
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "data" / "autoencoder_bootstrap" / "checkpoints_v5_fixed256"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "autoencoder_bootstrap" / "entrega_matheus"

OUTPUT_CSV_NAME = "training_metrics_v6.csv"
OUTPUT_JSON_NAME = "training_metrics_v6.json"
OUTPUT_SUMMARY_NAME = "summary_v6.txt"


def read_raw_rows(csv_path: Path) -> list:
    """Rows as written by train_autoencoder.py, values still strings -- passed
    through verbatim into the output CSV so re-serializing can't introduce a
    float-formatting mismatch against what was actually logged during training."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != METRICS_CSV_COLUMNS:
            raise SystemExit(
                f"{csv_path} has columns {reader.fieldnames}, expected {METRICS_CSV_COLUMNS} "
                "-- was this file written by a different version of train_autoencoder.py?"
            )
        return list(reader)


def typed_row(raw: dict) -> dict:
    """Same row, cast to the types a plotting library actually wants."""
    return {
        "epoch": int(raw["epoch"]),
        "epoch_total": int(raw["epoch_total"]),
        "train_loss": float(raw["train_loss"]),
        "val_loss": float(raw["val_loss"]),
        "is_best": raw["is_best"] == "True",
        "time_seconds": float(raw["time_seconds"]),
    }


def build_summary_text(typed_rows: list, checkpoint: dict) -> str:
    last = typed_rows[-1]
    total_epochs = last["epoch"]
    stopped_early = last["epoch"] < last["epoch_total"]
    total_seconds = sum(r["time_seconds"] for r in typed_rows)

    ckpt_args = checkpoint["args"]
    lines = [
        f"Total de epocas rodadas: {total_epochs}",
        f"Melhor epoca: {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.6f})",
        f"Motivo da parada: {'early stopping' if stopped_early else 'teto de epocas (epoch_total atingido)'}",
        f"Tempo total de treino: {total_seconds / 3600:.2f} horas ({total_seconds:.1f} segundos)",
        "",
        "Configuracao:",
        f"  latent_dim: {ckpt_args.get('latent_dim')}",
        f"  dense_weight: {ckpt_args.get('dense_weight')}",
        f"  dense_bce_ratio: {ckpt_args.get('dense_bce_ratio')}",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    marker_path = args.checkpoint_dir / TRAINING_COMPLETE_MARKER_NAME
    if not marker_path.exists():
        raise SystemExit(
            f"{marker_path} not found -- training isn't finished yet (or was interrupted). "
            "Refusing to package metrics from a run that never completed."
        )

    csv_path = args.checkpoint_dir / METRICS_CSV_FILENAME
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found.")
    raw_rows = read_raw_rows(csv_path)
    if not raw_rows:
        raise SystemExit(f"{csv_path} has a header but no epoch rows.")
    typed_rows = [typed_row(r) for r in raw_rows]

    checkpoint_path = args.checkpoint_dir / DEFAULT_CHECKPOINT_NAME
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_out = args.output_dir / OUTPUT_CSV_NAME
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(raw_rows)

    json_out = args.output_dir / OUTPUT_JSON_NAME
    with open(json_out, "w") as f:
        json.dump(typed_rows, f, indent=2)

    summary_out = args.output_dir / OUTPUT_SUMMARY_NAME
    summary_out.write_text(build_summary_text(typed_rows, checkpoint))

    print(f"Wrote {csv_out} ({len(raw_rows)} rows)")
    print(f"Wrote {json_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
