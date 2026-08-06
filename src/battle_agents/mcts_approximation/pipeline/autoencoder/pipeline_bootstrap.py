"""
Bootstraps the frozen fused_features autoencoder checkpoint (issue #10, PROPOSTA_10.md)
that the training model's build_models() loads. Meant to run once, before the AlphaZero
generational loop starts (see run_pipeline.py) -- self-play data generation and NN
training both depend on the same feature encoder, so it must exist before generation 1.

Runs the three stages documented in PROPOSTA_10.md by calling their Python entry
points directly (in-process, not via subprocess):
  1. generate_dataset(agent_type="random") (generate_data.py) -> data/genrandom_bootstrap/
     (PROPOSTA_10.md section 2.1: 49,999 games).
  2. generate_synthetic_dataset.main() -> data/autoencoder_bootstrap/fused_features_synthetic.npy
     (PROPOSTA_10.md section 2.2: 2,000,000 rows, default seeds/sizes).
  3. train_autoencoder.main() with the v5 hyperparameters (PROPOSTA_10.md section 7.5:
     latent_dim=256, dense-weight=70, dense-bce-ratio=0.1, epochs=140, patience=5) ->
     the checkpoint itself.

Each of the first two stages' own artifact (game files, the .npy) is its resumability
marker: a stage that already produced its output is skipped, so re-running after an
interrupted bootstrap does not redo completed work.

The training stage is different on purpose: train_autoencoder.py overwrites the .pt
checkpoint on every val_loss improvement, not just at the end, so the checkpoint file
alone can't tell a finished run from one interrupted mid-training. I gate on a separate
TRAINING_COMPLETE.marker file, written only after train_autoencoder.py's epoch loop
returns normally (full epoch budget or early stopping, not a crash/interrupt). If the
checkpoint exists but the marker doesn't, I treat the checkpoint as a partial, unusable
artifact and retrain from scratch rather than trying to resume it -- keeps this stage's
idempotency check as simple as "does the marker exist", nothing more.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT / "src"))

from battle_agents.mcts_approximation.pipeline.autoencoder import (
    generate_synthetic_dataset as gsd)
from battle_agents.mcts_approximation.pipeline.autoencoder import \
    train_autoencoder as ta
from battle_agents.mcts_approximation.pipeline.generate_data import \
    generate_dataset

NUM_BOOTSTRAP_GAMES = 49999  # PROPOSTA_10.md section 2.1

DEFAULT_CHECKPOINT_PATH = (PROJECT_ROOT / "data" / "autoencoder_bootstrap" /
                            "checkpoints_v5_fixed256" / ta.DEFAULT_CHECKPOINT_NAME)

TRAINING_COMPLETE_MARKER_NAME = "TRAINING_COMPLETE.marker"

LOG_PATH = PROJECT_ROOT / "data" / "autoencoder_bootstrap" / "pipeline_bootstrap.log"

# v5 hyperparameters, PROPOSTA_10.md section 7.5
V5_LATENT_DIM = 256
V5_DENSE_WEIGHT = 70.0
V5_DENSE_BCE_RATIO = 0.1
V5_EPOCHS = 140
V5_PATIENCE = 5
V5_BATCH_SIZE = 4096


def _log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class _TeeStdout:
    """Duplicates stdout to the bootstrap log file for a stage's duration, so
    tail -f on LOG_PATH shows the stage's own progress prints (per-epoch training
    loss, per-N-files scan progress) without needing this process attached."""

    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._original = None
        self._file = None

    def write(self, data):
        try:
            self._original.write(data)
        except (BrokenPipeError, ValueError):
            pass
        self._file.write(data)

    def flush(self):
        try:
            self._original.flush()
        except (BrokenPipeError, ValueError):
            pass
        self._file.flush()

    def isatty(self):
        return False

    def __enter__(self):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._log_path, "a")
        self._original = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._original
        self._file.close()


class _Argv:
    """Temporarily replaces sys.argv so an argparse-based main() can be called
    in-process with a fixed set of flags, instead of via subprocess."""

    def __init__(self, args):
        self._args = args
        self._original = None

    def __enter__(self):
        self._original = sys.argv
        sys.argv = [self._original[0] if self._original else "pipeline_bootstrap.py", *self._args]

    def __exit__(self, exc_type, exc, tb):
        sys.argv = self._original


def _write_completion_marker(checkpoint_path: Path, marker_path: Path) -> None:
    """Reads the epoch/val_loss train_autoencoder.py actually saved into the
    checkpoint, so the marker records the real outcome rather than a guess."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    marker_path.write_text(json.dumps({
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": ckpt["epoch"],
        "val_loss": ckpt["val_loss"],
        "fused_dim": ckpt["fused_dim"],
    }, indent=2))


def _run_stage(name, fn) -> None:
    _log(f"--- Stage '{name}': starting ---")
    t0 = time.time()
    try:
        with _TeeStdout(LOG_PATH):
            fn()
    except (Exception, SystemExit) as e:
        _log(f"--- Stage '{name}': FAILED after {time.time() - t0:.0f}s: {e} ---")
        raise RuntimeError(f"Autoencoder bootstrap failed at stage '{name}': {e}") from e
    _log(f"--- Stage '{name}': done in {time.time() - t0:.0f}s ---")


def ensure_autoencoder_ready(checkpoint_path: str = None) -> None:
    """
    Ensures the frozen v5 fused_features autoencoder checkpoint exists on disk,
    bootstrapping it from scratch if not. Idempotent: if the training-complete
    marker (see module docstring) already exists next to checkpoint_path,
    returns immediately without doing any work.

    checkpoint_path: absolute path to the final checkpoint. Defaults to
    DEFAULT_CHECKPOINT_PATH (the same path model.py's build_models() loads by
    default, resolved absolutely here rather than relative to cwd).
    """
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    checkpoint_dir = checkpoint_path.parent
    marker_path = checkpoint_dir / TRAINING_COMPLETE_MARKER_NAME

    checkpoint_is_compatible = False
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(
                str(checkpoint_path),
                map_location="cpu",
            )
            checkpoint_is_compatible = checkpoint.get("fused_dim") == gsd.FUSED_DIM
        except Exception as exc:
            _log(f"Could not validate checkpoint {checkpoint_path}: {exc}")

    if marker_path.exists() and checkpoint_is_compatible:
        _log(
            f"Compatible training-complete checkpoint found at {checkpoint_path}. "
            "Nothing to do."
        )
        return

    if marker_path.exists():
        marker_path.unlink()
        _log(
            f"Removed stale completion marker because the checkpoint is not "
            f"compatible with fused_dim={gsd.FUSED_DIM}."
        )

    if checkpoint_path.exists() and not marker_path.exists():
        _log(
            f"Checkpoint exists at {checkpoint_path} but is incomplete or incompatible; "
            "retraining from scratch."
        )

    _log(f"Autoencoder not ready ({checkpoint_path}). Starting bootstrap.")

    def stage_generate_games():
        generate_dataset(
            num_games=NUM_BOOTSTRAP_GAMES,
            output_dir=str(gsd.DEFAULT_DATA_DIR),
            agent_type="random",
        )

    def stage_generate_synthetic_dataset():
        if gsd.DEFAULT_OUTPUT.exists():
            existing = np.load(str(gsd.DEFAULT_OUTPUT), mmap_mode="r")
            existing_dim = existing.shape[1] if existing.ndim == 2 else None
            del existing
            if existing_dim == gsd.FUSED_DIM:
                _log(
                    f"{gsd.DEFAULT_OUTPUT} already has fused_dim={gsd.FUSED_DIM}; "
                    "skipping synthetic dataset generation."
                )
                return
            _log(
                f"{gsd.DEFAULT_OUTPUT} has fused_dim={existing_dim}; regenerating "
                f"for fused_dim={gsd.FUSED_DIM}."
            )
        with _Argv([]):
            gsd.main()

    def stage_train_autoencoder():
        with _Argv([
            "--checkpoint-dir", str(checkpoint_dir),
            "--latent-dim", str(V5_LATENT_DIM),
            "--dense-weight", str(V5_DENSE_WEIGHT),
            "--dense-bce-ratio", str(V5_DENSE_BCE_RATIO),
            "--epochs", str(V5_EPOCHS),
            "--patience", str(V5_PATIENCE),
            "--batch-size", str(V5_BATCH_SIZE),
        ]):
            ta.main()

    _run_stage(f"generate_dataset (agent_type=random, {NUM_BOOTSTRAP_GAMES} games)", stage_generate_games)
    _run_stage("generate_synthetic_dataset", stage_generate_synthetic_dataset)
    _run_stage("train_autoencoder (v5 hyperparameters)", stage_train_autoencoder)

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Autoencoder bootstrap finished all stages but checkpoint is still missing at "
            f"{checkpoint_path}. Check {LOG_PATH} for details."
        )

    _write_completion_marker(checkpoint_path, marker_path)
    _log(f"Autoencoder bootstrap complete. Checkpoint ready at {checkpoint_path}, "
         f"marker written to {marker_path}.")
