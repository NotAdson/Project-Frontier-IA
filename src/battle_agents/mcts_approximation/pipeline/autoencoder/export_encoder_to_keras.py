"""
Recreates the trained v5 encoder (checkpoints_v5_fixed256) as native Keras
Dense layers, copying its PyTorch weights over via set_weights() -- Option A
from PROPOSTA_10.md section 8. This script only rebuilds a frozen Keras
encoder from the checkpoint and validates that it reproduces the PyTorch
encoder's output numerically. It does not touch train_nn.py and does not
train anything.

Backend selection:
  Set KERAS_BACKEND env-var before running (see train_nn.py):
    KERAS_BACKEND=torch   (the only backend confirmed installed in this
                            environment -- tensorflow is not installed)

Usage:
    KERAS_BACKEND=torch python src/battle_agents/mcts_approximation/pipeline/autoencoder/export_encoder_to_keras.py
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

# Configure PyTorch device before Keras is imported if using torch backend --
# same guard as train_nn.py, kept here for consistency even though this
# script never trains (only a forward pass through both encoders).
if os.environ.get("KERAS_BACKEND") == "torch":
    if torch.cuda.is_available():
        print("[PyTorch Backend] CUDA GPU detected.")
    else:
        print("[PyTorch Backend] CUDA GPU not detected. Defaulting to 'cpu'.")

import keras

from battle_agents.mcts_approximation.pipeline.autoencoder.model import (
    FusedFeaturesAutoencoder, encoder_dims)
from battle_agents.mcts_approximation.pipeline.autoencoder.train_autoencoder import \
    split_indices

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CHECKPOINT = (PROJECT_ROOT / "data" / "autoencoder_bootstrap" /
                       "checkpoints_v5_fixed256" / "fused_autoencoder_best.pt")

# The 4 Linear layers inside model.py's _mlp() Sequential -- indices 0/2/4/6
# because ReLU (no parameters) sits at the odd indices in between.
ENCODER_LINEAR_KEYS = ["encoder.0", "encoder.2", "encoder.4", "encoder.6"]

MSE_TOLERANCE = 1e-6


def build_keras_encoder(fused_dim: int, latent_dim: int, state_dict: dict) -> keras.Sequential:
    """Rebuilds the trained encoder as a keras.Sequential of frozen Dense
    layers, with weights copied straight from state_dict. ReLU on the first
    3 layers, no activation on the last (mirrors model.py's _mlp())."""
    keras_model = keras.Sequential(name="fused_encoder_v5")
    keras_model.add(keras.layers.Input(shape=(fused_dim,)))
    for i, key in enumerate(ENCODER_LINEAR_KEYS):
        out_features = state_dict[f"{key}.weight"].shape[0]
        is_last = i == len(ENCODER_LINEAR_KEYS) - 1
        keras_model.add(keras.layers.Dense(
            out_features,
            activation=None if is_last else "relu",
            name=f"encoder_dense_{i}",
        ))

    for i, key in enumerate(ENCODER_LINEAR_KEYS):
        weight = state_dict[f"{key}.weight"].numpy()  # PyTorch Linear: [out, in]
        bias = state_dict[f"{key}.bias"].numpy()
        kernel = weight.T  # Keras Dense.kernel: [in, out]
        keras_model.layers[i].set_weights([kernel, bias])
        keras_model.layers[i].trainable = False

    return keras_model


def load_validation_rows(data_path: Path, n_total: int, val_fraction: float,
                          seed: int, num_rows: int) -> np.ndarray:
    """Pulls the first `num_rows` indices of the held-out validation split
    (same split_indices() used by train_autoencoder.py / test_reconstruction.py,
    same seed/val_fraction read from the checkpoint's own args) directly out
    of the memmapped .npy file."""
    _, val_idx = split_indices(n_total, val_fraction, seed)
    sample_idx = val_idx[:num_rows]
    data = np.load(str(data_path), mmap_mode="r")
    rows = np.array(data[sample_idx], dtype=np.float32, copy=True)
    del data
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-path", type=Path, default=None,
                         help="Defaults to the dataset path recorded in the checkpoint's args.")
    parser.add_argument("--num-val-rows", type=int, default=1000,
                         help="How many rows of the held-out validation split to check numerically.")
    args = parser.parse_args()

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    ckpt_args = checkpoint["args"]
    fused_dim = checkpoint["fused_dim"]
    latent_dim = checkpoint["latent_dim"]
    data_path = args.data_path if args.data_path is not None else Path(ckpt_args["data_path"])

    print(f"Checkpoint: {args.checkpoint}")
    print(f"fused_dim={fused_dim}  latent_dim={latent_dim}")

    torch_model = FusedFeaturesAutoencoder(fused_dim=fused_dim, latent_dim=latent_dim)
    torch_model.load_state_dict(checkpoint["model_state_dict"])
    torch_model.eval()

    encoder_state = {k: v for k, v in checkpoint["model_state_dict"].items() if k.startswith("encoder.")}
    real_widths = [fused_dim] + [encoder_state[f"{k}.weight"].shape[0] for k in ENCODER_LINEAR_KEYS]
    expected_widths = encoder_dims(latent_dim, fused_dim)
    print(f"Encoder widths from checkpoint tensors: {real_widths}")
    print(f"encoder_dims({latent_dim}, {fused_dim}):        {expected_widths}")
    assert real_widths == expected_widths, (
        f"checkpoint's actual encoder shapes {real_widths} don't match "
        f"model.py's encoder_dims() {expected_widths} -- architecture mismatch"
    )
    print("Shapes match encoder_dims() exactly.")

    keras_encoder = build_keras_encoder(fused_dim, latent_dim, encoder_state)

    rows = load_validation_rows(
        data_path=data_path,
        n_total=checkpoint["n_total"],
        val_fraction=ckpt_args["val_fraction"],
        seed=ckpt_args["seed"],
        num_rows=args.num_val_rows,
    )
    print(f"Validation rows pulled from held-out split: {rows.shape[0]} (first {args.num_val_rows} of val_idx)")

    with torch.no_grad():
        latent_torch = torch_model.encode(torch.from_numpy(rows)).numpy()

    latent_keras = keras_encoder.predict(rows, verbose=0)

    mse = float(np.mean((latent_torch - latent_keras) ** 2))
    max_abs_diff = float(np.max(np.abs(latent_torch - latent_keras)))

    print(f"Latent shapes: torch={latent_torch.shape}  keras={latent_keras.shape}")
    print(f"MSE(torch_latent, keras_latent) = {mse:.10e}")
    print(f"max |torch_latent - keras_latent| = {max_abs_diff:.10e}")

    if mse < MSE_TOLERANCE:
        print(f"PASS: MSE {mse:.10e} < tolerance {MSE_TOLERANCE:.0e}")
    else:
        print(f"FAIL: MSE {mse:.10e} >= tolerance {MSE_TOLERANCE:.0e}")

    return keras_encoder, mse


if __name__ == "__main__":
    main()
