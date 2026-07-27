"""
Autoencoder architecture for compressing train_nn.py's "fused_features" tensor
(3074-dim, see train_nn.py:429) into a small latent code (issue #10).

Encoder: 3074 -> encoder_dims(latent_dim)  (Linear + ReLU between layers, no
                                             activation on the last layer)
Decoder: mirror of the encoder (decoder_dims(latent_dim))

The 3 hidden widths between 3074 and latent_dim scale WITH latent_dim (see
encoder_dims()/decoder_dims() below) so that latent_dim is always the
network's true bottleneck. For latent_dim <= 128 this is exactly
3074 -> 512 -> 256 -> 128 -> latent_dim, same as every checkpoint trained
before this scaling was added (v1/v2's latent_dim=64, v3's latent_dim=128).
For latent_dim > 128 (e.g. 256) the hidden widths grow too, instead of
silently staying at a fixed 512/256/128 with the true bottleneck stuck at
128 regardless of latent_dim -- which is what happened before this fix (see
encoder_dims()'s docstring for the checkpoint evidence).

Decoder output activation is SEGMENTED by feature type within the dense
block (see `compute_dense_binary_mask()` below, derived from
state_encoder.py's offsets — not hardcoded column numbers):
  - continuous dense features (hp_ratio, level, stats, move power/accuracy),
    embeddings, and meta_plan: linear, no activation (unbounded targets).
  - binary dense features (fainted, the 6 status flags, is_active, the 18
    type flags, and each move's is_physical/is_special/is_status flags —
    within OWN_TEAM_DENSE and the opponent's mirrored block): sigmoid.

IMPORTANT design decision: `decode()`/`forward()` intentionally return RAW
(unactivated) values at the binary positions too — i.e. logits, not
probabilities in [0, 1]. This is so train_autoencoder.py's
`nn.BCEWithLogitsLoss` (numerically stable, expects raw logits) can consume
`model(x)` directly without a redundant/duplicate sigmoid. Use
`reconstruct(x)` instead of `forward(x)` when you want the actual segmented-
activation reconstruction (sigmoid already applied at binary positions) —
e.g. for inference or manual inspection. test_reconstruction.py's raw-MSE
evaluation also applies `torch.sigmoid` itself at the binary positions
before comparing, for the same reason.
"""
import numpy as np
import torch
import torch.nn as nn

from battle_agents.mcts_approximation.state_encoder import (
    NUM_BENCH, NUM_DENSE_FEATURES, NUM_MOVES, OFF_FAINTED, OFF_IS_ACTIVE,
    OFF_MOVES_DENSE, OFF_STATUSES, OFF_TYPES, OPP_TEAM_START, PER_MON_DENSE)
from battle_agents.mcts_approximation.state_encoder import NUM_STATUS as _NUM_STATUS
from battle_agents.mcts_approximation.state_encoder import NUM_TYPES as _NUM_TYPES

FUSED_DIM = 3074
LATENT_DIM = 64

# Widths of the 3 hidden layers between FUSED_DIM and latent_dim (mirrored for
# encoder/decoder). Each is max(default, a multiple of latent_dim) so that no
# hidden layer is ever narrower than latent_dim -- otherwise an earlier
# fixed-width layer becomes the network's *real* compression bottleneck
# instead of the nominal latent layer. This is not a hypothetical: with the
# old hardcoded [512, 256, 128] intermediates, latent_dim=256 (checkpoints_v4_segloss)
# still bottlenecked at 128 (the fixed penultimate width), making it
# functionally identical in capacity to latent_dim=128 (checkpoints_v3_latent128)
# -- confirmed by inspecting both checkpoints' encoder.4/encoder.6 weight shapes
# directly. The multipliers (2x, 1.5x, 1x of latent_dim) keep the funnel
# monotonically decreasing relative to each other as latent_dim grows past 128;
# for every latent_dim <= 128 already trained (v1/v2's 64, v3's 128) this
# produces the exact original [512, 256, 128] shape, so those checkpoints
# remain architecturally identical and loadable.
def _intermediate_dims(latent_dim: int) -> list:
    return [
        max(512, latent_dim * 2),
        max(256, int(latent_dim * 1.5)),
        max(128, latent_dim),
    ]


def encoder_dims(latent_dim: int = LATENT_DIM, fused_dim: int = FUSED_DIM) -> list:
    """Actual encoder layer widths used by FusedFeaturesAutoencoder.__init__ --
    the real source of truth (not a separate, possibly-stale constant).
    fused_dim is a parameter (not always FUSED_DIM=3074) because --dense-only
    mode builds a 758-dim autoencoder over just the dense block."""
    return [fused_dim, *_intermediate_dims(latent_dim), latent_dim]


def decoder_dims(latent_dim: int = LATENT_DIM, fused_dim: int = FUSED_DIM) -> list:
    """Mirror of encoder_dims(); actual decoder layer widths used by __init__."""
    return [latent_dim, *_intermediate_dims(latent_dim)[::-1], fused_dim]

# Offset of is_physical/is_special/is_status within a single move's 5-value
# dense block ([power, accuracy, is_physical, is_special, is_status] — see
# state_encoder.py's _encode_own_team/_encode_opp_team_revealed).
_MOVE_CAT_FLAG_OFFSETS = (2, 3, 4)


def _per_mon_binary_offsets() -> list:
    """Offsets (within a single PER_MON_DENSE=53 block) that are binary
    (0/1) features: fainted, the 6 one-hot status flags, is_active, the 18
    multi-label type flags, and each of the 4 moves' 3 category flags
    (is_physical/is_special/is_status). Everything else in the block
    (hp_ratio, level, the 5 stats, move power/accuracy) is continuous."""
    offs = [OFF_FAINTED]
    offs += list(range(OFF_STATUSES, OFF_STATUSES + _NUM_STATUS))
    offs.append(OFF_IS_ACTIVE)
    offs += list(range(OFF_TYPES, OFF_TYPES + _NUM_TYPES))
    for j in range(NUM_MOVES):
        move_base = OFF_MOVES_DENSE + j * 5
        offs += [move_base + f for f in _MOVE_CAT_FLAG_OFFSETS]
    return offs


def compute_dense_binary_indices() -> list:
    """Absolute indices (within the NUM_DENSE_FEATURES=758 dense block) that
    are binary features, covering both OWN_TEAM_DENSE (offset 0) and the
    opponent's mirrored block (offset OPP_TEAM_START) — the boosts, field,
    and PP blocks are left out entirely (they stay continuous/MSE)."""
    per_mon = _per_mon_binary_offsets()
    indices = []
    for block_start in (0, OPP_TEAM_START):
        for i in range(NUM_BENCH):
            mon_start = block_start + i * PER_MON_DENSE
            indices.extend(mon_start + off for off in per_mon)
    return sorted(indices)


def compute_dense_binary_mask() -> np.ndarray:
    """Boolean mask, shape (NUM_DENSE_FEATURES,) — True at binary positions."""
    mask = np.zeros(NUM_DENSE_FEATURES, dtype=bool)
    mask[compute_dense_binary_indices()] = True
    return mask


def compute_binary_mask(fused_dim: int) -> np.ndarray:
    """Boolean mask, shape (fused_dim,) — True at binary positions within the
    dense block, False everywhere else (continuous dense features, and, for
    fused_dim > NUM_DENSE_FEATURES, the embeddings + meta_plan positions,
    which are always linear/continuous)."""
    mask = np.zeros(fused_dim, dtype=bool)
    dense_mask = compute_dense_binary_mask()
    n = min(fused_dim, NUM_DENSE_FEATURES)
    mask[:n] = dense_mask[:n]
    return mask


def _mlp(dims: list) -> nn.Sequential:
    """Linear + ReLU between every pair of dims; no activation after the final Linear."""
    layers = []
    n_pairs = len(dims) - 1
    for i in range(n_pairs):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < n_pairs - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class FusedFeaturesAutoencoder(nn.Module):
    def __init__(self, fused_dim: int = FUSED_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.fused_dim = fused_dim
        self.latent_dim = latent_dim
        self.encoder = _mlp(encoder_dims(latent_dim, fused_dim))
        self.decoder = _mlp(decoder_dims(latent_dim, fused_dim))
        binary_mask = torch.from_numpy(compute_binary_mask(fused_dim))
        self.register_buffer("binary_mask", binary_mask)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Raw decoder output — logits at binary_mask positions, linear
        elsewhere. See module docstring for why sigmoid is NOT applied here."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Full reconstruction with the segmented output activation actually
        applied: sigmoid at binary_mask positions, linear elsewhere. Prefer
        this over forward()/decode() outside of training."""
        raw = self.forward(x)
        out = raw.clone()
        out[..., self.binary_mask] = torch.sigmoid(raw[..., self.binary_mask])
        return out
