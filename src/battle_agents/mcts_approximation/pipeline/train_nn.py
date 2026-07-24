"""Wrapper entrypoint for the refactored MCTS approximation training pipeline."""
import os

# Configure PyTorch device before Keras is imported if using torch backend.
# This is preserved so backend selection works the same way as before.
if os.environ.get('KERAS_BACKEND') == 'torch':
    import torch

    if torch.cuda.is_available():
        print('[PyTorch Backend] CUDA GPU detected.')
    else:
        print('[PyTorch Backend] CUDA GPU NOT detected. Defaulting to \'cpu\'.')

from .train import train


if __name__ == '__main__':
    train()
