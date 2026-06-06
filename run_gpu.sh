#!/bin/bash
# Run the passed command with PyTorch backend for GPU acceleration
KERAS_BACKEND=torch python3 "$@"
