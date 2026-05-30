# Pokémon AI Framework

An AI simulation framework for Pokémon battles, powered by the Pokémon Showdown engine. This project allows developers to connect classical search algorithms (like Monte Carlo Tree Search) and custom Python agents to a robust Pokémon battle engine.

## Architecture

The project is split into two main layers:
1. **Engine Layer (`engine/`)**: A Node.js backend (`bridge.js`) that wraps the official `pokemon-showdown` simulator. It exposes the engine over standard I/O streams using a JSON messaging protocol.
2. **Python Layer (`src/`)**: A decoupled Object-Oriented AI architecture. It treats the Pokémon battle as a formal Search Problem, allowing you to instantiate and pit different `Agent` classes against each other.

## Quick Start

### Prerequisites
- Node.js (for the Showdown engine)
- Python 3.x

### Running a Simulation
To run a simulation pitting two agents against each other:

```bash
cd src
python3 simulate.py
```

The script will automatically execute the game turn-by-turn and output a `replay.html` file in the root directory. Open this file in your web browser to watch the battle animation!

## Agents

The framework natively supports a plugin-style architecture for agents. You can find the currently implemented algorithms inside `src/agents/`:
- **Random Agent**: A baseline naive algorithm that picks uniform random moves.
- **MCTS Agent**: A Monte Carlo Tree Search algorithm that evaluates moves by running deep statistical simulations into the future.

### Contributing an Agent
Want to add Minimax, Expectimax, or a Neural Network? Check out [HOW_TO_ADD_SEARCH_ALGORITHM.md](HOW_TO_ADD_SEARCH_ALGORITHM.md) for a complete tutorial on how to submit a Pull Request!

## Backend Configuration (Neural MCTS)

The neural network component uses **[Keras 3](https://keras.io/)**, which is backend-agnostic. You can run training and inference on **TensorFlow**, **PyTorch**, or **JAX** without changing any code — just set the `KERAS_BACKEND` environment variable and install the matching package.

### Install a backend

```bash
# TensorFlow (default)
pip install keras tensorflow

# PyTorch
pip install keras torch

# JAX (CPU)
pip install keras jax

# JAX (GPU — CUDA 12)
pip install keras "jax[cuda12]"
```

### Run with a specific backend

```bash
# Training pipeline
KERAS_BACKEND=torch python src/battle_agents/mcts_approximation/run_pipeline.py

# One-off training
KERAS_BACKEND=tensorflow python src/battle_agents/mcts_approximation/train_nn.py
```

> **Note:** A model checkpoint saved with one backend cannot be loaded by a different backend. If you switch backends, delete `data/mcts_model.keras` and retrain from scratch.

