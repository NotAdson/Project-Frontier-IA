# Pokémon AI Framework

An AI simulation framework for Pokémon battles, powered by the Pokémon Showdown engine. This project allows developers to connect classical search algorithms (like Monte Carlo Tree Search) and custom Python agents to a robust Pokémon battle engine.

## Architecture

The project is split into two main layers:
1. **Engine Layer (`engine/`)**: A Node.js backend (`bridge.js`) that wraps the official `pokemon-showdown` simulator. It exposes the engine over standard I/O streams using a JSON messaging protocol.
2. **Python Layer (`src/`)**: A decoupled Object-Oriented AI architecture. It treats the Pokémon battle as a formal Search Problem, allowing you to instantiate and pit different `Agent` classes against each other.

## Repository Setup

### 1. Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install keras torch   # or tensorflow / jax
```

### 2. Engine (Pokémon Showdown)

The battle simulator is a Node.js project inside `engine/`. Install its
dependencies before running the pipeline:

```bash
cd engine
npm install
cd ..
```

The installation also builds `engine/dist/`. Both `node_modules/` and `dist/`
are generated locally and ignored by Git.

## Quick Start

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

## Running the Training Pipeline

The AlphaZero-style pipeline cycles through:

1. **Self-Play** — Agents play each other, recording (state, visit-count policy, outcome)
2. **Training** — Neural network fine-tunes on collected data (value + policy + auxiliary heads)
3. **Champion Gating** — New model vs previous champion; promoted if win rate ≥ 55%
4. **Next generation** — repeats with the new champion

All commands run from the **project root**:

```bash
KERAS_BACKEND=torch python src/battle_agents/mcts_approximation/pipeline/run_pipeline.py
```

### Customizing

```bash
KERAS_BACKEND=torch python src/battle_agents/mcts_approximation/pipeline/run_pipeline.py \
    --num_games 50 --mcts_iterations 100 --epochs 50 --num_generations 5
```

### Resuming

Auto-resumes from the last incomplete generation. To start fresh, pass `--wipe`.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ONNX model not found` | Ensure `src/data/mcts_model.onnx` exists or don't use `--wipe` |
| `Cannot import get_num_species` | Run from project root, not `src/` |
| `JIT compilation failed` | Use `KERAS_BACKEND=torch` or set `CUDA_VISIBLE_DEVICES=-1` |
| `node: command not found` | Install Node.js from https://nodejs.org |

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
KERAS_BACKEND=torch python src/battle_agents/mcts_approximation/pipeline/run_pipeline.py

# One-off training
KERAS_BACKEND=tensorflow python src/battle_agents/mcts_approximation/train_nn.py
```

> **Note:** A model checkpoint saved with one backend cannot be loaded by a different backend. If you switch backends, delete `data/mcts_model.keras` and retrain from scratch.

