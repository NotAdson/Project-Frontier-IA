# MCTS Approximation Agent

## Overview and Intuition
The **MCTS Approximation Agent** is an advanced AI designed for Pokémon Showdown. It extends the traditional Monte Carlo Tree Search (MCTS) algorithm by introducing a Deep Learning component to dramatically improve decision-making quality and efficiency.

To understand why this is necessary, we must first look at how standard MCTS operates and why it struggles in the domain of Pokémon.

### The Problem with Random Rollouts
Standard MCTS evaluates the "goodness" of a game state using a method called a **Rollout** (or Simulation). From a given state, the algorithm plays completely random moves for both players until the end of the game. If the rollout results in a win, the state is considered good (+1); if it's a loss, the state is considered bad (0).

While this works incredibly well in games with simple mechanics and short horizons (like Tic-Tac-Toe or Connect 4), it fails spectacularly in Pokémon for several reasons:
1. **High Strategic Depth:** Pokémon relies heavily on long-term strategy, typing advantages, and switching. A random rollout might inexplicably switch out a setup sweeper (like a +6 Dragon Dance Rayquaza) for a weak Pokémon, throwing the game away. 
2. **Deep Game Trees:** A Pokémon battle can easily last over 50 turns. Simulating 50 turns of random actions introduces massive variance (noise) into the evaluation. 
3. **Computational Expense:** Simulating the Showdown engine to the end of a match hundreds of times per turn is extremely slow.

### The Solution: Neural Network Value Approximation
Inspired by architectures like DeepMind's AlphaZero, the **MCTS Approximation Agent** removes the random rollout phase entirely. 

Instead of asking, *"If we play randomly from here, do we win?"*, the algorithm asks a pre-trained **Neural Network**: *"Based on thousands of past games, what is the probability of winning from this exact state?"*

This provides two massive advantages:
- **Instantaneous Evaluation:** We no longer need to simulate 50 turns forward. We just perform one fast matrix multiplication.
- **Smarter Evaluation:** The Neural Network has learned to recognize strong positions (e.g., having a type advantage, high HP, or a setup sweeper) and will evaluate them highly, whereas a random rollout might accidentally ruin that strong position.

---

## How the Algorithm Works in Detail

Traditional MCTS has four phases: Selection, Expansion, Simulation, and Backpropagation. This agent modifies the third phase:

1. **Selection:** The algorithm traverses down the search tree from the root (current state) to a leaf node by choosing moves that maximize the Upper Confidence Bound (UCB). This balances *exploitation* (choosing known good moves) with *exploration* (trying untested moves).
2. **Expansion:** If the leaf node does not represent the end of the game, a new child node (a valid move) is generated.
3. **Approximation (Replaces Simulation):** 
   - The raw JSON `PokemonState` of this new child node is passed to our `StateEncoder`.
   - The encoder formats the state into a fixed-size numerical array (vector).
   - This vector is fed into a **TensorFlow Neural Network** (the Value Network).
   - The network outputs a single floating-point number between `0.0` and `1.0`. This represents the predicted probability of winning from this state.
4. **Backpropagation:** The predicted value (e.g., `0.85`) is passed back up the tree, updating the average win rate of all parent nodes that led to this state.

---

## Heuristics Used

### 1. State Encoding
To feed Pokémon states into a standard Dense Neural Network, we must carefully encode the environment. The `state_encoder.py` script extracts a rich set of heuristics from the `PokemonState`, specifically adhering to "Blind" rules (only looking at public information regarding the opponent).

The extracted features include:
- **Player Active Pokémon:** HP ratio, normalized base stats (Atk, Def, SpA, SpD, Spe), and Status conditions (Tox, Brn, Par, Slp, Frz) represented as one-hot vectors.
- **Player Active Moves:** For all 4 move slots, the encoder queries a local database (`moves.json`) to extract the move's Base Power (normalized), Accuracy, and Category (Physical, Special, or Status).
- **Opponent Active Pokémon (Public):** The opponent's visible HP percentage and visible Status conditions.

### 2. Value Target
The Neural Network is trained via Self-Play. In `generate_data.py`, two agents play against each other, recording the encoded state at every turn. 
Crucially, the Neural Network is **not** trained to predict the MCTS's internal rollout estimates. It is trained to predict the **Actual Final Game Outcome** (`1.0` for a win, `0.0` for a loss). By training on the true outcome of the game, the network learns the genuine value of a board state over thousands of examples.

---

## External Dependencies

This algorithm requires the following Python packages (listed in the root `requirements.txt`):
- `tensorflow>=2.10.0`: The Deep Learning framework used to build, load, and run the Value Network.
- `numpy>=1.23.0`: Used for efficient array manipulations and vectorizing the game state before feeding it to TensorFlow.

---

## Usage

To utilize this agent, you must first generate self-play data and train your model.

1. **Generate Data:** Run self-play games to build a dataset. This outputs individual `.json` files to `data/games/`.
   ```bash
   python3 -m src.agents.mcts_approximation.generate_data
   ```
2. **Train the Network:** Parse the generated data and train the TensorFlow model. This saves the weights to `data/mcts_model.h5`.
   ```bash
   python3 -m src.agents.mcts_approximation.train_nn
   ```
3. **Play:** Load the `MCTSApproximationAgent` in your tournament suite or `simulate.py`. The agent will automatically detect and use `data/mcts_model.h5`.
