# Battle Agents

This directory contains various Artificial Intelligence agents designed to play Pokémon. The system is designed to support multiple different methods and architectures.

## Current Agents
- **`random/`**: A baseline agent that selects a uniformly random valid action.
- **`blind_mcts/`**: A Monte Carlo Tree Search agent that uses random rollouts to evaluate states.
- **`mcts_approximation/`**: An AlphaZero-style agent that uses a Neural Network to evaluate states instead of random rollouts, allowing for much faster and deeper strategic thinking.

## Adding a New Agent
To add a new search algorithm or AI method:
1. Create a new directory here (e.g., `minimax/`).
2. Create a class that inherits from `Agent` (defined in `src/core/agent.py`).
3. Implement the `get_action(state, player)` method.
