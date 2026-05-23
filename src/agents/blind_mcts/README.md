# Blind Monte Carlo Tree Search Agent

The `BlindMCTSAgent` is an "honest" variant of the classic Monte Carlo Tree Search algorithm designed specifically for incomplete information games like Pokémon.

## The Problem with standard MCTS
In a traditional MCTS, the algorithm requests the set of legal moves for both Player 1 and Player 2 in order to properly simulate the game into the future (the rollout phase). However, in Pokémon, a player is not mathematically supposed to know what moves the opponent can legally make, because it reveals whether they are Trapped, Choice-locked, or what their underlying moveset is. 

The standard MCTS inherently "cheats" by looking at the opponent's `request_dict` to find their valid moves.

## How Blind MCTS Works
The `BlindMCTSAgent` strictly enforces imperfect information boundaries. It never calls `problem.actions()` for the opponent. 

Instead, when it needs to simulate an opponent's turn during a node expansion or random rollout, it explicitly passes `None` to the Node.js battle engine. The engine natively catches this missing action and forces the opponent to play a `default` fallback move (acting randomly behind closed doors).

This ensures the Python MCTS AI algorithm can statistically evaluate future states without ever once looking at the opponent's restricted data!

## Usage
To use this agent in the simulation, simply import it and pass it to your runtime environment:

```python
from agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent

# Play with 100 iterations and a max depth of 50 turns per rollout
agent = BlindMCTSAgent(problem, iterations=100, max_rollout_depth=50)
```
