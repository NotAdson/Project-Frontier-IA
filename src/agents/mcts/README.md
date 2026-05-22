# Monte Carlo Tree Search (MCTS)

This is the baseline implementation of Monte Carlo Tree Search for the Pokémon simulation framework. 

## How It Works
The algorithm balances exploration and exploitation using the **UCB1 formula**. 
During the simulation phase, it treats the Pokémon battle as a Simultaneous Move Game. To solve for this without knowing the opponent's true AI, it internally models the opponent picking a uniformly random action.

## Cheating
**Cheat: True**
Because this algorithm simulates future nodes by directly calling `problem.result()`, it relies on the actual Pokémon Showdown engine to compute future state transitions. The engine inherently utilizes perfect information (e.g., it knows the opponent's unrevealed Pokémon, items, and stats). A true "non-cheating" MCTS would need to simulate against a Belief State (a randomized distribution of possible opponent teams) rather than the true engine state.
