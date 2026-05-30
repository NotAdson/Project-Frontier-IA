# Random Agent

This is the baseline naive agent for the Pokémon simulation framework.

## How It Works
The algorithm queries the Pokémon Showdown engine for a list of valid choices (e.g., `["move 1", "move 2", "switch 3"]`) and selects one using a uniform random distribution.

## Cheating
**Cheat: False**
This algorithm only requires knowledge of its own available valid actions. It does not inspect the engine's hidden variables, simulate forward using perfect information, or rely on any privileged state data.
