# Core Module

The `core` directory contains the foundational logic that connects our AI algorithms to the Pokémon Showdown engine.

## Structure
- **`client/`**: Contains `ShowdownClient`, which is responsible for establishing a subprocess connection to the Node.js Showdown engine, sending commands, and asynchronously reading the output logs and state.
- **`problem/`**: Contains `PokemonProblem`. This class wraps the client and exposes an interface similar to classic AI Search Problems (like `initial_state`, `actions(state)`, `result(state, action)`, `is_terminal(state)`). This makes it easy for any standard search algorithm to interact with the game.
- **`benchmark.py`**: The abstract base class for running tournaments.
- **`agent.py`**: The abstract base class for all battle agents.
