# How to Add a Search Algorithm

We welcome contributions! The repository uses a plugin-style Object-Oriented architecture, making it incredibly easy to add new search algorithms or heuristics without needing to touch the core game engine.

Follow this tutorial to create your own algorithm and submit it via a Pull Request.

## Step 1: Create Your Plugin Folder
Every algorithm lives in its own dedicated folder inside `src/agents/`.
Choose a short, descriptive, lower-case name for your algorithm. For example, if you are building Minimax:

```bash
mkdir src/agents/minimax
touch src/agents/minimax/__init__.py
```

## Step 2: Implement the `Agent` Class
Inside your new folder, create your python script (e.g., `minimax_agent.py`).
Your class MUST inherit from `core.agent.Agent` and implement the `get_action` method.

```python
from core.agent import Agent
import random

class MinimaxAgent(Agent):
    def __init__(self, problem, depth=3):
        super().__init__(problem)
        self.depth = depth

    def get_action(self, state, player="p1") -> str:
        # 1. Ask the problem wrapper for the valid actions available to you
        actions = self.problem.actions(state, player)
        
        if not actions:
            return "pass"
            
        # 2. Implement your AI logic here!
        # Example: best_action = self.minimax_search(...)
        best_action = random.choice(actions) # Placeholder
        
        return best_action
```

### Useful Core API Methods
When building your logic, you will primarily interact with the `self.problem` object:
- `self.problem.actions(state, player)`: Returns a list of valid string actions.
- `self.problem.result(state, p1_action, p2_action)`: Returns a new `PokemonState` simulating the engine forward.
- `self.problem.is_terminal(state)`: Returns `True` if the battle is over.
- `self.problem.is_goal(state)`: Returns `True` if Player 1 won.

## Step 3: Add Metadata
To help users understand what your algorithm does and its technical constraints, you MUST include a `metadata.yml` file in your agent's folder.

Create `src/agents/minimax/metadata.yml`:
```yaml
name: "Minimax Agent"
variant: "Alpha-Beta Pruning"
cheat: True
author: "Your Name"
description: "A deep adversarial search algorithm that calculates optimal game theoretic moves."
```

**Understanding `cheat`:** 
If your algorithm uses `problem.result()` to simulate the game engine forward, or inspects the `state.state_dict` JSON to view the opponent's unrevealed Pokémon or stats, set `cheat: True`. If your algorithm only plays based on perfectly public information, set `cheat: False`.

## Step 4: Write a README (Optional but Recommended)
Include a `README.md` inside your agent's folder explaining how your algorithm works, the heuristics it uses, and any external dependencies.

## Step 5: Test Your Agent
Open `src/simulate.py` and import your new agent. Put it up against the Random Agent or MCTS to verify it works!

```python
from agents.minimax.minimax_agent import MinimaxAgent
from agents.random.random_agent import RandomAgent

# ...
p1_agent = MinimaxAgent(problem, depth=3)
p2_agent = RandomAgent(problem)
```

Run `python3 simulate.py`. If everything runs smoothly and the replay generates correctly, you are ready to open a Pull Request!
