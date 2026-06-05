import random

from core.agent import Agent


class RandomAgent(Agent):
    """
    An agent that takes uniformly random actions from the available valid choices.
    """
    def get_action(self, state, player="p1") -> str:
        actions = self.problem.actions(state, player)
        if not actions:
            return "pass"
        return random.choice(actions)
