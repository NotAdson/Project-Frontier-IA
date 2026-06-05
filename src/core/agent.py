from abc import ABC, abstractmethod

from core.problem.aima_problem import Problem


class Agent(ABC):
    """
    Abstract base class for an AI agent in the environment.
    """
    def __init__(self, problem: Problem):
        self.problem = problem

    @abstractmethod
    def get_action(self, state, player="p1") -> str:
        """
        Calculates and returns the best action to take for the given state.
        
        Args:
            state: The current state object.
            
        Returns:
            str: The chosen action string.
        """
        pass
