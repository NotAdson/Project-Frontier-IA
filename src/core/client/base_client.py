from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseClient(ABC):
    """
    Abstract base class defining the interface for all Pokemon battle engine clients.
    """
    
    @abstractmethod
    def init_battle(self, formatid: str = 'gen3randombattle', p1_team: Optional[list] = None, p2_team: Optional[list] = None) -> Dict[str, Any]:
        """
        Initializes a new battle.
        """
        pass

    @abstractmethod
    def get_result(self, state: Dict[str, Any], p1_action: str, p2_action: Optional[str] = None, state_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes a turn and gets the resulting state.
        """
        pass

    @abstractmethod
    def rollout(self, state: Dict[str, Any], player: str, max_depth: int, state_id: Optional[int] = None) -> float:
        """
        Performs a rollout (simulation) from the current state to a terminal state.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Cleans up resources.
        """
        pass

    def clear_cache(self) -> None:
        """
        Clears any state caches to free memory.
        """
        pass
