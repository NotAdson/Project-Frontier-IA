import importlib

__all__ = ["MCTSApproximationAgent"]


def __getattr__(name):
    if name == "MCTSApproximationAgent":
        module = importlib.import_module(".mcts_approximation_agent", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
