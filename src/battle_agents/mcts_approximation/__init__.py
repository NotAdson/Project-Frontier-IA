"""MCTS approximation agent package.

The agent depends on ONNX Runtime, while utility submodules such as the
plotters do not.  Loading the public agent lazily keeps those tools usable in
lightweight analysis environments (for example a Colab plotting notebook).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mcts_approximation_agent import MCTSApproximationAgent


def __getattr__(name):
    if name == "MCTSApproximationAgent":
        import importlib

        return MCTSApproximationAgent
    raise AttributeError(name)


__all__ = ["MCTSApproximationAgent"]


def __getattr__(name):
    if name == "MCTSApproximationAgent":
        module = importlib.import_module(".mcts_approximation_agent", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
