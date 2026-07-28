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
        from .mcts_approximation_agent import MCTSApproximationAgent

        return MCTSApproximationAgent
    raise AttributeError(name)


__all__ = ["MCTSApproximationAgent"]
