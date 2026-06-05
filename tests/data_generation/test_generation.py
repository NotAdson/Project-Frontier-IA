import os
from unittest.mock import patch

from battle_agents.mcts_approximation.pipeline.generate_data import generate_dataset, run_simulation

def test_generate_dataset_signature():
    assert callable(generate_dataset)
    assert callable(run_simulation)
