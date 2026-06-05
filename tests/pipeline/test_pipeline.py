import os
from unittest.mock import patch

from battle_agents.mcts_approximation.pipeline.train_nn import load_data_from_files, build_model

def test_train_nn_signatures():
    assert callable(load_data_from_files)
    assert callable(build_model)
