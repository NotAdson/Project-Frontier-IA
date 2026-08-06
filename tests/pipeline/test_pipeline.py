from battle_agents.mcts_approximation.pipeline.train_nn import load_data_from_files, build_models, get_custom_objects

def test_train_nn_signatures():
    assert callable(load_data_from_files)
    assert callable(build_models)
    assert callable(get_custom_objects)
