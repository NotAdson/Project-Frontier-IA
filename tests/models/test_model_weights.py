import os
import tempfile
import numpy as np

from battle_agents.mcts_approximation.state_encoder import NUM_DENSE_FEATURES


def test_model_weights_loading():
    import keras
    from battle_agents.mcts_approximation.db.python.database import db
    from battle_agents.mcts_approximation.pipeline.train_nn import build_model

    num_species = db.get_num_species()
    num_moves = db.get_num_moves()
    num_items = db.get_num_items()
    num_abilities = db.get_num_abilities()

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = build_model(
            num_dense=NUM_DENSE_FEATURES,
            num_moves=num_moves,
            num_species=num_species,
            num_items=num_items,
            num_abilities=num_abilities,
        )

        model_path = os.path.join(tmp_dir, "temp_model.keras")
        model.save(model_path)
        assert os.path.exists(model_path), "Model file was not saved"

        loaded = keras.models.load_model(model_path, compile=False, safe_mode=False)

        dummy_input = {}
        for inp in loaded.inputs:
            name = inp.name.split(":")[0]
            shape = [1 if (dim is None or dim == -1) else dim for dim in inp.shape]
            dtype = np.int32 if "int" in str(inp.dtype) else np.float32
            dummy_input[name] = np.zeros(shape, dtype=dtype)

        preds = loaded(dummy_input, training=False)
        assert len(preds) in (2, 18), f"Expected 2 or 18 outputs, got {len(preds)}"
