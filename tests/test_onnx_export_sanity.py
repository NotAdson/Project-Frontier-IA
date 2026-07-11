"""Sanity test – ensure ONNX export reproduces Keras inference for batch‑size 1."""

import numpy as np
import onnxruntime as ort

from battle_agents.mcts_approximation.pipeline.train_nn import build_model, export_to_onnx
from battle_agents.mcts_approximation.state_encoder import NUM_DENSE_FEATURES, ACTION_SPACE
from battle_agents.mcts_approximation.db.python.database import db


def _make_dummy_inputs():
    return {
        "dense_features": np.zeros((1, NUM_DENSE_FEATURES), dtype=np.float32),
        "species_indices": np.zeros((1, 12), dtype=np.int32),
        "move_indices": np.zeros((1, 48), dtype=np.int32),
        "item_indices": np.zeros((1, 12), dtype=np.int32),
        "ability_indices": np.zeros((1, 12), dtype=np.int32),
        "action_mask": np.ones((1, len(ACTION_SPACE)), dtype=np.float32),
    }


def test_onnx_export_matches_keras(tmp_path):
    # 1. Build and compile the model
    model = build_model(
        num_dense=NUM_DENSE_FEATURES,
        num_moves=db.get_num_moves(),
        num_species=db.get_num_species(),
        num_items=db.get_num_items(),
        num_abilities=db.get_num_abilities(),
    )
    model.compile()

    # 2. Export to ONNX
    onnx_file = tmp_path / "model.onnx"
    export_to_onnx(model, str(onnx_file))
    assert onnx_file.exists(), "ONNX file was not created"

    # 3. Prepare deterministic dummy inputs (batch=1)
    dummy_inputs = _make_dummy_inputs()

    # 4. Keras inference
    keras_outputs = model.predict(dummy_inputs)

    # 5. ONNX inference
    sess = ort.InferenceSession(str(onnx_file))
    ort_inputs = {inp.name: dummy_inputs[inp.name] for inp in sess.get_inputs()}
    onnx_outputs = sess.run(None, ort_inputs)

    # 6. Compare each output
    for i, (k_out, o_out) in enumerate(zip(keras_outputs, onnx_outputs)):
        assert np.allclose(k_out, o_out, rtol=1e-5, atol=1e-5), \
            f"Output {i} mismatch between Keras and ONNX"
