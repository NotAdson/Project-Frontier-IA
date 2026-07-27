"""Sanity test – ensure ONNX export reproduces Keras inference for batch‑size 1."""

import numpy as np
import onnxruntime as ort

from battle_agents.mcts_approximation.pipeline.train_nn import build_models, export_to_onnx
from battle_agents.mcts_approximation.state_encoder import ACTION_SPACE, NUM_ACTIVE, NUM_DENSE_FEATURES, NUM_MOVES
from battle_agents.mcts_approximation.db.python.database import db

def _make_dummy_inputs():
    return {
        "dense_features": np.zeros((1, NUM_DENSE_FEATURES), dtype=np.float32),
        "species_indices": np.zeros((1, NUM_ACTIVE), dtype=np.int32),
        "move_indices": np.zeros((1, NUM_ACTIVE * NUM_MOVES), dtype=np.int32),
        "item_indices": np.zeros((1, NUM_ACTIVE), dtype=np.int32),
        "ability_indices": np.zeros((1, NUM_ACTIVE), dtype=np.int32),
        "action_mask": np.ones((1, len(ACTION_SPACE)), dtype=np.float32),
    }


def test_onnx_export_matches_keras(tmp_path):
    # 1. Build both model views
    inference_model, _ = build_models(
        num_dense=NUM_DENSE_FEATURES,
        num_moves=db.get_num_moves(),
        num_species=db.get_num_species(),
        num_items=db.get_num_items(),
        num_abilities=db.get_num_abilities(),
    )

    # 2. Export the inference model used by the agent
    onnx_file = tmp_path / "model.onnx"

    export_succeeded = export_to_onnx(inference_model, str(onnx_file))

    assert export_succeeded, (
        "export_to_onnx returned False. "
        "Check the export warnings above."
    )

    assert onnx_file.is_file(), (
        f"ONNX file was not created at {onnx_file}"
    )

    # 3. Prepare deterministic dummy inputs
    dummy_inputs = _make_dummy_inputs()

    # 4. Keras inference
    keras_outputs = inference_model.predict(dummy_inputs, verbose=0)

    # 5. ONNX inference.
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    assert [output.name for output in session.get_outputs()] == list(inference_model.output_names)

    ort_inputs = {
        input_info.name: dummy_inputs[input_info.name]
        for input_info in session.get_inputs()
    }

    onnx_outputs = session.run(None, ort_inputs)

    # Do not allow zip() to silently ignore extra outputs.
    assert len(keras_outputs) == len(onnx_outputs), (
        "Keras and ONNX returned different numbers "
        f"of outputs: {len(keras_outputs)} and "
        f"{len(onnx_outputs)}"
    )

    # 6. Compare every output.
    for index, (keras_output, onnx_output) in enumerate(
        zip(keras_outputs, onnx_outputs)
    ):
        assert np.allclose(
            keras_output,
            onnx_output,
            rtol=1e-5,
            atol=1e-5,
        ), f"Output {index} mismatch between Keras and ONNX"
