import os
import tempfile

import numpy as np

from battle_agents.mcts_approximation.state_encoder import NUM_DENSE_FEATURES


def test_model_weights_loading():
    import keras
    import torch

    from battle_agents.mcts_approximation.db.python.database import db
    from battle_agents.mcts_approximation.pipeline.autoencoder.model import (
        FUSED_DIM,
        FusedFeaturesAutoencoder,
    )
    from battle_agents.mcts_approximation.pipeline.train_nn import build_models, get_custom_objects

    num_species = db.get_num_species()
    num_moves = db.get_num_moves()
    num_items = db.get_num_items()
    num_abilities = db.get_num_abilities()

    with tempfile.TemporaryDirectory() as tmp_dir:
        encoder_path = os.path.join(tmp_dir, "fused_autoencoder_best.pt")
        autoencoder = FusedFeaturesAutoencoder(
            fused_dim=FUSED_DIM,
            latent_dim=16,
        )
        torch.save(
            {
                "model_state_dict": autoencoder.state_dict(),
                "fused_dim": FUSED_DIM,
                "latent_dim": 16,
            },
            encoder_path,
        )

        inference_model, _ = build_models(
            num_dense=NUM_DENSE_FEATURES,
            num_moves=num_moves,
            num_species=num_species,
            num_items=num_items,
            num_abilities=num_abilities,
            encoder_checkpoint_path=encoder_path,
        )

        assert "meta_plan" not in inference_model.output_names
        assert "meta_attention" not in {
            layer.name for layer in inference_model.layers
        }

        dummy_input = {}

        for model_input in inference_model.inputs:
            name = model_input.name.split(":")[0]

            shape = tuple(
                1 if dimension is None else int(dimension)
                for dimension in model_input.shape
            )

            dtype = np.int32 if "int" in str(model_input.dtype) else np.float32

            dummy_input[name] = np.zeros(shape, dtype=dtype)

        predictions_before_saving = inference_model.predict(
            dummy_input,
            verbose=0,
        )

        model_path = os.path.join(
            tmp_dir,
            "temp_model.keras",
        )

        inference_model.save(model_path)

        assert os.path.exists(model_path), (
            "Model file was not saved"
        )

        loaded_model = keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
            custom_objects=get_custom_objects(),
        )

        assert len(loaded_model.outputs) == 20, (
            "Expected 20 outputs, "
            f"got {len(loaded_model.outputs)}"
        )

        expected_dynamic_outputs = {
            "pred_weight_species",
            "pred_weight_stats",
            "pred_weight_type",
        }

        assert expected_dynamic_outputs.issubset(
            set(loaded_model.output_names)
        ), (
            "The loaded model is missing one or more "
            "dynamic weight outputs"
        )

        predictions_after_loading = loaded_model.predict(
            dummy_input,
            verbose=0,
        )

        assert len(predictions_before_saving) == len(
            predictions_after_loading
        )

        for output_index, (
            prediction_before,
            prediction_after,
        ) in enumerate(
            zip(
                predictions_before_saving,
                predictions_after_loading,
            )
        ):
            np.testing.assert_allclose(
                prediction_before,
                prediction_after,
                rtol=1e-5,
                atol=1e-5,
                err_msg=(
                    "Output "
                    f"{output_index} changed after "
                    "saving and loading the model"
                ),
            )
