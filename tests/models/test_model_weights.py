import os
import tensorflow as tf
import numpy as np
from battle_agents.mcts_approximation.evaluator import NeuralStateEvaluator

def test_model_weights_loading():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/data/mcts_model.keras'))
    
    # Check if file exists
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        assert False, f"Failed to load model weights: {e}"
        
    assert model is not None
    
    # Check basic input shape expecting 5 inputs
    dummy_input = {
        "dense_features": np.zeros((1, 654), dtype=np.float32),
        "species_indices": np.zeros((1, 12), dtype=np.int32),
        "move_indices": np.zeros((1, 48), dtype=np.int32),
        "item_indices": np.zeros((1, 12), dtype=np.int32),
        "ability_indices": np.zeros((1, 12), dtype=np.int32)
    }
    try:
        predictions = model(dummy_input, training=False)
        assert len(predictions) == 2, "Model should return (policy, value)"
    except Exception as e:
        assert False, f"Model inference failed with dummy input: {e}"
