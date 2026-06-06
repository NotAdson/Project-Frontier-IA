import os
import tensorflow as tf
import numpy as np
from battle_agents.mcts_approximation.evaluator import NeuralStateEvaluator

def test_model_weights_loading():
    import keras
    from battle_agents.mcts_approximation.pipeline.train_nn import build_model
    from battle_agents.mcts_approximation.db.moves_db import get_num_moves
    from battle_agents.mcts_approximation.db.species_db import (
        get_num_abilities, get_num_items, get_num_species
    )
    
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/data/mcts_model.keras'))
    
    # Check if file exists
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    
    # Load model
    model = None
    try:
        model = keras.models.load_model(model_path, compile=False, safe_mode=False)
    except Exception as e_load:
        # Fallback to build_model and load weights directly, mirroring production evaluator behavior
        try:
            num_species = get_num_species()
            num_moves = get_num_moves()
            num_items = get_num_items()
            num_abilities = get_num_abilities()
            
            # Try with 744 (NUM_DENSE_FEATURES)
            model = build_model(
                num_dense=744,
                num_moves=num_moves,
                num_species=num_species,
                num_items=num_items,
                num_abilities=num_abilities
            )
            model.load_weights(model_path)
        except Exception as e_fallback:
            assert False, f"Failed to load model weights via both load_model ({e_load}) and fallback weights loading ({e_fallback})"
        
    assert model is not None
    
    # Check basic inputs dynamically to support both old and new model architectures
    dummy_input = {}
    for inp in model.inputs:
        # Get input name (removing tensor index suffixes if any)
        name = inp.name.split(':')[0]
        shape = list(inp.shape)
        # Replace batch/dynamic dimensions with 1
        shape = [1 if (dim is None or dim == -1) else dim for dim in shape]
        
        # Determine numpy dtype
        dtype = np.float32
        if "int" in str(inp.dtype):
            dtype = np.int32
            
        dummy_input[name] = np.zeros(shape, dtype=dtype)
        
    try:
        predictions = model(dummy_input, training=False)
        assert len(predictions) in (2, 18), f"Expected 2 or 18 outputs, got {len(predictions)}"
    except Exception as e:
        assert False, f"Model inference failed with dummy input: {e}"


