import os
import math
import random
import numpy as np

# Keras is imported lazily inside __init__ as a fallback to avoid backend loading overhead
keras = None

from battle_agents.mcts_approximation.state_encoder import (
    encode_state,
    NUM_DENSE_FEATURES,
    NUM_SPECIES_INDICES, NUM_MOVE_INDICES, NUM_ITEM_INDICES, NUM_ABILITY_INDICES,
    OFF_SPECIES, OFF_MOVES, OFF_ITEMS, OFF_ABILITIES,
    ACTION_SPACE,
)


class BaseStateEvaluator:
    """
    Abstract Base Class for state evaluation in Neural Monte Carlo Tree Search.
    Decouples tree search from the execution of the Neural Network.
    """
    def evaluate(self, state, player: str, valid_actions: list) -> tuple[float, dict[str, float]]:
        """
        Evaluates a game state from a specific player's perspective.
        
        Args:
            state: The Showdown PokemonState.
            player: 'p1' or 'p2'.
            valid_actions: List of strings of valid actions for the player.
            
        Returns:
            A tuple of (value, action_probs) where:
              - value: float in [0.0, 1.0] representing probability of winning.
              - action_probs: dict mapping valid action strings to prior probabilities.
        """
        raise NotImplementedError


class NeuralStateEvaluator(BaseStateEvaluator):
    """
    Concrete State Evaluator that uses a Keras or ONNX Neural Network
    to estimate state values and move probabilities.
    """
    def __init__(self, model_path="data/mcts_model.keras"):
        self.model = None
        self.onnx_session = None

        # Pre-allocate feature input buffers for zero-allocation builder
        self._buf_dense = np.zeros((1, NUM_DENSE_FEATURES), dtype=np.float32)
        self._buf_species = np.zeros((1, NUM_SPECIES_INDICES), dtype=np.int32)
        self._buf_moves = np.zeros((1, NUM_MOVE_INDICES), dtype=np.int32)
        self._buf_items = np.zeros((1, NUM_ITEM_INDICES), dtype=np.int32)
        self._buf_abilities = np.zeros((1, NUM_ABILITY_INDICES), dtype=np.int32)

        # Check for ONNX model first
        onnx_path = None
        if model_path:
            possible_onnx_paths = [
                model_path,
                model_path.replace(".keras", ".onnx").replace(".h5", ".onnx")
            ]
            for p in possible_onnx_paths:
                if p and p.endswith(".onnx") and os.path.exists(p):
                    onnx_path = p
                    break

        if onnx_path:
            try:
                import onnxruntime as ort
                
                # Configure thread pools to avoid "pthread_create failed" (EAGAIN / Resource temporarily unavailable) in VMs
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                
                # Load with CPU Execution Provider to ensure strictly CPU execution
                self.onnx_session = ort.InferenceSession(
                    onnx_path, 
                    sess_options=opts, 
                    providers=['CPUExecutionProvider']
                )
                self.input_names = [inp.name for inp in self.onnx_session.get_inputs()]
                self.output_names = [out.name for out in self.onnx_session.get_outputs()]
                print(f"[NeuralStateEvaluator] Loaded ONNX model from {onnx_path} using onnxruntime (CPU, 1 thread)")
            except Exception as e:
                print(f"[Warning] Failed to load ONNX model using onnxruntime: {e}. Falling back to Keras.")

        if self.onnx_session is None:
            # Fallback to Keras
            try:
                import keras as lazy_keras
                global keras
                keras = lazy_keras
            except ImportError:
                keras = None

            if keras is not None:
                fallback = "data/mcts_model.h5"
                actual_path = (model_path if os.path.exists(model_path)
                               else fallback if os.path.exists(fallback)
                               else None)
                if actual_path:
                    try:
                        self.model = keras.models.load_model(actual_path, compile=False)
                        print(f"[NeuralStateEvaluator] Loaded Keras model from {actual_path} as fallback")
                    except Exception as err_keras:
                        print(f"[Warning] Failed to load Keras model due to Keras version mismatch/deserialization issues: {err_keras}. "
                              "Search will continue with default (0.5 value) predictions.")
                else:
                    print(f"[Warning] MCTS Approximation model not found at {model_path}. "
                          "Will predict 0.5 for all states.")
            else:
                print("[Error] Keras is not installed. Cannot use Neural Network fallback. "
                      "Please pip install keras and a backend (tensorflow / torch / jax).")

    def _build_inputs(self, features: np.ndarray) -> dict:
        """
        Splits the flat feature vector into the 5 named model inputs.
        Zero allocations due to pre-allocated buffers.
        """
        n = NUM_DENSE_FEATURES
        
        # In-place copy to pre-allocated buffers
        self._buf_dense[0, :] = features[:n]
        self._buf_species[0, :] = features[n + OFF_SPECIES : n + OFF_SPECIES + NUM_SPECIES_INDICES]
        self._buf_moves[0, :] = features[n + OFF_MOVES : n + OFF_MOVES + NUM_MOVE_INDICES]
        self._buf_items[0, :] = features[n + OFF_ITEMS : n + OFF_ITEMS + NUM_ITEM_INDICES]
        self._buf_abilities[0, :] = features[n + OFF_ABILITIES : n + OFF_ABILITIES + NUM_ABILITY_INDICES]

        return {
            "dense_features":   self._buf_dense,
            "species_indices":  self._buf_species,
            "move_indices":     self._buf_moves,
            "item_indices":     self._buf_items,
            "ability_indices":  self._buf_abilities,
        }

    def evaluate(self, state, player: str, valid_actions: list) -> tuple[float, dict[str, float]]:
        action_probs = {}
        
        if self.onnx_session is not None:
            features = encode_state(state, player)
            inputs = self._build_inputs(features)
            
            # Map Python dictionary inputs to exact ONNX session input names
            ort_inputs = {}
            for name in self.input_names:
                clean_name = name.split(':')[0]
                if clean_name in inputs:
                    ort_inputs[name] = inputs[clean_name]
                else:
                    # Fuzzy match fallback
                    matched = False
                    for k in inputs:
                        if k in name or name in k:
                            ort_inputs[name] = inputs[k]
                            matched = True
                            break
                    if not matched:
                        pass

            try:
                ort_outs = self.onnx_session.run(self.output_names, ort_inputs)
                
                # Match value and policy outputs
                value_pred = None
                policy_pred = None
                for name, out in zip(self.output_names, ort_outs):
                    if "value" in name:
                        value_pred = out
                    elif "policy" in name:
                        policy_pred = out
                
                # Dynamic shape/fallback matching
                if value_pred is None or policy_pred is None:
                    for out in ort_outs:
                        if out.shape == (1, 1):
                            value_pred = out
                        elif out.shape == (1, len(ACTION_SPACE)):
                            policy_pred = out

                # absolute final fallback by index
                if value_pred is None:
                    value_pred = ort_outs[0]
                if policy_pred is None:
                    policy_pred = ort_outs[1]

                reward = float(value_pred[0][0])
                policy_probs = policy_pred[0]

                for a in valid_actions:
                    idx = ACTION_SPACE.index(a) if a in ACTION_SPACE else ACTION_SPACE.index("pass")
                    action_probs[a] = float(policy_probs[idx])

                s = sum(action_probs.values())
                if s > 0:
                    for a in action_probs: action_probs[a] /= s
                else:
                    for a in action_probs: action_probs[a] = 1.0 / len(valid_actions)

            except Exception as e:
                print(f"[Error] ONNX inference failed: {e}. Falling back to default predictions.")
                reward = 0.5
                for a in valid_actions:
                    action_probs[a] = 1.0 / len(valid_actions)

        elif self.model is not None:
            features = encode_state(state, player)
            inputs = self._build_inputs(features)
            # Direct model call with training=False
            pred = self.model(inputs, training=False)

            # pred is a list of tensors starting with: [value (1,1), policy (1, ACTION_SPACE)]
            if isinstance(pred, (list, tuple)) and len(pred) >= 2:
                value_pred, policy_pred = pred
                reward = float(np.array(value_pred)[0][0])
                policy_probs = np.array(policy_pred)[0]

                for a in valid_actions:
                    idx = ACTION_SPACE.index(a) if a in ACTION_SPACE else ACTION_SPACE.index("pass")
                    action_probs[a] = float(policy_probs[idx])

                s = sum(action_probs.values())
                if s > 0:
                    for a in action_probs: action_probs[a] /= s
                else:
                    for a in action_probs: action_probs[a] = 1.0 / len(valid_actions)
            else:
                # Fallback: single-output model (value only)
                reward = float(np.array(pred)[0][0])
                for a in valid_actions:
                    action_probs[a] = 1.0 / len(valid_actions)
        else:
            reward = 0.5
            for a in valid_actions:
                action_probs[a] = 1.0 / len(valid_actions)
                
        return reward, action_probs
