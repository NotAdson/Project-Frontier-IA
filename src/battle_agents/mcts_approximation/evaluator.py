import math
import os
import random

import numpy as np

# Keras is imported lazily inside __init__ as a fallback to avoid backend loading overhead
keras = None

from battle_agents.mcts_approximation.state_encoder import (
    ACTION_SPACE, NUM_ABILITY_INDICES, NUM_DENSE_FEATURES, NUM_ITEM_INDICES,
    NUM_MOVE_INDICES, NUM_SPECIES_INDICES, OFF_ABILITIES, OFF_ITEMS, OFF_MOVES,
    OFF_SPECIES, encode_state)


def _load_weights_mapped_keras(model, legacy_model_path):
    import io
    import zipfile

    import h5py
    print(f"[NeuralStateEvaluator] Loading mapped weights from {legacy_model_path}...")
    with zipfile.ZipFile(legacy_model_path, "r") as z:
        weights_bytes = z.read("model.weights.h5")
        
    name_map = {
        "embedding": "meta_emb_species",
        "embedding_1": "meta_emb_items",
        "embedding_2": "meta_emb_abilities",
        "embedding_3": "meta_emb_moves",
        "embedding_4": "emb_species",
        "embedding_5": "emb_moves",
        "embedding_6": "emb_items",
        "embedding_7": "emb_abilities",
        "dense": "token_score_projection",
        "dense_1": "dense",
        "dense_2": "dense_1",
        "dense_3": "dense_2",
        "dense_4": "policy_logits",
        "dense_5": "value",
        "dense_6": "aux_field",
        "dense_7": "aux_own_hp",
        "dense_8": "aux_opp_hp",
        "dense_9": "aux_own_statuses",
        "dense_10": "aux_opp_statuses",
        "dense_11": "aux_own_boosts",
        "dense_12": "aux_opp_boosts",
        "dense_13": "aux_own_stats",
        "dense_14": "aux_opp_stats",
        "dense_15": "aux_own_types",
        "dense_16": "aux_opp_types",
        "dense_17": "aux_own_species",
        "dense_18": "aux_opp_species",
        "layer_normalization": "layer_normalization",
        "batch_normalization": "batch_normalization",
        "batch_normalization_1": "batch_normalization_1",
        "batch_normalization_2": "batch_normalization_2"
    }
    
    attn_map = {
        "multi_head_attention/query_dense": ("meta_attention", "query_dense"),
        "multi_head_attention/key_dense": ("meta_attention", "key_dense"),
        "multi_head_attention/value_dense": ("meta_attention", "value_dense"),
        "multi_head_attention/output_dense": ("meta_attention", "output_dense")
    }
    
    with h5py.File(io.BytesIO(weights_bytes), "r") as f:
        # Load standard layers
        for saved_name, target_name in name_map.items():
            group_path = f"layers/{saved_name}"
            if group_path not in f:
                continue
            
            datasets = {}
            def find_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets[name] = obj[:]
            f[group_path].visititems(find_datasets)
            
            if not datasets:
                continue
                
            try:
                layer = model.get_layer(target_name)
            except ValueError:
                continue
                
            sorted_keys = sorted(datasets.keys(), key=lambda x: int(x.split('/')[-1]))
            weights = [datasets[k] for k in sorted_keys]
            
            target_weights = layer.get_weights()
            adjusted_weights = []
            for tw, w in zip(target_weights, weights):
                if tw.shape != w.shape:
                    if len(tw.shape) == 2 and len(w.shape) == 2:
                        adjusted = np.zeros(tw.shape, dtype=tw.dtype)
                        min_rows = min(tw.shape[0], w.shape[0])
                        min_cols = min(tw.shape[1], w.shape[1])
                        adjusted[:min_rows, :min_cols] = w[:min_rows, :min_cols]
                        adjusted_weights.append(adjusted)
                    elif len(tw.shape) == 1 and len(w.shape) == 1:
                        adjusted = np.zeros(tw.shape, dtype=tw.dtype)
                        min_len = min(tw.shape[0], w.shape[0])
                        adjusted[:min_len] = w[:min_len]
                        adjusted_weights.append(adjusted)
                    else:
                        adjusted_weights.append(tw)
                else:
                    adjusted_weights.append(w)
            layer.set_weights(adjusted_weights)
            
        # Load attention sub-layers
        for saved_sub, (target_layer_name, sub_attr) in attn_map.items():
            group_path = f"layers/{saved_sub}"
            if group_path not in f:
                continue
                
            datasets = {}
            def find_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets[name] = obj[:]
            f[group_path].visititems(find_datasets)
            
            if not datasets:
                continue
                
            try:
                parent_layer = model.get_layer(target_layer_name)
                sub_layer = getattr(parent_layer, sub_attr)
            except (ValueError, AttributeError):
                continue
                
            sorted_keys = sorted(datasets.keys(), key=lambda x: int(x.split('/')[-1]))
            weights = [datasets[k] for k in sorted_keys]
            sub_layer.set_weights(weights)
    print("[NeuralStateEvaluator] Mapped weight loading completed successfully.")


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
                actual_path = (model_path if (model_path and os.path.exists(model_path))
                               else fallback if os.path.exists(fallback)
                               else None)
                if actual_path:
                    try:
                        self.model = keras.models.load_model(actual_path, compile=False, safe_mode=False)
                        print(f"[NeuralStateEvaluator] Loaded Keras model from {actual_path} as fallback")
                    except Exception as err_keras:
                        print(f"[Warning] Failed to load Keras model due to Keras version mismatch/deserialization issues: {err_keras}. "
                              "Attempting build_model + load_weights fallback...")
                        try:
                            from battle_agents.mcts_approximation.db.moves_db import \
                                get_num_moves
                            from battle_agents.mcts_approximation.db.species_db import (
                                get_num_abilities, get_num_items,
                                get_num_species)
                            from battle_agents.mcts_approximation.pipeline.train_nn import \
                                build_model
                            
                            num_species = get_num_species()
                            num_moves = get_num_moves()
                            num_items = get_num_items()
                            num_abilities = get_num_abilities()
                            
                            # First try modern shape 744
                            try:
                                self.model = build_model(
                                    num_dense=NUM_DENSE_FEATURES,
                                    num_moves=num_moves,
                                    num_species=num_species,
                                    num_items=num_items,
                                    num_abilities=num_abilities
                                )
                                try:
                                    self.model.load_weights(actual_path)
                                except Exception:
                                    _load_weights_mapped_keras(self.model, actual_path)
                                self.expected_dense_dim = NUM_DENSE_FEATURES
                                print(f"[NeuralStateEvaluator] Loaded Keras weights from {actual_path} successfully (744 features)!")
                            except Exception as e_shape:
                                # Retry with legacy shape 654
                                print(f"[Warning] Failed loading with 744 features: {e_shape}. Retrying with legacy 654 features...")
                                self.model = build_model(
                                    num_dense=654,
                                    num_moves=num_moves,
                                    num_species=num_species,
                                    num_items=num_items,
                                    num_abilities=num_abilities
                                )
                                try:
                                    self.model.load_weights(actual_path)
                                except Exception:
                                    _load_weights_mapped_keras(self.model, actual_path)
                                self.expected_dense_dim = 654
                                print(f"[NeuralStateEvaluator] Loaded Keras weights from {actual_path} successfully (654 features)!")
                        except Exception as err_weights:
                            print(f"[Warning] Failed to load weights directly: {err_weights}. "
                                  "Search will continue with default (0.5 value) predictions.")
                else:
                    print(f"[Warning] MCTS Approximation model not found at {model_path}. "
                          "Will predict 0.5 for all states.")
            else:
                print("[Error] Keras is not installed. Cannot use Neural Network fallback. "
                      "Please pip install keras and a backend (tensorflow / torch / jax).")

        # Determine expected dense_features dimension from the loaded model/session
        self.expected_dense_dim = NUM_DENSE_FEATURES
        if self.onnx_session is not None:
            try:
                for inp in self.onnx_session.get_inputs():
                    if "dense_features" in inp.name:
                        self.expected_dense_dim = inp.shape[1]
                        break
            except Exception as e:
                print(f"[Warning] Could not inspect ONNX input shape: {e}. Defaulting to {NUM_DENSE_FEATURES}")
        elif self.model is not None:
            try:
                if hasattr(self.model, "input_shape"):
                    if isinstance(self.model.input_shape, dict):
                        self.expected_dense_dim = self.model.input_shape.get("dense_features", [None, NUM_DENSE_FEATURES])[1]
                    elif isinstance(self.model.input_shape, list):
                        self.expected_dense_dim = self.model.input_shape[0][1]
                    else:
                        self.expected_dense_dim = self.model.input_shape[1]
            except Exception as e:
                print(f"[Warning] Could not inspect Keras model input shape: {e}. Defaulting to {NUM_DENSE_FEATURES}")
        print(f"[NeuralStateEvaluator] Configured expected dense dimension: {self.expected_dense_dim}")

    def _build_inputs(self, features: np.ndarray) -> dict:
        """
        Splits the flat feature vector into the 5 named model inputs.
        Supports down-converting the expanded features for legacy formats (702 or 654).
        Zero allocations due to pre-allocated buffers.
        """
        if self.expected_dense_dim == 702:
            # Down-convert 744 to 702 (symmetrically removing the 42 new volatiles at indices [654:696])
            features_dense = np.concatenate([features[:654], features[696:744]])
            features_embed = features[744:]
        elif self.expected_dense_dim == 654:
            # Down-convert 744 to 654 (removing PP features and new volatiles)
            features_dense = features[:654]
            features_embed = features[744:]
        else:
            # New shape 744 dense
            features_dense = features[:NUM_DENSE_FEATURES]
            features_embed = features[NUM_DENSE_FEATURES:]
        
        # Copy to pre-allocated buffers
        self._buf_dense[0, :self.expected_dense_dim] = features_dense
        self._buf_species[0, :] = features_embed[OFF_SPECIES : OFF_SPECIES + NUM_SPECIES_INDICES]
        self._buf_moves[0, :] = features_embed[OFF_MOVES : OFF_MOVES + NUM_MOVE_INDICES]
        self._buf_items[0, :] = features_embed[OFF_ITEMS : OFF_ITEMS + NUM_ITEM_INDICES]
        self._buf_abilities[0, :] = features_embed[OFF_ABILITIES : OFF_ABILITIES + NUM_ABILITY_INDICES]

        return {
            "dense_features":   self._buf_dense[:, :self.expected_dense_dim],
            "species_indices":  self._buf_species,
            "move_indices":     self._buf_moves,
            "item_indices":     self._buf_items,
            "ability_indices":  self._buf_abilities,
        }

    def evaluate(self, state, player: str, valid_actions: list) -> tuple[float, dict[str, float]]:
        action_probs = {}
        
        # Build binary action mask
        mask_array = np.zeros((1, len(ACTION_SPACE)), dtype=np.float32)
        for a in valid_actions:
            if a in ACTION_SPACE:
                mask_array[0, ACTION_SPACE.index(a)] = 1.0
        
        if self.onnx_session is not None:
            features = encode_state(state, player)
            inputs = self._build_inputs(features)
            inputs["action_mask"] = mask_array
            
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
            if self.expected_dense_dim == NUM_DENSE_FEATURES or (hasattr(self.model, "input_names") and "action_mask" in self.model.input_names):
                inputs["action_mask"] = mask_array
            
            # Direct model call with training=False
            pred = self.model(inputs, training=False)

            # pred is a list of tensors starting with: [value (1,1), policy (1, ACTION_SPACE)]
            if isinstance(pred, (list, tuple)) and len(pred) >= 2:
                value_pred = pred[0]
                policy_pred = pred[1]
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

    def predict_opponent_active(self, state, player: str = "p1") -> dict:
        """
        Predicts the opponent's active Pokémon using the model's auxiliary heads.
        Returns the closest matching species from the Knowledge Base, including predicted moves.
        """
        from battle_agents.mcts_approximation.db.knowledge_base import \
            find_closest_species
        from battle_agents.mcts_approximation.db.moves_db import (_load_db,
                                                                  _move_to_idx)

        # Build action mask (unused but needed for model call shape matching)
        mask_array = np.zeros((1, len(ACTION_SPACE)), dtype=np.float32)
        mask_array[0, ACTION_SPACE.index("pass")] = 1.0
        
        features = encode_state(state, player)
        inputs = self._build_inputs(features)
        
        opp_stats = None
        opp_types = None
        opp_species = None
        opp_moves = None
        
        if self.onnx_session is not None:
            if hasattr(self, "input_names") and "action_mask" in self.input_names:
                inputs["action_mask"] = mask_array
            else:
                # Find matching input name for action_mask
                for name in self.input_names:
                    if "action_mask" in name:
                        inputs[name] = mask_array
                        
            ort_inputs = {}
            for name in self.input_names:
                clean_name = name.split(':')[0]
                if clean_name in inputs:
                    ort_inputs[name] = inputs[clean_name]
                else:
                    matched = False
                    for k in inputs:
                        if k in name or name in k:
                            ort_inputs[name] = inputs[k]
                            matched = True
                            break
            
            try:
                ort_outs = self.onnx_session.run(self.output_names, ort_inputs)
                
                # Retrieve by output name
                for name, out in zip(self.output_names, ort_outs):
                    if "aux_opp_stats" in name:
                        opp_stats = out[0]
                    elif "aux_opp_types" in name:
                        opp_types = out[0]
                    elif "aux_opp_species" in name:
                        opp_species = out[0]
                    elif "aux_opp_moves" in name:
                        opp_moves = out[0]
                        
                # Fallback by output shape if names are decorated or mismatches
                if opp_stats is None or opp_types is None or opp_species is None or opp_moves is None:
                    for out in ort_outs:
                        if out.shape == (1, 5):
                            opp_stats = out[0]
                        elif out.shape == (1, 18):
                            opp_types = out[0]
                        elif len(out.shape) == 2 and out.shape[0] == 1 and out.shape[1] > 100:
                            opp_species = out[0]
                        elif len(out.shape) == 2 and out.shape[0] == 1 and out.shape[1] > 300:
                            opp_moves = out[0]
                            
                # Index-based absolute fallback (based on standard order with 18 outputs)
                if (opp_stats is None or opp_types is None or opp_species is None or opp_moves is None) and len(ort_outs) >= 17:
                    opp_stats = ort_outs[10][0]
                    opp_types = ort_outs[12][0]
                    opp_species = ort_outs[14][0]
                    opp_moves = ort_outs[16][0]
                elif (opp_stats is None or opp_types is None or opp_species is None) and len(ort_outs) >= 15:
                    opp_stats = ort_outs[10][0]
                    opp_types = ort_outs[12][0]
                    opp_species = ort_outs[14][0]
            except Exception as e:
                print(f"[Warning] ONNX active prediction failed: {e}")
                
        elif self.model is not None:
            if self.expected_dense_dim == NUM_DENSE_FEATURES or (hasattr(self.model, "input_names") and "action_mask" in self.model.input_names):
                inputs["action_mask"] = mask_array
                
            try:
                pred = self.model(inputs, training=False)
                # Model returns outputs as list of tensors
                if isinstance(pred, (list, tuple)):
                    if len(pred) >= 17:
                        opp_stats = np.array(pred[10])[0]
                        opp_types = np.array(pred[12])[0]
                        opp_species = np.array(pred[14])[0]
                        opp_moves = np.array(pred[16])[0]
                    elif len(pred) >= 15:
                        opp_stats = np.array(pred[10])[0]
                        opp_types = np.array(pred[12])[0]
                        opp_species = np.array(pred[14])[0]
            except Exception as e:
                print(f"[Warning] Keras active prediction failed: {e}")
                
        if opp_stats is not None and opp_types is not None and opp_species is not None:
            # We assign high weight to physical coordinate dimensions (stats & types) to drive matching
            matches = find_closest_species(
                opp_species, opp_stats, opp_types,
                weight_species=1.0, weight_stats=5.0, weight_types=5.0,
                top_k=1
            )
            if matches:
                res = dict(matches[0])
                if opp_moves is not None:
                    _load_db()
                    idx_to_move = {idx: mid for mid, idx in _move_to_idx.items()}
                    top_move_idxs = np.argsort(opp_moves)[::-1]
                    top_moves = []
                    for idx in top_move_idxs:
                        if idx in idx_to_move and idx != 0:
                            top_moves.append(idx_to_move[idx])
                            if len(top_moves) == 4:
                                break
                    res["predicted_moves"] = top_moves
                else:
                    res["predicted_moves"] = []
                return res
                
        return None

