from battle_agents.mcts_approximation.state_encoder import (
    ACTION_SPACE, NUM_ABILITY_INDICES, NUM_DENSE_FEATURES, NUM_ITEM_INDICES,
    NUM_MOVE_INDICES, NUM_SPECIES_INDICES, OFF_ABILITIES, OFF_ITEMS, OFF_MOVES,
    OFF_SPECIES, encode_state)

import os
import numpy as np

import onnxruntime as ort

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
    def __init__(self, model_path=None):
        """Create a NeuralStateEvaluator.

        Args:
            model_path (str|None): Path to an ONNX model file. If ``None`` (the default) the evaluator
                will operate without a model and return the neutral default prediction (value=0.5,
                uniform policy). Supplying a non‑None path that does not point to an existing ``.onnx``
                file will raise ``FileNotFoundError``.
        """
        self.model = None
        self.onnx_session = None

        # Pre-allocate feature input buffers for zero-allocation builder
        self._buf_dense = np.zeros((1, NUM_DENSE_FEATURES), dtype=np.float32)
        self._buf_species = np.zeros((1, NUM_SPECIES_INDICES), dtype=np.int32)
        self._buf_moves = np.zeros((1, NUM_MOVE_INDICES), dtype=np.int32)
        self._buf_items = np.zeros((1, NUM_ITEM_INDICES), dtype=np.int32)
        self._buf_abilities = np.zeros((1, NUM_ABILITY_INDICES), dtype=np.int32)

        # Load ONNX model
        onnx_path = None
        if model_path:
            if model_path.endswith(".onnx") and os.path.exists(model_path):
                onnx_path = model_path
            else:
                raise FileNotFoundError(f"ONNX model not found or invalid at '{model_path}'. Expected a .onnx file.")

        if onnx_path:
            try:
                # By default we use CPUExecutionProvider for batch-size 1 because it is significantly faster.
                providers = ['CPUExecutionProvider']

                # Configure ONNX Runtime to run single‑threaded (fastest for batch‑size‑1 inference
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

                self.onnx_session = ort.InferenceSession(
                    onnx_path,
                    sess_options=opts,
                    providers=providers,
                )
                self.input_names = [inp.name for inp in self.onnx_session.get_inputs()]
                self.output_names = [out.name for out in self.onnx_session.get_outputs()]
                active_providers = self.onnx_session.get_providers()
                provider_str = ", ".join(active_providers)
                print(f"[NeuralStateEvaluator] Loaded ONNX model from {onnx_path} using onnxruntime ({provider_str})")
            except Exception as e:
                print(f"[Warning] Failed to load ONNX model using onnxruntime: {e}.")
        
        
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

        print(f"[NeuralStateEvaluator] Configured expected dense dimension: {self.expected_dense_dim}")

        # Configure embedding limits
        self.max_move_idx = 357
        self.max_species_idx = 388
        self.max_item_idx = 200
        self.max_ability_idx = 200

    def _build_inputs(self, features: np.ndarray) -> dict:
        """
        Split the flat feature vector into the model's inputs.
        Uses the pre‑allocated buffers for zero‑allocation slicing.
        """
        # Dense features – always the full NUM_DENSE_FEATURES (744)
        features_dense = features[:NUM_DENSE_FEATURES]
        # Embedding part follows the dense block
        features_embed = features[NUM_DENSE_FEATURES:]

        # Slice the embedding sections according to offsets
        species_in = features_embed[OFF_SPECIES : OFF_SPECIES + NUM_SPECIES_INDICES]
        moves_in = features_embed[OFF_MOVES : OFF_MOVES + NUM_MOVE_INDICES]
        items_in = features_embed[OFF_ITEMS : OFF_ITEMS + NUM_ITEM_INDICES]
        abilities_in = features_embed[OFF_ABILITIES : OFF_ABILITIES + NUM_ABILITY_INDICES]

        # Clip indices exceeding limits to 0 (padding/unknown)
        species_in = np.where(species_in < self.max_species_idx, species_in, 0)
        moves_in = np.where(moves_in < self.max_move_idx, moves_in, 0)
        items_in = np.where(items_in < self.max_item_idx, items_in, 0)
        abilities_in = np.where(abilities_in < self.max_ability_idx, abilities_in, 0)

        # Populate pre‑allocated buffers
        self._buf_dense[0, :self.expected_dense_dim] = features_dense
        self._buf_species[0, :] = species_in
        self._buf_moves[0, :] = moves_in
        self._buf_items[0, :] = items_in
        self._buf_abilities[0, :] = abilities_in

        return {
            "dense_features":   self._buf_dense[:, :self.expected_dense_dim],
            "species_indices":  self._buf_species,
            "move_indices":     self._buf_moves,
            "item_indices":     self._buf_items,
            "ability_indices":  self._buf_abilities,
        }

    def _resolve_onnx_outputs(self, ort_outs):
        """Extract opponent auxiliary predictions from ONNX outputs using output name mapping.

        Returns a tuple (opp_stats, opp_types, opp_species, opp_moves). If a particular
        output is not found, the corresponding entry will be ``None``.
        """
        # Initialise all as None
        opp_stats = opp_types = opp_species = opp_moves = None
        # Mapping from identifier substring to variable name
        name_map = {
            "aux_opp_stats": "opp_stats",
            "aux_opp_types": "opp_types",
            "aux_opp_species": "opp_species",
            "aux_opp_moves": "opp_moves",
        }
        # Iterate over output names and assign when a key matches
        for idx, name in enumerate(self.output_names):
            for key, var in name_map.items():
                if key in name:
                    if var == "opp_stats":
                        opp_stats = ort_outs[idx][0]
                    elif var == "opp_types":
                        opp_types = ort_outs[idx][0]
                    elif var == "opp_species":
                        opp_species = ort_outs[idx][0]
                    elif var == "opp_moves":
                        opp_moves = ort_outs[idx][0]
        return opp_stats, opp_types, opp_species, opp_moves

        

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
                
                # Resolve ONNX outputs using explicit name mapping
                value_pred, policy_pred = ort_outs[0], ort_outs[1]

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
                # Move to CPU before numpy conversion (handles CUDA tensors)
                def _to_np(t):
                    try:
                        if hasattr(t, 'cpu'):
                            return t.cpu().detach().numpy()
                        return np.array(t)
                    except Exception:
                        return np.array(t)
                reward = float(_to_np(value_pred)[0][0])
                policy_probs = _to_np(policy_pred)[0]

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
                
                # Retrieve opponent auxiliary predictions via helper
                opp_stats, opp_types, opp_species, opp_moves = self._resolve_onnx_outputs(ort_outs)
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
