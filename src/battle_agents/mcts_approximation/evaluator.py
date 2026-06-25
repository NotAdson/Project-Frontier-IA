from battle_agents.mcts_approximation.state_encoder import (
    ACTION_SPACE, NUM_ABILITY_INDICES, NUM_DENSE_FEATURES, NUM_ITEM_INDICES,
    NUM_MOVE_INDICES, NUM_SPECIES_INDICES, OFF_ABILITIES, OFF_ITEMS, OFF_MOVES,
    OFF_SPECIES, encode_state)

import os
import numpy as np
import logging

import onnxruntime as ort

from battle_agents.mcts_approximation.db.database import db

logger = logging.getLogger(__name__)

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
    def __init__(self, model_path):
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

        # Load model
        if not (model_path.endswith(".onnx") and os.path.exists(model_path)):
            raise FileNotFoundError(f"ONNX model not found or invalid at '{model_path}'. Expected a .onnx file.")

        # By default we use CPUExecutionProvider for batch-size 1 because it is significantly faster.
        providers = ['CPUExecutionProvider']

        # Configure ONNX Runtime to run single‑threaded (fastest for batch‑size‑1 inference
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.onnx_session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers,
        )
        self.input_names = [inp.name for inp in self.onnx_session.get_inputs()]
        self.output_names = [out.name for out in self.onnx_session.get_outputs()]
        
        # Determine expected dense_features dimension from the loaded model/session
        found = False
        for inp in self.onnx_session.get_inputs():
            if "dense_features" in inp.name:
                self.expected_dense_dim = inp.shape[1]
                found = True
                break
        if not found:
            raise RuntimeError("ONNX model does not contain a 'dense_features' input")


        print(f"[NeuralStateEvaluator] Configured expected dense dimension: {self.expected_dense_dim}")

    def _prepare_inputs_base(self, state, player: str) -> dict:
        """Encode state and build input buffers (without action mask)."""
        features = encode_state(state, player)
        return self._build_inputs(features)

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

        # Warn about out-of-range indices (helps detect data issues)
        def _warn_if_invalid(arr, name, max_idx):
            invalid = arr[arr >= max_idx]
            if invalid.size:
                logger.warning(
                    "%s contains %d out-of-range indices (>= %d): %s",
                    name, invalid.size, max_idx, invalid
                )

        _warn_if_invalid(species_in, "species_indices", db.get_num_species())
        _warn_if_invalid(moves_in, "move_indices", db.get_num_moves())
        _warn_if_invalid(items_in, "item_indices", db.get_num_items())
        _warn_if_invalid(abilities_in, "ability_indices", db.get_num_abilities())

        # Clip indices exceeding limits to 0 (padding/unknown)
        species_in = np.where(species_in < db.get_num_species(), species_in, 0)
        moves_in = np.where(moves_in < db.get_num_moves(), moves_in, 0)
        items_in = np.where(items_in < db.get_num_items(), items_in, 0)
        abilities_in = np.where(abilities_in < db.get_num_abilities(), abilities_in, 0)

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
        opp_stats = opp_types = opp_species = opp_moves = None
        name_map = {
            "aux_opp_stats": "opp_stats",
            "aux_opp_types": "opp_types",
            "aux_opp_species": "opp_species",
            "aux_opp_moves": "opp_moves",
        }

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
        # Prepare base inputs
        inputs = self._prepare_inputs_base(state, player)
        
        # Build binary action mask
        mask_array = np.zeros((1, len(ACTION_SPACE)), dtype=np.float32)
        for a in valid_actions:
            if a in ACTION_SPACE:
                mask_array[0, ACTION_SPACE.index(a)] = 1.0
        inputs["action_mask"] = mask_array

        # Run the model (ONNX only)
        if self.onnx_session is None:
            raise RuntimeError("ONNX model not loaded. Please provide a valid .onnx file path.")
        
        ort_inputs = {}
        for name in self.input_names:
            clean_name = name.split(":")[0]
            if clean_name not in inputs:
                raise ValueError(f"Missing ONNX input '{name}' (clean: '{clean_name}')")
            ort_inputs[name] = inputs[clean_name]
        ort_outs = self.onnx_session.run(self.output_names, ort_inputs)
        value_pred, policy_pred = ort_outs[0], ort_outs[1]

        # Process outputs
        reward = float(value_pred[0][0])
        policy_probs = policy_pred[0]

        action_probs = {}
        for a in valid_actions:
            idx = ACTION_SPACE.index(a) if a in ACTION_SPACE else ACTION_SPACE.index("pass")
            action_probs[a] = float(policy_probs[idx])

        total = sum(action_probs.values())
        if total > 0:
            for a in action_probs:
                action_probs[a] /= total
        else:
            uniform = 1.0 / len(valid_actions)
            for a in action_probs:
                action_probs[a] = uniform

        return reward, action_probs

    def predict_opponent_active(self, state, player: str = "p1") -> dict:
        """
        Predicts the opponent's active Pokémon using the model's auxiliary heads.
        Returns the closest matching species from the Knowledge Base, including predicted moves.
        If the model does not provide the required auxiliary outputs, returns None.
        """
        from battle_agents.mcts_approximation.db.knowledge_base import find_closest_species
        from battle_agents.mcts_approximation.db.database import db

        # Prepare base inputs
        inputs = self._prepare_inputs_base(state, player)
        
        # Build action mask (unused but needed for model call shape matching)
        mask_array = np.zeros((1, len(ACTION_SPACE)), dtype=np.float32)
        mask_array[0, ACTION_SPACE.index("pass")] = 1.0
        # Check if the model expects an action_mask input (by substring in the input name)
        if any("action_mask" in name.split(":")[0] for name in self.input_names):
            inputs["action_mask"] = mask_array

        if self.onnx_session is None:
            logger.warning("No ONNX model loaded for opponent active prediction.")
            return None

        # Build ONNX inputs with exact name matching
        ort_inputs = {}
        for name in self.input_names:
            clean_name = name.split(":")[0]  # "dense_features:index" → "dense_features"
            if clean_name not in inputs:
                raise ValueError(f"Missing ONNX input '{name}' (clean: '{clean_name}')")
            ort_inputs[name] = inputs[clean_name]
        ort_outs = self.onnx_session.run(self.output_names, ort_inputs)
        opp_stats, opp_types, opp_species, opp_moves = self._resolve_onnx_outputs(ort_outs)

        # If we have the three core outputs, proceed to find the closest species
        if opp_stats is not None and opp_types is not None and opp_species is not None:
            matches = find_closest_species(
                opp_species, opp_stats, opp_types,
                weight_species=1.0, weight_stats=5.0, weight_types=5.0,
                top_k=1
            )
            if matches:
                res = dict(matches[0])
                if opp_moves is not None:
                    db.load_moves()
                    idx_to_move = {idx: mid for mid, idx in db.move_to_idx.items()}
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
        else:
            # Missing one or more of opp_stats, opp_types, opp_species
            logger.warning(
                "Opponent auxiliary outputs incomplete: stats=%s, types=%s, species=%s",
                opp_stats is not None, opp_types is not None, opp_species is not None
            )
