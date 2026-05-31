import math
import random
import os
import numpy as np
import numba

# Keras is imported lazily inside __init__ as a fallback to avoid backend loading overhead
keras = None

from core.agent import Agent
from battle_agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent, MCTSNode
from battle_agents.mcts_approximation.state_encoder import (
    encode_state,
    NUM_DENSE_FEATURES,
    NUM_SPECIES_INDICES, NUM_ITEM_INDICES, NUM_ABILITY_INDICES,
    NUM_BENCH_MOVE_INDICES, NUM_ACTIVE_MOVE_INDICES,
    NUM_OPP_SPECIES_INDICES, NUM_OPP_MOVE_INDICES,
    OFF_SPECIES, OFF_ITEMS, OFF_ABILITIES, OFF_BENCH_MOVES, OFF_ACTIVE_MOVES,
    OFF_OPP_SPECIES, OFF_OPP_MOVES,
)

@numba.njit(fastmath=True, cache=True)
def _numba_best_child_idx(visits_arr, values_arr, priors_arr, parent_visits, c_param):
    best_score = -1e9
    best_idx = 0
    parent_visits_sqrt = math.sqrt(parent_visits)
    for i in range(len(visits_arr)):
        v = visits_arr[i]
        q = values_arr[i] / v if v > 0.0 else 0.0
        u = c_param * priors_arr[i] * parent_visits_sqrt / (1.0 + v)
        score = q + u
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


class MCTSApproximationNode:
    def __init__(self, state, parent=None, action=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.is_expanded = False
        
        # Zero-allocation numpy arrays for JIT child selection
        self.child_index = 0
        self._visits_arr = None
        self._values_arr = None
        self._priors_arr = None

    def best_child(self, c_param=1.414):
        if not self.children:
            return None
        best_idx = _numba_best_child_idx(self._visits_arr, self._values_arr, self._priors_arr, float(self.visits), c_param)
        return self.children[best_idx]


class MCTSApproximationAgent(BlindMCTSAgent):
    """
    MCTS Approximation Agent.
    Replaces random rollouts with a Neural Network evaluation of the state.
    The model uses 8 inputs and outputs both a value (probability of winning) 
    and a policy (probability distribution over actions).
    """
    def __init__(self, problem, iterations=50, model_path="data/mcts_model.keras", **kwargs):
        super().__init__(problem, iterations=iterations, max_rollout_depth=0)

        self.model = None
        self.onnx_session = None

        # Pre-allocate feature input buffers for zero-allocation builder
        self._buf_dense = np.zeros((1, 163), dtype=np.float32)
        self._buf_species = np.zeros((1, 6), dtype=np.int32)
        self._buf_items = np.zeros((1, 6), dtype=np.int32)
        self._buf_abilities = np.zeros((1, 6), dtype=np.int32)
        self._buf_bench_moves = np.zeros((1, 24), dtype=np.int32)
        self._buf_moves = np.zeros((1, 4), dtype=np.int32)
        self._buf_opp_species = np.zeros((1, 6), dtype=np.int32)
        self._buf_opp_moves = np.zeros((1, 24), dtype=np.int32)

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
                print(f"[MCTSApproximationAgent] Loaded ONNX model from {onnx_path} using onnxruntime (CPU, 1 thread)")
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
                        print(f"[MCTSApproximationAgent] Loaded Keras model from {actual_path} as fallback")
                    except Exception as err_keras:
                        print(f"[Warning] Failed to load Keras model due to Keras version mismatch/deserialization issues: {err_keras}. "
                              "Search will continue with default (0.5 value) predictions.")
                else:
                    print(f"[Warning] MCTS Approximation model not found at {model_path}. "
                          "Will predict 0.5 for all states.")
            else:
                print("[Error] Keras is not installed. Cannot use Neural Network fallback. "
                      "Please pip install keras and a backend (tensorflow / torch / jax).")

    # ─── Feature vector → model inputs ─────────────────────────────────────

    def _build_inputs(self, features: np.ndarray) -> dict:
        """
        Splits the feature vector into the 8 named model inputs.

        Feature layout:
            [0:163]   dense_features
            [163:169] species_indices     (6)
            [169:175] item_indices        (6)
            [175:181] ability_indices     (6)
            [181:205] bench_move_indices  (24)
            [205:209] move_indices        (4)
            [209:215] opp_species_indices (6, 0 if not revealed)
            [215:239] opp_move_indices    (24, 0 if not yet seen)
        """
        n = NUM_DENSE_FEATURES  # 163
        
        # In-place copy to pre-allocated buffers (zero allocation!)
        self._buf_dense[0, :] = features[:n]
        self._buf_species[0, :] = features[n + OFF_SPECIES : n + OFF_SPECIES + NUM_SPECIES_INDICES]
        self._buf_items[0, :] = features[n + OFF_ITEMS : n + OFF_ITEMS + NUM_ITEM_INDICES]
        self._buf_abilities[0, :] = features[n + OFF_ABILITIES : n + OFF_ABILITIES + NUM_ABILITY_INDICES]
        self._buf_bench_moves[0, :] = features[n + OFF_BENCH_MOVES : n + OFF_BENCH_MOVES + NUM_BENCH_MOVE_INDICES]
        self._buf_moves[0, :] = features[n + OFF_ACTIVE_MOVES : n + OFF_ACTIVE_MOVES + NUM_ACTIVE_MOVE_INDICES]
        self._buf_opp_species[0, :] = features[n + OFF_OPP_SPECIES : n + OFF_OPP_SPECIES + NUM_OPP_SPECIES_INDICES]
        self._buf_opp_moves[0, :] = features[n + OFF_OPP_MOVES : n + OFF_OPP_MOVES + NUM_OPP_MOVE_INDICES]

        return {
            "dense_features":      self._buf_dense,
            "species_indices":     self._buf_species,
            "item_indices":        self._buf_items,
            "ability_indices":     self._buf_abilities,
            "bench_move_indices":  self._buf_bench_moves,
            "move_indices":        self._buf_moves,
            "opp_species_indices": self._buf_opp_species,
            "opp_move_indices":    self._buf_opp_moves,
        }

    # ─── MCTS loop ──────────────────────────────────────────────────────────

    def get_action(self, state, player="p1", return_probs=False, temperature=0.0, **kwargs):
        valid_actions = self.problem.actions(state, player)
        if len(valid_actions) <= 1:
            action = valid_actions[0] if valid_actions else "pass"
            if return_probs:
                return action, {action: 1.0}
            return action

        from battle_agents.mcts_approximation.state_encoder import ACTION_SPACE
        
        root = MCTSApproximationNode(state=state)
        
        # Initial expansion
        self._expand(root, player, ACTION_SPACE)

        for _ in range(self.iterations):
            node = root
            
            # 1. Selection
            while node.is_expanded and not self.problem.is_terminal(node.state):
                node = node.best_child()
                # Lazy state generation to avoid expensive engine calls for all leaves
                if node.state is None:
                    opp_player = "p2" if player == "p1" else "p1"
                    opp_actions = self.problem.actions(node.parent.state, opp_player)
                    opp_action = random.choice(opp_actions) if opp_actions else None
                    if player == "p1":
                        node.state = self.problem.result(node.parent.state, p1_action=node.action, p2_action=opp_action)
                    else:
                        node.state = self.problem.result(node.parent.state, p1_action=opp_action, p2_action=node.action)

            # 2. Expansion & Evaluation
            if self.problem.is_terminal(node.state):
                p1_won = self.problem.is_goal(node.state)
                reward = 1.0 if (p1_won and player == "p1") or (not p1_won and player == "p2") else 0.0
            else:
                reward = self._expand(node, player, ACTION_SPACE)

            # 3. Backpropagation
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value += reward
                if curr.parent is not None:
                    curr.parent._visits_arr[curr.child_index] = curr.visits
                    curr.parent._values_arr[curr.child_index] = curr.value
                curr = curr.parent

        if not root.children:
            actions = self.problem.actions(state, player)
            best_action = random.choice(actions) if actions else "pass"
            return (best_action, {best_action: 1.0}) if return_probs else best_action

        total_visits = sum(c.visits for c in root.children)
        
        # 1. Calculate raw visit probabilities (Training labels)
        if total_visits == 0:
            prob = 1.0 / len(root.children)
            action_probs = {c.action: prob for c in root.children}
        else:
            action_probs = {c.action: c.visits / total_visits for c in root.children}

        # 2. Select final action applying temperature
        if temperature > 0.0 and total_visits > 0:
            weights = [math.pow(c.visits, 1.0 / temperature) for c in root.children]
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            chosen_action = np.random.choice([c.action for c in root.children], p=probs)
        else:
            chosen_action = max(root.children, key=lambda c: c.visits).action
        
        return (chosen_action, action_probs) if return_probs else chosen_action

    def _expand(self, node, player, action_space):
        """Evaluates node with NN and creates lazily-evaluated children."""
        valid_actions = self.problem.actions(node.state, player)
        if not valid_actions:
            node.is_expanded = True
            return 0.5
            
        action_probs = {}
        if self.onnx_session is not None:
            features = encode_state(node.state, player)
            inputs = self._build_inputs(features)
            
            # Map Python dictionary inputs to exact ONNX session input names (robust to name suffixes)
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
                        # Default zero array fallback if input not provided
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
                        elif out.shape == (1, len(action_space)):
                            policy_pred = out

                # absolute final fallback by index
                if value_pred is None:
                    value_pred = ort_outs[0]
                if policy_pred is None:
                    policy_pred = ort_outs[1]

                reward = float(value_pred[0][0])
                policy_probs = policy_pred[0]

                for a in valid_actions:
                    idx = action_space.index(a) if a in action_space else action_space.index("pass")
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
            features = encode_state(node.state, player)
            inputs = self._build_inputs(features)
            # Direct model call with training=False — works on all Keras 3 backends
            # and is equivalent in speed to the deprecated predict_on_batch
            pred = self.model(inputs, training=False)

            # pred is a list of two tensors: [value (1,1), policy (1, ACTION_SPACE)]
            if isinstance(pred, (list, tuple)) and len(pred) == 2:
                value_pred, policy_pred = pred
                reward = float(np.array(value_pred)[0][0])
                policy_probs = np.array(policy_pred)[0]

                for a in valid_actions:
                    idx = action_space.index(a) if a in action_space else action_space.index("pass")
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
                
        for idx, a in enumerate(valid_actions):
            child = MCTSApproximationNode(state=None, parent=node, action=a, prior_prob=action_probs[a])
            child.child_index = idx
            node.children.append(child)
            
        # Pre-allocate numpy arrays for JIT child selection
        n_children = len(node.children)
        node._visits_arr = np.zeros(n_children, dtype=np.float64)
        node._values_arr = np.zeros(n_children, dtype=np.float64)
        node._priors_arr = np.array([c.prior_prob for c in node.children], dtype=np.float64)
            
        node.is_expanded = True
        return reward
