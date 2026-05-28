import math
import random
import os
import numpy as np

try:
    import tensorflow as tf
    # Enable memory growth so multiprocessing doesn't crash from OOM
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except ImportError:
    tf = None
except Exception as e:
    print(f"[Warning] Failed to configure GPU memory growth: {e}")

from core.agent import Agent
from agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent, MCTSNode
from agents.mcts_approximation.state_encoder import (
    encode_state,
    NUM_DENSE_FEATURES,
    NUM_SPECIES_INDICES, NUM_ITEM_INDICES, NUM_ABILITY_INDICES,
    NUM_BENCH_MOVE_INDICES, NUM_ACTIVE_MOVE_INDICES,
    NUM_OPP_SPECIES_INDICES, NUM_OPP_MOVE_INDICES,
    OFF_SPECIES, OFF_ITEMS, OFF_ABILITIES, OFF_BENCH_MOVES, OFF_ACTIVE_MOVES,
    OFF_OPP_SPECIES, OFF_OPP_MOVES,
)


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

    def best_child(self, c_param=1.414):
        best_score = -float('inf')
        best_c = None
        for c in self.children:
            q = c.value / c.visits if c.visits > 0 else 0.0
            u = c_param * c.prior_prob * math.sqrt(self.visits) / (1 + c.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_c = c
        return best_c


class MCTSApproximationAgent(BlindMCTSAgent):
    """
    MCTS Approximation Agent.
    Replaces random rollouts with a Neural Network evaluation of the state.
    The model uses 8 inputs and outputs both a value (probability of winning) 
    and a policy (probability distribution over actions).
    """
    def __init__(self, problem, iterations=50, max_rollout_depth=0,
                 model_path="data/mcts_model.keras"):
        super().__init__(problem, iterations=iterations, max_rollout_depth=0)

        self.model = None
        if tf is not None:
            fallback = "data/mcts_model.h5"
            actual_path = (model_path if os.path.exists(model_path)
                           else fallback if os.path.exists(fallback)
                           else None)
            if actual_path:
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                self.model = tf.keras.models.load_model(actual_path, compile=False)
            else:
                print(f"[Warning] MCTS Approximation model not found at {model_path}. "
                      "Will predict 0.5 for all states.")
        else:
            print("[Error] TensorFlow not installed. Cannot use Neural Network. "
                  "Please pip install tensorflow.")

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
        dense        = features[:n]
        species      = features[n + OFF_SPECIES     : n + OFF_SPECIES     + NUM_SPECIES_INDICES    ].astype(np.int32)
        items        = features[n + OFF_ITEMS       : n + OFF_ITEMS       + NUM_ITEM_INDICES       ].astype(np.int32)
        abilities    = features[n + OFF_ABILITIES   : n + OFF_ABILITIES   + NUM_ABILITY_INDICES    ].astype(np.int32)
        bench_moves  = features[n + OFF_BENCH_MOVES : n + OFF_BENCH_MOVES + NUM_BENCH_MOVE_INDICES ].astype(np.int32)
        moves        = features[n + OFF_ACTIVE_MOVES: n + OFF_ACTIVE_MOVES + NUM_ACTIVE_MOVE_INDICES].astype(np.int32)
        opp_species  = features[n + OFF_OPP_SPECIES : n + OFF_OPP_SPECIES + NUM_OPP_SPECIES_INDICES].astype(np.int32)
        opp_moves    = features[n + OFF_OPP_MOVES   : n + OFF_OPP_MOVES   + NUM_OPP_MOVE_INDICES   ].astype(np.int32)

        return {
            "dense_features":      np.expand_dims(dense,       axis=0),
            "species_indices":     np.expand_dims(species,     axis=0),
            "item_indices":        np.expand_dims(items,       axis=0),
            "ability_indices":     np.expand_dims(abilities,   axis=0),
            "bench_move_indices":  np.expand_dims(bench_moves, axis=0),
            "move_indices":        np.expand_dims(moves,       axis=0),
            "opp_species_indices": np.expand_dims(opp_species, axis=0),
            "opp_move_indices":    np.expand_dims(opp_moves,   axis=0),
        }

    # ─── MCTS loop ──────────────────────────────────────────────────────────

    def get_action(self, state, player="p1", return_probs=False, add_noise=False):
        from agents.mcts_approximation.state_encoder import ACTION_SPACE
        
        root = MCTSApproximationNode(state=state)
        
        # Initial expansion
        self._expand(root, player, ACTION_SPACE)
        
        # Add Dirichlet noise to the root node for exploration during self-play
        if add_noise and root.children:
            alpha = 0.3
            epsilon = 0.25
            noise = np.random.dirichlet([alpha] * len(root.children))
            for i, child in enumerate(root.children):
                child.prior_prob = (1 - epsilon) * child.prior_prob + epsilon * noise[i]

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
                curr = curr.parent

        if not root.children:
            actions = self.problem.actions(state, player)
            best_action = random.choice(actions) if actions else "pass"
            if return_probs:
                return best_action, {best_action: 1.0}
            return best_action

        best_node = max(root.children, key=lambda c: c.visits)
        
        if return_probs:
            total_visits = sum(c.visits for c in root.children)
            if total_visits == 0:
                prob = 1.0 / len(root.children)
                action_probs = {c.action: prob for c in root.children}
            else:
                action_probs = {c.action: c.visits / total_visits for c in root.children}
            return best_node.action, action_probs
            
        return best_node.action

    def _expand(self, node, player, action_space):
        """Evaluates node with NN and creates lazily-evaluated children."""
        valid_actions = self.problem.actions(node.state, player)
        if not valid_actions:
            node.is_expanded = True
            return 0.5
            
        action_probs = {}
        if self.model is not None:
            features = encode_state(node.state, player)
            inputs = self._build_inputs(features)
            # predict_on_batch is roughly 10x faster than predict() for single items on CPUs
            pred = self.model.predict_on_batch(inputs)
            
            # Handle both single output (old model) and multiple outputs (new model) gracefully
            if isinstance(pred, list) and len(pred) == 2:
                value_pred, policy_pred = pred
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
            else:
                reward = float(pred[0][0]) if not isinstance(pred, list) else float(pred[0][0][0])
                for a in valid_actions:
                    action_probs[a] = 1.0 / len(valid_actions)
        else:
            reward = 0.5
            for a in valid_actions:
                action_probs[a] = 1.0 / len(valid_actions)
                
        for a in valid_actions:
            child = MCTSApproximationNode(state=None, parent=node, action=a, prior_prob=action_probs[a])
            node.children.append(child)
            
        node.is_expanded = True
        return reward
