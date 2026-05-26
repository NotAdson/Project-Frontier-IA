import math
import random
import os
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

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


class MCTSApproximationAgent(BlindMCTSAgent):
    """
    MCTS Approximation Agent.
    Replaces random rollouts with a Neural Network evaluation of the state.
    The model uses 5 inputs:
        dense_features, species_indices, item_indices, ability_indices, move_indices
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
        Splits the 221-element feature vector into the 8 named model inputs.

        Feature layout:
            [0:145]   dense_features
            [145:151] species_indices     (6)
            [151:157] item_indices        (6)
            [157:163] ability_indices     (6)
            [163:187] bench_move_indices  (24)
            [187:191] move_indices        (4)
            [191:197] opp_species_indices (6, 0 if not revealed)
            [197:221] opp_move_indices    (24, 0 if not yet seen)
        """
        n = NUM_DENSE_FEATURES  # 145
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

    def get_action(self, state, player="p1", return_probs=False):
        root = MCTSNode(state=state)
        root.untried_actions = self.problem.actions(root.state, player)

        for _ in range(self.iterations):
            node = root

            # 1. Selection
            while node.is_fully_expanded() and not self.problem.is_terminal(node.state):
                node = node.best_child()

            # 2. Expansion
            if not self.problem.is_terminal(node.state) and len(node.untried_actions) > 0:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)

                # Sample a random but valid opponent action (improvement over None/engine-default).
                # The engine default is likely random anyway, but this ensures we explore
                # realistic simultaneous-action pairs during tree expansion.
                opp_player = "p2" if player == "p1" else "p1"
                opp_actions = self.problem.actions(node.state, opp_player)
                opp_action = random.choice(opp_actions) if opp_actions else None

                if player == "p1":
                    next_state = self.problem.result(node.state, p1_action=action, p2_action=opp_action)
                else:
                    next_state = self.problem.result(node.state, p1_action=opp_action, p2_action=action)

                child = MCTSNode(state=next_state, parent=node, action=action)
                child.untried_actions = self.problem.actions(child.state, player)
                node.children.append(child)
                node = child

            # 3. Approximation (Neural Network instead of rollout)
            current_state = node.state

            if self.problem.is_terminal(current_state):
                p1_won = self.problem.is_goal(current_state)
                reward = 1.0 if (p1_won and player == "p1") or (not p1_won and player == "p2") else 0.0
            else:
                if self.model is not None:
                    features = encode_state(current_state, player)
                    inputs = self._build_inputs(features)
                    prediction = self.model.predict(inputs, verbose=0)[0][0]
                    reward = float(prediction)
                else:
                    reward = 0.5

            # 4. Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

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
