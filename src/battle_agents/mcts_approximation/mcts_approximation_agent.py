import math
import random
import os
import numpy as np
import numba

from core.agent import Agent
from battle_agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent, MCTSNode
from battle_agents.mcts_approximation.state_encoder import ACTION_SPACE

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
    Evaluations are delegated to a pluggable StateEvaluator.
    """
    def __init__(self, problem, iterations=50, evaluator=None, model_path="data/mcts_model.keras", **kwargs):
        super().__init__(problem, iterations=iterations, max_rollout_depth=0)

        # Pluggable state evaluator
        if evaluator is not None:
            self.evaluator = evaluator
        else:
            from battle_agents.mcts_approximation.evaluator import NeuralStateEvaluator
            self.evaluator = NeuralStateEvaluator(model_path)

    # ─── MCTS loop ──────────────────────────────────────────────────────────

    def get_action(self, state, player="p1", return_probs=False, temperature=0.0, **kwargs):
        valid_actions = self.problem.actions(state, player)
        if len(valid_actions) <= 1:
            action = valid_actions[0] if valid_actions else "pass"
            if return_probs:
                return action, {action: 1.0}
            return action
        
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
                    
                    # Censor opponent state to prevent cheating (no access to unrevealed info)
                    censored_parent_state = self._censor_opponent_state(node.parent.state, player)
                    opp_actions = self.problem.actions(censored_parent_state, opp_player)
                    
                    opp_action = None
                    if opp_actions:
                        # Policy-guided opponent action selection using censored state
                        _, opp_action_probs = self.evaluator.evaluate(censored_parent_state, opp_player, opp_actions)
                        opp_probs = [opp_action_probs.get(a, 0.0) for a in opp_actions]
                        sum_probs = sum(opp_probs)
                        if sum_probs > 0:
                            opp_probs = [p / sum_probs for p in opp_probs]
                            opp_action = np.random.choice(opp_actions, p=opp_probs)
                        else:
                            opp_action = random.choice(opp_actions)
                            
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
        """Evaluates node with evaluator and creates lazily-evaluated children."""
        valid_actions = self.problem.actions(node.state, player)
        if not valid_actions:
            node.is_expanded = True
            return 0.5
            
        reward, action_probs = self.evaluator.evaluate(node.state, player, valid_actions)
                
        for idx, a in enumerate(valid_actions):
            child = MCTSApproximationNode(state=None, parent=node, action=a, prior_prob=action_probs.get(a, 0.0))
            child.child_index = idx
            node.children.append(child)
            
        # Pre-allocate numpy arrays for JIT child selection
        n_children = len(node.children)
        node._visits_arr = np.zeros(n_children, dtype=np.float64)
        node._values_arr = np.zeros(n_children, dtype=np.float64)
        node._priors_arr = np.array([c.prior_prob for c in node.children], dtype=np.float64)
            
        node.is_expanded = True
        return reward

    def _censor_opponent_state(self, state, player):
        """
        Returns a new PokemonState with a copy of state_dict where the opponent's
        unrevealed Pokémon, moves, and items are stripped out.
        This ensures MCTS opponent evaluation uses only public information.
        """
        import copy
        from core.problem.pokemon_problem import PokemonState
        
        state_dict = state.state_dict
        censored_dict = copy.deepcopy(state_dict)
        
        opp_player = "p2" if player == "p1" else "p1"
        opp_idx = 1 if opp_player == "p2" else 0
        
        sides = censored_dict.get("sides", [])
        if len(sides) > opp_idx:
            opp_side = sides[opp_idx]
            pokemon_list = opp_side.get("pokemon", [])
            for p in pokemon_list:
                # Determine if this Pokémon has ever been revealed
                is_revealed = (
                    p.get("isActive", False)
                    or p.get("previouslySwitchedIn", 0) > 0
                    or p.get("fainted", False)
                )
                if not is_revealed:
                    # Clear completely (unknown Pokémon)
                    p["details"] = ""
                    p["hp"] = 0
                    p["maxhp"] = 1
                    p["condition"] = "0 fnt"
                    p["fainted"] = False
                    p["status"] = ""
                    p["moveSlots"] = []
                    p["item"] = ""
                    p["ability"] = ""
                    p["baseAbility"] = ""
                else:
                    # Keep only the moves that have been used/revealed in the battle
                    move_slots = p.get("moveSlots", [])
                    p["moveSlots"] = [m for m in move_slots if m.get("used", False)]
                    
        return PokemonState(
            censored_dict, 
            request_dict=state.request_dict, 
            p2_request_dict=state.p2_request_dict, 
            log=state.log, 
            winner=state.winner
        )
