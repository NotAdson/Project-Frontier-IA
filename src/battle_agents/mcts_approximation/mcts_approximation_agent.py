import math
import random
import os
import numpy as np

from core.agent import Agent
from battle_agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent
from battle_agents.mcts_approximation.state_encoder import ACTION_SPACE


def select_p1_action(node, c_param=1.414):
    best_score = -1e9
    best_action = None
    parent_visits_sqrt = math.sqrt(node.visits) if node.visits > 0 else 1.0
    for a in node.p1_actions:
        v = node.p1_visits[a]
        q = node.p1_values[a] / v if v > 0.0 else 0.5
        u = c_param * node.p1_priors.get(a, 0.0) * parent_visits_sqrt / (1.0 + v)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = a
    return best_action if best_action is not None else random.choice(node.p1_actions)


def select_p2_action(node, c_param=1.414):
    best_score = -1e9
    best_action = None
    parent_visits_sqrt = math.sqrt(node.visits) if node.visits > 0 else 1.0
    for a in node.p2_actions:
        v = node.p2_visits[a]
        # Opponent maximizes (1.0 - reward)
        q = node.p2_values[a] / v if v > 0.0 else 0.5
        u = c_param * node.p2_priors.get(a, 0.0) * parent_visits_sqrt / (1.0 + v)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = a
    return best_action if best_action is not None else random.choice(node.p2_actions)


class MCTSApproximationNode:
    def __init__(self, state, parent=None, joint_action=None):
        self.state = state
        self.parent = parent
        self.joint_action = joint_action  # (a1, a2) that led to this node
        self.children = {}  # dict mapping (a1, a2) -> MCTSApproximationNode
        self.visits = 0
        self.value = 0.0  # from P1's perspective
        self.is_expanded = False
        
        # Player 1 action statistics
        self.p1_actions = []
        self.p1_visits = {}
        self.p1_values = {}
        self.p1_priors = {}
        
        # Player 2 action statistics
        self.p2_actions = []
        self.p2_visits = {}
        self.p2_values = {}
        self.p2_priors = {}


class MCTSApproximationAgent(BlindMCTSAgent):
    """
    MCTS Approximation Agent using Decoupled UCT (DUCT) to support simultaneous decision-making.
    Replaces random rollouts with a Neural Network evaluation of the state.
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
        self._expand(root)

        for _ in range(self.iterations):
            node = root
            
            # 1. Selection
            while node.is_expanded and not self.problem.is_terminal(node.state):
                a1 = select_p1_action(node)
                a2 = select_p2_action(node)
                joint_action = (a1, a2)
                
                if joint_action in node.children:
                    node = node.children[joint_action]
                else:
                    # Transition to a new child state using both P1 and P2 actions
                    child_state = self.problem.result(node.state, p1_action=a1, p2_action=a2)
                    child_node = MCTSApproximationNode(state=child_state, parent=node, joint_action=joint_action)
                    node.children[joint_action] = child_node
                    node = child_node
                    break

            # 2. Expansion & Evaluation
            if self.problem.is_terminal(node.state):
                p1_won = self.problem.is_goal(node.state)
                reward = 1.0 if p1_won else 0.0
            else:
                reward = self._expand(node)

            # 3. Backpropagation (Zero-sum: P1 gets reward, P2 gets 1.0 - reward)
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value += reward
                parent = curr.parent
                if parent is not None:
                    a1, a2 = curr.joint_action
                    parent.p1_visits[a1] += 1
                    parent.p1_values[a1] += reward
                    parent.p2_visits[a2] += 1
                    parent.p2_values[a2] += (1.0 - reward)
                curr = parent

        # Determine controlled player action statistics at root
        if player == "p1":
            actions_list = root.p1_actions
            visits_dict = root.p1_visits
        else:
            actions_list = root.p2_actions
            visits_dict = root.p2_visits

        total_visits = sum(visits_dict[a] for a in actions_list)
        
        # 1. Calculate raw visit probabilities (Training labels)
        if total_visits == 0:
            prob = 1.0 / len(actions_list)
            action_probs = {a: prob for a in actions_list}
        else:
            action_probs = {a: visits_dict[a] / total_visits for a in actions_list}

        # 2. Select final action applying temperature
        if temperature > 0.0 and total_visits > 0:
            weights = [math.pow(visits_dict[a], 1.0 / temperature) for a in actions_list]
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            chosen_action = np.random.choice(actions_list, p=probs)
        else:
            chosen_action = max(actions_list, key=lambda a: visits_dict[a])
        
        return (chosen_action, action_probs) if return_probs else chosen_action

    def _expand(self, node):
        """Evaluates node state and initializes choice lists and priors for both players."""
        node.p1_actions = self.problem.actions(node.state, "p1")
        if not node.p1_actions:
            node.p1_actions = ["pass"]
            
        node.p2_actions = self.problem.actions(node.state, "p2")
        if not node.p2_actions:
            node.p2_actions = ["pass"]

        # Evaluate P1 priors and state value from P1 perspective
        p1_val, p1_probs = self.evaluator.evaluate(node.state, "p1", node.p1_actions)
        node.p1_priors = p1_probs
        
        # Evaluate P2 priors using censored state to avoid opponent cheating
        censored_state = self._censor_opponent_state(node.state, "p2")
        _, p2_probs = self.evaluator.evaluate(censored_state, "p2", node.p2_actions)
        node.p2_priors = p2_probs
        
        # Initialize visits and values to 0 for all valid actions
        for a in node.p1_actions:
            node.p1_visits[a] = 0.0
            node.p1_values[a] = 0.0
        for a in node.p2_actions:
            node.p2_visits[a] = 0.0
            node.p2_values[a] = 0.0
            
        node.is_expanded = True
        return p1_val

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
