import math
import random
from core.agent import Agent

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0 and len(self.children) > 0

    def best_child(self, c_param=1.414):
        choices_weights = [
            (c.value / c.visits) + c_param * math.sqrt((2 * math.log(self.visits) / c.visits))
            for c in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

class BlindMCTSAgent(Agent):
    """
    Blind Monte Carlo Tree Search Agent. 
    Unlike standard MCTS, it does NOT peek at the opponent's valid actions to model them.
    Instead, it passes the opponent's turn back to the engine to automatically simulate a generic fallback.
    """
    def __init__(self, problem, iterations=50, max_rollout_depth=20):
        super().__init__(problem)
        self.iterations = iterations
        self.max_rollout_depth = max_rollout_depth

    def get_action(self, state, player="p1", return_probs=False):
        root = MCTSNode(state=state)
        root.untried_actions = self.problem.actions(root.state, player)
        
        for i in range(self.iterations):
            node = root
            
            # 1. Selection
            while node.is_fully_expanded() and not self.problem.is_terminal(node.state):
                node = node.best_child()
                
            # 2. Expansion
            if not self.problem.is_terminal(node.state) and len(node.untried_actions) > 0:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                # BLIND MCTS: Do NOT query opponent actions. 
                # Let the engine handle the opponent via its `default` fallback logic.
                if player == "p1":
                    next_state = self.problem.result(node.state, p1_action=action, p2_action=None)
                else:
                    next_state = self.problem.result(node.state, p1_action=None, p2_action=action)
                    
                child = MCTSNode(state=next_state, parent=node, action=action)
                child.untried_actions = self.problem.actions(child.state, player)
                
                node.children.append(child)
                node = child
                
            # 3. Simulation (Rollout)
            if hasattr(self.problem, 'rollout'):
                reward = self.problem.rollout(node.state, player, self.max_rollout_depth)
            else:
                current_state = node.state
                depth = 0
                while not self.problem.is_terminal(current_state) and depth < self.max_rollout_depth:
                    p_actions = self.problem.actions(current_state, player)
                    p_act = random.choice(p_actions) if p_actions else "pass"
                    
                    # BLIND MCTS: Again, we pass None for the opponent.
                    if player == "p1":
                        current_state = self.problem.result(current_state, p1_action=p_act, p2_action=None)
                    else:
                        current_state = self.problem.result(current_state, p1_action=None, p2_action=p_act)
                        
                    depth += 1
                    
                # 4. Backpropagation
                if self.problem.is_terminal(current_state):
                    p1_won = self.problem.is_goal(current_state)
                    reward = 1.0 if (p1_won and player == "p1") or (not p1_won and player == "p2") else 0.0
                else:
                    reward = 0.5
                
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
            # If total_visits is 0 (which shouldn't happen with valid iterations), fallback to uniform
            if total_visits == 0:
                prob = 1.0 / len(root.children)
                action_probs = {c.action: prob for c in root.children}
            else:
                action_probs = {c.action: c.visits / total_visits for c in root.children}
            return best_node.action, action_probs
            
        # print(f"  [BlindMCTS {player}] Evaluated {self.iterations} rollouts. Selected '{best_node.action}' (Visits: {best_node.visits}/{self.iterations}, Value: {best_node.value/best_node.visits:.2f})")
        return best_node.action
