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
        # UCB1 formula: exploit (value/visits) + explore (c * sqrt(2*ln(N)/n))
        choices_weights = [
            (c.value / c.visits) + c_param * math.sqrt((2 * math.log(self.visits) / c.visits))
            for c in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

class MCTSAgent(Agent):
    """
    Monte Carlo Tree Search Agent. 
    Simulates random rollouts to find the action with the highest win confidence.
    """
    def __init__(self, problem, iterations=50, max_rollout_depth=20):
        super().__init__(problem)
        self.iterations = iterations
        self.max_rollout_depth = max_rollout_depth

    def get_action(self, state, player="p1") -> str:
        opponent = "p2" if player == "p1" else "p1"
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
                
                # MCTS explicitly models the opponent making a random choice
                opp_actions = self.problem.actions(node.state, opponent)
                opp_action = random.choice(opp_actions) if opp_actions else "pass"
                
                # Query the engine to simulate the combined actions
                if player == "p1":
                    next_state = self.problem.result(node.state, p1_action=action, p2_action=opp_action)
                else:
                    next_state = self.problem.result(node.state, p1_action=opp_action, p2_action=action)
                    
                child = MCTSNode(state=next_state, parent=node, action=action)
                child.untried_actions = self.problem.actions(child.state, player)
                
                node.children.append(child)
                node = child
                
            # 3. Simulation (Rollout)
            current_state = node.state
            depth = 0
            while not self.problem.is_terminal(current_state) and depth < self.max_rollout_depth:
                p1_actions = self.problem.actions(current_state, "p1")
                p1_act = random.choice(p1_actions) if p1_actions else "pass"
                
                p2_actions = self.problem.actions(current_state, "p2")
                p2_act = random.choice(p2_actions) if p2_actions else "pass"
                
                current_state = self.problem.result(current_state, p1_action=p1_act, p2_action=p2_act)
                depth += 1
                
            # 4. Backpropagation
            if self.problem.is_terminal(current_state):
                # 1.0 for a win, 0.0 for a loss
                p1_won = self.problem.is_goal(current_state)
                reward = 1.0 if (p1_won and player == "p1") or (not p1_won and player == "p2") else 0.0
            else:
                # 0.5 neutral reward if we hit the max depth cutoff without a clear winner
                reward = 0.5
                
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent
                
        if not root.children:
            actions = self.problem.actions(state, player)
            return random.choice(actions) if actions else "pass"
            
        # Pick the most robustly visited child
        best_node = max(root.children, key=lambda c: c.visits)
        
        print(f"  [MCTS {player}] Evaluated {self.iterations} rollouts. Selected '{best_node.action}' "
              f"(Visits: {best_node.visits}/{root.visits}, Value: {best_node.value/best_node.visits:.2f})")
        
        return best_node.action
