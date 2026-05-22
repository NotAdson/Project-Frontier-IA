import json
import os
from core.problem.aima_problem import Problem
from core.client.showdown_client import ShowdownClient

class PokemonState:
    def __init__(self, state_dict, request_dict=None, p2_request_dict=None, log=None, winner=None):
        self.state_dict = state_dict
        self.request_dict = request_dict
        self.p2_request_dict = p2_request_dict
        self.log = log or []
        self.winner = winner
        
    def __eq__(self, other):
        # Using string comparison for simplicity, though feature extraction is better for large scale search.
        return json.dumps(self.state_dict, sort_keys=True) == json.dumps(other.state_dict, sort_keys=True)
        
    def __hash__(self):
        return hash(json.dumps(self.state_dict, sort_keys=True))

class PokemonProblem(Problem):
    def __init__(self, client: ShowdownClient, formatid='gen3randombattle'):
        self.client = client
        resp = self.client.init_battle(formatid=formatid)
        initial_state = PokemonState(resp['state'], resp.get('request'), resp.get('p2_request'), resp.get('log'), resp.get('winner'))
        super().__init__(initial_state)

    def actions(self, state: PokemonState, player="p1"):
        """ Returns valid actions from the given state for the specified player. """
        actions = []
        request = state.request_dict if player == "p1" else state.p2_request_dict
        
        if not request: return ["pass"]
        
        if request.get('forceSwitch'):
            # Player is forced to switch
            side = request.get('side', {})
            for i, p in enumerate(side.get('pokemon', [])):
                if not p.get('active') and p.get('condition') != '0 fnt':
                    actions.append(f"switch {i+1}")
        elif request.get('active'):
            # Player can choose a move or switch
            for i, move in enumerate(request['active'][0].get('moves', [])):
                # Note: 'disabled' attribute usually determines if a move can't be used (e.g. Taunt, Choice locked)
                # But Showdown usually filters them in activeRequest or marks them.
                if not move.get('disabled', False):
                    actions.append(f"move {i+1}")
            
            # Player can switch
            side = request.get('side', {})
            # Trapping check
            trapped = request['active'][0].get('trapped', False)
            if not trapped:
                for i, p in enumerate(side.get('pokemon', [])):
                    if not p.get('active') and p.get('condition') != '0 fnt':
                        actions.append(f"switch {i+1}")
                    
        if not actions:
            actions.append("pass")
            
        return actions

    def result(self, state: PokemonState, p1_action: str, p2_action: str = None):
        """ Returns the next state after executing the actions. """
        resp = self.client.get_result(state.state_dict, p1_action=p1_action, p2_action=p2_action)
        return PokemonState(resp['state'], resp.get('request'), resp.get('p2_request'), resp.get('log'), resp.get('winner'))

    def is_terminal(self, state: PokemonState):
        """ Returns True if the match has ended. """
        return state.winner is not None

    def is_goal(self, state: PokemonState):
        """ Returns True if Player 1 won the match. """
        return state.winner == 'Player 1'

if __name__ == "__main__":
    # Simple Breadth First Search simulation wrapper test
    engine_path = os.path.abspath("../battle_engine")
    client = ShowdownClient(engine_path)
    try:
        print("Initializing Pokemon Problem...")
        problem = PokemonProblem(client, formatid="gen3randombattle")
        state = problem.initial
        
        print("Initial Actions available:", problem.actions(state))
        
        if not problem.is_goal(state):
            action = problem.actions(state)[0]
            print(f"Executing action: {action}")
            next_state = problem.result(state, action)
            print("Next state actions:", problem.actions(next_state))
            print("Is goal?", problem.is_goal(next_state))
    finally:
        client.close()
