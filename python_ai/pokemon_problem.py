import json
import os
from aima_problem import Problem
from showdown_client import ShowdownClient

class PokemonState:
    def __init__(self, state_dict, request_dict=None, log=None):
        self.state_dict = state_dict
        self.request_dict = request_dict
        self.log = log or []
        
    def __eq__(self, other):
        # Using string comparison for simplicity, though feature extraction is better for large scale search.
        return json.dumps(self.state_dict, sort_keys=True) == json.dumps(other.state_dict, sort_keys=True)
        
    def __hash__(self):
        return hash(json.dumps(self.state_dict, sort_keys=True))

class PokemonProblem(Problem):
    def __init__(self, client: ShowdownClient, formatid='gen3randombattle'):
        self.client = client
        resp = self.client.init_battle(formatid=formatid)
        initial_state = PokemonState(resp['state'], resp.get('request'), resp.get('log'))
        super().__init__(initial_state)

    def actions(self, state: PokemonState):
        """ Returns valid actions from the given state for Player 1. """
        actions = []
        request = state.request_dict
        
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

    def result(self, state: PokemonState, action: str):
        """ Returns the next state after executing the action. """
        resp = self.client.get_result(state.state_dict, p1_action=action)
        return PokemonState(resp['state'], resp.get('request'), resp.get('log'))

    def is_goal(self, state: PokemonState):
        """ Returns True if Player 1 won the match. """
        # Check if Player 2 has any alive pokemon
        p2_side = state.state_dict.get('sides', [{}, {}])[1]
        
        all_fainted = True
        for p in p2_side.get('pokemon', []):
            # 'hp' is 0 if fainted, or condition is '0 fnt'
            if p.get('hp', 0) > 0:
                all_fainted = False
                break
        
        return all_fainted

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
