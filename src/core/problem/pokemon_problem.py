import json
import os

from core.client.showdown_client import ShowdownClient
from core.problem.aima_problem import Problem


class PokemonState:
    def __init__(self, state_dict, request_dict=None, p2_request_dict=None, log=None, winner=None, state_id=None):
        self.state_dict = state_dict
        self.request_dict = request_dict
        self.p2_request_dict = p2_request_dict
        self.log = log or []
        self.winner = winner
        self.state_id = state_id
        
    def __eq__(self, other):
        if self.state_id is not None and other.state_id is not None:
            return self.state_id == other.state_id
        return json.dumps(self.state_dict, sort_keys=True) == json.dumps(other.state_dict, sort_keys=True)
        
    def __hash__(self):
        if self.state_id is not None:
            return hash(self.state_id)
        return hash(json.dumps(self.state_dict, sort_keys=True))

class PokemonProblem(Problem):
    def __init__(self, client: ShowdownClient, formatid='gen3ou', p1_team=None, p2_team=None):
        self.client = client
        resp = self.client.init_battle(formatid=formatid, p1_team=p1_team, p2_team=p2_team)
        initial_state = PokemonState(
            resp['state'], 
            resp.get('request'), 
            resp.get('p2_request'), 
            resp.get('log'), 
            resp.get('winner'),
            state_id=resp.get('state_id')
        )
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
        resp = self.client.get_result(state.state_dict, p1_action=p1_action, p2_action=p2_action, state_id=state.state_id)
        return PokemonState(
            resp['state'], 
            resp.get('request'), 
            resp.get('p2_request'), 
            resp.get('log'), 
            resp.get('winner'),
            state_id=resp.get('state_id')
        )

    def rollout(self, state: PokemonState, player: str, max_depth: int):
        return self.client.rollout(state.state_dict, player, max_depth, state_id=state.state_id)

    def is_terminal(self, state: PokemonState):
        """ Returns True if the match has ended (winner set, or 1000-turn draw). """
        if state.winner is not None:
            return True
        # Fallback turn limit: mirrors Battle::MAX_TURNS = 1000 in C++ engine
        turn = state.state_dict.get('turn', 0)
        return turn >= 1000

    def is_goal(self, state: PokemonState):
        """ Returns True if Player 1 won the match. """
        return state.winner == 'Player 1'

if __name__ == "__main__":
    # Simple Breadth First Search simulation wrapper test
    engine_path = os.path.abspath("../battle_engine")
    client = ShowdownClient(engine_path)
    try:
        print("Initializing Pokemon Problem...")
        problem = PokemonProblem(client, formatid="gen3ou")
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
