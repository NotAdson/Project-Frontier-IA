import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify

# Add src to path so we can import core and battle_agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.client.showdown_client import ShowdownClient
from core.problem.pokemon_problem import PokemonProblem
from battle_agents.mcts_approximation.mcts_approximation_agent import MCTSApproximationAgent

from battle_agents.mcts_approximation.db.moves_db import get_move_data

class GameController:
    def __init__(self):
        self.client = None
        self.problem = None
        self.agent = None
        self.current_state = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.is_thinking = False
        self.precomputed_p2_action = None

    def init_game(self):
        if self.client is not None:
            self.client.close()
            
        engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        model_path = os.path.join(data_dir, "mcts_model.keras")
        
        self.client = ShowdownClient(engine_path)
        self.problem = PokemonProblem(self.client, formatid="gen3randombattle")
        self.agent = MCTSApproximationAgent(self.problem, iterations=400, max_rollout_depth=0, model_path=model_path)
        self.current_state = self.problem.initial
        self.precomputed_p2_action = None
        self.is_thinking = False
        
        # Start precomputing the very first turn of the game in background
        self.start_precompute()

    def reset(self):
        self.init_game()

    def start_precompute(self):
        """Starts background calculation of the AI move for the current state."""
        if self.current_state is None or self.problem.is_terminal(self.current_state):
            return
        self.precomputed_p2_action = None
        self.executor.submit(self._precompute_worker, self.current_state)

    def _precompute_worker(self, state):
        try:
            action = self.agent.get_action(state, player="p2")
            self.precomputed_p2_action = action
        except Exception as e:
            print(f"Error in precomputing AI move: {e}")

game = GameController()
app = Flask(__name__)

@app.route("/")
def index():
    if game.current_state is None:
        game.init_game()
    return render_template("index.html")

def parse_pokemon_condition(condition):
    if not condition or condition == '0 fnt':
        return {"hp": 0, "max_hp": 100, "status": "fnt"}
    
    parts = condition.split(' ')
    hp_fraction = parts[0]
    status = parts[1] if len(parts) > 1 else ""
    
    if '/' in hp_fraction:
        hp, max_hp = hp_fraction.split('/')
        return {"hp": int(hp), "max_hp": int(max_hp), "status": status}
    return {"hp": 0, "max_hp": 100, "status": "fnt"}

def get_base_species_name(ident):
    if ': ' in ident:
        return ident.split(': ')[1]
    return ident

@app.route("/state")
def get_state():
    if game.current_state is None:
        return jsonify({"error": "Game not initialized"})
        
    p1_active = None
    p2_active = None
    p1_bench = []
    p2_party_count = 6
    p1_moves = []
    
    # Parse P1
    if game.current_state.request_dict:
        req = game.current_state.request_dict
        if 'side' in req and 'pokemon' in req['side']:
            for p in req['side']['pokemon']:
                display_name = get_base_species_name(p['ident'])
                name = display_name.lower().replace(" ", "").replace("-", "")
                cond = parse_pokemon_condition(p['condition'])
                
                pkmn_data = {
                    "name": name,
                    "display_name": display_name,
                    "hp": cond["hp"],
                    "max_hp": cond["max_hp"],
                    "status": cond["status"],
                    "item": p.get("item", ""),
                    "moves": p.get("moves", [])
                }
                
                if p.get('active'):
                    p1_active = pkmn_data
                else:
                    p1_bench.append(pkmn_data)
                    
        if 'active' in req and req['active']:
            for m in req['active'][0].get('moves', []):
                move_id = m.get('id', '')
                move_db_data = get_move_data(move_id)
                p1_moves.append({
                    "id": move_id,
                    "move": m.get('move', ''),
                    "pp": m.get('pp', 0),
                    "maxpp": m.get('maxpp', 0),
                    "type": move_db_data.get('type', 'Normal'),
                    "basePower": move_db_data.get('basePower', 0),
                    "category": move_db_data.get('category', 'Physical')
                })

    # Parse P2
    if game.current_state.p2_request_dict:
        req2 = game.current_state.p2_request_dict
        if 'side' in req2 and 'pokemon' in req2['side']:
            p2_party_count = 0
            for p in req2['side']['pokemon']:
                cond = parse_pokemon_condition(p['condition'])
                if cond["hp"] > 0:
                    p2_party_count += 1
                    
                if p.get('active'):
                    display_name = get_base_species_name(p['ident'])
                    name = display_name.lower().replace(" ", "").replace("-", "")
                    p2_active = {
                        "name": name,
                        "display_name": display_name,
                        "hp": cond["hp"],
                        "max_hp": cond["max_hp"],
                        "status": cond["status"]
                    }
                
    valid_actions_raw = game.problem.actions(game.current_state, "p1")
    valid_actions = []
    
    req = game.current_state.request_dict
    for action in valid_actions_raw:
        text = action
        if action.startswith("move ") and req and 'active' in req:
            try:
                idx = int(action.split(' ')[1]) - 1
                move_name = req['active'][0]['moves'][idx].get('move', 'Unknown')
                text = f"{move_name}"
            except Exception:
                pass
        elif action.startswith("switch ") and req and 'side' in req and 'pokemon' in req['side']:
            try:
                idx = int(action.split(' ')[1]) - 1
                pkmn_ident = req['side']['pokemon'][idx].get('ident', '')
                pkmn_name = get_base_species_name(pkmn_ident)
                text = f"Switch: {pkmn_name}"
            except Exception:
                pass
                
        valid_actions.append({"id": action, "text": text})
    
    # Process log
    clean_log = []
    for line in game.current_state.log:
        if line.startswith("|") and len(line) > 1:
            clean_log.append(line)
            
    is_terminal = game.problem.is_terminal(game.current_state)
    winner = game.current_state.winner if is_terminal else None

    return jsonify({
        "p1_active": p1_active,
        "p2_active": p2_active,
        "p1_bench": p1_bench,
        "p1_moves": p1_moves,
        "p2_party_count": p2_party_count,
        "valid_actions": valid_actions,
        "log": clean_log,
        "is_terminal": is_terminal,
        "winner": winner,
        "is_thinking": game.is_thinking
    })

import time

def do_action_bg(p1_action, current_state):
    game.is_thinking = True
    try:
        # Since the ThreadPoolExecutor has max_workers=1, Task 2 (this resolver task)
        # runs strictly after Task 1 (precompute worker). Thus, game.precomputed_p2_action
        # is guaranteed to be finished and set here.
        p2_action = game.precomputed_p2_action
        if p2_action is None:
            # Fallback in case of race condition or unexpected execution order
            p2_action = game.agent.get_action(current_state, player="p2")
            
        game.current_state = game.problem.result(current_state, p1_action=p1_action, p2_action=p2_action)
        
        if game.problem.is_terminal(game.current_state):
            history_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "history"))
            os.makedirs(history_dir, exist_ok=True)
            filename = f"game_{int(time.time())}.txt"
            with open(os.path.join(history_dir, filename), "w") as f:
                f.write("\n".join(game.current_state.log))
        else:
            # Precompute the AI's move for the next turn in the background
            game.start_precompute()
                
    except Exception as e:
        print(f"Error in background AI thread: {e}")
    finally:
        game.is_thinking = False

@app.route("/action", methods=["POST"])
def do_action():
    data = request.json
    p1_action = data.get("action")
    
    if game.problem.is_terminal(game.current_state):
        return jsonify({"error": "Game over"})
        
    # Start background execution to process and advance turn state
    game.is_thinking = True
    game.executor.submit(do_action_bg, p1_action, game.current_state)
    
    return jsonify({"success": True})

@app.route("/reset", methods=["POST"])
def reset():
    game.reset()
    return jsonify({"success": True})

if __name__ == "__main__":
    game.init_game()
    app.run(debug=True, host="0.0.0.0", port=5000)
