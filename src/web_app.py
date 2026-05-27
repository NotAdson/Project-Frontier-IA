import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.client.showdown_client import ShowdownClient
from core.problem.pokemon_problem import PokemonProblem
from agents.mcts_approximation.mcts_approximation_agent import MCTSApproximationAgent
from agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent

app = Flask(__name__)

# Globals to hold our game state
client = None
problem = None
agent = None
current_state = None
executor = ThreadPoolExecutor(max_workers=1)
ai_future = None

def init_game():
    global client, problem, agent, current_state, ai_future
    if client is not None:
        client.close()
        
    engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine"))
    client = ShowdownClient(engine_path)
    problem = PokemonProblem(client, formatid="gen3randombattle")
    # Using BlindMCTSAgent temporarily so it's a competent opponent while your Neural Network trains!
    # (Since BlindMCTS actually does rollouts to the end of the game to find real wins/losses)
    # Bumping iterations up to 1000 so it actually poses a challenge!
    agent = BlindMCTSAgent(problem, iterations=1000, max_rollout_depth=150) 
    current_state = problem.initial
    
    # Start AI thinking immediately
    if not problem.is_terminal(current_state):
        ai_future = executor.submit(agent.get_action, current_state, player="p2")

@app.route("/")
def index():
    if current_state is None:
        init_game()
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
    # ident is something like "p1a: Pikachu"
    name = ident.split(': ')[1]
    # Handle weird showdown names or forms if necessary
    return name

@app.route("/state")
def get_state():
    if current_state is None:
        return jsonify({"error": "Game not initialized"})
        
    p1_active = None
    p2_active = None
    
    # Parse P1 Active
    if current_state.request_dict and 'side' in current_state.request_dict:
        for p in current_state.request_dict['side']['pokemon']:
            if p.get('active'):
                display_name = get_base_species_name(p['ident'])
                name = display_name.lower().replace(" ", "").replace("-", "")
                cond = parse_pokemon_condition(p['condition'])
                p1_active = {
                    "name": name,
                    "display_name": display_name,
                    "hp": cond["hp"],
                    "max_hp": cond["max_hp"],
                    "status": cond["status"]
                }
                break
                
    # Parse P2 Active
    if current_state.p2_request_dict and 'side' in current_state.p2_request_dict:
        for p in current_state.p2_request_dict['side']['pokemon']:
            if p.get('active'):
                display_name = get_base_species_name(p['ident'])
                name = display_name.lower().replace(" ", "").replace("-", "")
                cond = parse_pokemon_condition(p['condition'])
                p2_active = {
                    "name": name,
                    "display_name": display_name,
                    "hp": cond["hp"],
                    "max_hp": cond["max_hp"],
                    "status": cond["status"]
                }
                break
                
    valid_actions_raw = problem.actions(current_state, "p1")
    valid_actions = []
    
    req = current_state.request_dict
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
                pkmn_name = get_base_species_name(pkmn_ident) if ':' in pkmn_ident else 'Unknown'
                text = f"Switch: {pkmn_name}"
            except Exception:
                pass
                
        valid_actions.append({"id": action, "text": text})
    
    # Process log to look nice
    clean_log = []
    for line in current_state.log:
        if line.startswith("|") and len(line) > 1:
            clean_log.append(line)
            
    is_terminal = problem.is_terminal(current_state)
    winner = current_state.winner if is_terminal else None

    return jsonify({
        "p1_active": p1_active,
        "p2_active": p2_active,
        "valid_actions": valid_actions,
        "log": clean_log,
        "is_terminal": is_terminal,
        "winner": winner
    })

@app.route("/action", methods=["POST"])
def do_action():
    global current_state, ai_future
    data = request.json
    p1_action = data.get("action")
    
    if problem.is_terminal(current_state):
        return jsonify({"error": "Game over"})
        
    print(f"P1 chose: {p1_action}")
    
    # Wait for background AI calculation to finish (if it hasn't already)
    p2_action = ai_future.result() if ai_future else agent.get_action(current_state, player="p2")
    print(f"P2 (AI) chose: {p2_action}")
    
    # Execute actions
    current_state = problem.result(current_state, p1_action=p1_action, p2_action=p2_action)
    
    # Immediately start thinking about the NEXT turn
    if not problem.is_terminal(current_state):
        ai_future = executor.submit(agent.get_action, current_state, player="p2")
    else:
        ai_future = None
        
    return jsonify({"success": True})

@app.route("/reset", methods=["POST"])
def reset():
    global ai_future
    if ai_future is not None:
        ai_future.cancel()
    init_game()
    return jsonify({"success": True})

if __name__ == "__main__":
    init_game()
    app.run(debug=True, host="0.0.0.0", port=5000)
