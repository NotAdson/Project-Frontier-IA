import html
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from battle_agents.mcts.mcts_agent import MCTSAgent
from battle_agents.random.random_agent import RandomAgent
from core.client.showdown_client import ShowdownClient
from core.problem.pokemon_problem import PokemonProblem

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Pokemon Battle Simulation Replay</title>
<style>
    body { background: #222; color: #fff; font-family: sans-serif; }
</style>
</head><body>
<h2 style="text-align: center;">Random Blind Search AI Battle</h2>
<script type="text/plain" class="battle-log-data">{BATTLE_LOG_TEXT}</script>
<script src="https://play.pokemonshowdown.com/js/replay-embed.js"></script>
</head><body>
"""

def generate_replay_html(full_log, filename="replay.html"):
    # Join log lines and escape HTML characters (including quotes for the attribute)
    log_text = "\n".join(full_log)
    escaped_log = html.escape(log_text)
    
    html_content = HTML_TEMPLATE.replace("{BATTLE_LOG_TEXT}", escaped_log)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Replay saved to {filename}. Open this file in your browser!")

if __name__ == "__main__":
    engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
    client = ShowdownClient(engine_path)
    
    try:
        print("Initializing Battle Simulator...")
        problem = PokemonProblem(client, formatid="gen3randombattle")
        
        print("Running Monte Carlo Tree Search (MCTS) Simulation...")
        state = problem.initial
        turn = 1
        
        # Instantiate our multi-agent framework
        from battle_agents.mcts_approximation.mcts_approximation_agent import \
            MCTSApproximationAgent
        p1_agent = MCTSApproximationAgent(problem, iterations=80)
        p2_agent = MCTSAgent(problem, iterations=50, max_rollout_depth=20)
        
        while not problem.is_terminal(state):
            print(f"\n--- Turn {turn} ---")
            p1_action = p1_agent.get_action(state, player="p1")
            p2_action = p2_agent.get_action(state, player="p2")
            print(f"  P1 action: {p1_action} | P2 action: {p2_action}")
            
            state = problem.result(state, p1_action=p1_action, p2_action=p2_action)
            
            # Print active Pokemon status
            p1_active = next(p for p in state.state_dict['sides'][0]['pokemon'] if p['isActive'])
            p2_active = next(p for p in state.state_dict['sides'][1]['pokemon'] if p['isActive'])
            print(f"  P1 {p1_active['name']}: {p1_active['hp']}/{p1_active['maxhp']} ({p1_active['status']})")
            print(f"  P2 {p2_active['name']}: {p2_active['hp']}/{p2_active['maxhp']} ({p2_active['status']})")
            
            turn += 1
            
        print(f"\nSimulation ended at turn {turn-1}.")
        if problem.is_goal(state):
            print("Player 1 won the match!")
        else:
            print("Match ended or Player 1 lost.")
            
        # Extract full battle log from the final state
        full_log = state.log if state.log else []
        
        # Generate the replay
        replay_filename = "replay.html"
        generate_replay_html(full_log, replay_filename)
        
    finally:
        client.close()
