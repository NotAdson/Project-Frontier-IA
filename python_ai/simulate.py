import os
import html
from pokemon_problem import PokemonProblem
from showdown_client import ShowdownClient
from search_algorithms import random_blind_search

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
</body></html>
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
    engine_path = os.path.abspath("../battle_engine")
    client = ShowdownClient(engine_path)
    
    try:
        print("Initializing Battle Simulator...")
        problem = PokemonProblem(client, formatid="gen3randombattle")
        
        print("Running Random Blind Search Simulation...")
        final_node = random_blind_search(problem, max_depth=1000)
        
        print(f"Simulation ended at depth {final_node.depth}.")
        if problem.is_goal(final_node.state):
            print("Player 1 won the match!")
        else:
            print("Match ended or Player 1 lost.")
            
        # Extract full battle log from the final state
        # The battle engine returns the complete log history at each step
        full_log = final_node.state.log if final_node.state.log else []
                
        # Generate the replay
        replay_filename = "replay.html"
        generate_replay_html(full_log, replay_filename)
        
    finally:
        client.close()
