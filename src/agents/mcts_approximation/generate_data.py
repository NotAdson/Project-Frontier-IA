import os
from pathlib import Path
import json
import uuid
import multiprocessing
import traceback
from core.client.showdown_client import ShowdownClient
from core.problem.pokemon_problem import PokemonProblem
from agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent
from agents.mcts_approximation.state_encoder import encode_state

def run_simulation(args):
    """
    Runs a single full game of Pokemon Showdown.
    Player 1 is BlindMCTS, Player 2 is also BlindMCTS (Self-play).
    Returns a list of tuples (encoded_state_list, p1_win)
    """
    engine_path, formatid, mcts_iterations, mcts_depth = args
    
    client = ShowdownClient(engine_path)
    try:
        problem = PokemonProblem(client, formatid=formatid)
        
        agent_p1 = BlindMCTSAgent(problem, iterations=mcts_iterations, max_rollout_depth=mcts_depth)
        agent_p2 = BlindMCTSAgent(problem, iterations=mcts_iterations, max_rollout_depth=mcts_depth)
        
        state = problem.initial
        states_history_p1 = []
        states_history_p2 = []
        
        while not problem.is_terminal(state):
            # Record state for P1 perspective
            encoded_p1 = encode_state(state, player="p1")
            
            # Record state for P2 perspective
            encoded_p2 = encode_state(state, player="p2")
            
            action_p1, probs_p1 = agent_p1.get_action(state, player="p1", return_probs=True)
            action_p2, probs_p2 = agent_p2.get_action(state, player="p2", return_probs=True)
            
            states_history_p1.append((encoded_p1.tolist(), probs_p1))
            states_history_p2.append((encoded_p2.tolist(), probs_p2))
            
            state = problem.result(state, p1_action=action_p1, p2_action=action_p2)
            
        p1_won = problem.is_goal(state)
        return (states_history_p1, states_history_p2, p1_won)
    except Exception as e:
        print(f"Simulation failed: {e}")
        traceback.print_exc()
        return None
    finally:
        client.close()

def generate_dataset(num_games=10000, processes=None, output_dir="data/games"):
    if processes is None:
        # Use all available CPUs for cloud environments (Colab/Kaggle)
        processes = max(1, multiprocessing.cpu_count())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check how many games already exist so we can pause/resume
    existing_files = len(list(output_path.glob("game_*.json")))
    remaining_games = num_games - existing_files
    
    if remaining_games <= 0:
        print(f"Already generated {existing_files} games. Target of {num_games} reached!")
        return
        
    print(f"Found {existing_files} existing games. Generating {remaining_games} more.")
    
    # Path(__file__) resolves to src/agents/mcts_approximation/generate_data.py
    # .parents[3] takes us up to the project root (Pokemon/)
    engine_path = str(Path(__file__).resolve().parents[3] / "engine")
    
    # HIGH QUALITY DATA PARAMS:
    # 200 iterations gives the MCTS a lot of time to build a smart tree.
    # 150 depth ensures rollouts almost always hit a terminal Win/Loss state 
    # instead of timing out and returning a useless 0.5 (draw) evaluation.
    args = [(engine_path, "gen3randombattle", 200, 150) for _ in range(remaining_games)]
    
    print(f"Starting {remaining_games} simulations using {processes} processes...")
    
    saved_count = 0
    # Use imap_unordered to process and save games AS SOON as they finish.
    # This ensures no data is lost if the cloud instance times out.
    with multiprocessing.Pool(processes) as pool:
        for res in pool.imap_unordered(run_simulation, args):
            if res is None:
                continue
                
            states_p1, states_p2, p1_won = res
            
            game_data = []
            # Save P1 transitions
            for s, p in states_p1:
                game_data.append({
                    "features": s,
                    "value": 1.0 if p1_won else 0.0,
                    "policy": p
                })
                
            # Save P2 transitions
            for s, p in states_p2:
                game_data.append({
                    "features": s,
                    "value": 0.0 if p1_won else 1.0,
                    "policy": p
                })
                
            game_id = str(uuid.uuid4())
            file_path = output_path / f"game_{game_id}.json"
            with open(file_path, 'w') as f:
                json.dump(game_data, f)
                
            saved_count += 1
            if saved_count % 10 == 0 or saved_count == remaining_games:
                print(f"Progress: Generated {existing_files + saved_count}/{num_games} total games...")
                
    print(f"Successfully generated and saved {saved_count} new games to {output_dir}")

if __name__ == "__main__":
    generate_dataset()
