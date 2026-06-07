import json
import multiprocessing
import os
import random
import traceback
import uuid
from pathlib import Path

from tqdm import tqdm

from battle_agents.mcts.mcts_agent import MCTSAgent
from battle_agents.mcts_approximation.mcts_approximation_agent import \
    MCTSApproximationAgent
from battle_agents.mcts_approximation.state_encoder import encode_state
from core.client.showdown_client import ShowdownClient
from core.problem.pokemon_problem import PokemonProblem


def run_simulation(args):
    """Runs a single full game of Pokemon Showdown.
    Player 1 and Player 2 are self‑playing agents (MCTSApproximationAgent or MCTSAgent).
    Returns a list of tuples (encoded_state_list, p1_win)
    """
    engine_path, formatid, mcts_iterations, use_cheating_mcts, model_path = args

    client = ShowdownClient(engine_path)
    try:
        problem = PokemonProblem(client, formatid=formatid)
        
        if use_cheating_mcts:
            agent_p1 = MCTSAgent(problem, iterations=mcts_iterations)
            agent_p2 = MCTSAgent(problem, iterations=mcts_iterations)
        else:
            agent_p1 = MCTSApproximationAgent(problem, iterations=mcts_iterations, model_path=model_path)
            agent_p2 = MCTSApproximationAgent(problem, iterations=mcts_iterations, model_path=model_path)
        
        state = problem.initial
        states_history_p1 = []
        states_history_p2 = []
        
        turn_count = 0
        while not problem.is_terminal(state):
            # Record state for P1 perspective
            encoded_p1 = encode_state(state, player="p1")
            
            # Record state for P2 perspective
            encoded_p2 = encode_state(state, player="p2")
            
            if use_cheating_mcts:
                action_p1, probs_p1 = agent_p1.get_action(state, player="p1", return_probs=True)
                action_p2, probs_p2 = agent_p2.get_action(state, player="p2", return_probs=True)
            else:
                # Exponential decay for temperature.
                temp = max(0.01, (0.85 ** turn_count))
                action_p1, probs_p1 = agent_p1.get_action(state, player="p1", return_probs=True, temperature=temp)
                action_p2, probs_p2 = agent_p2.get_action(state, player="p2", return_probs=True, temperature=temp)
            
            turn_count += 1
            
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


def generate_dataset(num_games=1000, processes=None, output_dir="data/games", mcts_iterations=100, use_cheating_mcts=False, model_path="data/mcts_model.onnx"):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if processes is None:
        processes = max(1, multiprocessing.cpu_count() - 2)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check how many games already exist so we can pause/resume
    existing_files = len(list(output_path.glob("game_*.json")))
    remaining_games = num_games - existing_files

    if remaining_games <= 0:
        print(f"Already generated {existing_files} games. Target of {num_games} reached!")
        return

    print(f"Found {existing_files} existing games. Generating {remaining_games} more.")

    engine_path = str(Path(__file__).resolve().parents[4] / "engine")

    args = [(engine_path, "gen3randombattle", mcts_iterations, use_cheating_mcts, model_path) for _ in range(remaining_games)]

    print(f"Starting {remaining_games} simulations using {processes} processes...")

    saved_count = 0
    with multiprocessing.Pool(processes) as pool:
        pbar = tqdm(pool.imap_unordered(run_simulation, args), total=remaining_games, desc="Generating games")
        for res in pbar:
            if res is None:
                continue
            states_p1, states_p2, p1_won = res
            game_data = []
            for s, p in states_p1:
                game_data.append({"features": s, "value": 1.0 if p1_won else 0.0, "policy": p})
            for s, p in states_p2:
                game_data.append({"features": s, "value": 0.0 if p1_won else 1.0, "policy": p})
            game_id = str(uuid.uuid4())
            file_path = output_path / f"game_{game_id}.json"
            with open(file_path, 'w') as f:
                json.dump(game_data, f)
            saved_count += 1
    print(f"Successfully generated and saved {saved_count} new games to {output_dir}")


if __name__ == "__main__":
    generate_dataset()
