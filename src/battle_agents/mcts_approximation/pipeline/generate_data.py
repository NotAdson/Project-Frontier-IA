import json
import multiprocessing
import os
import random
import traceback
import uuid
from pathlib import Path

from tqdm import tqdm

from battle_agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent
from battle_agents.mcts.mcts_agent import MCTSAgent
from battle_agents.mcts_approximation.mcts_approximation_agent import \
    MCTSApproximationAgent
from battle_agents.mcts_approximation.db import teams as teams_db
from battle_agents.mcts_approximation.state_encoder import encode_state
from battle_agents.random.random_agent import RandomAgent
from core.client.showdown_client import ShowdownClient
from core.problem.pokemon_problem import PokemonProblem


def run_simulation(args):
    """Runs a single full game of Pokemon Showdown.
    Player 1 and Player 2 are self‑playing agents. ``agent_type`` selects the pair:
    "random" (RandomAgent, no model/search needed), "blind_mcts" (BlindMCTSAgent,
    no model needed), or the legacy behavior (MCTSAgent if use_cheating_mcts else
    MCTSApproximationAgent, which requires model_path to point at an existing .onnx).
    Returns a list of tuples (encoded_state_list, p1_win)

    Team strings (multi-line format, consumed by Teams.import in the engine) are
    pre-selected in the parent process and passed as args so that each simulation
    uses proper competitive Gen 3 OU teams instead of engine-generated random teams.
    """
    engine_path, formatid, mcts_iterations, use_cheating_mcts, model_path, agent_type, p1_team, p2_team = args

    client = ShowdownClient(engine_path)
    try:
        problem = PokemonProblem(client, formatid=formatid, p1_team=p1_team, p2_team=p2_team)

        if agent_type == "random":
            agent_p1 = RandomAgent(problem)
            agent_p2 = RandomAgent(problem)
        elif agent_type == "blind_mcts":
            agent_p1 = BlindMCTSAgent(problem, iterations=mcts_iterations)
            agent_p2 = BlindMCTSAgent(problem, iterations=mcts_iterations)
        elif use_cheating_mcts:
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
            
            if agent_type == "random":
                action_p1 = agent_p1.get_action(state, player="p1")
                action_p2 = agent_p2.get_action(state, player="p2")
                probs_p1 = {action_p1: 1.0}
                probs_p2 = {action_p2: 1.0}
            elif agent_type == "blind_mcts":
                action_p1, probs_p1 = agent_p1.get_action(state, player="p1", return_probs=True)
                action_p2, probs_p2 = agent_p2.get_action(state, player="p2", return_probs=True)
            else:
                # Turn-decaying temperature: same tau used for both action
                # selection and the saved policy target (training label).
                tau = max(0.4, 0.92 ** turn_count)
                action_p1, probs_p1 = agent_p1.get_action(state, player="p1", return_probs=True, temperature=tau)
                action_p2, probs_p2 = agent_p2.get_action(state, player="p2", return_probs=True, temperature=tau)
            
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


def generate_dataset(num_games=1000, processes=None, output_dir="data/games", mcts_iterations=100,
                      use_cheating_mcts=False, model_path=None, agent_type=None):
    """
    agent_type: None (legacy: MCTSAgent if use_cheating_mcts else MCTSApproximationAgent),
        "random" (RandomAgent both sides, no model/search), or "blind_mcts"
        (BlindMCTSAgent both sides, no model needed). "random"/"blind_mcts" ignore
        use_cheating_mcts and model_path entirely.
    """
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if agent_type is None and model_path is None:
        model_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "mcts_model.onnx")

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

    # Pre‑select competitive Gen 3 OU teams for every simulation, so the engine
    # uses real Metamon teams instead of generating random Pokémon.
    team_pairs = [
        (teams_db.get_random_team("gen3ou"), teams_db.get_random_team("gen3ou"))
        for _ in range(remaining_games)
    ]
    args = [
        (engine_path, "gen3ou", mcts_iterations, use_cheating_mcts, model_path, agent_type, p1, p2)
        for p1, p2 in team_pairs
    ]

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
            tmp_path = output_path / f".{file_path.name}.tmp"
            with open(tmp_path, 'w') as f:
                json.dump(game_data, f)
            tmp_path.replace(file_path)
            saved_count += 1
    print(f"Successfully generated and saved {saved_count} new games to {output_dir}")


if __name__ == "__main__":
    generate_dataset()
