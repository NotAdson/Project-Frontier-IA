import os
from core.client.showdown_client import ShowdownClient
from benchmarks.round_robin import RoundRobinBenchmark
from benchmarks.tournament import TournamentBenchmark
from battle_agents.random.random_agent import RandomAgent
from battle_agents.mcts.mcts_agent import MCTSAgent
from battle_agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    print("Initializing Showdown Engine for Benchmarking...")
    engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
    client = ShowdownClient(engine_path)
    
    try:
        # Define our agents using factory lambdas. 
        # This ensures every game creates a fresh agent connected to that game's specific Problem instance.
        agents = {
            "MCTS (Cheater)": lambda prob: MCTSAgent(prob, iterations=50, max_rollout_depth=20),
            "MCTS (Blind)": lambda prob: BlindMCTSAgent(prob, iterations=50, max_rollout_depth=20),
            "Random Agent": lambda prob: RandomAgent(prob)
        }
        
        # Run a Bracket Tournament!
        benchmark = TournamentBenchmark(client, agents, games_per_matchup=1, shuffle=True)
        benchmark.run()
        
        # Automatically format and print the statistics table
        benchmark.print_report()
        
    except KeyboardInterrupt:
        print("\nBenchmark manually interrupted. Printing partial results...")
        benchmark.print_report()
    finally:
        client.close()
