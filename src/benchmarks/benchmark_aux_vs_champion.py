import os
import json
import sys
from collections import defaultdict

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.client.showdown_client import ShowdownClient
from benchmarks.round_robin import RoundRobinBenchmark
from battle_agents.mcts_approximation.mcts_approximation_agent import MCTSApproximationAgent

if __name__ == "__main__":
    print("Initializing Showdown Engine for Benchmark...")
    engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
    client = ShowdownClient(engine_path)
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    champion_path = os.path.join(data_dir, "mcts_model.keras")
    aux_path = os.path.join(data_dir, "mcts_model_aux.keras")
    
    print(f"Loading Champion from: {champion_path}")
    print(f"Loading Aux Model from: {aux_path}")
    
    try:
        # We run MCTS with a smaller number of iterations (e.g. 30) or standard iterations (300) to keep it fast but representative.
        # Let's use 30 iterations.
        iterations = 30
        agents = {
            "Standard Champion": lambda prob, p=champion_path: MCTSApproximationAgent(
                prob, iterations=iterations, model_path=p
            ),
            "Aux Model": lambda prob, p=aux_path: MCTSApproximationAgent(
                prob, iterations=iterations, model_path=p
            )
        }
        
        # Run 10 games matchup (5 games as player 1, 5 games as player 2 for balance)
        print("Running head-to-head match (10 games)...")
        benchmark = RoundRobinBenchmark(client, agents, games_per_matchup=5)
        benchmark.run()
        
        # Format and print the report
        print("\n" + "="*80)
        print(" "*25 + "AUX MODEL VS CHAMPION BENCHMARK REPORT")
        print("="*80)
        print(f"Total Matches Played: {len(benchmark.results)}")
        
        stats = defaultdict(lambda: {'wins': 0, 'matches': 0, 'total_time': 0})
        for r in benchmark.results:
            p1, p2, winner = r['p1'], r['p2'], r['winner']
            stats[p1]['matches'] += 1
            stats[p2]['matches'] += 1
            stats[p1]['total_time'] += r['p1_avg_time']
            stats[p2]['total_time'] += r['p2_avg_time']
            if winner == p1:
                stats[p1]['wins'] += 1
            elif winner == p2:
                stats[p2]['wins'] += 1
                
        print(f"\n{'Agent Name':<20} | {'Win Rate':<10} | {'Avg Time/Turn':<15}")
        print("-" * 80)
        for agent, data in stats.items():
            win_rate = (data['wins'] / data['matches']) * 100 if data['matches'] > 0 else 0
            avg_time = data['total_time'] / data['matches'] if data['matches'] > 0 else 0
            print(f"{agent:<20} | {win_rate:>8.1f}% | {avg_time:>13.3f}s")
        print("="*80)
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
    finally:
        client.close()
