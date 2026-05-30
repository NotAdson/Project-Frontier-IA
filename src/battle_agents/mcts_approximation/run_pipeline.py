import os
import glob
import shutil
import sys
from pathlib import Path

# Fix path to allow importing from core/battle_agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from battle_agents.mcts_approximation.generate_data import generate_dataset
from battle_agents.mcts_approximation.train_nn import train

from core.client.showdown_client import ShowdownClient
from benchmarks.round_robin import RoundRobinBenchmark
from battle_agents.random.random_agent import RandomAgent
from battle_agents.mcts_approximation.mcts_approximation_agent import MCTSApproximationAgent


def run_pipeline(num_games=1000, num_generations=1, processes=None):
    print(f"=== AlphaZero Pipeline Automation ({num_generations} Generations) ===")
    
    for gen_idx in range(num_generations):
        # 1. Identify Generation
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        gen_folders = glob.glob(os.path.join(data_dir, "gen*"))
        gen_nums = []
        for folder in gen_folders:
            try:
                num = int(os.path.basename(folder).replace("gen", ""))
                gen_nums.append(num)
            except ValueError:
                pass
                
        if not gen_nums:
            next_gen = 1
        else:
            next_gen = max(gen_nums) + 1
            
        next_gen_dir = os.path.join(data_dir, f"gen{next_gen}")
        os.makedirs(next_gen_dir, exist_ok=True)
        print(f"\n=============================================")
        print(f"Starting Generation {next_gen} (Loop {gen_idx + 1}/{num_generations})")
        print(f"=============================================")
        
        # 2. Self-Play
        print(f"\n--- Phase 1: Generating Data into {next_gen_dir} ---")
        generate_dataset(num_games=num_games, output_dir=next_gen_dir, processes=processes)
        
        # 3. Train
        print("\n--- Phase 2: Training Neural Network ---")
        model_save_path = os.path.join(data_dir, "mcts_model.keras")
        train(data_dir=data_dir, model_save_path=model_save_path, max_games_buffer=10000)
        
        # 4. Archive
        print(f"\n--- Phase 3: Archiving Model to {next_gen_dir} ---")
        shutil.copy(model_save_path, os.path.join(next_gen_dir, "mcts_model.keras"))
        
        # 5. Benchmark
        print("\n--- Phase 4: Benchmarking against previous generations ---")
        engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "engine"))
            
        client = ShowdownClient(engine_path)
        
        try:
            agents = {
                "Random": lambda prob: RandomAgent(prob)
            }
            
            # Load all available generation models
            for i in range(1, next_gen + 1):
                mp = os.path.join(data_dir, f"gen{i}", "mcts_model.keras")
                if os.path.exists(mp):
                    # Use default arguments trick to capture the variable `mp` properly in the lambda
                    agents[f"Gen{i}"] = lambda prob, p=mp: MCTSApproximationAgent(prob, iterations=50, max_rollout_depth=0, model_path=p)
            
            benchmark = RoundRobinBenchmark(client, agents, games_per_matchup=5, shuffle=True)
            benchmark.run()
            
            report_path = os.path.join(next_gen_dir, "benchmark_report.txt")
            print(f"Saving benchmark report to {report_path}")
            
            # Redirect stdout to save report
            original_stdout = sys.stdout
            with open(report_path, "w") as f:
                sys.stdout = f
                benchmark.print_report()
                sys.stdout = original_stdout
                
            print("Benchmark Complete! Check the report.")
            
        finally:
            client.close()
            
        print(f"\n=== Pipeline for Generation {next_gen} Finished successfully! ===")

if __name__ == "__main__":
    # processes=None uses (CPU_CORES - 2). You can hardcode it (e.g., processes=4) to restrict CPU usage.
    run_pipeline(num_games=2500, num_generations=8, processes=None)