import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
            latest_gen = max(gen_nums)
            latest_gen_dir = os.path.join(data_dir, f"gen{latest_gen}")
            existing_games = len(glob.glob(os.path.join(latest_gen_dir, "game_*.json")))
            
            if existing_games < num_games:
                print(f"\n[Resuming] Latest generation {latest_gen} is incomplete: {existing_games}/{num_games} games. Continuing it.")
                next_gen = latest_gen
            else:
                print(f"\n[New Gen] Latest generation {latest_gen} is complete: {existing_games}/{num_games} games. Starting next generation.")
                next_gen = latest_gen + 1
            
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
        onnx_save_path = model_save_path.replace(".keras", ".onnx")
        if os.path.exists(onnx_save_path):
            shutil.copy(onnx_save_path, os.path.join(next_gen_dir, "mcts_model.onnx"))
            
        print(f"\n=== Pipeline for Generation {next_gen} Finished successfully! ===")

if __name__ == "__main__":
    # processes=None uses (CPU_CORES - 2). You can hardcode it (e.g., processes=4) to restrict CPU usage.
    run_pipeline(num_games=2500, num_generations=8, processes=None)