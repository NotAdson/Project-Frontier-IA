import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import glob
import shutil
import sys
import json
from pathlib import Path
from collections import defaultdict

# Fix path to allow importing from core/battle_agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from battle_agents.mcts_approximation.pipeline.generate_data import generate_dataset
from battle_agents.mcts_approximation.pipeline.train_nn import train
from battle_agents.mcts_approximation.mcts_approximation_agent import MCTSApproximationAgent
from battle_agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent
from battle_agents.random.random_agent import RandomAgent

from core.client.showdown_client import ShowdownClient
from benchmarks.round_robin import RoundRobinBenchmark


def check_generation_won(gen_dir, gen_idx):
    """
    Parses the tournament report inside gen_dir to check if Model Gen gen_idx 
    outperformed or tied both static baselines (Blind MCTS and Random Agent).
    Returns True if the model won/tied them, False otherwise.
    """
    report_path = os.path.join(gen_dir, "benchmark_report.json")
    if not os.path.exists(report_path):
        return False
    try:
        with open(report_path, "r") as f:
            results = json.load(f)
        if not results:
            return False
            
        stats = defaultdict(lambda: {'wins': 0, 'matches': 0})
        for r in results:
            p1, p2, winner = r['p1'], r['p2'], r['winner']
            stats[p1]['matches'] += 1
            stats[p2]['matches'] += 1
            if winner == p1:
                stats[p1]['wins'] += 1
            elif winner == p2:
                stats[p2]['wins'] += 1
                
        rates = {}
        for agent, data in stats.items():
            rates[agent] = data['wins'] / data['matches'] if data['matches'] > 0 else 0
            
        current_model_name = f"Model Gen {gen_idx}"
        if current_model_name not in rates:
            return False
            
        current_rate = rates[current_model_name]
        
        # Check if current model's win rate is >= the rates of the static baselines
        for agent, rate in rates.items():
            if agent != current_model_name:
                # We check against Blind MCTS and Random Agent baselines
                if ("Blind MCTS" in agent or "Random Agent" in agent) and current_rate < rate:
                    return False
        return True
    except Exception as e:
        print(f"[Warning] Error parsing benchmark report for Gen {gen_idx}: {e}")
        return False


def run_pipeline(num_games=5, num_generations=3, mcts_iterations=15, epochs=2, wipe=False, games_per_matchup=2, max_rollout_depth=20, processes=None):
    """
    Runs the AlphaZero-style training pipeline end-to-end.
    
    Arguments:
      num_games: Number of self-play games to ensure are generated for each generation.
      num_generations: Number of incremental generational loops to run.
      mcts_iterations: Search iterations per move during self-play and evaluation.
      epochs: Training epochs per generation.
      wipe: If True, deletes all prior model files and generation folders before starting.
            If False, resumes seamlessly from where the last completed generation folder left off.
      games_per_matchup: Number of matches played between each pair of agents during round-robin tournament evaluation.
      max_rollout_depth: The rollout search depth limit for the Blind MCTS agent.
    """
    print(f"=== AlphaZero Pipeline Training ===")
    print(f"Parameters: num_games={num_games}, mcts_iterations={mcts_iterations}, epochs={epochs}")
    print(f"Wipe: {wipe} | games_per_matchup={games_per_matchup} | max_rollout_depth={max_rollout_depth}")
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    model_path = os.path.join(data_dir, "mcts_model.keras")
    onnx_path = os.path.join(data_dir, "mcts_model.onnx")
    
    # --- PHASE 0: Clean slate if wipe=True ---
    if wipe:
        print("\n[Wiping Prior Data] Deleting existing generation data and models as requested...")
        for gen_dir in glob.glob(os.path.join(data_dir, "gen*")):
            print(f"Removing generation directory: {gen_dir}")
            shutil.rmtree(gen_dir, ignore_errors=True)
            
        if os.path.exists(model_path):
            print(f"Removing Keras model: {model_path}")
            os.remove(model_path)
        if os.path.exists(onnx_path):
            print(f"Removing ONNX model: {onnx_path}")
            os.remove(onnx_path)
            
    engine_path = str(Path(__file__).resolve().parents[4] / "engine")
    
    # Determine the starting generation number
    start_gen = 1
    if not wipe:
        gen_folders = glob.glob(os.path.join(data_dir, "gen*"))
        gen_nums = []
        for folder in gen_folders:
            try:
                num = int(os.path.basename(folder).replace("gen", ""))
                gen_nums.append(num)
            except ValueError:
                pass
        if gen_nums:
            latest_gen = max(gen_nums)
            latest_gen_dir = os.path.join(data_dir, f"gen{latest_gen}")
            existing_games = len(glob.glob(os.path.join(latest_gen_dir, "game_*.json")))
            model_archived = os.path.exists(os.path.join(latest_gen_dir, "mcts_model.keras"))
            
            if existing_games >= num_games and model_archived:
                # The latest folder is fully completed. Start next generation.
                start_gen = latest_gen + 1
                print(f"\n[Resume] Latest Gen {latest_gen} is fully complete. Starting on Gen {start_gen}.")
            else:
                # Latest generation is incomplete or not trained. Resume this generation.
                start_gen = latest_gen
                print(f"\n[Resume] Resuming incomplete/untrained Gen {latest_gen}.")
                
    end_gen = start_gen + num_generations - 1
    
    for gen in range(start_gen, end_gen + 1):
        next_gen_dir = os.path.join(data_dir, f"gen{gen}")
        os.makedirs(next_gen_dir, exist_ok=True)
        
        print(f"\n========================================================")
        print(f" Starting Generation {gen} / {end_gen}")
        print(f"========================================================")
        
        # Check if this generation's self-play dataset is already complete
        games_count = len(glob.glob(os.path.join(next_gen_dir, "game_*.json")))
        
        if games_count >= num_games:
            print(f"[Skip Self-Play] Gen {gen} already has {games_count}/{num_games} games.")
        else:
            # --- PHASE 1: Self-Play Data Generation ---
            print(f"\n--- [Gen {gen}] Phase 1: Generating Data ({games_count}/{num_games} complete) ---")
            generate_dataset(
                num_games=num_games, 
                processes=processes, 
                output_dir=next_gen_dir, 
                mcts_iterations=mcts_iterations,
                use_cheating_mcts=(gen == 1)
            )
            
        # Check if this generation is already trained and archived
        model_archived = os.path.exists(os.path.join(next_gen_dir, "mcts_model.keras"))
        if model_archived:
            print(f"[Skip Training] Gen {gen} model already exists in archive.")
            # Restore model to root for next generation's self-play
            shutil.copy(os.path.join(next_gen_dir, "mcts_model.keras"), model_path)
            if os.path.exists(os.path.join(next_gen_dir, "mcts_model.onnx")):
                shutil.copy(os.path.join(next_gen_dir, "mcts_model.onnx"), onnx_path)
        else:
            # --- PHASE 2: Neural Network Training ---
            print(f"\n--- [Gen {gen}] Phase 2: Training Neural Network ---")
            train(
                data_dir=data_dir, 
                model_save_path=model_path, 
                max_games_buffer=10000, 
                epochs=epochs
            )
            
            # --- PHASE 3: Archive Model ---
            print(f"\n--- [Gen {gen}] Phase 3: Archiving Model ---")
            shutil.copy(model_path, os.path.join(next_gen_dir, "mcts_model.keras"))
            if os.path.exists(onnx_path):
                shutil.copy(onnx_path, os.path.join(next_gen_dir, "mcts_model.onnx"))
                
        # --- PHASE 4: Round-Robin Tournament evaluation ---
        benchmark_json_path = os.path.join(next_gen_dir, "benchmark_report.json")
        benchmark_txt_path = os.path.join(next_gen_dir, "benchmark_report.txt")
        
        if os.path.exists(benchmark_json_path) and os.path.exists(benchmark_txt_path):
            print(f"[Skip Benchmark] Gen {gen} benchmark already completed and saved.")
        else:
            print(f"\n--- [Gen {gen}] Phase 4: Running Generational Round-Robin Tournament ---")
            client = ShowdownClient(engine_path)
            try:
                # Build agent factories dynamically
                agent_factories = {
                    f"Model Gen {gen}": lambda prob: MCTSApproximationAgent(
                        prob, 
                        iterations=mcts_iterations, 
                        model_path=model_path
                    ),
                    "Blind MCTS": lambda prob: BlindMCTSAgent(
                        prob, 
                        iterations=mcts_iterations,
                        max_rollout_depth=max_rollout_depth
                    ),
                    "Random Agent": lambda prob: RandomAgent(prob)
                }
                
                # If there was a previous generation model, include it!
                if gen > 1:
                    prev_model_path = os.path.join(data_dir, f"gen{gen-1}", "mcts_model.keras")
                    if os.path.exists(prev_model_path):
                        agent_factories[f"Model Gen {gen-1}"] = lambda prob, p=prev_model_path: MCTSApproximationAgent(
                            prob, 
                            iterations=mcts_iterations, 
                            model_path=p
                        )
                
                # Match them up in a Round Robin tournament
                benchmark = RoundRobinBenchmark(client, agent_factories, games_per_matchup=games_per_matchup)
                benchmark.run()
                benchmark.print_report()
                
                # Save results to JSON
                with open(benchmark_json_path, "w") as f:
                    json.dump(benchmark.results, f, indent=4)
                print(f"Saved benchmark JSON report to: {benchmark_json_path}")
                
                # Save formatted report to text
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
                        
                report_lines = []
                report_lines.append("="*80)
                report_lines.append(" "*24 + f"GENERATION {gen} ROUND ROBIN REPORT")
                report_lines.append("="*80)
                report_lines.append(f"Total Matches Played: {len(benchmark.results)}")
                report_lines.append(f"\n{'Agent Name':<20} | {'Win Rate':<10} | {'Avg Time/Turn':<15}")
                report_lines.append("-" * 80)
                for agent, data in stats.items():
                    win_rate = (data['wins'] / data['matches']) * 100 if data['matches'] > 0 else 0
                    avg_time = data['total_time'] / data['matches'] if data['matches'] > 0 else 0
                    report_lines.append(f"{agent:<20} | {win_rate:>5.1f}%     | {avg_time:>10.3f}s")
                report_lines.append("="*80 + "\n")
                
                report_text = "\n".join(report_lines)
                with open(benchmark_txt_path, "w") as f:
                    f.write(report_text)
                print(f"Saved benchmark text report to: {benchmark_txt_path}")
                
            except Exception as e:
                print(f"[Warning] Tournament benchmark failed: {e}")
            finally:
                client.close()
                
        # --- Early Stopping Evaluation: learning quality check ---
        if gen >= 3:
            history_checks = []
            for g in range(gen - 2, gen + 1):
                g_dir = os.path.join(data_dir, f"gen{g}")
                history_checks.append(check_generation_won(g_dir, g))
                
            print(f"\nLearning Check (Last 3 Generations): {['Won' if c else 'Lost' for c in history_checks]}")
            if not any(history_checks):
                print("\n========================================================")
                print(" [Early Stopping] Pipeline Halted Early!")
                print(" Reason: The model failed to outperform/tie static baselines")
                print(" (Blind MCTS & Random Agent) in all of the last 3 generations.")
                print("========================================================\n")
                break
                
        print(f"\n=== Generation {gen} Completed Successfully! ===")


if __name__ == "__main__":
    # Test settings: 4 self-play games per gen, 3 generations, 15 MCTS iterations, 2 training epochs, wipe=False
    run_pipeline(
        num_games=4, 
        num_generations=3, 
        mcts_iterations=15, 
        epochs=2, 
        wipe=False,
        games_per_matchup=2,
        max_rollout_depth=20,
        processes=None
    )