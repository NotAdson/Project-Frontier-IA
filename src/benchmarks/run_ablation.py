"""
Meta-Planner ablation experiment.

Runs the AlphaZero-style training pipeline twice sequentially:
  1. Variant A: WITH the Meta-Planner (default architecture)
  2. Variant B: WITHOUT the Meta-Planner (ablation -- no per-mon self-attention,
                 no meta_plan output, no counterfactual supervision)

Then runs a round-robin tournament between the two trained variants plus two
static baselines (Blind MCTS, Random Agent) and prints a win-rate report.

The two variants write to separate model files so they don't clobber each other:
  - Variant A: data/mcts_model.{keras,onnx}        + data/champion.json
  - Variant B: data/mcts_model_no_meta.{keras,onnx} + data/champion_no_meta.json

Usage:
    python src/benchmarks/run_ablation.py
    python src/benchmarks/run_ablation.py --num-games 50 --num-generations 5 --epochs 3
    python src/benchmarks/run_ablation.py --skip-training  # only re-run the tournament

Decision rule (printed at the end):
    win-rate difference <  2%  -> Meta-Planner has no measurable effect; remove it.
    win-rate difference >= 2%  -> Meta-Planner contributes; keep it and apply deeper fixes.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

# Add src to python path (same convention as run_benchmark.py / benchmark_aux_vs_champion.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from battle_agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent
from battle_agents.mcts_approximation.mcts_approximation_agent import \
    MCTSApproximationAgent
from battle_agents.mcts_approximation.pipeline.run_pipeline import run_pipeline
from battle_agents.random.random_agent import RandomAgent
from benchmarks.round_robin import RoundRobinBenchmark
from core.client.showdown_client import ShowdownClient


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
ENGINE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

# Decision threshold for the ablation verdict (in percentage points of win rate).
DECISION_THRESHOLD_PCT = 2.0


def run_training_phase(label: str, use_meta_planner: bool, args):
    """Run one full training pipeline for the given variant."""
    print("\n" + "=" * 80)
    print(f"  TRAINING PHASE: {label}")
    print(f"  use_meta_planner={use_meta_planner}")
    print("=" * 80 + "\n")

    run_pipeline(
        num_games=args.num_games,
        num_generations=args.num_generations,
        mcts_iterations=args.mcts_iterations,
        epochs=args.epochs,
        wipe=args.wipe,
        games_per_matchup=args.games_per_matchup,
        max_rollout_depth=args.max_rollout_depth,
        processes=args.processes,
        use_meta_planner=use_meta_planner,
    )


def run_tournament(args):
    """Round-robin between the two trained variants + static baselines."""
    print("\n" + "=" * 80)
    print("  ABLATION TOURNAMENT")
    print("=" * 80 + "\n")

    onnx_with_meta = os.path.join(DATA_DIR, "mcts_model.onnx")
    onnx_no_meta = os.path.join(DATA_DIR, "mcts_model_no_meta.onnx")

    missing = []
    if not os.path.exists(onnx_with_meta):
        missing.append(onnx_with_meta)
    if not os.path.exists(onnx_no_meta):
        missing.append(onnx_no_meta)
    if missing:
        print("[Error] Missing trained models -- cannot run tournament:")
        for m in missing:
            print(f"  - {m}")
        print("Run without --skip-training first, or check that the pipeline completed.")
        return None

    print(f"Loading WITH-Meta model from: {onnx_with_meta}")
    print(f"Loading NO-Meta  model from: {onnx_no_meta}")
    print(f"Engine: {ENGINE_PATH}\n")

    iterations = args.benchmark_mcts_iterations
    agent_factories = {
        "With Meta-Planner": lambda prob, p=onnx_with_meta: MCTSApproximationAgent(
            prob, iterations=iterations, model_path=p
        ),
        "No Meta-Planner": lambda prob, p=onnx_no_meta: MCTSApproximationAgent(
            prob, iterations=iterations, model_path=p
        ),
        "Blind MCTS": lambda prob: BlindMCTSAgent(
            prob, iterations=iterations, max_rollout_depth=args.max_rollout_depth
        ),
        "Random Agent": lambda prob: RandomAgent(prob),
    }

    client = ShowdownClient(ENGINE_PATH)
    try:
        benchmark = RoundRobinBenchmark(
            client, agent_factories, games_per_matchup=args.benchmark_games_per_matchup
        )
        benchmark.run()
        benchmark.print_report()
        return benchmark.results
    finally:
        client.close()


def print_verdict(results):
    """Compute win rates, decide the ablation verdict, and print a report."""
    if not results:
        return

    stats = defaultdict(lambda: {"wins": 0, "matches": 0, "total_time": 0.0})
    for r in results:
        p1, p2, winner = r["p1"], r["p2"], r["winner"]
        stats[p1]["matches"] += 1
        stats[p2]["matches"] += 1
        stats[p1]["total_time"] += r["p1_avg_time"]
        stats[p2]["total_time"] += r["p2_avg_time"]
        if winner == p1:
            stats[p1]["wins"] += 1
        elif winner == p2:
            stats[p2]["wins"] += 1

    print("\n" + "=" * 80)
    print(" " * 28 + "ABLATION REPORT")
    print("=" * 80)
    print(f"Total Matches Played: {len(results)}\n")
    print(f"{'Agent':<22} | {'Win Rate':<10} | {'Avg Time/Turn':<15}")
    print("-" * 80)
    for agent, data in stats.items():
        win_rate = (data["wins"] / data["matches"]) * 100 if data["matches"] > 0 else 0.0
        avg_time = data["total_time"] / data["matches"] if data["matches"] > 0 else 0.0
        print(f"{agent:<22} | {win_rate:>7.1f}%  | {avg_time:>13.3f}s")
    print("=" * 80)

    # Head-to-head between the two variants
    wr_with = (stats["With Meta-Planner"]["wins"] / stats["With Meta-Planner"]["matches"]) * 100 \
        if stats["With Meta-Planner"]["matches"] > 0 else 0.0
    wr_without = (stats["No Meta-Planner"]["wins"] / stats["No Meta-Planner"]["matches"]) * 100 \
        if stats["No Meta-Planner"]["matches"] > 0 else 0.0
    delta = wr_with - wr_without

    print("\n" + "-" * 80)
    print(f"  With Meta-Planner win rate: {wr_with:.1f}%")
    print(f"  No  Meta-Planner win rate: {wr_without:.1f}%")
    print(f"  Delta (with - no):          {delta:+.1f} percentage points")
    print("-" * 80)

    print("\nVerdict:")
    if abs(delta) < DECISION_THRESHOLD_PCT:
        print(f"  |delta| < {DECISION_THRESHOLD_PCT:.1f}%  ->  Meta-Planner has no "
              "measurable effect.")
        print("  Recommendation: REMOVE the Meta-Planner (reclaim simplicity, "
              "training speed, inference cost).")
    elif delta >= DECISION_THRESHOLD_PCT:
        print(f"  delta >= +{DECISION_THRESHOLD_PCT:.1f}%  ->  Meta-Planner helps.")
        print("  Recommendation: KEEP the Meta-Planner and apply deeper fixes "
              "(EMA targets, ranking loss, positional encoding, loss_weight sweep).")
    else:
        print(f"  delta <= -{DECISION_THRESHOLD_PCT:.1f}%  ->  Meta-Planner HURTS.")
        print("  Recommendation: REMOVE the Meta-Planner -- it actively degrades "
              "performance.")
    print("=" * 80 + "\n")

    # Persist the report to data/ so it survives across runs
    report_path = os.path.join(DATA_DIR, "ablation_report.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "results": results,
            "win_rates": {a: (d["wins"] / d["matches"]) * 100 if d["matches"] > 0 else 0.0
                          for a, d in stats.items()},
            "delta_with_minus_no": delta,
            "threshold_pct": DECISION_THRESHOLD_PCT,
        }, f, indent=2)
    print(f"Saved ablation report to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Meta-Planner ablation: train both variants, then benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Training phase args (forwarded to run_pipeline)
    parser.add_argument("--num-games", type=int, default=5,
                        help="Self-play games per generation (default: 5).")
    parser.add_argument("--num-generations", type=int, default=3,
                        help="Generations per variant (default: 3).")
    parser.add_argument("--mcts-iterations", type=int, default=15,
                        help="MCTS iterations per move during self-play (default: 15).")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs per generation (default: 2).")
    parser.add_argument("--wipe", action="store_true",
                        help="Wipe prior data/models before training each variant.")
    parser.add_argument("--games-per-matchup", type=int, default=5,
                        help="Games per matchup in the pipeline's internal tournament (default: 5).")
    parser.add_argument("--max-rollout-depth", type=int, default=20,
                        help="Rollout depth limit for Blind MCTS (default: 20).")
    parser.add_argument("--processes", type=int, default=None,
                        help="Parallel self-play processes (default: None = auto).")
    # Tournament phase args
    parser.add_argument("--benchmark-mcts-iterations", type=int, default=30,
                        help="MCTS iterations per move during the final tournament (default: 30).")
    parser.add_argument("--benchmark-games-per-matchup", type=int, default=10,
                        help="Games per matchup in the final tournament (default: 10). "
                             "Higher = less noisy win-rate signal.")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip both training phases and only run the tournament "
                             "(assumes both .onnx files already exist).")
    args = parser.parse_args()

    if not args.skip_training:
        # Phase 1: train WITH Meta-Planner (default architecture)
        run_training_phase("Variant A (WITH Meta-Planner)", use_meta_planner=True, args=args)
        # Phase 2: train WITHOUT Meta-Planner (ablation)
        run_training_phase("Variant B (NO Meta-Planner)", use_meta_planner=False, args=args)
    else:
        print("[--skip-training] Skipping training phases; only running the tournament.")

    # Phase 3: round-robin tournament between both variants + baselines
    results = run_tournament(args)
    if results is not None:
        print_verdict(results)


if __name__ == "__main__":
    main()
