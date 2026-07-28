from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[3]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from battle_agents.mcts_approximation.pipeline.plot_training import (
    load_training_log,
)
from benchmarks.plotting import (
    plot_matchup_heatmap,
    plot_win_rates,
    resolve_winner,
)


_GENERATION_PATTERN = re.compile(r"gen(\d+)$", re.IGNORECASE)
_MODEL_PATTERN = re.compile(r"Model Gen (\d+)$", re.IGNORECASE)
_RATING_KEYS = ("rating", "elo", "elo_rating")
_AUTOENCODER_KEYS = (
    "autoencoder_error",
    "autoencoder_loss",
    "reconstruction_error",
    "reconstruction_loss",
    "encoder_loss",
    "val_autoencoder_loss",
    "val_reconstruction_loss",
)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _generation_number(path: Path) -> int:
    directory = path if path.is_dir() else path.parent
    match = _GENERATION_PATTERN.fullmatch(directory.name)
    if match is None:
        raise ValueError(f"Path is not inside a genN directory: {path}")
    return int(match.group(1))


def discover_generation_reports(data_dir: str | Path) -> list[Path]:
    """Find benchmark reports ordered by their numeric generation suffix."""

    reports = [
        report
        for report in Path(data_dir).glob("gen*/benchmark_report.json")
        if _GENERATION_PATTERN.fullmatch(report.parent.name)
    ]
    return sorted(reports, key=_generation_number)


def _load_report(path: Path) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as report_file:
        loaded = json.load(report_file)
    if isinstance(loaded, list):
        return [item for item in loaded if isinstance(item, Mapping)], {}
    if isinstance(loaded, Mapping):
        raw_results = loaded.get("results", loaded.get("matches", []))
        if isinstance(raw_results, list):
            return [
                item for item in raw_results if isinstance(item, Mapping)
            ], loaded
    raise ValueError(f"Expected match results in {path}")


def _players(result: Mapping[str, Any]) -> tuple[str, str] | None:
    p1 = result.get("p1")
    p2 = result.get("p2")
    if p1 is None or p2 is None:
        return None
    return str(p1), str(p2)


def _extract_rating(
    metadata: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    candidate: str,
) -> float | None:
    for key in _RATING_KEYS:
        direct = _number(metadata.get(key))
        if direct is not None:
            return direct
        ratings = metadata.get(f"{key}s")
        if isinstance(ratings, Mapping):
            candidate_rating = _number(ratings.get(candidate))
            if candidate_rating is not None:
                return candidate_rating
    observed = []
    for result in results:
        players = _players(result)
        if players is None or candidate not in players:
            continue
        seat = "p1" if players[0] == candidate else "p2"
        for key in _RATING_KEYS:
            value = _number(result.get(f"{seat}_{key}", result.get(key)))
            if value is not None:
                observed.append(value)
                break
    return fmean(observed) if observed else None


def summarize_generation(report_path: str | Path) -> dict[str, Any]:
    """Summarize one candidate's benchmark against its incumbent/baselines."""

    path = Path(report_path)
    generation = _generation_number(path)
    results, metadata = _load_report(path)
    candidate = f"Model Gen {generation}"
    comparison_wins = 0
    comparison_matches = 0
    overall_wins = 0
    overall_matches = 0
    opponents: set[str] = set()
    baseline_counts: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "matches": 0}
    )
    turns: list[float] = []
    survivors: list[float] = []

    for result in results:
        turn_count = _number(result.get("turns"))
        if turn_count is not None and turn_count >= 0:
            turns.append(turn_count)
        survivor_count = _number(result.get("survivors"))
        if survivor_count is not None and survivor_count >= 0:
            survivors.append(survivor_count)

        players = _players(result)
        if players is None or candidate not in players:
            continue
        overall_matches += 1
        winner = resolve_winner(result)
        if winner == candidate:
            overall_wins += 1

        opponent = players[1] if players[0] == candidate else players[0]
        model_match = _MODEL_PATTERN.fullmatch(opponent)
        if model_match and opponent != candidate:
            opponents.add(opponent)
            comparison_matches += 1
            if winner == candidate:
                comparison_wins += 1
        elif opponent != candidate:
            baseline_counts[opponent]["matches"] += 1
            if winner == candidate:
                baseline_counts[opponent]["wins"] += 1

    opponent_names = sorted(
        opponents,
        key=lambda name: int(_MODEL_PATTERN.fullmatch(name).group(1)),
    )
    baseline_win_rates = {
        name: counts["wins"] / counts["matches"]
        for name, counts in sorted(baseline_counts.items())
        if counts["matches"]
    }
    return {
        "generation": generation,
        "candidate": candidate,
        "opponents": opponent_names,
        "opponent": ", ".join(opponent_names) if opponent_names else None,
        "wins": comparison_wins,
        "matches": comparison_matches,
        "win_rate": (
            comparison_wins / comparison_matches if comparison_matches else None
        ),
        "overall_wins": overall_wins,
        "overall_matches": overall_matches,
        "overall_win_rate": (
            overall_wins / overall_matches if overall_matches else None
        ),
        "baseline_win_rates": baseline_win_rates,
        "avg_turns": fmean(turns) if turns else None,
        "avg_survivors": fmean(survivors) if survivors else None,
        "rating": _extract_rating(metadata, results, candidate),
    }


def _save(fig: plt.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def _empty(ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def plot_champion_evolution(
    summaries: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot candidate win rate against the active incumbent champion."""

    comparable = [
        summary
        for summary in summaries
        if summary.get("matches") and summary.get("win_rate") is not None
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if not comparable:
        _empty(
            ax,
            "Evolução do champion",
            "Sem partidas candidate vs champion disponíveis",
        )
        return _save(fig, output_path)

    generations = [int(summary["generation"]) for summary in comparable]
    rates = [float(summary["win_rate"]) * 100 for summary in comparable]
    ax.plot(generations, rates, marker="o", linewidth=2, color="#4472C4")
    ax.axhline(50, color="#7F7F7F", linestyle="--", label="Paridade (50%)")
    ax.set_title("Champion: candidate vs incumbent")
    ax.set_xlabel("Geração do candidate")
    ax.set_ylabel("Win-rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(generations)
    ax.grid(alpha=0.25)
    ax.legend()
    for generation, rate, summary in zip(generations, rates, comparable):
        opponent = summary.get("opponent") or "champion"
        ax.annotate(
            f"{rate:.1f}% vs {opponent}\n(n={summary['matches']})",
            (generation, rate),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    return _save(fig, output_path)


def plot_rating_or_baselines(
    summaries: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot ELO/rating when present, otherwise candidate win-rate by baseline."""

    rated = [item for item in summaries if item.get("rating") is not None]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if rated:
        generations = [int(item["generation"]) for item in rated]
        ratings = [float(item["rating"]) for item in rated]
        ax.plot(generations, ratings, marker="o", linewidth=2, color="#7030A0")
        ax.set_title("Evolução de rating/ELO")
        ax.set_ylabel("Rating")
        ax.set_xticks(generations)
    else:
        baselines = sorted(
            {
                name
                for item in summaries
                for name in item.get("baseline_win_rates", {})
            }
        )
        plotted = False
        for baseline in baselines:
            points = [
                (int(item["generation"]), item["baseline_win_rates"][baseline])
                for item in summaries
                if baseline in item.get("baseline_win_rates", {})
            ]
            if points:
                ax.plot(
                    [point[0] for point in points],
                    [point[1] * 100 for point in points],
                    marker="o",
                    label=baseline,
                )
                plotted = True
        if not plotted:
            _empty(
                ax,
                "Rating ou win-rate vs baselines",
                "Sem rating e sem partidas contra baselines",
            )
            return _save(fig, output_path)
        ax.axhline(50, color="#7F7F7F", linestyle="--", linewidth=1)
        ax.set_title("Win-rate do candidate vs baselines")
        ax.set_ylabel("Win-rate (%)")
        ax.set_ylim(0, 105)
        ax.legend()
    ax.set_xlabel("Geração")
    ax.grid(alpha=0.25)
    return _save(fig, output_path)


def plot_battle_metrics(
    summaries: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot average battle length and winner survivors by generation."""

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    definitions = (
        ("avg_turns", "Turnos médios", "#ED7D31"),
        ("avg_survivors", "Survivors médios do vencedor", "#70AD47"),
    )
    for ax, (key, title, color) in zip(axes, definitions):
        points = [
            (int(item["generation"]), float(item[key]))
            for item in summaries
            if item.get(key) is not None
        ]
        if not points:
            _empty(ax, title, "Métrica ausente nos benchmarks")
            continue
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            color=color,
        )
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Geração")
    fig.tight_layout()
    return _save(fig, output_path)


def _latest_training_session(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """Return the suffix after the final epoch reset in a cumulative CSV log."""

    start = 0
    previous: float | None = None
    for index, row in enumerate(rows):
        epoch = row.get("epoch")
        if (
            previous is not None
            and epoch is not None
            and math.isfinite(epoch)
            and epoch <= previous
        ):
            start = index
        if epoch is not None and math.isfinite(epoch):
            previous = epoch
    return rows[start:]


def _last_metric(rows: Sequence[Mapping[str, float]], names: Sequence[str]) -> float | None:
    for name in names:
        for row in reversed(rows):
            value = _number(row.get(name))
            if value is not None:
                return value
    return None


def _metadata_metrics(generation_dir: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in (
        "generation_metrics.json",
        "metrics.json",
        "autoencoder_metrics.json",
    ):
        path = generation_dir / name
        if not path.is_file():
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, Mapping):
            merged.update(loaded)
            for container_name in ("metrics", "training", "autoencoder"):
                nested = loaded.get(container_name)
                if isinstance(nested, Mapping):
                    merged.update(nested)
    return merged


def summarize_training_metrics(data_dir: str | Path) -> list[dict[str, Any]]:
    """Extract the final train/validation losses for each archived generation."""

    summaries = []
    directories = [
        path
        for path in Path(data_dir).glob("gen*")
        if path.is_dir() and _GENERATION_PATTERN.fullmatch(path.name)
    ]
    for generation_dir in sorted(directories, key=_generation_number):
        log_path = generation_dir / "training_log.csv"
        rows: list[dict[str, float]] = []
        if log_path.is_file():
            try:
                rows = _latest_training_session(load_training_log(log_path))
            except (OSError, ValueError):
                rows = []
        metadata = _metadata_metrics(generation_dir)
        autoencoder = _last_metric(rows, _AUTOENCODER_KEYS)
        if autoencoder is None:
            autoencoder = _last_metric([metadata], _AUTOENCODER_KEYS)
        summaries.append(
            {
                "generation": _generation_number(generation_dir),
                "policy_loss": _last_metric(
                    rows, ("val_policy_loss", "policy_loss")
                ),
                "value_loss": _last_metric(
                    rows, ("val_value_loss", "value_loss")
                ),
                "autoencoder_error": autoencoder,
            }
        )
    return summaries


def plot_generation_losses(
    summaries: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot policy, value and autoencoder errors across generations."""

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    definitions = (
        ("policy_loss", "Policy loss", "#4472C4"),
        ("value_loss", "Value loss", "#ED7D31"),
        ("autoencoder_error", "Erro do autoencoder", "#70AD47"),
    )
    for ax, (key, title, color) in zip(axes, definitions):
        points = [
            (int(item["generation"]), float(item[key]))
            for item in summaries
            if item.get(key) is not None
        ]
        if not points:
            message = (
                "Encoder congelado: nenhuma métrica por geração registrada"
                if key == "autoencoder_error"
                else "Métrica ausente nos training_log.csv"
            )
            _empty(ax, title, message)
            continue
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            color=color,
        )
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Geração")
    fig.tight_layout()
    return _save(fig, output_path)


def _per_generation_benchmark_plots(
    reports: Sequence[Path], output_dir: Path
) -> tuple[list[Path], list[Mapping[str, Any]]]:
    paths: list[Path] = []
    consolidated: list[Mapping[str, Any]] = []
    for report in reports:
        results, _ = _load_report(report)
        generation = _generation_number(report)
        consolidated.extend(results)
        paths.extend(
            (
                plot_win_rates(
                    results,
                    output_dir / f"gen{generation}_agent_win_rates.png",
                ),
                plot_matchup_heatmap(
                    results,
                    output_dir / f"gen{generation}_matchup_heatmap.png",
                ),
            )
        )
    return paths, consolidated


def plot_generations(
    data_dir: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate aggregate and per-generation PNG reports."""

    reports = discover_generation_reports(data_dir)
    summaries = [summarize_generation(report) for report in reports]
    training_summaries = summarize_training_metrics(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_champion_evolution(
            summaries, output / "champion_win_rate_evolution.png"
        ),
        plot_rating_or_baselines(
            summaries, output / "rating_or_baseline_evolution.png"
        ),
        plot_battle_metrics(
            summaries, output / "battle_metrics_evolution.png"
        ),
        plot_generation_losses(
            training_summaries, output / "losses_by_generation.png"
        ),
    ]
    benchmark_paths, consolidated = _per_generation_benchmark_plots(
        reports, output
    )
    paths.extend(benchmark_paths)
    if consolidated:
        paths.extend(
            (
                plot_win_rates(
                    consolidated, output / "consolidated_agent_win_rates.png"
                ),
                plot_matchup_heatmap(
                    consolidated, output / "consolidated_matchup_heatmap.png"
                ),
            )
        )
    return paths


def parse_args() -> argparse.Namespace:
    project_data = Path(__file__).resolve().parents[3] / "data"
    parser = argparse.ArgumentParser(
        description="Plot pipeline evolution from genN archives."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_data,
        help="Directory containing genN folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PNG output directory (default: <data-dir>/generation_plots).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.data_dir / "generation_plots"
    for plot_path in plot_generations(args.data_dir, output_dir):
        print(f"Generation plot saved to: {plot_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "discover_generation_reports",
    "plot_battle_metrics",
    "plot_champion_evolution",
    "plot_generation_losses",
    "plot_generations",
    "plot_rating_or_baselines",
    "summarize_generation",
    "summarize_training_metrics",
]
