from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


Result = Mapping[str, Any]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def resolve_winner(result: Result) -> str | None:
    """Return the winning agent name, normalizing common player labels.

    Draws, missing winners and unrecognized values return ``None``.  They
    still count as matches when win rates are calculated.
    """

    p1 = result.get("p1")
    p2 = result.get("p2")
    winner = result.get("winner")

    if p1 is not None and winner == p1:
        return str(p1)
    if p2 is not None and winner == p2:
        return str(p2)
    if not isinstance(winner, str):
        return None

    normalized = winner.strip().lower()
    if normalized in {"p1", "player 1", "player1"} and p1 is not None:
        return str(p1)
    if normalized in {"p2", "player 2", "player2"} and p2 is not None:
        return str(p2)
    return None


def _players(result: Result) -> tuple[str, str] | None:
    p1 = result.get("p1")
    p2 = result.get("p2")
    if p1 is None or p2 is None:
        return None
    return str(p1), str(p2)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize_results(results: Sequence[Result]) -> dict[str, Any]:
    """Aggregate win rates, match counts and decision times by agent."""

    wins: defaultdict[str, int] = defaultdict(int)
    matches: defaultdict[str, int] = defaultdict(int)
    turn_times: defaultdict[str, list[float]] = defaultdict(list)

    for result in results:
        players = _players(result)
        if players is None:
            continue
        p1, p2 = players
        matches[p1] += 1
        matches[p2] += 1

        winner = resolve_winner(result)
        if winner in players:
            wins[winner] += 1

        p1_time = _finite_number(result.get("p1_avg_time"))
        p2_time = _finite_number(result.get("p2_avg_time"))
        if p1_time is not None and p1_time >= 0:
            turn_times[p1].append(p1_time)
        if p2_time is not None and p2_time >= 0:
            turn_times[p2].append(p2_time)

    agents = sorted(matches)
    win_rates = {
        agent: wins[agent] / matches[agent] if matches[agent] else 0.0
        for agent in agents
    }
    avg_turn_times = {
        agent: fmean(turn_times[agent]) if turn_times[agent] else 0.0
        for agent in agents
    }
    turn_time_stddev = {
        agent: pstdev(turn_times[agent]) if len(turn_times[agent]) > 1 else 0.0
        for agent in agents
    }

    return {
        "agents": agents,
        "wins": {agent: wins[agent] for agent in agents},
        "matches": {agent: matches[agent] for agent in agents},
        "win_rates": win_rates,
        "turn_times": {agent: list(turn_times[agent]) for agent in agents},
        "avg_turn_times": avg_turn_times,
        "turn_time_stddev": turn_time_stddev,
    }


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_and_close(fig: plt.Figure, output_path: str | Path) -> Path:
    path = _prepare_output_path(output_path)
    try:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def _empty_plot(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        "No benchmark data available",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_axis_off()


def plot_win_rates(
    results: Sequence[Result], output_path: str | Path
) -> Path:
    """Create a win-rate bar chart for every agent."""

    summary = summarize_results(results)
    agents = summary["agents"]
    fig, ax = plt.subplots(figsize=(max(7, len(agents) * 1.4), 5))

    if not agents:
        _empty_plot(ax, "Win rate by agent")
        return _save_and_close(fig, output_path)

    rates = [summary["win_rates"][agent] * 100 for agent in agents]
    bars = ax.bar(agents, rates, color="#4472C4")
    ax.set_title("Win rate by agent")
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    for bar, agent, rate in zip(bars, agents, rates):
        ax.annotate(
            f"{rate:.1f}%\n(n={summary['matches'][agent]})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    return _save_and_close(fig, output_path)


def plot_turns_boxplot(
    results: Sequence[Result], output_path: str | Path
) -> Path:
    """Create a turn-count box plot grouped by unordered matchup."""

    turns_by_matchup: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for result in results:
        players = _players(result)
        turns = _finite_number(result.get("turns"))
        if players is None or turns is None or turns < 0:
            continue
        turns_by_matchup[tuple(sorted(players))].append(turns)

    matchups = sorted(turns_by_matchup)
    fig_width = max(7, len(matchups) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    if not matchups:
        _empty_plot(ax, "Turns per match")
        return _save_and_close(fig, output_path)

    labels = [f"{p1} vs {p2}" for p1, p2 in matchups]
    values = [turns_by_matchup[matchup] for matchup in matchups]
    ax.boxplot(values, patch_artist=True)
    ax.set_xticks(range(1, len(labels) + 1), labels=labels)
    for patch in ax.patches:
        patch.set_facecolor("#70AD47")
        patch.set_alpha(0.75)
    ax.set_title("Distribution of turns by matchup")
    ax.set_ylabel("Turns")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    return _save_and_close(fig, output_path)


def plot_avg_turn_time(
    results: Sequence[Result], output_path: str | Path
) -> Path:
    """Create a mean decision-time chart with population standard deviation."""

    summary = summarize_results(results)
    agents = summary["agents"]
    fig, ax = plt.subplots(figsize=(max(7, len(agents) * 1.4), 5))

    if not agents:
        _empty_plot(ax, "Average time per turn")
        return _save_and_close(fig, output_path)

    averages = [summary["avg_turn_times"][agent] for agent in agents]
    deviations = [summary["turn_time_stddev"][agent] for agent in agents]
    bars = ax.bar(
        agents,
        averages,
        yerr=deviations,
        capsize=5,
        color="#ED7D31",
    )
    ax.set_title("Average time per turn")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    for bar, agent, average in zip(bars, agents, averages):
        ax.annotate(
            f"{average:.3f}s\n(n={len(summary['turn_times'][agent])})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    return _save_and_close(fig, output_path)


def _matchup_matrix(
    results: Sequence[Result], agents: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    size = len(agents)
    index = {agent: position for position, agent in enumerate(agents)}
    wins = np.zeros((size, size), dtype=float)
    matches = np.zeros((size, size), dtype=int)

    for result in results:
        players = _players(result)
        if players is None:
            continue
        p1, p2 = players
        if p1 not in index or p2 not in index:
            continue
        p1_index, p2_index = index[p1], index[p2]
        matches[p1_index, p2_index] += 1
        matches[p2_index, p1_index] += 1

        winner = resolve_winner(result)
        if winner == p1:
            wins[p1_index, p2_index] += 1
        elif winner == p2:
            wins[p2_index, p1_index] += 1

    matrix = np.full((size, size), np.nan, dtype=float)
    np.divide(wins, matches, out=matrix, where=matches > 0)
    return matrix, matches


def plot_matchup_heatmap(
    results: Sequence[Result], output_path: str | Path
) -> Path:
    """Create an asymmetric agent-versus-agent win-rate heat map."""

    agents = summarize_results(results)["agents"]
    size = max(6, len(agents) * 1.15)
    fig, ax = plt.subplots(figsize=(size, size))

    if not agents:
        _empty_plot(ax, "Head-to-head win rate")
        return _save_and_close(fig, output_path)

    matrix, matches = _matchup_matrix(results, agents)
    color_map = plt.get_cmap("RdYlGn").with_extremes(bad="#E7E6E6")
    image = ax.imshow(matrix, cmap=color_map, vmin=0, vmax=1)
    color_bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    color_bar.set_label("Win rate")

    ax.set_title("Head-to-head win rate")
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent")
    ax.set_xticks(range(len(agents)), labels=agents, rotation=35, ha="right")
    ax.set_yticks(range(len(agents)), labels=agents)

    for row in range(len(agents)):
        for column in range(len(agents)):
            value = matrix[row, column]
            label = (
                "N/A"
                if np.isnan(value)
                else f"{value * 100:.0f}%\n(n={matches[row, column]})"
            )
            text_color = (
                "white"
                if not np.isnan(value) and (value < 0.2 or value > 0.8)
                else "black"
            )
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    return _save_and_close(fig, output_path)


def _safe_prefix(prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix)).strip("._")
    return cleaned or "benchmark"


def plot_benchmark_report(
    results: Sequence[Result],
    output_dir: str | Path | None = None,
    prefix: str = "benchmark",
) -> list[Path]:
    """Generate all benchmark charts and return their PNG paths."""

    output_directory = (
        DEFAULT_RESULTS_DIR if output_dir is None else Path(output_dir)
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    filename_prefix = _safe_prefix(prefix)

    plotters = (
        ("win_rates", plot_win_rates),
        ("turns_boxplot", plot_turns_boxplot),
        ("avg_turn_time", plot_avg_turn_time),
        ("matchup_heatmap", plot_matchup_heatmap),
    )
    return [
        plotter(results, output_directory / f"{filename_prefix}_{suffix}.png")
        for suffix, plotter in plotters
    ]


__all__ = [
    "DEFAULT_RESULTS_DIR",
    "plot_avg_turn_time",
    "plot_benchmark_report",
    "plot_matchup_heatmap",
    "plot_turns_boxplot",
    "plot_win_rates",
    "resolve_winner",
    "summarize_results",
]
