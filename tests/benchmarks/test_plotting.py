import json

import pytest

from battle_agents.mcts_approximation.pipeline.plot_generations import (
    discover_generation_reports,
    plot_generations,
    summarize_generation,
    summarize_training_metrics,
)
from battle_agents.mcts_approximation.pipeline.plot_training import (
    _head_loss_names,
    load_training_log,
    plot_training,
)
from benchmarks.plotting import (
    plot_benchmark_report,
    resolve_winner,
    summarize_results,
)
from core.benchmark import BaseBenchmark

import matplotlib.pyplot as plt


RESULTS = [
    {
        "p1": "Random",
        "p2": "Neural",
        "winner": "Neural",
        "turns": 20,
        "p1_avg_time": 0.01,
        "p2_avg_time": 0.10,
    },
    {
        "p1": "Neural",
        "p2": "Random",
        "winner": "p1",
        "turns": 25,
        "p1_avg_time": 0.12,
        "p2_avg_time": 0.03,
    },
    {
        "p1": "Random",
        "p2": "Neural",
        "winner": None,
        "turns": 30,
        "p1_avg_time": 0.02,
        "p2_avg_time": 0.11,
    },
]


def test_summarize_results_tracks_seats_and_draws():
    summary = summarize_results(RESULTS)

    assert summary["agents"] == ["Neural", "Random"]
    assert summary["matches"] == {"Neural": 3, "Random": 3}
    assert summary["wins"] == {"Neural": 2, "Random": 0}
    assert summary["win_rates"]["Neural"] == pytest.approx(2 / 3)
    assert summary["avg_turn_times"]["Neural"] == pytest.approx(0.11)
    assert summary["avg_turn_times"]["Random"] == pytest.approx(0.02)


def test_resolve_winner_normalizes_player_labels():
    assert resolve_winner({"p1": "A", "p2": "B", "winner": "Player 1"}) == "A"
    assert resolve_winner({"p1": "A", "p2": "B", "winner": "p2"}) == "B"
    assert resolve_winner({"p1": "A", "p2": "B", "winner": "draw"}) is None


@pytest.mark.parametrize("results", [RESULTS, []])
def test_plot_benchmark_report_creates_four_pngs_and_closes_figures(
    tmp_path, results
):
    paths = plot_benchmark_report(results, tmp_path, prefix="synthetic benchmark")

    assert len(paths) == 4
    assert all(path.suffix == ".png" for path in paths)
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert plt.get_fignums() == []


def test_base_benchmark_plot_report_uses_shared_plotter(tmp_path):
    class DummyBenchmark(BaseBenchmark):
        def run(self):
            return self.results

    benchmark = DummyBenchmark(client=None)
    benchmark.results = list(RESULTS)

    paths = benchmark.plot_report(tmp_path, prefix="dummy")

    assert len(paths) == 4
    assert all(path.name.startswith("dummy_") for path in paths)


def _write_report(data_dir, generation, results):
    generation_dir = data_dir / f"gen{generation}"
    generation_dir.mkdir()
    report_path = generation_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(results), encoding="utf-8")
    return report_path


def test_generation_reports_are_numeric_and_plot_champion_win_rate(tmp_path):
    gen10 = _write_report(
        tmp_path,
        10,
        [
            {
                "p1": "Model Gen 10",
                "p2": "Model Gen 8",
                "winner": "Model Gen 10",
            }
        ],
    )
    gen2 = _write_report(
        tmp_path,
        2,
        [
            {
                "p1": "Model Gen 1",
                "p2": "Model Gen 2",
                "winner": "p2",
            },
            {
                "p1": "Model Gen 2",
                "p2": "Model Gen 1",
                "winner": None,
            },
        ],
    )

    assert discover_generation_reports(tmp_path) == [gen2, gen10]
    summary = summarize_generation(gen2)
    assert summary["opponent"] == "Model Gen 1"
    assert summary["matches"] == 2
    assert summary["win_rate"] == pytest.approx(0.5)

    paths = plot_generations(tmp_path, tmp_path / "plots")
    assert len(paths) == 10
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert {path.name for path in paths} >= {
        "champion_win_rate_evolution.png",
        "losses_by_generation.png",
        "gen2_agent_win_rates.png",
        "gen10_matchup_heatmap.png",
        "consolidated_agent_win_rates.png",
    }
    assert plt.get_fignums() == []


def _write_training_log(path):
    path.write_text(
        "\n".join(
            [
                "epoch,loss,policy_loss,value_loss,aux_field_loss,"
                "policy_accuracy,val_loss,val_policy_loss,val_value_loss,"
                "val_aux_field_loss,val_policy_accuracy",
                "0,4.0,0.50,0.80,0.3,0.40,4.5,0.55,0.90,0.4,0.35",
                "1,3.5,0.40,0.70,0.2,0.50,4.0,0.50,0.80,0.3,0.45",
                "0,3.0,0.30,0.60,0.1,0.60,3.5,0.40,0.70,0.2,0.55",
            ]
        ),
        encoding="utf-8",
    )


def test_training_plots_support_dynamic_heads_and_resumed_epochs(tmp_path):
    csv_path = tmp_path / "training_log.csv"
    _write_training_log(csv_path)

    rows = load_training_log(csv_path)
    assert [row["global_epoch"] for row in rows] == [1, 2, 3]

    paths = plot_training(csv_path, tmp_path / "training_plots")
    assert len(paths) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert plt.get_fignums() == []


def test_generation_training_summary_uses_latest_csv_session(tmp_path):
    generation_dir = tmp_path / "gen3"
    generation_dir.mkdir()
    _write_training_log(generation_dir / "training_log.csv")
    (generation_dir / "generation_metrics.json").write_text(
        '{"autoencoder_error": 0.012}',
        encoding="utf-8",
    )

    summaries = summarize_training_metrics(tmp_path)
    assert summaries == [
        {
            "generation": 3,
            "policy_loss": pytest.approx(0.40),
            "value_loss": pytest.approx(0.70),
            "autoencoder_error": pytest.approx(0.012),
        }
    ]


def test_generation_summary_uses_all_battles_for_battle_metrics(tmp_path):
    generation_dir = tmp_path / "gen4"
    generation_dir.mkdir()
    report = {
        "ratings": {"Model Gen 4": 1612},
        "results": [
            {
                "p1": "Model Gen 4",
                "p2": "Random Agent",
                "winner": "Model Gen 4",
                "turns": 10,
                "survivors": 2,
            },
            {
                "p1": "Random Agent",
                "p2": "Blind MCTS",
                "winner": "Blind MCTS",
                "turns": 30,
                "survivors": 4,
            },
        ],
    }
    report_path = generation_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    summary = summarize_generation(report_path)
    assert summary["rating"] == pytest.approx(1612)
    assert summary["baseline_win_rates"] == {"Random Agent": 1.0}
    assert summary["avg_turns"] == pytest.approx(20)
    assert summary["avg_survivors"] == pytest.approx(3)


def test_validation_only_auxiliary_loss_is_discovered():
    rows = [
        {
            "global_epoch": 1.0,
            "val_aux_future_loss": 0.25,
        }
    ]
    assert _head_loss_names(rows) == ["aux_future"]


def test_nested_autoencoder_metadata_is_supported(tmp_path):
    generation_dir = tmp_path / "gen7"
    generation_dir.mkdir()
    (generation_dir / "generation_metrics.json").write_text(
        '{"metrics": {"reconstruction_error": 0.0042}}',
        encoding="utf-8",
    )

    summary = summarize_training_metrics(tmp_path)
    assert summary[0]["autoencoder_error"] == pytest.approx(0.0042)
