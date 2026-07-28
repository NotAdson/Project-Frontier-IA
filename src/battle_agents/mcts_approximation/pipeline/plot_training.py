from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MetricRows = list[dict[str, float]]


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def load_training_log(csv_path: str | Path) -> MetricRows:
    """Load numeric metrics and add a monotonic ``global_epoch`` column.

    Keras starts the ``epoch`` column at zero on each invocation even when
    ``CSVLogger(append=True)`` is used.  Row position is therefore the reliable
    x-axis across resumed training sessions and generations.
    """

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training log not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(f"Training log has no header: {path}")
        rows = [
            {
                **{name: _number(raw.get(name)) for name in reader.fieldnames},
                "global_epoch": float(index + 1),
            }
            for index, raw in enumerate(reader)
            if raw
        ]

    if not rows:
        raise ValueError(f"Training log contains no metric rows: {path}")
    return rows


def _columns(rows: Sequence[Mapping[str, float]]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for name in row:
            if name not in {"epoch", "global_epoch"}:
                seen.setdefault(name, None)
    return list(seen)


def _values(rows: Sequence[Mapping[str, float]], column: str) -> np.ndarray:
    return np.asarray([row.get(column, math.nan) for row in rows], dtype=float)


def _has_values(rows: Sequence[Mapping[str, float]], column: str) -> bool:
    return column in _columns(rows) and np.isfinite(_values(rows, column)).any()


def _with_derived_primary(rows: MetricRows) -> MetricRows:
    for row in rows:
        if not math.isfinite(row.get("primary_loss", math.nan)):
            policy = row.get("policy_loss", math.nan)
            value = row.get("value_loss", math.nan)
            if math.isfinite(policy) and math.isfinite(value):
                row["primary_loss"] = 5.0 * policy + value
        if not math.isfinite(row.get("val_primary_loss", math.nan)):
            policy = row.get("val_policy_loss", math.nan)
            value = row.get("val_value_loss", math.nan)
            if math.isfinite(policy) and math.isfinite(value):
                row["val_primary_loss"] = 5.0 * policy + value
    return rows


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


def _plot_metric(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, float]],
    column: str,
    *,
    label: str | None = None,
    linestyle: str = "-",
) -> bool:
    values = _values(rows, column)
    valid = np.isfinite(values)
    if not valid.any():
        return False
    epochs = _values(rows, "global_epoch")
    ax.plot(
        epochs[valid],
        values[valid],
        label=label or column,
        linewidth=1.8,
        linestyle=linestyle,
    )
    return True


def plot_total_and_primary_losses(
    rows: MetricRows, output_path: str | Path
) -> Path:
    """Plot total loss and the weighted primary loss (5×policy + value)."""

    rows = _with_derived_primary(rows)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = False
    for column, label, style in (
        ("loss", "total — treino", "-"),
        ("val_loss", "total — validação", "--"),
        ("primary_loss", "primário — treino", "-"),
        ("val_primary_loss", "primário — validação", "--"),
    ):
        plotted |= _plot_metric(ax, rows, column, label=label, linestyle=style)

    if not plotted:
        _empty(ax, "Loss total e primário", "Nenhuma coluna de loss encontrada")
    else:
        ax.set_title("Loss total e primário (primário = 5×policy + value)")
        ax.set_xlabel("Epoch global")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()
    return _save(fig, output_path)


def _head_loss_names(rows: Sequence[Mapping[str, float]]) -> list[str]:
    names: set[str] = set()
    for column in _columns(rows):
        normalized = column[4:] if column.startswith("val_") else column
        if not normalized.endswith("_loss"):
            continue
        if normalized in {"loss", "primary_loss"}:
            continue
        head = normalized[: -len("_loss")]
        if _has_values(rows, f"{head}_loss") or _has_values(
            rows, f"val_{head}_loss"
        ):
            names.add(head)
    preferred = {"policy": 0, "value": 1}
    return sorted(names, key=lambda name: (preferred.get(name, 2), name))


def plot_head_losses(rows: MetricRows, output_path: str | Path) -> Path:
    """Plot train/validation loss for every discovered model head."""

    heads = _head_loss_names(rows)
    if not heads:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _empty(ax, "Loss por cabeça", "Nenhuma loss por cabeça encontrada")
        return _save(fig, output_path)

    columns = min(4, max(2, math.ceil(math.sqrt(len(heads)))))
    row_count = math.ceil(len(heads) / columns)
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(columns * 4.2, row_count * 3.2),
        squeeze=False,
        sharex=True,
    )
    for ax, head in zip(axes.flat, heads):
        _plot_metric(ax, rows, f"{head}_loss", label="treino")
        _plot_metric(
            ax,
            rows,
            f"val_{head}_loss",
            label="validação",
            linestyle="--",
        )
        ax.set_title(head.replace("_", " "))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(heads) :]:
        ax.set_axis_off()
    fig.supxlabel("Epoch global")
    fig.supylabel("Loss")
    fig.suptitle("Loss por cabeça", fontsize=14)
    fig.tight_layout()
    return _save(fig, output_path)


def _policy_accuracy_columns(rows: Sequence[Mapping[str, float]]) -> tuple[str | None, str | None]:
    columns = _columns(rows)

    def choose(validation: bool) -> str | None:
        candidates = []
        for name in columns:
            is_validation = name.startswith("val_")
            normalized = name[4:] if is_validation else name
            if is_validation != validation or "policy" not in normalized:
                continue
            if "accuracy" in normalized or normalized.endswith("_acc"):
                candidates.append(name)
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda name: ("categorical" in name, len(name)),
        )[0]

    return choose(False), choose(True)


def plot_policy_accuracy(rows: MetricRows, output_path: str | Path) -> Path:
    """Plot policy accuracy for training and validation when available."""

    train_column, validation_column = _policy_accuracy_columns(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    if train_column:
        plotted |= _plot_metric(ax, rows, train_column, label="treino")
    if validation_column:
        plotted |= _plot_metric(
            ax, rows, validation_column, label="validação", linestyle="--"
        )

    if not plotted:
        _empty(ax, "Policy accuracy", "Nenhuma métrica de policy accuracy encontrada")
    else:
        ax.set_title("Policy accuracy")
        ax.set_xlabel("Epoch global")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()
    return _save(fig, output_path)


def plot_overfitting_check(rows: MetricRows, output_path: str | Path) -> Path:
    """Compare train and validation curves for the principal losses."""

    rows = _with_derived_primary(rows)
    metrics = [
        metric
        for metric in ("loss", "primary_loss", "policy_loss", "value_loss")
        if _has_values(rows, metric) or _has_values(rows, f"val_{metric}")
    ]
    if not metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _empty(ax, "Treino vs validação", "Nenhum par treino/validação encontrado")
        return _save(fig, output_path)

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(10, max(4.5, 3.2 * len(metrics))),
        squeeze=False,
        sharex=True,
    )
    for ax, metric in zip(axes.flat, metrics):
        _plot_metric(ax, rows, metric, label="treino")
        _plot_metric(
            ax, rows, f"val_{metric}", label="validação", linestyle="--"
        )
        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()
    axes.flat[-1].set_xlabel("Epoch global")
    fig.suptitle("Overfitting check: treino vs validação", fontsize=14)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_training(
    csv_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate the complete training report and return the PNG paths."""

    rows = load_training_log(csv_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return [
        plot_total_and_primary_losses(rows, output / "loss_total_primary.png"),
        plot_head_losses(rows, output / "loss_by_head.png"),
        plot_policy_accuracy(rows, output / "policy_accuracy.png"),
        plot_overfitting_check(rows, output / "train_vs_validation.png"),
    ]


def parse_args() -> argparse.Namespace:
    project_data = Path(__file__).resolve().parents[3] / "data"
    parser = argparse.ArgumentParser(description="Plot Keras training_log.csv metrics.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=project_data / "training_log.csv",
        help="Path to training_log.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_data / "training_plots",
        help="Directory in which PNG files are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for plot_path in plot_training(args.csv_path, args.output_dir):
        print(f"Training plot saved to: {plot_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "load_training_log",
    "plot_head_losses",
    "plot_overfitting_check",
    "plot_policy_accuracy",
    "plot_total_and_primary_losses",
    "plot_training",
]
