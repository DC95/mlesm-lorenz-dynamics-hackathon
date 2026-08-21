from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .compare_matrix import (
    METRIC_SPECS,
    MatrixRecord,
    evaluation_delta_t,
    finite_mean_and_std,
    forecast_window,
    load_matrix_records,
    padded_curves,
    records_for_rho,
)


DEFAULT_IN_DISTRIBUTION_LABEL = "in_distribution_rho28"
DEFAULT_CHANGED_LABEL = "multirho_unseen_rho24_30"
REGIME_ORDER = (
    ("28", "In-distribution control"),
    ("30", "Unseen interpolation"),
    ("24", "Out-of-range extrapolation"),
)
DISPLAY_METRICS = (
    ("one_step_nrmse", "One-step NRMSE"),
    ("useful_horizon", "Useful horizon"),
    ("finite_fraction", "Finite fraction"),
    ("bounded_fraction", "Bounded fraction"),
    ("growth_error", "Growth-rate error"),
    ("variance_error", "Variance-ratio error"),
    ("mean_wasserstein", "Mean Wasserstein"),
    ("switch_error", "Switch-rate error"),
)
METRIC_DIRECTIONS = {key: direction for key, _, direction in METRIC_SPECS}


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing matrix summary: {path}")
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary.get("by_rho"), dict):
        raise ValueError(f"Matrix summary has no by_rho mapping: {path}")
    return summary


def record_key(record: MatrixRecord) -> tuple[str, int]:
    return record.config_path, record.seed


def validate_inputs(
    records_by_label: dict[str, list[MatrixRecord]],
    summaries_by_label: dict[str, dict[str, Any]],
    in_distribution_label: str,
    changed_label: str,
) -> None:
    labels = list(records_by_label)
    reference_records = records_by_label[labels[0]]
    reference_keys = [record_key(record) for record in reference_records]
    if len({record.config_path for record in reference_records}) != 2:
        raise ValueError("Team B regime summary requires exactly two configurations")

    reference_hashes = {
        record_key(record): record.result.get("checkpoint_sha256")
        for record in reference_records
    }
    reference_settings = summaries_by_label[labels[0]].get("evaluation_settings")
    for label in labels[1:]:
        records = records_by_label[label]
        if [record_key(record) for record in records] != reference_keys:
            raise ValueError("Team B evaluations do not use the same configs and seeds")
        hashes = {
            record_key(record): record.result.get("checkpoint_sha256")
            for record in records
        }
        if hashes != reference_hashes:
            raise ValueError("Team B evaluations do not use the same checkpoints")
        if summaries_by_label[label].get("evaluation_settings") != reference_settings:
            raise ValueError("Team B evaluations use different evaluation settings")

    delta_times: list[float] = []
    for rho, _ in REGIME_ORDER:
        label = in_distribution_label if rho == "28" else changed_label
        selected = records_for_rho(records_by_label[label], rho)
        if not selected:
            raise ValueError(f"Team B evaluation {label!r} has no rho={rho}")
        delta_times.append(evaluation_delta_t(selected[0][0], selected[0][1]))
    if not np.allclose(delta_times, delta_times[0], rtol=0.0, atol=1e-12):
        raise ValueError("Team B regimes use different exposed time intervals")


def plot_forecast_panel(
    axis: plt.Axes,
    records: list[MatrixRecord],
    rho: str,
    subtitle: str,
    colors_by_config: dict[str, Any],
) -> None:
    selected = records_for_rho(records, rho)
    config_order = list(dict.fromkeys(record.config_path for record, _ in selected))

    for config in config_order:
        config_rows = [
            (record, result)
            for record, result in selected
            if record.config_path == config
        ]
        curves = padded_curves(
            [result["forecast"]["nrmse_by_lead"] for _, result in config_rows]
        )
        mean, std = finite_mean_and_std(curves)
        delta_t = evaluation_delta_t(config_rows[0][0], config_rows[0][1])
        times = np.arange(curves.shape[1]) * delta_t
        color = colors_by_config[config]
        axis.plot(
            times,
            mean,
            color=color,
            linewidth=2.5,
            label=config_rows[0][0].label,
        )
        axis.fill_between(
            times,
            np.maximum(0.0, mean - std),
            mean + std,
            color=color,
            alpha=0.18,
        )

    persistence_curves = padded_curves(
        [result["forecast"]["persistence_nrmse_by_lead"] for _, result in selected]
    )
    persistence_mean, _ = finite_mean_and_std(persistence_curves)
    delta_t = evaluation_delta_t(selected[0][0], selected[0][1])
    times = np.arange(len(persistence_mean)) * delta_t
    axis.plot(
        times,
        persistence_mean,
        color="black",
        linestyle=":",
        linewidth=1.7,
        label="Persistence",
    )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axis.set_title(f"rho={rho}\n{subtitle}", fontsize=12)
    axis.set_xlabel("Lead time")
    axis.grid(alpha=0.24)
    axis.set_ylim(bottom=0.0)


def format_metric_summary(
    metric_key: str,
    summary: dict[str, Any],
    window: float,
) -> str:
    mean = summary.get("mean")
    population_std = summary.get("population_std")
    censored = int(summary.get("not_crossed_within_window", 0) or 0)
    missing = int(summary.get("missing", 0) or 0)

    if mean is None:
        if metric_key == "useful_horizon" and censored:
            return f">={window:g} ({censored})"
        return "NA"

    text = f"{float(mean):.3g}"
    if population_std is not None:
        text += f" +/- {float(population_std):.2g}"
    if metric_key == "useful_horizon" and censored:
        text += f"; {censored} >= {window:g}"
    if missing:
        text += f"; {missing} NA"
    return text


def improvement_count(metric_key: str, rho_summary: dict[str, Any]) -> str:
    paired = rho_summary.get("paired_changes")
    if not isinstance(paired, dict):
        return "NA"
    metric = paired.get("metrics", {}).get(metric_key, {})
    finite_changes = [
        float(item["change"])
        for item in metric.get("per_seed", [])
        if item.get("change") is not None and np.isfinite(float(item["change"]))
    ]
    if not finite_changes:
        return "NA"
    direction = METRIC_DIRECTIONS[metric_key]
    if direction == "lower":
        improved = sum(change < 0.0 for change in finite_changes)
    else:
        improved = sum(change > 0.0 for change in finite_changes)
    return f"{improved}/{len(finite_changes)}"


def plot_metric_table(
    axis: plt.Axes,
    rho_summary: dict[str, Any],
    config_order: list[str],
    window: float,
) -> None:
    axis.axis("off")
    config_summary = rho_summary["config_summary"]
    config_labels = [config_summary[config]["label"] for config in config_order]
    rows = []
    for metric_key, metric_label in DISPLAY_METRICS:
        direction = "down" if METRIC_DIRECTIONS[metric_key] == "lower" else "up"
        values = [
            format_metric_summary(
                metric_key,
                config_summary[config]["metrics"][metric_key],
                window,
            )
            for config in config_order
        ]
        rows.append(
            [
                f"{metric_label} ({direction})",
                *values,
                improvement_count(metric_key, rho_summary),
            ]
        )

    table = axis.table(
        cellText=rows,
        colLabels=["Metric", config_labels[0], config_labels[1], "B2 better\nseeds"],
        colWidths=[0.34, 0.25, 0.25, 0.16],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.45)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("0.82")
        if row == 0:
            cell.set_facecolor("0.93")
            cell.set_text_props(weight="bold")
        elif column == 0:
            cell.set_text_props(ha="left")


def create_team_b_regime_summary(
    matrix_path: Path,
    comparison_root: Path,
    output_dir: Path,
    in_distribution_label: str = DEFAULT_IN_DISTRIBUTION_LABEL,
    changed_label: str = DEFAULT_CHANGED_LABEL,
) -> dict[str, Any]:
    labels = (in_distribution_label, changed_label)
    summaries_by_label = {
        label: load_summary(comparison_root / label / "matrix_summary.json")
        for label in labels
    }
    records_by_label = {
        label: load_matrix_records(matrix_path, label) for label in labels
    }

    validate_inputs(
        records_by_label,
        summaries_by_label,
        in_distribution_label,
        changed_label,
    )

    reference_records = records_by_label[in_distribution_label]
    config_order = list(dict.fromkeys(record.config_path for record in reference_records))
    colors = plt.get_cmap("tab10").colors
    colors_by_config = {
        config: colors[index % len(colors)]
        for index, config in enumerate(config_order)
    }

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18.0, 9.0),
        sharex="row",
        sharey="row",
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )

    combined_by_rho: dict[str, Any] = {}
    for column, (rho, subtitle) in enumerate(REGIME_ORDER):
        label = in_distribution_label if rho == "28" else changed_label
        records = records_by_label[label]
        summary = summaries_by_label[label]
        rho_summary = summary["by_rho"].get(rho)
        if rho_summary is None:
            raise ValueError(f"Matrix summary {label!r} has no rho={rho}")
        selected = records_for_rho(records, rho)
        window = forecast_window(selected[0][0], selected[0][1])
        plot_forecast_panel(
            axes[0, column], records, rho, subtitle, colors_by_config
        )
        plot_metric_table(axes[1, column], rho_summary, config_order, window)
        combined_by_rho[rho] = {
            "scientific_role": subtitle,
            "evaluation_label": label,
            "source_data": summary.get("data"),
            "source_data_sha256": summary.get("data_sha256"),
            "summary": rho_summary,
        }

    axes[0, 0].set_ylabel("Normalized RMSE")
    handles, labels_for_legend = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels_for_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Team B: does explicit rho conditioning help across regimes?",
        fontsize=16,
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Curves and tables show mean +/- population SD across matched seeds. "
        "'B2 better seeds' counts directional improvements among finite pairs; "
        "it is not an overall score or significance test.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.91))

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_name = "team_b_three_regime_summary.png"
    figure_path = output_dir / figure_name
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    combined_summary = {
        "schema_version": 1,
        "matrix": str(matrix_path),
        "rho_order": [rho for rho, _ in REGIME_ORDER],
        "configs": summaries_by_label[in_distribution_label].get("configs"),
        "evaluation_settings": summaries_by_label[in_distribution_label].get(
            "evaluation_settings"
        ),
        "sources": {
            label: str(comparison_root / label / "matrix_summary.json")
            for label in labels
        },
        "by_rho": combined_by_rho,
        "artifacts": {"three_regime_figure": figure_name},
    }
    summary_path = output_dir / "team_b_three_regime_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(combined_summary, handle, indent=2, allow_nan=False)

    print(f"Wrote {figure_path}")
    print(f"Wrote {summary_path}")
    return combined_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine Team B in-distribution, interpolation, and extrapolation "
            "evidence in one presentation-ready figure."
        )
    )
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--in-distribution-label", default=DEFAULT_IN_DISTRIBUTION_LABEL
    )
    parser.add_argument("--changed-label", default=DEFAULT_CHANGED_LABEL)
    args = parser.parse_args()
    create_team_b_regime_summary(
        matrix_path=args.matrix,
        comparison_root=args.comparison_root,
        output_dir=args.output_dir,
        in_distribution_label=args.in_distribution_label,
        changed_label=args.changed_label,
    )


if __name__ == "__main__":
    main()
