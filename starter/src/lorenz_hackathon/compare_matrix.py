from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIG_LABELS = {
    "train_a0_direct_onestep": "A0: one-step loss",
    "train_a1_direct_multistep": "A1: four-step loss",
    "train_b1_multirho_state_only": "B1: state only",
    "train_b2_multirho_conditioned": "B2: state + rho",
    "train_linear": "Linear baseline",
}

METRIC_SPECS = (
    ("one_step_nrmse", "One-step NRMSE", "lower"),
    ("useful_horizon", "Useful horizon [time]", "higher"),
    ("finite_fraction", "Finite fraction", "higher"),
    ("bounded_fraction", "Bounded fraction", "higher"),
    ("growth_error", "Growth-rate error [1/time]", "lower"),
    ("variance_error", "Variance-ratio error", "lower"),
    ("occupancy_error", "Lobe-occupancy error", "lower"),
    ("switch_error", "Switch-rate error [1/time]", "lower"),
    ("residence_error", "Residence-time error [time]", "lower"),
    ("mean_wasserstein", "Mean Wasserstein [state]", "lower"),
)


@dataclass(frozen=True)
class MatrixRecord:
    config_path: str
    label: str
    seed: int
    output_directory: Path
    result_path: Path
    result: dict[str, Any]


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def finite_mean(values: Any) -> float | None:
    converted = [finite_float(value) for value in values]
    finite = [value for value in converted if value is not None]
    return None if len(finite) != len(converted) or not finite else float(np.mean(finite))


def absolute_difference(left: Any, right: Any) -> float | None:
    left_value = finite_float(left)
    right_value = finite_float(right)
    if left_value is None or right_value is None:
        return None
    return abs(left_value - right_value)


def config_label(config_path: str) -> str:
    stem = Path(config_path).stem
    return CONFIG_LABELS.get(stem, stem.replace("train_", "").replace("_", " "))


def parse_matrix(matrix_path: Path) -> list[tuple[str, int, Path]]:
    rows: list[tuple[str, int, Path]] = []
    seen: set[tuple[str, int]] = set()
    with matrix_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 3:
                raise ValueError(
                    f"{matrix_path}:{line_number} must contain "
                    "CONFIG SEED OUTPUT_DIRECTORY"
                )
            config_path, seed_text, output_directory = fields
            try:
                seed = int(seed_text)
            except ValueError as exc:
                raise ValueError(
                    f"{matrix_path}:{line_number} has invalid seed {seed_text!r}"
                ) from exc
            key = (config_path, seed)
            if key in seen:
                raise ValueError(
                    f"{matrix_path}:{line_number} duplicates config={config_path} "
                    f"seed={seed}"
                )
            seen.add(key)
            rows.append((config_path, seed, Path(output_directory)))
    if not rows:
        raise ValueError(f"Matrix contains no experiments: {matrix_path}")
    return rows


def load_matrix_records(
    matrix_path: Path,
    evaluation_label: str,
) -> list[MatrixRecord]:
    records: list[MatrixRecord] = []
    for config_path, seed, output_directory in parse_matrix(matrix_path):
        result_path = (
            output_directory
            / "evaluation"
            / evaluation_label
            / "benchmark_results.json"
        )
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing evaluation result: {result_path}")
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if not isinstance(result.get("by_rho"), dict) or not result["by_rho"]:
            raise ValueError(f"Evaluation result has no by_rho mapping: {result_path}")
        records.append(
            MatrixRecord(
                config_path=config_path,
                label=config_label(config_path),
                seed=seed,
                output_directory=output_directory,
                result_path=result_path,
                result=result,
            )
        )

    reference = records[0].result
    for record in records[1:]:
        if record.result.get("data_sha256") != reference.get("data_sha256"):
            raise ValueError("Matrix results use different evaluation datasets")
        if record.result.get("evaluation_settings") != reference.get(
            "evaluation_settings"
        ):
            raise ValueError("Matrix results use different evaluation settings")
    return records


def extract_metrics(rho_result: dict[str, Any]) -> dict[str, float | None]:
    one_step = rho_result["one_step"]
    forecast = rho_result["forecast"]
    stability = rho_result["stability"]
    perturbation = rho_result["perturbation"]
    climate = rho_result["climate"]

    variance_ratio = [finite_float(value) for value in stability["variance_ratio"]]
    variance_error = (
        None
        if any(value is None for value in variance_ratio)
        else float(np.mean([abs(float(value) - 1.0) for value in variance_ratio]))
    )
    mean_wasserstein = finite_mean(climate["wasserstein_by_variable"].values())

    return {
        "one_step_nrmse": finite_float(one_step["nrmse"]),
        "useful_horizon": finite_float(forecast["useful_horizon_nrmse_1"]),
        "finite_fraction": finite_float(stability["finite_trajectory_fraction"]),
        "bounded_fraction": finite_float(
            stability["within_reference_bound_fraction"]
        ),
        "growth_error": absolute_difference(
            perturbation["model_effective_growth_rate"],
            perturbation["reference_effective_growth_rate"],
        ),
        "variance_error": variance_error,
        "occupancy_error": absolute_difference(
            climate["model_positive_x_fraction"],
            climate["reference_positive_x_fraction"],
        ),
        "switch_error": absolute_difference(
            climate["model_switch_rate"], climate["reference_switch_rate"]
        ),
        "residence_error": absolute_difference(
            climate["model_mean_residence_time"],
            climate["reference_mean_residence_time"],
        ),
        "mean_wasserstein": mean_wasserstein,
    }


def summarize_values(values: list[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    return {
        "mean": None if not len(finite) else float(np.mean(finite)),
        "population_std": None if not len(finite) else float(np.std(finite, ddof=0)),
        "count": int(len(finite)),
        "missing": int(len(values) - len(finite)),
    }


def summarize_metric(
    metric_key: str, values: list[float | None]
) -> dict[str, float | int | None]:
    summary = summarize_values(values)
    if metric_key == "useful_horizon":
        summary["not_crossed_within_window"] = summary["missing"]
        summary["missing"] = 0
    return summary


def rho_sort_key(rho: str) -> tuple[int, float | str]:
    try:
        return (0, float(rho))
    except ValueError:
        return (1, rho)


def records_for_rho(
    records: list[MatrixRecord], rho: str
) -> list[tuple[MatrixRecord, dict[str, Any]]]:
    selected: list[tuple[MatrixRecord, dict[str, Any]]] = []
    for record in records:
        rho_result = record.result["by_rho"].get(rho)
        if rho_result is not None:
            selected.append((record, rho_result))
    return selected


def build_summary(
    matrix_path: Path,
    evaluation_label: str,
    records: list[MatrixRecord],
) -> dict[str, Any]:
    config_order = list(dict.fromkeys(record.config_path for record in records))
    config_labels = {
        record.config_path: record.label
        for record in records
    }
    rho_values = sorted(
        {rho for record in records for rho in record.result["by_rho"]},
        key=rho_sort_key,
    )
    by_rho: dict[str, Any] = {}

    for rho in rho_values:
        selected = records_for_rho(records, rho)
        per_seed = []
        metrics_by_config: dict[str, dict[int, dict[str, float | None]]] = {
            config: {} for config in config_order
        }
        for record, rho_result in selected:
            metrics = extract_metrics(rho_result)
            metrics_by_config[record.config_path][record.seed] = metrics
            per_seed.append(
                {
                    "config": record.config_path,
                    "label": record.label,
                    "seed": record.seed,
                    "result": str(record.result_path),
                    "metrics": metrics,
                }
            )

        config_summary: dict[str, Any] = {}
        for config in config_order:
            seed_metrics = metrics_by_config[config]
            config_summary[config] = {
                "label": config_labels[config],
                "seeds": sorted(seed_metrics),
                "metrics": {
                    key: summarize_metric(
                        key,
                        [seed_metrics[seed].get(key) for seed in sorted(seed_metrics)]
                    )
                    for key, _, _ in METRIC_SPECS
                },
            }

        paired_changes = None
        if len(config_order) == 2:
            first, second = config_order
            common_seeds = sorted(
                set(metrics_by_config[first]) & set(metrics_by_config[second])
            )
            paired_changes = {
                "comparison": f"{config_labels[second]} minus {config_labels[first]}",
                "metrics": {},
            }
            for key, _, direction in METRIC_SPECS:
                changes = []
                for seed in common_seeds:
                    first_value = metrics_by_config[first][seed].get(key)
                    second_value = metrics_by_config[second][seed].get(key)
                    change = (
                        None
                        if first_value is None or second_value is None
                        else float(second_value - first_value)
                    )
                    changes.append({"seed": seed, "change": change})
                paired_changes["metrics"][key] = {
                    "direction": direction,
                    "per_seed": changes,
                    "summary": summarize_values(
                        [item["change"] for item in changes]
                    ),
                }

        by_rho[rho] = {
            "per_seed": per_seed,
            "config_summary": config_summary,
            "paired_changes": paired_changes,
        }

    return {
        "schema_version": 1,
        "matrix": str(matrix_path),
        "evaluation_label": evaluation_label,
        "data": records[0].result.get("data"),
        "data_sha256": records[0].result.get("data_sha256"),
        "evaluation_settings": records[0].result.get("evaluation_settings"),
        "metric_directions": {
            key: direction for key, _, direction in METRIC_SPECS
        },
        "configs": [
            {
                "config": config,
                "label": config_labels[config],
                "seeds": sorted(
                    record.seed for record in records if record.config_path == config
                ),
            }
            for config in config_order
        ],
        "by_rho": by_rho,
    }


def padded_curves(curves: list[list[Any]]) -> np.ndarray:
    max_length = max(len(curve) for curve in curves)
    values = np.full((len(curves), max_length), np.nan, dtype=float)
    for row, curve in enumerate(curves):
        converted = [np.nan if value is None else float(value) for value in curve]
        values[row, : len(converted)] = converted
    return values


def evaluation_delta_t(
    record: MatrixRecord, rho_result: dict[str, Any]
) -> float:
    direct = finite_float(rho_result.get("delta_t"))
    if direct is not None:
        return direct

    settings = record.result.get("evaluation_settings", {})
    from_settings = finite_float(settings.get("delta_t"))
    if from_settings is not None:
        return from_settings

    data_value = record.result.get("data")
    data_path = Path(data_value) if isinstance(data_value, str) else None
    if data_path is not None and data_path.is_file():
        with np.load(data_path, allow_pickle=False) as data:
            if "metadata_json" in data:
                metadata = json.loads(str(data["metadata_json"].item()))
                from_metadata = finite_float(
                    metadata.get("reference", {}).get("delta_t")
                )
                if from_metadata is not None:
                    return from_metadata

    raise ValueError(
        f"Cannot determine delta_t for {record.result_path}; expected it in "
        "the rho result, evaluation settings, or dataset metadata"
    )


def finite_mean_and_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked = np.ma.masked_invalid(values)
    return (
        np.ma.mean(masked, axis=0).filled(np.nan),
        np.ma.std(masked, axis=0, ddof=0).filled(np.nan),
    )


def plot_labels(labels: list[str]) -> list[str]:
    return [label.replace(": ", "\n", 1) for label in labels]


def plot_forecast_comparison(
    records: list[MatrixRecord],
    rho: str,
    output_path: Path,
) -> None:
    selected = records_for_rho(records, rho)
    config_order = list(dict.fromkeys(record.config_path for record, _ in selected))
    colors = plt.get_cmap("tab10").colors
    figure, axis = plt.subplots(figsize=(9.5, 6.0))

    for config_index, config in enumerate(config_order):
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
        color = colors[config_index % len(colors)]
        for record, result in config_rows:
            curve = np.asarray(
                [np.nan if value is None else float(value) for value in result["forecast"]["nrmse_by_lead"]]
            )
            axis.plot(
                np.arange(len(curve)) * delta_t,
                curve,
                color=color,
                alpha=0.18,
                linewidth=1.0,
            )
        axis.plot(
            times,
            mean,
            color=color,
            linewidth=2.4,
            label=f"{config_rows[0][0].label} (mean)",
        )
        axis.fill_between(
            times,
            np.maximum(0.0, mean - std),
            mean + std,
            color=color,
            alpha=0.16,
        )

    persistence_curves = padded_curves(
        [result["forecast"]["persistence_nrmse_by_lead"] for _, result in selected]
    )
    persistence_mean, _ = finite_mean_and_std(persistence_curves)
    delta_t = evaluation_delta_t(selected[0][0], selected[0][1])
    persistence_times = np.arange(len(persistence_mean)) * delta_t
    axis.plot(
        persistence_times,
        persistence_mean,
        color="black",
        linestyle=":",
        linewidth=1.8,
        label="Persistence",
    )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    axis.set(
        xlabel="Lead time",
        ylabel="Normalized RMSE",
        title=f"Matrix forecast comparison at rho={rho}",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.text(
        0.5,
        0.01,
        "Thin lines: individual seeds; solid lines and shading: mean +/- population SD",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def forecast_window(record: MatrixRecord, rho_result: dict[str, Any]) -> float:
    curve = rho_result["forecast"]["nrmse_by_lead"]
    return (len(curve) - 1) * evaluation_delta_t(record, rho_result)


def plot_metric_comparison(
    records: list[MatrixRecord],
    rho: str,
    output_path: Path,
) -> None:
    selected = records_for_rho(records, rho)
    config_order = list(dict.fromkeys(record.config_path for record, _ in selected))
    labels_by_config = {
        record.config_path: record.label for record, _ in selected
    }
    metrics: dict[tuple[str, int], dict[str, float | None]] = {}
    rho_results: dict[tuple[str, int], dict[str, Any]] = {}
    matrix_records: dict[tuple[str, int], MatrixRecord] = {}
    for record, rho_result in selected:
        key = (record.config_path, record.seed)
        metrics[key] = extract_metrics(rho_result)
        rho_results[key] = rho_result
        matrix_records[key] = record

    x_positions = np.arange(len(config_order), dtype=float)
    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(2, 5, figsize=(17.5, 7.2), squeeze=False)
    seeds = sorted({record.seed for record, _ in selected})

    for axis, (metric_key, title, direction) in zip(axes.ravel(), METRIC_SPECS):
        for seed in seeds:
            paired_values: list[float] = []
            paired_x: list[float] = []
            for config_index, config in enumerate(config_order):
                value = metrics.get((config, seed), {}).get(metric_key)
                if value is not None:
                    paired_x.append(x_positions[config_index])
                    paired_values.append(value)
            if len(paired_values) > 1:
                axis.plot(
                    paired_x,
                    paired_values,
                    color="0.72",
                    linewidth=0.9,
                    zorder=1,
                )

        for config_index, config in enumerate(config_order):
            color = colors[config_index % len(colors)]
            actual_values: list[float] = []
            censored_values: list[float] = []
            missing = 0
            for seed in seeds:
                key = (config, seed)
                if key not in metrics:
                    continue
                value = metrics[key].get(metric_key)
                if value is not None:
                    actual_values.append(value)
                    axis.scatter(
                        x_positions[config_index],
                        value,
                        color=color,
                        s=32,
                        zorder=3,
                    )
                elif metric_key == "useful_horizon":
                    censored = forecast_window(matrix_records[key], rho_results[key])
                    censored_values.append(censored)
                    axis.scatter(
                        x_positions[config_index],
                        censored,
                        facecolors="none",
                        edgecolors=color,
                        marker="^",
                        s=48,
                        linewidth=1.4,
                        zorder=3,
                    )
                else:
                    missing += 1

            if actual_values:
                mean = float(np.mean(actual_values))
                std = float(np.std(actual_values, ddof=0))
                axis.errorbar(
                    x_positions[config_index],
                    mean,
                    yerr=std,
                    fmt="D",
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    capsize=3,
                    markersize=6,
                    zorder=4,
                )
            elif censored_values:
                axis.text(
                    x_positions[config_index],
                    max(censored_values),
                    " all >= window",
                    color=color,
                    fontsize=7,
                    rotation=90,
                    va="bottom",
                    ha="center",
                )
            if missing:
                axis.text(
                    x_positions[config_index],
                    0.03,
                    f"NA x{missing}",
                    transform=axis.get_xaxis_transform(),
                    color=color,
                    fontsize=7,
                    ha="center",
                )

        arrow = "higher is better" if direction == "higher" else "lower is better"
        axis.set_title(f"{title}\n{arrow}", fontsize=9)
        axis.set_xticks(x_positions, plot_labels([labels_by_config[c] for c in config_order]))
        axis.tick_params(axis="x", labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(axis="y", alpha=0.25)
        if metric_key in {"finite_fraction", "bounded_fraction"}:
            axis.set_ylim(-0.05, 1.05)

    figure.suptitle(f"Matched-seed matrix diagnostics at rho={rho}", fontsize=14)
    figure.text(
        0.5,
        0.012,
        "Circles: seeds; diamonds: mean; bars: +/- population SD; lines: matched seeds; "
        "open triangles: useful-horizon threshold not crossed within the window",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def create_matrix_comparison(
    matrix_path: Path,
    evaluation_label: str,
    output_dir: Path,
) -> dict[str, Any]:
    records = load_matrix_records(matrix_path, evaluation_label)
    summary = build_summary(matrix_path, evaluation_label, records)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, dict[str, str]] = {}
    for rho in summary["by_rho"]:
        forecast_name = f"forecast_comparison_rho_{rho}.png"
        metrics_name = f"metric_comparison_rho_{rho}.png"
        plot_forecast_comparison(records, rho, output_dir / forecast_name)
        plot_metric_comparison(records, rho, output_dir / metrics_name)
        artifacts[rho] = {
            "forecast_figure": forecast_name,
            "metric_figure": metrics_name,
        }
        print(f"Wrote {output_dir / forecast_name}")
        print(f"Wrote {output_dir / metrics_name}")

    summary["artifacts"] = artifacts
    summary_path = output_dir / "matrix_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(f"Wrote {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate a completed evaluation matrix and plot matched-seed comparisons."
    )
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--evaluation-label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    create_matrix_comparison(
        matrix_path=args.matrix,
        evaluation_label=args.evaluation_label,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
