from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .baselines import persistence_rollout
from .dynamics import LorenzParameters, integrate_lorenz
from .models import FlowModel, build_model
from .train import select_device


def model_step(
    model: FlowModel,
    states: np.ndarray,
    parameters: np.ndarray,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    state_mean = checkpoint["state_mean"]
    state_std = checkpoint["state_std"]
    parameter_mean = checkpoint["parameter_mean"]
    parameter_std = checkpoint["parameter_std"]
    normalized_states = (states - state_mean) / state_std
    normalized_parameters = (parameters - parameter_mean) / parameter_std
    state_tensor = torch.as_tensor(normalized_states, dtype=torch.float32, device=device)
    parameter_tensor = torch.as_tensor(
        normalized_parameters, dtype=torch.float32, device=device
    )
    with torch.no_grad():
        predicted = model(
            state_tensor, parameter_tensor if model.conditioned else None
        ).cpu().numpy()
    return predicted * state_std + state_mean


def rollout_model(
    model: FlowModel,
    initial_states: np.ndarray,
    parameters: np.ndarray,
    num_steps: int,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    trajectory = np.empty((len(initial_states), num_steps + 1, 3), dtype=np.float64)
    trajectory[:, 0] = initial_states
    current = initial_states.astype(np.float64, copy=True)
    for step in range(num_steps):
        current = model_step(model, current, parameters, checkpoint, device)
        trajectory[:, step + 1] = current
    return trajectory


def quantile_wasserstein(reference: np.ndarray, prediction: np.ndarray) -> float:
    quantiles = np.linspace(0.001, 0.999, 999)
    return float(
        np.mean(
            np.abs(
                np.quantile(reference, quantiles)
                - np.quantile(prediction, quantiles)
            )
        )
    )


def switch_rate(trajectories: np.ndarray, delta_t: float) -> float:
    signs = trajectories[..., 0] >= 0.0
    switches = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1)
    duration = (trajectories.shape[1] - 1) * delta_t
    return float(np.mean(switches / duration))


def mean_residence_time(trajectories: np.ndarray, delta_t: float) -> float:
    durations: list[float] = []
    for trajectory in trajectories:
        signs = trajectory[:, 0] >= 0.0
        boundaries = np.flatnonzero(signs[1:] != signs[:-1]) + 1
        run_lengths = np.diff(np.concatenate(([0], boundaries, [len(signs)])))
        durations.extend((run_lengths * delta_t).tolist())
    return float(np.mean(durations))


def first_threshold_time(values: np.ndarray, threshold: float, delta_t: float) -> float | None:
    indices = np.flatnonzero(values >= threshold)
    return None if len(indices) == 0 else float(indices[0] * delta_t)


def effective_growth_rate(
    times: np.ndarray,
    median_distance: np.ndarray,
    initial_distance: float,
    state_std: np.ndarray,
) -> float | None:
    lower = max(initial_distance * 5.0, 1e-12)
    upper = 0.1 * float(np.linalg.norm(state_std))
    mask = (median_distance >= lower) & (median_distance <= upper)
    if np.count_nonzero(mask) < 3:
        return None
    slope, _ = np.polyfit(times[mask], np.log(median_distance[mask]), 1)
    return float(slope)


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def evaluate_rho_group(
    rho: float,
    states: np.ndarray,
    parameters: np.ndarray,
    model: FlowModel,
    checkpoint: dict[str, Any],
    device: torch.device,
    reference_metadata: dict[str, Any],
    output_dir: Path,
    long_steps: int,
    perturbation_steps: int,
) -> dict[str, Any]:
    reference = reference_metadata["reference"]
    delta_t = float(reference["delta_t"])
    dt_reference = float(reference["dt_reference"])
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float64)

    forecast_steps = min(200, states.shape[1] - 1)
    forecast_truth = states[:, : forecast_steps + 1].astype(np.float64)
    forecast_model = rollout_model(
        model,
        forecast_truth[:, 0],
        parameters,
        forecast_steps,
        checkpoint,
        device,
    )
    normalized_error = (forecast_model - forecast_truth) / state_std
    nrmse_by_lead = np.sqrt(np.nanmean(normalized_error**2, axis=(0, 2)))
    forecast_persistence = persistence_rollout(
        forecast_truth[:, 0], forecast_steps
    )
    persistence_normalized_error = (
        forecast_persistence - forecast_truth
    ) / state_std
    persistence_nrmse_by_lead = np.sqrt(
        np.nanmean(persistence_normalized_error**2, axis=(0, 2))
    )

    one_step_inputs = states[:, :-1].reshape(-1, 3).astype(np.float64)
    one_step_targets = states[:, 1:].reshape(-1, 3).astype(np.float64)
    repeated_parameters = np.repeat(parameters, states.shape[1] - 1, axis=0)
    one_step_prediction = model_step(
        model, one_step_inputs, repeated_parameters, checkpoint, device
    )
    persistence_error = one_step_inputs - one_step_targets
    model_error = one_step_prediction - one_step_targets

    long_count = min(32, len(states))
    long_initial = states[:long_count, 0].astype(np.float64)
    long_parameters = parameters[:long_count].astype(np.float64)
    lorenz_parameters = LorenzParameters(
        sigma=float(long_parameters[0, 0]),
        rho=float(rho),
        beta=float(long_parameters[0, 2]),
    )
    long_reference = integrate_lorenz(
        long_initial,
        num_steps=long_steps,
        delta_t=delta_t,
        dt_reference=dt_reference,
        parameters=lorenz_parameters,
    )
    long_model = rollout_model(
        model,
        long_initial,
        long_parameters,
        long_steps,
        checkpoint,
        device,
    )
    evaluation_burn = min(400, long_steps // 5)
    reference_climate = long_reference[:, evaluation_burn:]
    model_climate = long_model[:, evaluation_burn:]

    reference_radius = np.linalg.norm(reference_climate, axis=-1)
    broad_bound = 5.0 * float(np.quantile(reference_radius, 0.999))
    finite_trajectories = np.all(np.isfinite(long_model), axis=(1, 2))
    within_bound = np.all(np.linalg.norm(long_model, axis=-1) <= broad_bound, axis=1)

    perturb_count = min(64, len(states))
    perturb_initial = states[:perturb_count, 0].astype(np.float64)
    rng = np.random.default_rng(7717 + int(round(rho * 10)))
    directions = rng.normal(size=perturb_initial.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    epsilon = 1e-5 * float(np.linalg.norm(state_std))
    perturbed_initial = perturb_initial + epsilon * directions
    perturb_parameters = parameters[:perturb_count].astype(np.float64)

    reference_a = integrate_lorenz(
        perturb_initial,
        perturbation_steps,
        delta_t,
        dt_reference,
        lorenz_parameters,
    )
    reference_b = integrate_lorenz(
        perturbed_initial,
        perturbation_steps,
        delta_t,
        dt_reference,
        lorenz_parameters,
    )
    model_a = rollout_model(
        model,
        perturb_initial,
        perturb_parameters,
        perturbation_steps,
        checkpoint,
        device,
    )
    model_b = rollout_model(
        model,
        perturbed_initial,
        perturb_parameters,
        perturbation_steps,
        checkpoint,
        device,
    )
    reference_distance = np.linalg.norm(reference_a - reference_b, axis=-1)
    model_distance = np.linalg.norm(model_a - model_b, axis=-1)
    median_reference_distance = np.median(reference_distance, axis=0)
    median_model_distance = np.median(model_distance, axis=0)
    perturbation_times = np.arange(perturbation_steps + 1) * delta_t

    variable_names = ("x", "y", "z")
    wasserstein = {
        name: quantile_wasserstein(
            reference_climate[..., index].ravel(),
            model_climate[..., index].ravel(),
        )
        for index, name in enumerate(variable_names)
    }
    reference_variance = np.var(reference_climate, axis=(0, 1))
    model_variance = np.var(model_climate, axis=(0, 1))

    result = {
        "rho": float(rho),
        "one_step": {
            "rmse_by_variable": np.sqrt(np.mean(model_error**2, axis=0)).tolist(),
            "nrmse": float(np.sqrt(np.mean((model_error / state_std) ** 2))),
            "persistence_nrmse": float(
                np.sqrt(np.mean((persistence_error / state_std) ** 2))
            ),
        },
        "forecast": {
            "useful_horizon_nrmse_1": first_threshold_time(
                nrmse_by_lead, 1.0, delta_t
            ),
            "persistence_useful_horizon_nrmse_1": first_threshold_time(
                persistence_nrmse_by_lead, 1.0, delta_t
            ),
            "nrmse_by_lead": [finite_or_none(value) for value in nrmse_by_lead],
            "persistence_nrmse_by_lead": [
                finite_or_none(value) for value in persistence_nrmse_by_lead
            ],
        },
        "stability": {
            "finite_trajectory_fraction": float(np.mean(finite_trajectories)),
            "within_reference_bound_fraction": float(np.mean(within_bound)),
            "reference_bound": broad_bound,
            "variance_ratio": [
                finite_or_none(value)
                for value in model_variance / np.maximum(reference_variance, 1e-12)
            ],
        },
        "perturbation": {
            "initial_distance": epsilon,
            "reference_effective_growth_rate": effective_growth_rate(
                perturbation_times,
                median_reference_distance,
                epsilon,
                state_std,
            ),
            "model_effective_growth_rate": effective_growth_rate(
                perturbation_times,
                median_model_distance,
                epsilon,
                state_std,
            ),
        },
        "climate": {
            "reference_mean": np.mean(reference_climate, axis=(0, 1)).tolist(),
            "model_mean": np.mean(model_climate, axis=(0, 1)).tolist(),
            "reference_std": np.std(reference_climate, axis=(0, 1)).tolist(),
            "model_std": np.std(model_climate, axis=(0, 1)).tolist(),
            "wasserstein_by_variable": wasserstein,
            "reference_positive_x_fraction": float(
                np.mean(reference_climate[..., 0] >= 0.0)
            ),
            "model_positive_x_fraction": float(np.mean(model_climate[..., 0] >= 0.0)),
            "reference_switch_rate": switch_rate(reference_climate, delta_t),
            "model_switch_rate": switch_rate(model_climate, delta_t),
            "reference_mean_residence_time": mean_residence_time(
                reference_climate, delta_t
            ),
            "model_mean_residence_time": mean_residence_time(model_climate, delta_t),
        },
    }

    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    lead_times = np.arange(forecast_steps + 1) * delta_t
    axes[0, 0].plot(lead_times, nrmse_by_lead, label="Learned emulator")
    axes[0, 0].plot(
        lead_times,
        persistence_nrmse_by_lead,
        label="Persistence",
        linestyle=":",
    )
    axes[0, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(xlabel="Lead time", ylabel="Normalized RMSE", title="Forecast skill")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        reference_climate[0, :, 0],
        reference_climate[0, :, 2],
        linewidth=0.7,
        label="RK4",
    )
    axes[0, 1].plot(
        model_climate[0, :, 0],
        model_climate[0, :, 2],
        linewidth=0.7,
        alpha=0.8,
        label="Emulator",
    )
    axes[0, 1].set(xlabel="x", ylabel="z", title="Long-term phase space")
    axes[0, 1].legend()

    axes[1, 0].semilogy(
        perturbation_times, median_reference_distance, label="RK4", linewidth=2
    )
    axes[1, 0].semilogy(
        perturbation_times, median_model_distance, label="Emulator", linewidth=2
    )
    axes[1, 0].set(
        xlabel="Time", ylabel="Median perturbation distance", title="Perturbation growth"
    )
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    common_min = min(reference_climate[..., 0].min(), model_climate[..., 0].min())
    common_max = max(reference_climate[..., 0].max(), model_climate[..., 0].max())
    bins = np.linspace(common_min, common_max, 60)
    axes[1, 1].hist(
        reference_climate[..., 0].ravel(), bins=bins, density=True, alpha=0.5, label="RK4"
    )
    axes[1, 1].hist(
        model_climate[..., 0].ravel(), bins=bins, density=True, alpha=0.5, label="Emulator"
    )
    axes[1, 1].set(xlabel="x", ylabel="Density", title="Long-term state distribution")
    axes[1, 1].legend()
    figure.suptitle(f"Lorenz emulator model autopsy: rho={rho:g}")
    figure.tight_layout()
    figure.savefig(output_dir / f"model_autopsy_rho_{rho:g}.png", dpi=180)
    plt.close(figure)
    return result


def evaluate(
    checkpoint_path: Path,
    data_path: Path,
    output_dir: Path,
    requested_device: str,
    long_steps: int,
    perturbation_steps: int,
) -> dict[str, Any]:
    device = select_device(requested_device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with np.load(data_path, allow_pickle=False) as data:
        states = data["public_test_states"]
        parameters = data["public_test_parameters"]
        metadata = json.loads(str(data["metadata_json"].item()))

    output_dir.mkdir(parents=True, exist_ok=True)
    by_rho: dict[str, Any] = {}
    for rho in np.unique(parameters[:, 1]):
        mask = np.isclose(parameters[:, 1], rho)
        by_rho[f"{float(rho):g}"] = evaluate_rho_group(
            float(rho),
            states[mask],
            parameters[mask],
            model,
            checkpoint,
            device,
            metadata,
            output_dir,
            long_steps,
            perturbation_steps,
        )

    results = {
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "by_rho": by_rho,
    }
    with (output_dir / "benchmark_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    print(f"Wrote {output_dir / 'benchmark_results.json'}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a learned Lorenz flow map.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--long-steps", type=int, default=4000)
    parser.add_argument("--perturbation-steps", type=int, default=200)
    args = parser.parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_dir=args.output_dir,
        requested_device=args.device,
        long_steps=args.long_steps,
        perturbation_steps=args.perturbation_steps,
    )


if __name__ == "__main__":
    main()
