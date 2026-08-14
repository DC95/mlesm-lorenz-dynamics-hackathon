from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dynamics import LorenzParameters, advance_flow_map, integrate_lorenz


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_burned_initial_states(
    count: int,
    seed: int,
    low: float,
    high: float,
    burn_in_time: float,
    dt_reference: float,
    parameters: LorenzParameters,
) -> np.ndarray:
    """Sample independent states and integrate away their initial transients."""

    rng = np.random.default_rng(seed)
    states = rng.uniform(low, high, size=(count, 3))
    burn_steps = int(round(burn_in_time / dt_reference))
    for _ in range(burn_steps):
        states = advance_flow_map(
            states,
            delta_t=dt_reference,
            dt_reference=dt_reference,
            parameters=parameters,
        )
    return states


def generate_split(
    split_config: dict[str, Any],
    reference_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    trajectories: list[np.ndarray] = []
    parameter_rows: list[np.ndarray] = []
    base_seed = int(split_config["seed"])
    count = int(split_config["trajectories_per_rho"])
    steps = int(split_config["steps"])
    sigma = float(reference_config["sigma"])
    beta = float(reference_config["beta"])

    for rho_index, rho_value in enumerate(split_config["rhos"]):
        parameters = LorenzParameters(sigma=sigma, rho=float(rho_value), beta=beta)
        initial_states = sample_burned_initial_states(
            count=count,
            seed=base_seed + 1009 * rho_index,
            low=float(reference_config["initial_low"]),
            high=float(reference_config["initial_high"]),
            burn_in_time=float(reference_config["burn_in_time"]),
            dt_reference=float(reference_config["dt_reference"]),
            parameters=parameters,
        )
        trajectories.append(
            integrate_lorenz(
                initial_states,
                num_steps=steps,
                delta_t=float(reference_config["delta_t"]),
                dt_reference=float(reference_config["dt_reference"]),
                parameters=parameters,
            ).astype(np.float32)
        )
        parameter_rows.append(
            np.repeat(
                np.array([[sigma, float(rho_value), beta]], dtype=np.float32),
                count,
                axis=0,
            )
        )

    return np.concatenate(trajectories, axis=0), np.concatenate(parameter_rows, axis=0)


def generate_benchmark(config: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    reference = config["reference"]
    for split_name, split_config in config["splits"].items():
        states, parameters = generate_split(split_config, reference)
        arrays[f"{split_name}_states"] = states
        arrays[f"{split_name}_parameters"] = parameters
    arrays["metadata_json"] = np.array(json.dumps(config))
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen Lorenz benchmark data.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = load_json(args.config)
    arrays = generate_benchmark(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    print(f"Wrote {args.output}")
    for name, array in arrays.items():
        if name != "metadata_json":
            print(f"  {name}: {array.shape}")


if __name__ == "__main__":
    main()
