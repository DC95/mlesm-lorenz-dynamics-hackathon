from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import load_json
from .models import FlowModel, build_model


class TrajectoryWindowDataset(Dataset):
    """Return fixed-length windows without mixing trajectories across splits."""

    def __init__(
        self,
        states: np.ndarray,
        parameters: np.ndarray,
        rollout_steps: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        parameter_mean: np.ndarray,
        parameter_std: np.ndarray,
    ) -> None:
        if states.ndim != 3 or states.shape[-1] != 3:
            raise ValueError("states must have shape (trajectory, time, 3).")
        if rollout_steps < 1 or rollout_steps >= states.shape[1]:
            raise ValueError("rollout_steps is incompatible with trajectory length.")
        self.states = (states - state_mean) / state_std
        self.parameters = (parameters - parameter_mean) / parameter_std
        self.rollout_steps = rollout_steps
        self.windows_per_trajectory = states.shape[1] - rollout_steps

    def __len__(self) -> int:
        return self.states.shape[0] * self.windows_per_trajectory

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory_index = index // self.windows_per_trajectory
        start = index % self.windows_per_trajectory
        stop = start + self.rollout_steps + 1
        sequence = torch.from_numpy(
            self.states[trajectory_index, start:stop].astype(np.float32, copy=False)
        )
        parameters = torch.from_numpy(
            self.parameters[trajectory_index].astype(np.float32, copy=False)
        )
        return sequence, parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def compute_normalization(
    train_states: np.ndarray,
    train_parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_samples = train_states[:, :-1].reshape(-1, 3).astype(np.float64)
    state_mean = state_samples.mean(axis=0)
    state_std = state_samples.std(axis=0)
    state_std = np.where(state_std < 1e-8, 1.0, state_std)

    parameter_mean = train_parameters.astype(np.float64).mean(axis=0)
    parameter_std = train_parameters.astype(np.float64).std(axis=0)
    parameter_std = np.where(parameter_std < 1e-8, 1.0, parameter_std)
    return state_mean, state_std, parameter_mean, parameter_std


def rollout_loss(
    model: FlowModel,
    sequences: torch.Tensor,
    parameters: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    prediction = sequences[:, 0]
    total = torch.zeros((), device=sequences.device)
    model_parameters = parameters if model.conditioned else None
    for lead in range(1, sequences.shape[1]):
        prediction = model(prediction, model_parameters)
        total = total + loss_fn(prediction, sequences[:, lead])
    return total / (sequences.shape[1] - 1)


def run_epoch(
    model: FlowModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for sequences, parameters in loader:
            sequences = sequences.to(device)
            parameters = parameters.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = rollout_loss(model, sequences, parameters, loss_fn)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = sequences.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_examples += batch_size
    return total_loss / total_examples


def train_from_config(config: dict[str, Any]) -> Path:
    seed = int(config["seed"])
    set_seed(seed)
    device = select_device(str(config.get("device", "auto")))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(config["data_path"], allow_pickle=False) as data:
        train_states = data["train_states"]
        train_parameters = data["train_parameters"]
        validation_states = data["validation_states"]
        validation_parameters = data["validation_parameters"]
        benchmark_metadata = str(data["metadata_json"].item())

    normalization = compute_normalization(train_states, train_parameters)
    state_mean, state_std, parameter_mean, parameter_std = normalization
    rollout_steps = int(config["training"]["rollout_steps"])

    train_dataset = TrajectoryWindowDataset(
        train_states,
        train_parameters,
        rollout_steps,
        state_mean,
        state_std,
        parameter_mean,
        parameter_std,
    )
    validation_dataset = TrajectoryWindowDataset(
        validation_states,
        validation_parameters,
        rollout_steps,
        state_mean,
        state_std,
        parameter_mean,
        parameter_std,
    )
    loader_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"].get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_kwargs
    )

    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    best_validation = float("inf")
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / "best_checkpoint.pt"
    epochs = int(config["training"]["epochs"])

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, device, None)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6e} "
            f"validation_loss={validation_loss:.6e}"
        )

        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config["model"],
                    "training_config": config,
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "parameter_mean": parameter_mean,
                    "parameter_std": parameter_std,
                    "benchmark_metadata": benchmark_metadata,
                    "best_epoch": epoch,
                    "best_validation_loss": validation_loss,
                },
                checkpoint_path,
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Best checkpoint: {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Lorenz flow-map emulator.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_json(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    train_from_config(config)


if __name__ == "__main__":
    main()
