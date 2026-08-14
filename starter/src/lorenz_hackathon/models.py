from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    activations: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


class FlowMLP(nn.Module):
    """MLP flow map in normalized coordinates."""

    def __init__(
        self,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        activation: str = "tanh",
        prediction_type: str = "direct",
        conditioned: bool = False,
    ) -> None:
        super().__init__()
        if prediction_type not in {"direct", "residual"}:
            raise ValueError("prediction_type must be 'direct' or 'residual'.")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least one.")

        self.prediction_type = prediction_type
        self.conditioned = conditioned
        # The lean Team B comparison conditions on rho only. Sigma and beta remain
        # fixed properties of the reference system and are not model inputs.
        input_dim = 4 if conditioned else 3
        activation_cls = _activation(activation)

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), activation_cls()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_cls()])
        layers.append(nn.Linear(hidden_dim, 3))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        states: torch.Tensor,
        parameters: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.conditioned:
            if parameters is None:
                raise ValueError("A conditioned model requires normalized parameters.")
            rho = parameters[..., 1:2]
            inputs = torch.cat((states, rho), dim=-1)
        else:
            inputs = states

        output = self.network(inputs)
        if self.prediction_type == "residual":
            output = states + output
        return output


class LinearFlow(nn.Module):
    """Linear direct or residual flow map used as a mandatory baseline."""

    def __init__(self, prediction_type: str = "residual", conditioned: bool = False) -> None:
        super().__init__()
        if prediction_type not in {"direct", "residual"}:
            raise ValueError("prediction_type must be 'direct' or 'residual'.")
        self.prediction_type = prediction_type
        self.conditioned = conditioned
        self.network = nn.Linear(4 if conditioned else 3, 3)

    def forward(
        self,
        states: torch.Tensor,
        parameters: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.conditioned:
            if parameters is None:
                raise ValueError("A conditioned model requires normalized parameters.")
            rho = parameters[..., 1:2]
            inputs = torch.cat((states, rho), dim=-1)
        else:
            inputs = states
        output = self.network(inputs)
        return states + output if self.prediction_type == "residual" else output


FlowModel = FlowMLP | LinearFlow


def build_model(config: dict) -> FlowModel:
    architecture = str(config.get("architecture", "mlp"))
    common = {
        "prediction_type": str(config["prediction_type"]),
        "conditioned": bool(config["conditioned"]),
    }
    if architecture == "linear":
        return LinearFlow(**common)
    if architecture != "mlp":
        raise ValueError(f"Unsupported model architecture: {architecture}")
    return FlowMLP(
        hidden_dim=int(config["hidden_dim"]),
        hidden_layers=int(config["hidden_layers"]),
        activation=str(config["activation"]),
        **common,
    )
