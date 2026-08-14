from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LorenzParameters:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


def _as_state_array(states: np.ndarray) -> np.ndarray:
    array = np.asarray(states, dtype=np.float64)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected final state dimension 3, received {array.shape}.")
    return array


def lorenz_rhs(
    states: np.ndarray,
    parameters: LorenzParameters = LorenzParameters(),
) -> np.ndarray:
    """Return Lorenz-63 tendencies for one state or a batch of states."""

    states = _as_state_array(states)
    x, y, z = np.moveaxis(states, -1, 0)
    dx = parameters.sigma * (y - x)
    dy = x * (parameters.rho - z) - y
    dz = x * y - parameters.beta * z
    return np.stack((dx, dy, dz), axis=-1)


def rk4_step(
    states: np.ndarray,
    dt: float,
    parameters: LorenzParameters = LorenzParameters(),
) -> np.ndarray:
    """Advance one or many states by one RK4 step in float64."""

    states = _as_state_array(states)
    k1 = lorenz_rhs(states, parameters)
    k2 = lorenz_rhs(states + 0.5 * dt * k1, parameters)
    k3 = lorenz_rhs(states + 0.5 * dt * k2, parameters)
    k4 = lorenz_rhs(states + dt * k3, parameters)
    return states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def advance_flow_map(
    states: np.ndarray,
    delta_t: float,
    dt_reference: float,
    parameters: LorenzParameters = LorenzParameters(),
) -> np.ndarray:
    """Advance states over one exposed ML forecast interval."""

    ratio = delta_t / dt_reference
    substeps = int(round(ratio))
    if substeps < 1 or not np.isclose(ratio, substeps, rtol=0.0, atol=1e-10):
        raise ValueError("delta_t must be an integer multiple of dt_reference.")

    result = _as_state_array(states).copy()
    for _ in range(substeps):
        result = rk4_step(result, dt_reference, parameters)
    return result


def integrate_lorenz(
    initial_states: np.ndarray,
    num_steps: int,
    delta_t: float = 0.05,
    dt_reference: float = 0.001,
    parameters: LorenzParameters = LorenzParameters(),
) -> np.ndarray:
    """Integrate a batch and return shape ``(batch, num_steps + 1, 3)``."""

    initial_states = _as_state_array(initial_states)
    squeeze = initial_states.ndim == 1
    if squeeze:
        initial_states = initial_states[None, :]
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape (3,) or (batch, 3).")
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")

    trajectory = np.empty(
        (initial_states.shape[0], num_steps + 1, 3), dtype=np.float64
    )
    trajectory[:, 0] = initial_states
    current = initial_states.copy()
    for step in range(num_steps):
        current = advance_flow_map(
            current,
            delta_t=delta_t,
            dt_reference=dt_reference,
            parameters=parameters,
        )
        trajectory[:, step + 1] = current

    return trajectory[0] if squeeze else trajectory

