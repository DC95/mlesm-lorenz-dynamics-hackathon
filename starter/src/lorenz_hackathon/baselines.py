from __future__ import annotations

import numpy as np


def persistence_rollout(initial_states: np.ndarray, num_steps: int) -> np.ndarray:
    """Repeat each initial state to form a no-change forecast baseline."""

    states = np.asarray(initial_states)
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("initial_states must have shape (trajectory, 3).")
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")
    return np.repeat(states[:, np.newaxis, :], num_steps + 1, axis=1)
