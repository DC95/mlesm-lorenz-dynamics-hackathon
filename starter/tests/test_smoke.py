import unittest

import numpy as np

from lorenz_hackathon.baselines import persistence_rollout
from lorenz_hackathon.data import generate_benchmark
from lorenz_hackathon.dynamics import LorenzParameters, integrate_lorenz, lorenz_rhs

try:
    import torch

    from lorenz_hackathon.models import FlowMLP, LinearFlow

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


class SmokeTests(unittest.TestCase):
    def test_rhs_and_integrator_shapes(self) -> None:
        states = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, 20.0]])
        tendencies = lorenz_rhs(states)
        trajectories = integrate_lorenz(
            states,
            num_steps=3,
            delta_t=0.01,
            dt_reference=0.001,
            parameters=LorenzParameters(),
        )
        self.assertEqual(tendencies.shape, states.shape)
        self.assertEqual(trajectories.shape, (2, 4, 3))
        self.assertTrue(np.all(np.isfinite(trajectories)))

    def test_persistence_rollout_repeats_the_initial_state(self) -> None:
        initial = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]])
        rollout = persistence_rollout(initial, num_steps=3)
        self.assertEqual(rollout.shape, (2, 4, 3))
        np.testing.assert_array_equal(rollout[:, 0], initial)
        np.testing.assert_array_equal(rollout[:, -1], initial)

        with self.assertRaises(ValueError):
            persistence_rollout(np.zeros((2, 4)), num_steps=3)
        with self.assertRaises(ValueError):
            persistence_rollout(initial, num_steps=-1)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_model_modes_and_conditioning(self) -> None:
        states = torch.zeros(5, 3)
        parameters = torch.zeros(5, 3)
        direct = FlowMLP(prediction_type="direct", conditioned=False)
        residual = FlowMLP(prediction_type="residual", conditioned=True)
        linear = LinearFlow(prediction_type="residual", conditioned=False)
        conditioned_linear = LinearFlow(prediction_type="direct", conditioned=True)
        self.assertEqual(direct(states).shape, (5, 3))
        self.assertEqual(residual(states, parameters).shape, (5, 3))
        self.assertEqual(linear(states).shape, (5, 3))
        self.assertEqual(conditioned_linear(states, parameters).shape, (5, 3))
        self.assertEqual(direct.network[0].in_features, 3)
        self.assertEqual(residual.network[0].in_features, 4)
        self.assertEqual(conditioned_linear.network.in_features, 4)

        same_rho_a = torch.tensor([[8.0, 28.0, 2.0]]).repeat(5, 1)
        same_rho_b = torch.tensor([[12.0, 28.0, 4.0]]).repeat(5, 1)
        torch.testing.assert_close(
            residual(states, same_rho_a), residual(states, same_rho_b)
        )

    def test_tiny_benchmark_has_disjoint_named_splits(self) -> None:
        config = {
            "reference": {
                "sigma": 10.0,
                "beta": 8.0 / 3.0,
                "dt_reference": 0.01,
                "delta_t": 0.02,
                "burn_in_time": 0.02,
                "initial_low": -2.0,
                "initial_high": 2.0,
            },
            "splits": {
                "train": {
                    "rhos": [28.0],
                    "trajectories_per_rho": 2,
                    "steps": 3,
                    "seed": 1,
                },
                "validation": {
                    "rhos": [28.0],
                    "trajectories_per_rho": 1,
                    "steps": 3,
                    "seed": 2,
                },
                "public_test": {
                    "rhos": [28.0],
                    "trajectories_per_rho": 1,
                    "steps": 3,
                    "seed": 3,
                },
            },
        }
        arrays = generate_benchmark(config)
        self.assertEqual(arrays["train_states"].shape, (2, 4, 3))
        self.assertEqual(arrays["validation_states"].shape, (1, 4, 3))
        self.assertFalse(
            np.array_equal(arrays["train_states"][0], arrays["validation_states"][0])
        )


if __name__ == "__main__":
    unittest.main()
