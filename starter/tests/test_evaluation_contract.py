from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
    from torch import nn

    from lorenz_hackathon.dynamics import LorenzParameters, integrate_lorenz
    from lorenz_hackathon.evaluate import (
        EVALUATION_BURN_STEPS,
        FORECAST_STEPS,
        LONG_ROLLOUT_STEPS,
        LONG_TRAJECTORY_COUNT,
        PERTURBATION_STEPS,
        PERTURBATION_TRAJECTORY_COUNT,
        REFERENCE_BOUND_MULTIPLIER,
        REFERENCE_BOUND_QUANTILE,
        USEFUL_HORIZON_THRESHOLD,
        effective_growth_rate,
        evaluate_rho_group,
        evaluation_settings,
        first_threshold_time,
        mean_residence_time,
        quantile_wasserstein,
        switch_rate,
    )

    EVALUATION_DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    EVALUATION_DEPENDENCIES_AVAILABLE = False


if EVALUATION_DEPENDENCIES_AVAILABLE:

    class AlwaysNonFiniteModel(nn.Module):
        conditioned = False

        def forward(
            self,
            states: torch.Tensor,
            parameters: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del parameters
            return torch.full_like(states, float("nan"))


@unittest.skipUnless(
    EVALUATION_DEPENDENCIES_AVAILABLE,
    "PyTorch and Matplotlib are required for evaluator contract tests",
)
class EvaluationContractTests(unittest.TestCase):
    def test_frozen_default_settings(self) -> None:
        settings = evaluation_settings(LONG_ROLLOUT_STEPS, PERTURBATION_STEPS)

        self.assertEqual(FORECAST_STEPS, 200)
        self.assertEqual(LONG_TRAJECTORY_COUNT, 32)
        self.assertEqual(LONG_ROLLOUT_STEPS, 4000)
        self.assertEqual(EVALUATION_BURN_STEPS, 400)
        self.assertEqual(PERTURBATION_TRAJECTORY_COUNT, 64)
        self.assertEqual(PERTURBATION_STEPS, 200)
        self.assertEqual(USEFUL_HORIZON_THRESHOLD, 1.0)
        self.assertEqual(REFERENCE_BOUND_QUANTILE, 0.999)
        self.assertEqual(REFERENCE_BOUND_MULTIPLIER, 5.0)
        self.assertEqual(settings["evaluation_burn_steps"], 400)

    def test_nonfinite_value_counts_as_threshold_failure(self) -> None:
        values = np.array([0.0, 0.3, np.nan, 0.1])
        self.assertEqual(first_threshold_time(values, 1.0, 0.05), 0.1)
        self.assertIsNone(
            first_threshold_time(np.array([0.0, 0.3, 0.9]), 1.0, 0.05)
        )

    def test_quantile_wasserstein(self) -> None:
        reference = np.linspace(-2.0, 2.0, 1001)
        self.assertAlmostEqual(quantile_wasserstein(reference, reference), 0.0)
        self.assertAlmostEqual(
            quantile_wasserstein(reference, reference + 2.0),
            2.0,
            places=12,
        )

    def test_lobe_switch_and_observed_residence_definitions(self) -> None:
        trajectories = np.zeros((1, 4, 3), dtype=np.float64)
        trajectories[0, :, 0] = [1.0, -1.0, -1.0, 1.0]

        self.assertAlmostEqual(switch_rate(trajectories, 0.5), 2.0 / 1.5)
        self.assertAlmostEqual(
            mean_residence_time(trajectories, 0.5),
            np.mean([0.5, 1.0, 0.5]),
        )

    def test_effective_growth_rate_uses_finite_fit_window(self) -> None:
        times = np.linspace(0.0, 10.0, 101)
        initial_distance = 1e-5
        rate = 0.8
        distance = initial_distance * np.exp(rate * times)

        fitted = effective_growth_rate(
            times,
            distance,
            initial_distance,
            np.ones(3),
        )
        self.assertIsNotNone(fitted)
        self.assertAlmostEqual(float(fitted), rate, places=12)

    def test_nonfinite_model_produces_diagnostic_result_and_figure(self) -> None:
        delta_t = 0.02
        parameters = np.array(
            [[10.0, 28.0, 8.0 / 3.0], [10.0, 28.0, 8.0 / 3.0]],
            dtype=np.float64,
        )
        initial_states = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, 20.0]])
        states = integrate_lorenz(
            initial_states,
            num_steps=4,
            delta_t=delta_t,
            dt_reference=0.01,
            parameters=LorenzParameters(),
        )
        checkpoint = {
            "state_mean": np.zeros(3),
            "state_std": np.ones(3),
            "parameter_mean": np.array([10.0, 28.0, 8.0 / 3.0]),
            "parameter_std": np.ones(3),
        }
        metadata = {
            "reference": {
                "delta_t": delta_t,
                "dt_reference": 0.01,
            }
        }

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            result = evaluate_rho_group(
                rho=28.0,
                states=states,
                parameters=parameters,
                model=AlwaysNonFiniteModel(),
                checkpoint=checkpoint,
                device=torch.device("cpu"),
                reference_metadata=metadata,
                output_dir=output_dir,
                long_steps=5,
                perturbation_steps=4,
            )

            self.assertIsNone(result["one_step"]["nrmse"])
            self.assertEqual(result["one_step"]["finite_prediction_fraction"], 0.0)
            self.assertEqual(
                result["forecast"]["useful_horizon_nrmse_1"], delta_t
            )
            self.assertEqual(
                result["stability"]["finite_trajectory_fraction"], 0.0
            )
            self.assertEqual(result["climate"]["model_mean"], [None, None, None])
            json.dumps(result, allow_nan=False)
            self.assertTrue(
                (output_dir / "model_autopsy_rho_28.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
