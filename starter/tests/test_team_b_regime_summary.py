from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lorenz_hackathon.compare_matrix import create_matrix_comparison
from lorenz_hackathon.compare_team_b_regimes import (
    create_team_b_regime_summary,
)


STARTER_ROOT = Path(__file__).resolve().parents[1]


def synthetic_rho_result(
    rho: float,
    one_step_nrmse: float,
    useful_horizon: float,
    metric_offset: float,
) -> dict:
    return {
        "rho": rho,
        "delta_t": 0.05,
        "one_step": {"nrmse": one_step_nrmse},
        "forecast": {
            "useful_horizon_nrmse_1": useful_horizon,
            "nrmse_by_lead": [0.0, one_step_nrmse, 2.0 * one_step_nrmse],
            "persistence_nrmse_by_lead": [0.0, 0.4, 0.8],
        },
        "stability": {
            "finite_trajectory_fraction": 1.0,
            "within_reference_bound_fraction": 1.0 - metric_offset,
            "variance_ratio": [
                1.0 + metric_offset,
                1.0 - metric_offset,
                1.0,
            ],
        },
        "perturbation": {
            "reference_effective_growth_rate": 0.8,
            "model_effective_growth_rate": 0.8 + metric_offset,
        },
        "climate": {
            "reference_positive_x_fraction": 0.5,
            "model_positive_x_fraction": 0.5 + metric_offset,
            "reference_switch_rate": 0.2,
            "model_switch_rate": 0.2 + metric_offset,
            "reference_mean_residence_time": 5.0,
            "model_mean_residence_time": 5.0 + metric_offset,
            "wasserstein_by_variable": {
                "x": metric_offset,
                "y": 2.0 * metric_offset,
                "z": 3.0 * metric_offset,
            },
        },
    }


class TeamBRegimeSummaryTests(unittest.TestCase):
    def write_results(
        self,
        output_directory: Path,
        evaluation_label: str,
        by_rho: dict[str, dict],
        data_sha256: str,
    ) -> None:
        evaluation_directory = output_directory / "evaluation" / evaluation_label
        evaluation_directory.mkdir(parents=True)
        result = {
            "checkpoint": str(output_directory / "best_checkpoint.pt"),
            "checkpoint_sha256": f"checkpoint-{output_directory.name}",
            "data": f"data/{data_sha256}.npz",
            "data_sha256": data_sha256,
            "evaluation_settings": {"forecast_steps": 2},
            "by_rho": by_rho,
        }
        (evaluation_directory / "benchmark_results.json").write_text(
            json.dumps(result), encoding="utf-8"
        )

    def test_combines_three_regimes_in_presentation_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix_team_b_seeds.txt"
            rows = []
            for config, prefix, base_error, base_offset in (
                (
                    "configs/train_b1_multirho_state_only.json",
                    "b1",
                    0.30,
                    0.05,
                ),
                (
                    "configs/train_b2_multirho_conditioned.json",
                    "b2",
                    0.20,
                    0.02,
                ),
            ):
                for seed_index, seed in enumerate((41, 42, 43)):
                    output_directory = root / f"{prefix}_seed{seed}"
                    rows.append(f"{config} {seed} {output_directory}")
                    error = base_error + 0.01 * seed_index
                    offset = base_offset + 0.002 * seed_index
                    self.write_results(
                        output_directory,
                        "in_distribution_rho28",
                        {
                            "28": synthetic_rho_result(
                                28.0, error, 1.0 + 0.1 * seed_index, offset
                            )
                        },
                        "standard-data",
                    )
                    self.write_results(
                        output_directory,
                        "multirho_unseen_rho24_30",
                        {
                            "24": synthetic_rho_result(
                                24.0, error + 0.04, 0.8, offset + 0.02
                            ),
                            "30": synthetic_rho_result(
                                30.0, error + 0.02, 0.9, offset + 0.01
                            ),
                        },
                        "multirho-data",
                    )

            matrix_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            comparison_root = root / "matrix_comparisons" / "matrix_team_b_seeds"
            create_matrix_comparison(
                matrix_path,
                "in_distribution_rho28",
                comparison_root / "in_distribution_rho28",
            )
            create_matrix_comparison(
                matrix_path,
                "multirho_unseen_rho24_30",
                comparison_root / "multirho_unseen_rho24_30",
            )

            output_dir = comparison_root / "three_regime_summary"
            summary = create_team_b_regime_summary(
                matrix_path=matrix_path,
                comparison_root=comparison_root,
                output_dir=output_dir,
            )

            self.assertEqual(summary["rho_order"], ["28", "30", "24"])
            self.assertEqual(list(summary["by_rho"]), ["28", "30", "24"])
            self.assertEqual(
                summary["by_rho"]["30"]["scientific_role"],
                "Unseen interpolation",
            )
            figure_path = output_dir / "team_b_three_regime_summary.png"
            json_path = output_dir / "team_b_three_regime_summary.json"
            self.assertGreater(figure_path.stat().st_size, 10_000)
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["rho_order"], ["28", "30", "24"])

    def test_evaluation_script_attempts_summary_after_both_labels(self) -> None:
        script = (STARTER_ROOT / "scripts" / "evaluate_matrix.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("lorenz_hackathon.compare_team_b_regimes", script)
        self.assertIn("in_distribution_rho28/matrix_summary.json", script)
        self.assertIn("multirho_unseen_rho24_30/matrix_summary.json", script)


if __name__ == "__main__":
    unittest.main()
