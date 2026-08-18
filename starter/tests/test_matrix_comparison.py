from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lorenz_hackathon.compare_matrix import create_matrix_comparison


STARTER_ROOT = Path(__file__).resolve().parents[1]


def synthetic_rho_result(
    one_step_nrmse: float,
    useful_horizon: float | None,
    metric_offset: float,
) -> dict:
    return {
        "rho": 28.0,
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


class MatrixComparisonTests(unittest.TestCase):
    def test_matrix_evaluation_automatically_runs_comparison(self) -> None:
        script = (STARTER_ROOT / "scripts" / "evaluate_matrix.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("python -m lorenz_hackathon.compare_matrix", script)
        self.assertIn(
            'comparison_directory="runs/matrix_comparisons/',
            script,
        )
        self.assertGreater(
            script.index("python -m lorenz_hackathon.compare_matrix"),
            script.index('done < "$matrix_file"'),
        )

    def write_result(
        self,
        output_directory: Path,
        evaluation_label: str,
        rho_result: dict,
        data_sha256: str = "frozen-data",
        data_path: str = "data/standard_benchmark.npz",
    ) -> None:
        self.write_results(
            output_directory,
            evaluation_label,
            {"28": rho_result},
            data_sha256,
            data_path,
        )

    def write_results(
        self,
        output_directory: Path,
        evaluation_label: str,
        by_rho: dict[str, dict],
        data_sha256: str = "frozen-data",
        data_path: str = "data/standard_benchmark.npz",
    ) -> None:
        evaluation_directory = output_directory / "evaluation" / evaluation_label
        evaluation_directory.mkdir(parents=True)
        result = {
            "checkpoint": str(output_directory / "best_checkpoint.pt"),
            "checkpoint_sha256": f"checkpoint-{output_directory.name}",
            "data": data_path,
            "data_sha256": data_sha256,
            "evaluation_settings": {"forecast_steps": 2},
            "by_rho": by_rho,
        }
        (evaluation_directory / "benchmark_results.json").write_text(
            json.dumps(result), encoding="utf-8"
        )

    def test_creates_seed_summary_and_both_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.txt"
            rows = []
            for config, prefix, base_error in (
                ("configs/train_a0_direct_onestep.json", "a0", 0.30),
                ("configs/train_a1_direct_multistep.json", "a1", 0.20),
            ):
                for seed_index, seed in enumerate((41, 42, 43)):
                    output_directory = root / f"{prefix}_seed{seed}"
                    rows.append(f"{config} {seed} {output_directory}")
                    self.write_result(
                        output_directory,
                        "team_a_rho28",
                        synthetic_rho_result(
                            one_step_nrmse=base_error + 0.01 * seed_index,
                            useful_horizon=1.0 + 0.1 * seed_index,
                            metric_offset=0.03 + 0.01 * seed_index,
                        ),
                    )
            matrix_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            output_dir = root / "comparison"

            summary = create_matrix_comparison(
                matrix_path,
                "team_a_rho28",
                output_dir,
            )

            rho_summary = summary["by_rho"]["28"]
            a0_summary = rho_summary["config_summary"][
                "configs/train_a0_direct_onestep.json"
            ]
            self.assertAlmostEqual(
                a0_summary["metrics"]["one_step_nrmse"]["mean"], 0.31
            )
            self.assertAlmostEqual(
                a0_summary["metrics"]["one_step_nrmse"]["population_std"],
                0.008164965809277268,
            )
            paired = rho_summary["paired_changes"]["metrics"]["one_step_nrmse"]
            self.assertAlmostEqual(paired["summary"]["mean"], -0.1)
            self.assertEqual(paired["summary"]["count"], 3)

            summary_path = output_dir / "matrix_summary.json"
            forecast_path = output_dir / "forecast_comparison_rho_28.png"
            metric_path = output_dir / "metric_comparison_rho_28.png"
            self.assertTrue(summary_path.is_file())
            self.assertGreater(forecast_path.stat().st_size, 1_000)
            self.assertGreater(metric_path.stat().st_size, 1_000)
            json.loads(summary_path.read_text(encoding="utf-8"))

    def test_recovers_delta_t_from_legacy_result_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "legacy_benchmark.npz"
            np.savez_compressed(
                data_path,
                metadata_json=np.asarray(
                    json.dumps({"reference": {"delta_t": 0.05}})
                ),
            )
            rho_result = synthetic_rho_result(0.2, 0.1, 0.02)
            del rho_result["delta_t"]
            output_directory = root / "a0_seed41"
            self.write_result(
                output_directory,
                "standard_rho28",
                rho_result,
                data_path=str(data_path),
            )
            matrix_path = root / "matrix.txt"
            matrix_path.write_text(
                "configs/train_a0_direct_onestep.json "
                f"41 {output_directory}\n",
                encoding="utf-8",
            )

            create_matrix_comparison(
                matrix_path, "standard_rho28", root / "comparison"
            )

            self.assertTrue(
                (root / "comparison" / "forecast_comparison_rho_28.png").is_file()
            )

    def test_multiple_rhos_create_separate_team_b_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix_team_b_seeds.txt"
            rows = []
            for config, prefix, error in (
                ("configs/train_b1_multirho_state_only.json", "b1", 0.30),
                ("configs/train_b2_multirho_conditioned.json", "b2", 0.20),
            ):
                output_directory = root / f"{prefix}_seed41"
                rows.append(f"{config} 41 {output_directory}")
                self.write_results(
                    output_directory,
                    "multirho_unseen_rho24_30",
                    {
                        "24": synthetic_rho_result(error, 1.0, 0.04),
                        "30": synthetic_rho_result(error + 0.02, 0.9, 0.05),
                    },
                )
            matrix_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            output_dir = root / "comparison"

            summary = create_matrix_comparison(
                matrix_path,
                "multirho_unseen_rho24_30",
                output_dir,
            )

            self.assertEqual(list(summary["by_rho"]), ["24", "30"])
            for rho in ("24", "30"):
                self.assertTrue(
                    (output_dir / f"forecast_comparison_rho_{rho}.png").is_file()
                )
                self.assertTrue(
                    (output_dir / f"metric_comparison_rho_{rho}.png").is_file()
                )

    def test_preserves_uncrossed_horizon_as_censored_summary_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_directory = root / "a0_seed41"
            self.write_result(
                output_directory,
                "benchmark",
                synthetic_rho_result(0.2, None, 0.02),
            )
            matrix_path = root / "matrix.txt"
            matrix_path.write_text(
                "configs/train_a0_direct_onestep.json "
                f"41 {output_directory}\n",
                encoding="utf-8",
            )

            summary = create_matrix_comparison(
                matrix_path, "benchmark", root / "comparison"
            )

            horizon = summary["by_rho"]["28"]["config_summary"][
                "configs/train_a0_direct_onestep.json"
            ]["metrics"]["useful_horizon"]
            self.assertIsNone(horizon["mean"])
            self.assertEqual(horizon["count"], 0)
            self.assertEqual(horizon["missing"], 0)
            self.assertEqual(horizon["not_crossed_within_window"], 1)

    def test_rejects_mixed_evaluation_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a0_seed41"
            second = root / "a1_seed41"
            self.write_result(
                first,
                "benchmark",
                synthetic_rho_result(0.2, 1.0, 0.02),
                data_sha256="dataset-a",
            )
            self.write_result(
                second,
                "benchmark",
                synthetic_rho_result(0.1, 1.5, 0.01),
                data_sha256="dataset-b",
            )
            matrix_path = root / "matrix.txt"
            matrix_path.write_text(
                "configs/train_a0_direct_onestep.json "
                f"41 {first}\n"
                "configs/train_a1_direct_multistep.json "
                f"41 {second}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "different evaluation datasets"):
                create_matrix_comparison(
                    matrix_path, "benchmark", root / "comparison"
                )


if __name__ == "__main__":
    unittest.main()
