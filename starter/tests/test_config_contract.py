import json
import unittest
from collections import defaultdict
from pathlib import Path


STARTER_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = STARTER_ROOT / "configs"


def load_config(filename: str) -> dict:
    with (CONFIG_ROOT / filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_matrix(filename: str) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    with (CONFIG_ROOT / filename).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 3:
                raise AssertionError(
                    f"{filename}:{line_number} must contain CONFIG SEED OUTPUT_DIRECTORY"
                )
            rows.append((fields[0], int(fields[1]), fields[2]))
    return rows


class ConfigurationContractTests(unittest.TestCase):
    def test_team_a_changes_only_rollout_loss_horizon(self) -> None:
        a0 = load_config("train_a0_direct_onestep.json")
        a1 = load_config("train_a1_direct_multistep.json")

        self.assertEqual(a0["data_path"], "data/standard_benchmark.npz")
        self.assertEqual(a0["data_path"], a1["data_path"])
        self.assertEqual(a0["model"], a1["model"])
        self.assertEqual(a0["model"]["prediction_type"], "direct")
        self.assertFalse(a0["model"]["conditioned"])

        a0_training = dict(a0["training"])
        a1_training = dict(a1["training"])
        a0_horizon = a0_training.pop("rollout_steps")
        a1_horizon = a1_training.pop("rollout_steps")
        self.assertEqual(a0_training, a1_training)
        self.assertEqual(a0_horizon, 1)
        self.assertGreater(a1_horizon, 1)

    def test_team_b_changes_only_rho_conditioning(self) -> None:
        b1 = load_config("train_b1_multirho_state_only.json")
        b2 = load_config("train_b2_multirho_conditioned.json")

        self.assertEqual(b1["data_path"], "data/multirho_benchmark.npz")
        self.assertEqual(b1["data_path"], b2["data_path"])
        self.assertEqual(b1["training"], b2["training"])
        self.assertEqual(b1["training"]["rollout_steps"], 1)

        b1_model = dict(b1["model"])
        b2_model = dict(b2["model"])
        self.assertFalse(b1_model.pop("conditioned"))
        self.assertTrue(b2_model.pop("conditioned"))
        self.assertEqual(b1_model, b2_model)
        self.assertEqual(b1_model["prediction_type"], "direct")

    def test_standard_and_multirho_splits_match_the_contract(self) -> None:
        standard = load_config("benchmark_standard.json")["splits"]
        multirho = load_config("benchmark_multirho.json")["splits"]

        for split in standard.values():
            self.assertEqual(split["rhos"], [28.0])
        self.assertEqual(len({split["seed"] for split in standard.values()}), 3)

        self.assertEqual(multirho["train"]["rhos"], [26.0, 28.0, 32.0])
        self.assertEqual(multirho["validation"]["rhos"], [26.0, 28.0, 32.0])
        self.assertEqual(multirho["public_test"]["rhos"], [24.0, 30.0])
        self.assertTrue(
            set(multirho["train"]["rhos"]).isdisjoint(
                multirho["public_test"]["rhos"]
            )
        )
        self.assertEqual(len({split["seed"] for split in multirho.values()}), 3)

    def test_team_matrices_contain_three_matched_seeds(self) -> None:
        expected = {
            "matrix_team_a_seeds.txt": {
                "configs/train_a0_direct_onestep.json",
                "configs/train_a1_direct_multistep.json",
            },
            "matrix_team_b_seeds.txt": {
                "configs/train_b1_multirho_state_only.json",
                "configs/train_b2_multirho_conditioned.json",
            },
        }
        for matrix_filename, expected_configs in expected.items():
            rows = load_matrix(matrix_filename)
            seeds_by_config: dict[str, set[int]] = defaultdict(set)
            for config_path, seed, output_directory in rows:
                self.assertTrue((STARTER_ROOT / config_path).is_file())
                self.assertTrue(output_directory.endswith(f"seed{seed}"))
                seeds_by_config[config_path].add(seed)
            self.assertEqual(set(seeds_by_config), expected_configs)
            for seeds in seeds_by_config.values():
                self.assertEqual(seeds, {41, 42, 43})

    def test_shared_baseline_matrix_contains_mlp_repeats_and_linear_reference(self) -> None:
        rows = load_matrix("matrix_shared_baseline_seeds.txt")
        seeds_by_config: dict[str, set[int]] = defaultdict(set)
        for config_path, seed, output_directory in rows:
            self.assertTrue((STARTER_ROOT / config_path).is_file())
            self.assertTrue(output_directory.endswith(f"seed{seed}"))
            seeds_by_config[config_path].add(seed)

        self.assertEqual(
            seeds_by_config["configs/train_a0_direct_onestep.json"], {41, 42, 43}
        )
        self.assertEqual(seeds_by_config["configs/train_linear.json"], {42})
        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
