from pathlib import Path
import unittest


STARTER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = STARTER_ROOT.parent
SCRIPTS_ROOT = STARTER_ROOT / "scripts"


class RuntimeSelectionTests(unittest.TestCase):
    def test_selector_offers_all_supported_runtimes(self):
        selector = (SCRIPTS_ROOT / "select_runtime.sh").read_text(
            encoding="utf-8"
        )
        for runtime in ("local", "colab", "jureca"):
            self.assertIn(runtime, selector)
        self.assertIn(".hackathon-runtime", selector)

    def test_portable_prepare_recreates_and_verifies_frozen_data(self):
        runner = (SCRIPTS_ROOT / "run.sh").read_text(encoding="utf-8")
        self.assertIn("benchmark_standard.json", runner)
        self.assertIn("benchmark_multirho.json", runner)
        self.assertIn("sha256sum -c SHA256SUMS", runner)
        self.assertIn("python -m pip install --editable .", runner)

    def test_portable_manifest_matches_frozen_benchmark_document(self):
        manifest = (
            STARTER_ROOT / "configs" / "frozen_data_SHA256SUMS"
        ).read_text(encoding="utf-8")
        contract = (
            REPOSITORY_ROOT / "docs" / "frozen_benchmark_v1.0.md"
        ).read_text(encoding="utf-8")
        for line in manifest.splitlines():
            checksum, filename = line.split()
            self.assertIn(checksum, contract)
            self.assertIn(filename, contract)

    def test_team_b_action_uses_same_matrix_and_three_regimes(self):
        runner = (SCRIPTS_ROOT / "run.sh").read_text(encoding="utf-8")
        self.assertIn("configs/matrix_team_b_seeds.txt", runner)
        self.assertIn("in_distribution_rho28", runner)
        self.assertIn("multirho_unseen_rho24_30", runner)

    def test_jureca_path_keeps_reservation_and_serial_dependencies(self):
        runner = (SCRIPTS_ROOT / "run.sh").read_text(encoding="utf-8")
        self.assertIn('--reservation="$HACKATHON_RESERVATION"', runner)
        self.assertIn("--dependency=afterok:", runner)
        self.assertIn("slurm/train_matrix.sbatch", runner)
        self.assertIn("slurm/evaluate_matrix.sbatch", runner)

    def test_local_and_colab_do_not_invoke_slurm_for_team_b(self):
        runner = (SCRIPTS_ROOT / "run.sh").read_text(encoding="utf-8")
        self.assertIn('if [[ "$runtime" == "jureca" ]]', runner)
        self.assertIn('bash scripts/train_matrix_worker.sh "$matrix"', runner)
        self.assertIn("bash scripts/evaluate_matrix.sh", runner)


if __name__ == "__main__":
    unittest.main()
