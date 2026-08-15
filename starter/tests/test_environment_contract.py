from pathlib import Path
import unittest


STARTER_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_ROOT = STARTER_ROOT / "environment"


class EnvironmentContractTests(unittest.TestCase):
    def test_rehearsed_jureca_modules_are_pinned(self):
        modules = (ENVIRONMENT_ROOT / "modules.sh").read_text(encoding="utf-8")

        self.assertIn("Stages/2026", modules)
        self.assertIn("GCCcore/14.3.0", modules)
        self.assertIn("PyTorch/2.9.1", modules)
        self.assertIn("matplotlib/3.10.5", modules)

    def test_activation_uses_the_current_checkout_source(self):
        activation = (ENVIRONMENT_ROOT / "activate.sh").read_text(encoding="utf-8")

        self.assertIn('starter_dir=$(cd -- "${script_dir}/.." && pwd)', activation)
        self.assertIn('${starter_dir}/src', activation)
        self.assertIn('export HACKATHON_ACTIVATE="${script_dir}/activate.sh"', activation)

    def test_fresh_activation_defines_per_user_run_root(self):
        config = (ENVIRONMENT_ROOT / "config.sh").read_text(encoding="utf-8")

        self.assertIn("HACKATHON_RUN_ROOT", config)
        self.assertIn("/p/scratch/training2635/${USER}/mlesm-lorenz-hackathon-2026", config)

    def test_shared_environment_does_not_install_one_team_branch(self):
        setup = (ENVIRONMENT_ROOT / "setup.sh").read_text(encoding="utf-8")

        self.assertNotIn("pip install -e", setup)
        self.assertNotIn("pip install --editable", setup)


if __name__ == "__main__":
    unittest.main()
