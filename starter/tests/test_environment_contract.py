from pathlib import Path
import unittest


STARTER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = STARTER_ROOT.parent
DOCS_ROOT = REPOSITORY_ROOT / "docs"
ENVIRONMENT_ROOT = STARTER_ROOT / "environment"
SCRIPTS_ROOT = STARTER_ROOT / "scripts"
SLURM_ROOT = STARTER_ROOT / "slurm"


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
        self.assertIn('new_pythonpath="${starter_dir}/src"', activation)
        self.assertIn('export PYTHONPATH="${new_pythonpath}"', activation)
        self.assertNotIn('case ":${PYTHONPATH:-}:"', activation)
        self.assertIn('export HACKATHON_ACTIVATE="${script_dir}/activate.sh"', activation)

    def test_fresh_activation_defines_per_user_run_root(self):
        config = (ENVIRONMENT_ROOT / "config.sh").read_text(encoding="utf-8")

        self.assertIn("HACKATHON_RUN_ROOT", config)
        self.assertIn("/p/scratch/training2635/${USER}/mlesm-lorenz-hackathon-2026", config)

    def test_event_reservations_are_mapped_to_their_calendar_days(self):
        config = (ENVIRONMENT_ROOT / "config.sh").read_text(encoding="utf-8")

        expected_reservations = {
            "2026-08-19": "challenge_3_and_5_day1",
            "2026-08-20": "challenge_3_and_5_day2",
            "2026-08-21": "challenge_3_and_5_day3",
        }
        for event_date, reservation in expected_reservations.items():
            self.assertIn(event_date, config)
            self.assertIn(reservation, config)
        self.assertIn('HACKATHON_ACCOUNT:-training2635', config)
        self.assertIn('HACKATHON_PARTITION:-dc-gpu', config)
        self.assertIn("HACKATHON_RESERVATION_OVERRIDE", config)

    def test_shared_environment_does_not_install_one_team_branch(self):
        setup = (ENVIRONMENT_ROOT / "setup.sh").read_text(encoding="utf-8")

        self.assertNotIn("pip install -e", setup)
        self.assertNotIn("pip install --editable", setup)

    def test_login_preflight_checks_data_tests_and_scratch(self):
        preflight = (SCRIPTS_ROOT / "preflight_login.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("(cd data && sha256sum -c SHA256SUMS)", preflight)
        self.assertNotIn("sha256sum -c data/SHA256SUMS", preflight)
        self.assertIn("python -m unittest discover -s tests -v", preflight)
        self.assertIn('touch "${test_file}"', preflight)
        self.assertIn("HACKATHON_RUN_ROOT", preflight)
        self.assertIn('scontrol show reservation "${HACKATHON_RESERVATION}"', preflight)

    def test_gpu_preflight_requests_and_exercises_cuda(self):
        preflight = (SLURM_ROOT / "preflight.sbatch").read_text(encoding="utf-8")

        self.assertIn("#SBATCH --partition=dc-gpu", preflight)
        self.assertIn("#SBATCH --gres=gpu:1", preflight)
        self.assertIn("torch.cuda.is_available()", preflight)
        self.assertIn("loss.backward()", preflight)

    def test_every_gpu_job_uses_one_production_node(self):
        gpu_jobs = (
            "preflight.sbatch",
            "train_matrix.sbatch",
            "evaluate.sbatch",
            "evaluate_matrix.sbatch",
        )
        for job_name in gpu_jobs:
            job = (SLURM_ROOT / job_name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --partition=dc-gpu", job)
            self.assertIn("#SBATCH --nodes=1", job)
            self.assertNotIn("dc-gpu-devel", job)

    def test_participant_commands_use_the_event_reservation(self):
        command_reference = (DOCS_ROOT / "commands_and_outputs.md").read_text(
            encoding="utf-8"
        )

        self.assertIn('--reservation="$HACKATHON_RESERVATION"', command_reference)
        self.assertIn('--partition="$HACKATHON_PARTITION"', command_reference)
        self.assertIn('challenge_3_and_5_day1', command_reference)
        self.assertIn('challenge_3_and_5_day2', command_reference)
        self.assertIn('challenge_3_and_5_day3', command_reference)
        self.assertNotIn("dc-gpu-devel", command_reference)

    def test_team_b_evaluations_are_serialized(self):
        command_reference = (DOCS_ROOT / "commands_and_outputs.md").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            '--dependency="afterok:$team_eval_unseen_job"', command_reference
        )


if __name__ == "__main__":
    unittest.main()
