# Participant Commands and Outputs

This is the detailed operational path from a fresh JURECA login to trained
models, evaluation artifacts, and figures. If JURECA is unavailable, use the
[local and Colab quick start](../starter/LOCAL_AND_COLAB_QUICKSTART.md); it
runs the same frozen configurations and evaluations without Slurm. Read the assigned
[Team A](team_a_rollout_fidelity.md) or
[Team B](team_b_changing_dynamics.md) guide for the scientific question and
the [evaluation evidence guide](evaluation_evidence_guide.md) before
interpreting results.

## 1. Start from a fresh JURECA login

Activate the hackathon project and clone the public repository over HTTPS:

```bash
jutil env activate -p training2635

echo "$PROJECT"
echo "$SCRATCH"

mkdir -p "$PROJECT/$USER"
cd "$PROJECT/$USER"

git clone https://github.com/DC95/mlesm-lorenz-dynamics-hackathon.git
cd mlesm-lorenz-dynamics-hackathon
```

Switch to the branch assigned by the organizer:

```bash
# Team A
git switch team-a-rollout-fidelity

# Team B: use this instead of the preceding command
git switch team-b-changing-dynamics
```

All remaining commands are run from `starter/`:

```bash
cd starter
source environment/activate.sh
umask 0027
```

Activation uses the shared, organizer-prepared Python environment but imports
`lorenz_hackathon` from the current checkout. Team branches therefore share
dependencies without sharing source code. It also selects the Challenge 3/5
reservation from the JURECA login node's calendar date:

| Day | Reservation | Access window |
|---|---|---|
| 19 August 2026 | `challenge_3_and_5_day1` | 13:00--18:00 |
| 20 August 2026 | `challenge_3_and_5_day2` | 09:00--18:00 |
| 21 August 2026 | `challenge_3_and_5_day3` | 09:00--12:30 |

All times are the scheduler's local time. Confirm the printed reservation
before continuing. Outside these dates, activation deliberately leaves
`HACKATHON_RESERVATION` empty and the login preflight stops. Re-source
`environment/activate.sh` at the start of every event day. The reservation is
fixed when `sbatch` is called: a job submitted to the Day 1 reservation cannot
roll into Day 2. Submit only work whose requested wall time fits the remaining
window, and submit unfinished stages under the next day's reservation.

## 2. Run the login-node preflight

```bash
bash scripts/preflight_login.sh
```

This creates the per-user output directories and `data`/`runs` links, verifies
the released dataset checksums, runs the unit tests, and confirms that the
scratch output directory is writable. Do not continue if it fails. Report the
exact failing command and its output to the organizer.

## 3. Confirm access to one GPU

Submit the ten-minute preflight job and remember its job ID:

```bash
preflight_job=$(sbatch --parsable \
    --account="$HACKATHON_ACCOUNT" \
    --partition="$HACKATHON_PARTITION" \
    --reservation="$HACKATHON_RESERVATION" \
    --job-name=lorenz-preflight \
    --chdir="$(pwd)" \
    --output="$HACKATHON_RUN_ROOT/slurm/lorenz-preflight-%j.out" \
    --error="$HACKATHON_RUN_ROOT/slurm/lorenz-preflight-%j.err" \
    slurm/preflight.sbatch)

echo "GPU preflight job: $preflight_job"
squeue -j "$preflight_job"
```

When the job leaves the queue, inspect its state and output:

```bash
sacct -j "$preflight_job" --format=JobID,JobName,State,Elapsed,ExitCode
cat "$HACKATHON_RUN_ROOT/slurm/lorenz-preflight-${preflight_job}.out"
```

Continue only when the job prints `GPU preflight passed` and its Slurm state is
`COMPLETED` with exit code `0:0`. The test constructs the A0 model and performs
one CUDA forward/backward pass. `torch.cuda.is_available()` may be false on a
login node; this batch-job result is the check that matters.

## 4. Understand the matrix files

Every non-comment row has three fields:

```text
TRAIN_CONFIG SEED OUTPUT_DIRECTORY
```

| File | Meaning |
|---|---|
| `configs/matrix_shared_baseline_seeds.txt` | A0 at seeds 41--43 and one linear model at seed 42 |
| `configs/matrix_team_a_a1_only_seeds.txt` | A1 at seeds 41--43, used after the shared baseline has produced A0 |
| `configs/matrix_team_a_seeds.txt` | Complete A0/A1 manifest used for Team A evaluation and comparison |
| `configs/matrix_team_b_seeds.txt` | Complete B1/B2 training and evaluation manifest |

Persistence is evaluated directly and does not require a checkpoint.

## 5. Train the shared baseline

Both teams first submit the same shared baseline:

```bash
baseline_job=$(sbatch --parsable \
    --account="$HACKATHON_ACCOUNT" \
    --partition="$HACKATHON_PARTITION" \
    --reservation="$HACKATHON_RESERVATION" \
    --job-name=lorenz-baseline \
    --chdir="$(pwd)" \
    --output="$HACKATHON_RUN_ROOT/slurm/lorenz-baseline-%j.out" \
    --error="$HACKATHON_RUN_ROOT/slurm/lorenz-baseline-%j.err" \
    slurm/train_matrix.sbatch \
    configs/matrix_shared_baseline_seeds.txt)

echo "Shared baseline job: $baseline_job"
```

## 6. Train the assigned team comparison

Team A trains only A1 here because the shared baseline has already produced
the three A0 checkpoints:

```bash
team_job=$(sbatch --parsable \
    --dependency="afterok:$baseline_job" \
    --account="$HACKATHON_ACCOUNT" \
    --partition="$HACKATHON_PARTITION" \
    --reservation="$HACKATHON_RESERVATION" \
    --job-name=lorenz-team-a \
    --chdir="$(pwd)" \
    --output="$HACKATHON_RUN_ROOT/slurm/lorenz-team-a-%j.out" \
    --error="$HACKATHON_RUN_ROOT/slurm/lorenz-team-a-%j.err" \
    slurm/train_matrix.sbatch \
    configs/matrix_team_a_a1_only_seeds.txt)

echo "Team A job: $team_job"
```

Team B uses this command instead:

```bash
team_job=$(sbatch --parsable \
    --dependency="afterok:$baseline_job" \
    --account="$HACKATHON_ACCOUNT" \
    --partition="$HACKATHON_PARTITION" \
    --reservation="$HACKATHON_RESERVATION" \
    --job-name=lorenz-team-b \
    --chdir="$(pwd)" \
    --output="$HACKATHON_RUN_ROOT/slurm/lorenz-team-b-%j.out" \
    --error="$HACKATHON_RUN_ROOT/slurm/lorenz-team-b-%j.err" \
    slurm/train_matrix.sbatch \
    configs/matrix_team_b_seeds.txt)

echo "Team B job: $team_job"
```

`afterok` prevents the team job from starting if the shared baseline fails.
Each team keeps all later work in the same dependency chain. Because every
supplied GPU job requests exactly one node, one active chain per team limits
Challenge 3 to at most two nodes. Do not submit an independent debug or
extension job while both team chains are active.

## 7. Monitor jobs and logs

```bash
squeue -j "$baseline_job"
squeue -j "$team_job"

sacct -j "$team_job" \
    --format=JobID,JobName,State,Elapsed,ExitCode

tail -f "$HACKATHON_RUN_ROOT/slurm/lorenz-team-a-${team_job}.out"
```

Team B should replace `team-a` with `team-b` in the log filename. Press
`Ctrl-C` to stop following a log. Cancel a job only when necessary:

```bash
scancel "$team_job"
```

Do not submit evaluation until the training job state is `COMPLETED` with
exit code `0:0`.

## 8. Evaluate the shared baseline

The final argument is an output label, not a physical parameter. It becomes
part of the evaluation directory name.

```bash
baseline_eval_job=$(sbatch --parsable \
    --dependency="afterok:$team_job" \
    --account="$HACKATHON_ACCOUNT" \
    --partition="$HACKATHON_PARTITION" \
    --reservation="$HACKATHON_RESERVATION" \
    --job-name=lorenz-baseline-eval \
    --chdir="$(pwd)" \
    --output="$HACKATHON_RUN_ROOT/slurm/lorenz-baseline-eval-%j.out" \
    --error="$HACKATHON_RUN_ROOT/slurm/lorenz-baseline-eval-%j.err" \
    slurm/evaluate_matrix.sbatch \
    configs/matrix_shared_baseline_seeds.txt \
    data/standard_benchmark.npz \
    shared_baseline_rho28)

echo "Shared baseline evaluation job: $baseline_eval_job"
```

## 9. Evaluate the assigned comparison

Team A:

```bash
team_eval_job=$(sbatch --parsable \
    --dependency="afterok:$baseline_eval_job" \
    --account="$HACKATHON_ACCOUNT" \
    --partition="$HACKATHON_PARTITION" \
    --reservation="$HACKATHON_RESERVATION" \
    --job-name=lorenz-team-a-eval \
    --chdir="$(pwd)" \
    --output="$HACKATHON_RUN_ROOT/slurm/lorenz-team-a-eval-%j.out" \
    --error="$HACKATHON_RUN_ROOT/slurm/lorenz-team-a-eval-%j.err" \
    slurm/evaluate_matrix.sbatch \
    configs/matrix_team_a_seeds.txt \
    data/standard_benchmark.npz \
    team_a_rho28)

echo "Team A evaluation job: $team_eval_job"
```

Team B requires separate changed-dynamics and in-distribution evaluations:

```bash
team_eval_unseen_job=$(sbatch --parsable \
    --dependency="afterok:$baseline_eval_job" \
    --account="$HACKATHON_ACCOUNT" \
    --partition="$HACKATHON_PARTITION" \
    --reservation="$HACKATHON_RESERVATION" \
    --job-name=lorenz-team-b-unseen \
    --chdir="$(pwd)" \
    --output="$HACKATHON_RUN_ROOT/slurm/lorenz-team-b-unseen-%j.out" \
    --error="$HACKATHON_RUN_ROOT/slurm/lorenz-team-b-unseen-%j.err" \
    slurm/evaluate_matrix.sbatch \
    configs/matrix_team_b_seeds.txt \
    data/multirho_benchmark.npz \
    multirho_unseen_rho24_30)

team_eval_rho28_job=$(sbatch --parsable \
    --dependency="afterok:$team_eval_unseen_job" \
    --account="$HACKATHON_ACCOUNT" \
    --partition="$HACKATHON_PARTITION" \
    --reservation="$HACKATHON_RESERVATION" \
    --job-name=lorenz-team-b-rho28 \
    --chdir="$(pwd)" \
    --output="$HACKATHON_RUN_ROOT/slurm/lorenz-team-b-rho28-%j.out" \
    --error="$HACKATHON_RUN_ROOT/slurm/lorenz-team-b-rho28-%j.err" \
    slurm/evaluate_matrix.sbatch \
    configs/matrix_team_b_seeds.txt \
    data/standard_benchmark.npz \
    in_distribution_rho28)

echo "Team B unseen-rho evaluation job: $team_eval_unseen_job"
echo "Team B rho=28 evaluation job: $team_eval_rho28_job"
```

## 10. Find and interpret outputs

Each trained experiment contains:

```text
runs/<experiment>/
├── best_checkpoint.pt
├── history.json
├── resolved_config.json
└── evaluation/<evaluation-label>/
    ├── benchmark_results.json
    └── model_autopsy_rho_*.png

runs/matrix_comparisons/<matrix-name>/<evaluation-label>/
├── matrix_summary.json
├── forecast_comparison_rho_*.png
└── metric_comparison_rho_*.png

runs/matrix_comparisons/matrix_team_b_seeds/three_regime_summary/
├── team_b_three_regime_summary.json
└── team_b_three_regime_summary.png
```

List all generated result files and figures with:

```bash
find runs -type f \
    \( -name 'benchmark_results.json' \
       -o -name 'matrix_summary.json' \
       -o -name 'team_b_three_regime_summary.json' \
       -o -name 'model_autopsy_rho_*.png' \
       -o -name 'forecast_comparison_rho_*.png' \
       -o -name 'metric_comparison_rho_*.png' \
       -o -name 'team_b_three_regime_summary.png' \) \
    -print | sort
```

`benchmark_results.json` contains the complete numerical evaluation and
provenance hashes. `model_autopsy_rho_*.png` contains forecast skill,
long-term phase-space behaviour, perturbation growth, and an x-distribution
comparison for one checkpoint. After the final row, the evaluation command
automatically produces matrix-level forecast and matched-seed diagnostic
figures. `matrix_summary.json` records the per-seed values, population
statistics, missing and censored counts, metric directions, and paired changes
used in those figures.

For Team B, the final evaluation also combines `rho=28`, `rho=30`, and
`rho=24` into `team_b_three_regime_summary.png`. Its top row shows B1/B2
forecast-NRMSE means and population-standard-deviation bands. The lower row
shows selected predictive, stability, sensitivity, and long-term metrics as
mean +/- population standard deviation, plus the number of matched seeds whose
change favors B2. This is a cross-regime reading aid, not a composite score.

If the two evaluation labels were completed in separate sessions and the
combined figure was deferred, regenerate it without rerunning evaluation:

```bash
python -m lorenz_hackathon.compare_team_b_regimes \
    --matrix configs/matrix_team_b_seeds.txt \
    --comparison-root runs/matrix_comparisons/matrix_team_b_seeds \
    --output-dir \
        runs/matrix_comparisons/matrix_team_b_seeds/three_regime_summary
```

## 11. Common failures

| Symptom | Meaning and action |
|---|---|
| `Shared environment not found` | The organizer has not created the shared environment or it is not readable by the project group. |
| `Permission denied` under the shared root | Stop and report the exact path; do not copy the shared environment. |
| Dataset checksum failure | Stop. Do not regenerate or modify participant datasets. |
| `CUDA available: False` on a login node | Expected; use the GPU preflight batch-job result. |
| `Missing checkpoint` during evaluation | Training did not complete successfully or the wrong matrix was supplied. |
| Slurm job state `DEPENDENCY` | The job is waiting for the job named in `--dependency`. |
| Slurm job state `DependencyNeverSatisfied` | An upstream job failed; inspect its log and submit a corrected job chain. |
| Non-finite evaluation metrics | Preserve the result. It is model evidence, not automatically an infrastructure failure. |

For scheduler or environment failures, preserve the job ID and log and report
them to the organizer. Do not count infrastructure failures as scientific
experiments.
