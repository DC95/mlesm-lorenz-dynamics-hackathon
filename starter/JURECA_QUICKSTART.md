# JURECA-DC quick start for organizers and participants

This page records the JURECA software stack rehearsed for the August 2026
event. The versions are pinned in `environment/modules.sh`; update and rehearse
that file before reusing the challenge on a later JSC software stage.

Read the assigned [Team A](../docs/team_a_rollout_fidelity.md) or
[Team B](../docs/team_b_changing_dynamics.md) guide before launching the
mandatory matrix. Record every job in the
[experiment ledger](../docs/templates/experiment_ledger.md).
For a single copy-paste path from login to outputs, use the
[participant command and output reference](../docs/commands_and_outputs.md).

Official system documentation:

- Batch system: <https://apps.fz-juelich.de/jsc/hps/jureca/batchsystem.html>
- GPU computing: <https://apps.fz-juelich.de/jsc/hps/jureca/gpu-computing.html>

## 1. Reproducible shared environment

The minimal environment workflow in `environment/` is adapted from the JSC
[`sc_venv_template`](https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template).
It pins the rehearsed JURECA module stack, creates one project-level virtual
environment, and keeps each checkout's source code branch-local.

The organizer creates the environment once:

```bash
cd starter
bash environment/setup.sh
```

The default shared location is:

```text
/p/project1/training2635/mlesm-lorenz-hackathon-2026/env
```

To reuse the challenge under another project, set `HACKATHON_SHARED_ROOT`
before running `setup.sh` and before every activation.

Participants and the organizer activate from their own checkout:

```bash
cd starter
source environment/activate.sh
export HACKATHON_ACCOUNT=training2635
umask 0027
```

Activation exports `HACKATHON_ACTIVATE` for the supplied Slurm scripts and
prepends the current checkout's `src` directory to `PYTHONPATH`. Team branches
therefore share dependencies without sharing source code. Do not install
packages inside production jobs; the `dc-gpu` production partition does not
provide internet access.

Create checkout-local links to shared immutable data and per-user scratch
outputs once:

```bash
mkdir -p "$HACKATHON_RUN_ROOT/runs" "$HACKATHON_RUN_ROOT/slurm"

if [[ ! -e data && ! -L data ]]; then
    ln -s "$HACKATHON_SHARED_ROOT/data" data
fi

if [[ ! -e runs && ! -L runs ]]; then
    ln -s "$HACKATHON_RUN_ROOT/runs" runs
fi

sha256sum -c data/SHA256SUMS
```

## 2. Smoke test

From the activated starter directory:

```bash
python -m unittest discover -s tests -v
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Repeat the CUDA check in a short development allocation. Use the project account supplied by the organizers.

## 3. Organizer-only dataset regeneration

Participants must use the shared, checksum-verified datasets above. The
following commands are retained only to reproduce a new organizer release:

```bash
mkdir -p data

sbatch --account="$HACKATHON_ACCOUNT" \
  slurm/generate_data.sbatch \
  configs/benchmark_standard.json \
  data/standard_benchmark.npz

sbatch --account="$HACKATHON_ACCOUNT" \
  slurm/generate_data.sbatch \
  configs/benchmark_multirho.json \
  data/multirho_benchmark.npz
```

The organizer must generate, inspect, and checksum both released datasets
before replacing the shared copies.

## 4. Train matched seed matrices on four GPUs

JURECA-DC GPU nodes contain four A100 GPUs and node allocations are exclusive. The supplied job starts four workers, each assigned one GPU. Six-row team matrices are processed in strides: after a worker finishes one row, it takes the next available row assigned to it.

```bash
# Shared A0 baseline at three seeds plus the linear reference
sbatch --account="$HACKATHON_ACCOUNT" \
  slurm/train_matrix.sbatch \
  configs/matrix_shared_baseline_seeds.txt

# Team A after the shared baseline: three A1 experiments
sbatch --account="$HACKATHON_ACCOUNT" \
  slurm/train_matrix.sbatch \
  configs/matrix_team_a_a1_only_seeds.txt

# Team B mandatory comparison: six experiments
sbatch --account="$HACKATHON_ACCOUNT" \
  slurm/train_matrix.sbatch \
  configs/matrix_team_b_seeds.txt
```

Each non-comment matrix row contains:

```text
TRAIN_CONFIG SEED OUTPUT_DIRECTORY
```

Create a new matrix for extensions. Do not silently edit a frozen organizer matrix after results have been reported.

## 5. Evaluate complete matrices

```bash
# Shared persistence, linear, and direct-MLP baselines at rho=28
sbatch --account="$HACKATHON_ACCOUNT" \
  slurm/evaluate_matrix.sbatch \
  configs/matrix_shared_baseline_seeds.txt \
  data/standard_benchmark.npz \
  shared_baseline_rho28

# Team A at rho=28
sbatch --account="$HACKATHON_ACCOUNT" \
  slurm/evaluate_matrix.sbatch \
  configs/matrix_team_a_seeds.txt \
  data/standard_benchmark.npz \
  team_a_rho28

# Team B on rho=24 and rho=30
sbatch --account="$HACKATHON_ACCOUNT" \
  slurm/evaluate_matrix.sbatch \
  configs/matrix_team_b_seeds.txt \
  data/multirho_benchmark.npz \
  multirho_unseen_rho24_30

# Team B in-distribution check at rho=28
sbatch --account="$HACKATHON_ACCOUNT" \
  slurm/evaluate_matrix.sbatch \
  configs/matrix_team_b_seeds.txt \
  data/standard_benchmark.npz \
  in_distribution_rho28
```

The supplied Slurm limits and evaluation lengths are the rehearsed settings.
Do not shorten scientific diagnostics to make a job finish sooner; report any
scheduler or infrastructure failure to the organizer.

Metric meanings, failure policies, and the minimum reporting standard are in
the [evaluation evidence guide](../docs/evaluation_evidence_guide.md).

The single-checkpoint `slurm/evaluate.sbatch` remains available for debugging.

## 6. Cluster-use rules

- Never train or perform heavy evaluation on login nodes.
- Never omit an explicit GPU request from GPU jobs.
- Never download packages or datasets from a production compute node.
- Preserve the Slurm job ID in the experiment ledger.
- Use one shared copy of each dataset.
- Submit coordinated experiment matrices instead of separate one-GPU allocations.
- Treat environment or scheduler failures as infrastructure failures, not as part of a team's experiment allowance.
