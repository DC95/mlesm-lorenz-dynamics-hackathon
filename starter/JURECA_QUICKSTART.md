# JURECA-DC quick start for organizers and participants

This page deliberately avoids hard-coding a JURECA software stage or module version. The organizer must create and rehearse one shared environment before the event, then give every participant the same activation script.

Official system documentation:

- Batch system: <https://apps.fz-juelich.de/jsc/hps/jureca/batchsystem.html>
- GPU computing: <https://apps.fz-juelich.de/jsc/hps/jureca/gpu-computing.html>

## 1. Organizer environment

Prepare a shared environment containing Python 3.10 or newer, NumPy, CUDA-enabled PyTorch, Matplotlib, and this starter package installed in editable mode. Participants set:

```bash
export HACKATHON_ACTIVATE=/absolute/path/to/activate_hackathon_env.sh
```

Do not install packages inside production jobs; the `dc-gpu` production partition does not provide internet access.

## 2. Smoke test

From the starter directory:

```bash
source "$HACKATHON_ACTIVATE"
python -m unittest discover -s tests -v
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Repeat the CUDA check in a short development allocation. Use the project account supplied by the organizers.

## 3. Generate and freeze the datasets

```bash
mkdir -p data

sbatch --account=PROJECT_ACCOUNT \
  slurm/generate_data.sbatch \
  configs/benchmark_standard.json \
  data/standard_benchmark.npz

sbatch --account=PROJECT_ACCOUNT \
  slurm/generate_data.sbatch \
  configs/benchmark_multirho.json \
  data/multirho_benchmark.npz
```

The organizer must generate, inspect, and checksum both released datasets. Participants should use the shared frozen copies instead of regenerating them.

## 4. Train matched seed matrices on four GPUs

JURECA-DC GPU nodes contain four A100 GPUs and node allocations are exclusive. The supplied job starts four workers, each assigned one GPU. Six-row team matrices are processed in strides: after a worker finishes one row, it takes the next available row assigned to it.

```bash
# Shared A0 baseline at three seeds plus the linear reference
sbatch --account=PROJECT_ACCOUNT \
  slurm/train_matrix.sbatch \
  configs/matrix_shared_baseline_seeds.txt

# Team A mandatory comparison: six experiments
sbatch --account=PROJECT_ACCOUNT \
  slurm/train_matrix.sbatch \
  configs/matrix_team_a_seeds.txt

# Team B mandatory comparison: six experiments
sbatch --account=PROJECT_ACCOUNT \
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
sbatch --account=PROJECT_ACCOUNT \
  slurm/evaluate_matrix.sbatch \
  configs/matrix_shared_baseline_seeds.txt \
  data/standard_benchmark.npz \
  standard_rho28

# Team A at rho=28
sbatch --account=PROJECT_ACCOUNT \
  slurm/evaluate_matrix.sbatch \
  configs/matrix_team_a_seeds.txt \
  data/standard_benchmark.npz \
  standard_rho28

# Team B on rho=24 and rho=30
sbatch --account=PROJECT_ACCOUNT \
  slurm/evaluate_matrix.sbatch \
  configs/matrix_team_b_seeds.txt \
  data/multirho_benchmark.npz \
  changed_dynamics

# Team B in-distribution check at rho=28
sbatch --account=PROJECT_ACCOUNT \
  slurm/evaluate_matrix.sbatch \
  configs/matrix_team_b_seeds.txt \
  data/standard_benchmark.npz \
  in_distribution_rho28
```

The current development-job wall time is provisional. If the complete matrix evaluation exceeds it, use a rehearsed production allocation rather than shortening the scientific diagnostics during the event.

The single-checkpoint `slurm/evaluate.sbatch` remains available for debugging.

## 6. Cluster-use rules

- Never train or perform heavy evaluation on login nodes.
- Never omit an explicit GPU request from GPU jobs.
- Never download packages or datasets from a production compute node.
- Preserve the Slurm job ID in the experiment ledger.
- Use one shared copy of each dataset.
- Submit coordinated experiment matrices instead of separate one-GPU allocations.
- Treat environment or scheduler failures as infrastructure failures, not as part of a team's experiment allowance.
