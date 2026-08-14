# JURECA-DC quick start for organizers and participants

This document deliberately avoids hard-coding a JURECA software-stage or module version. The organizer should create and test one shared environment before the event, then give every participant the same activation script.

Official system documentation:

- Batch system: <https://apps.fz-juelich.de/jsc/hps/jureca/batchsystem.html>
- GPU computing: <https://apps.fz-juelich.de/jsc/hps/jureca/gpu-computing.html>

## 1. Organizer preparation

Create a shared environment containing at minimum:

- Python 3.10 or newer;
- NumPy;
- PyTorch with CUDA support;
- Matplotlib;
- the local starter package installed in editable mode.

Create an activation script, for example in the project directory, that loads the tested JURECA software stack and activates the environment. Participants should set:

```bash
export HACKATHON_ACTIVATE=/absolute/path/to/activate_hackathon_env.sh
```

Do not place package installation commands inside production jobs. The `dc-gpu` production partition does not provide internet access.

## 2. First smoke test

From the starter-package directory:

```bash
source "$HACKATHON_ACTIVATE"
python -m unittest discover -s tests -v
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then request a short interactive development allocation and repeat the CUDA check on the compute node. Use the project account supplied by the hackathon organizers.

## 3. Generate public data

The data generation is CPU-based:

```bash
mkdir -p data
sbatch --account=PROJECT_ACCOUNT \
  slurm/generate_data.sbatch \
  configs/benchmark_standard.json \
  data/standard_benchmark.npz
```

For Team B:

```bash
sbatch --account=PROJECT_ACCOUNT \
  slurm/generate_data.sbatch \
  configs/benchmark_conditioned.json \
  data/conditioned_benchmark.npz
```

The organizer should generate and checksum the released datasets before the event. Participants should not independently regenerate nominally identical public data during the challenge.

## 4. Use all four GPUs in a production allocation

JURECA-DC GPU nodes contain four A100 GPUs and node allocations are exclusive. The provided matrix job launches four independent training processes, with one Slurm task assigned to each GPU:

```bash
sbatch --account=PROJECT_ACCOUNT \
  slurm/train_matrix.sbatch \
  configs/matrix_direct_seeds.txt
```

For the parameter-conditioned track:

```bash
sbatch --account=PROJECT_ACCOUNT \
  slurm/train_matrix.sbatch \
  configs/matrix_conditioned_seeds.txt
```

Each non-comment matrix row contains:

```text
TRAIN_CONFIG SEED OUTPUT_DIRECTORY
```

Keep exactly four active rows for the supplied four-task job. Create a new matrix file rather than silently editing the organizer baseline matrix after results have been reported.

## 5. Evaluation

Example:

```bash
sbatch --account=PROJECT_ACCOUNT \
  slurm/evaluate.sbatch \
  runs/direct_seed42/best_checkpoint.pt \
  data/standard_benchmark.npz \
  runs/direct_seed42/evaluation
```

The development partition is suitable for short evaluation and debugging. If evaluation runtime grows beyond the development limit, move coordinated evaluation jobs to a production allocation and use all allocated GPUs.

## 6. Event rules for cluster use

- Never train or perform heavy evaluation on login nodes.
- Never omit an explicit GPU request from GPU jobs.
- Never download packages or datasets from a production compute node.
- Preserve the Slurm job ID in the experiment ledger.
- Use one shared data copy rather than duplicating benchmark files per participant.
- Submit coordinated four-experiment matrices instead of four separate one-GPU allocations.
- Treat environment or scheduler failures as infrastructure failures, not as part of a team's experiment allowance.
