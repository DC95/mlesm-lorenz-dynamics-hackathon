#!/bin/bash -x
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

matrix_file=${1:-configs/matrix_shared_baseline_seeds.txt}

if [[ -z "${HACKATHON_ACTIVATE:-}" ]]; then
    echo "Set HACKATHON_ACTIVATE to the organizer-provided activation script." >&2
    exit 2
fi

mkdir -p runs
source "$HACKATHON_ACTIVATE"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count in allocation shell:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY

srun --exclusive bash scripts/train_matrix_worker.sh "$matrix_file"
