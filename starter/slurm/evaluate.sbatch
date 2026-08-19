#!/bin/bash -x
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

checkpoint_path=${1:?Usage: sbatch evaluate.sbatch CHECKPOINT DATA OUTPUT_DIR}
data_path=${2:?Usage: sbatch evaluate.sbatch CHECKPOINT DATA OUTPUT_DIR}
output_directory=${3:?Usage: sbatch evaluate.sbatch CHECKPOINT DATA OUTPUT_DIR}

if [[ -z "${HACKATHON_ACTIVATE:-}" ]]; then
    echo "Set HACKATHON_ACTIVATE to the organizer-provided activation script." >&2
    exit 2
fi

mkdir -p "$output_directory"
source "$HACKATHON_ACTIVATE"

python -m lorenz_hackathon.evaluate \
    --checkpoint "$checkpoint_path" \
    --data "$data_path" \
    --output-dir "$output_directory"
