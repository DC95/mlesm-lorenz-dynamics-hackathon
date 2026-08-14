#!/bin/bash -x
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

matrix_file=${1:?Usage: sbatch evaluate_matrix.sbatch MATRIX DATA [LABEL]}
data_path=${2:?Usage: sbatch evaluate_matrix.sbatch MATRIX DATA [LABEL]}
evaluation_label=${3:-benchmark}

if [[ -z "${HACKATHON_ACTIVATE:-}" ]]; then
    echo "Set HACKATHON_ACTIVATE to the organizer-provided activation script." >&2
    exit 2
fi

source "$HACKATHON_ACTIVATE"
bash scripts/evaluate_matrix.sh "$matrix_file" "$data_path" "$evaluation_label"
