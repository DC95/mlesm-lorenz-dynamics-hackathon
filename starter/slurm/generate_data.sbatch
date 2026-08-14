#!/bin/bash -x
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

benchmark_config=${1:-configs/benchmark_standard.json}
output_path=${2:-data/standard_benchmark.npz}

if [[ -z "${HACKATHON_ACTIVATE:-}" ]]; then
    echo "Set HACKATHON_ACTIVATE to the organizer-provided activation script." >&2
    exit 2
fi

mkdir -p data
source "$HACKATHON_ACTIVATE"

python -m lorenz_hackathon.data \
    --config "$benchmark_config" \
    --output "$output_path"
