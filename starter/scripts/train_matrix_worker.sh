#!/bin/bash
set -euo pipefail

matrix_file=${1:?Usage: train_matrix_worker.sh MATRIX_FILE}
mapfile -t experiment_rows < <(awk 'NF && $1 !~ /^#/' "$matrix_file")
worker_index=${SLURM_PROCID:-0}

if (( worker_index >= ${#experiment_rows[@]} )); then
    echo "No experiment row for worker ${worker_index}" >&2
    exit 2
fi

read -r config_path experiment_seed output_directory <<< "${experiment_rows[$worker_index]}"

echo "worker=${worker_index} gpu=${CUDA_VISIBLE_DEVICES:-unset} config=${config_path} seed=${experiment_seed}"
python -m lorenz_hackathon.train \
    --config "$config_path" \
    --seed "$experiment_seed" \
    --output-dir "$output_directory"

