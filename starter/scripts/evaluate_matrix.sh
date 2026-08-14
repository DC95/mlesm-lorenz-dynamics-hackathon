#!/bin/bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "Usage: evaluate_matrix.sh MATRIX_FILE DATA_FILE [EVALUATION_LABEL]" >&2
    exit 2
fi

matrix_file=$1
data_path=$2
evaluation_label=${3:-benchmark}

while read -r config_path experiment_seed output_directory extra; do
    if [[ -z "${config_path:-}" || "$config_path" == \#* ]]; then
        continue
    fi
    if [[ -n "${extra:-}" ]]; then
        echo "Invalid matrix row: expected CONFIG SEED OUTPUT_DIRECTORY" >&2
        exit 2
    fi

    checkpoint_path="$output_directory/best_checkpoint.pt"
    if [[ ! -f "$checkpoint_path" ]]; then
        echo "Missing checkpoint: $checkpoint_path" >&2
        exit 1
    fi

    evaluation_directory="$output_directory/evaluation/$evaluation_label"
    echo "Evaluating config=$config_path seed=$experiment_seed data=$data_path"
    python -m lorenz_hackathon.evaluate \
        --checkpoint "$checkpoint_path" \
        --data "$data_path" \
        --output-dir "$evaluation_directory"
done < "$matrix_file"
