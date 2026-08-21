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

matrix_filename=$(basename "$matrix_file")
matrix_name=${matrix_filename%.*}
comparison_directory="runs/matrix_comparisons/$matrix_name/$evaluation_label"

python -m lorenz_hackathon.compare_matrix \
    --matrix "$matrix_file" \
    --evaluation-label "$evaluation_label" \
    --output-dir "$comparison_directory"

# Team B's in-distribution and changed-dynamics evaluations are separate jobs.
# Once both matrix summaries exist, combine them into one presentation-ready
# three-regime overview. A stale result from an earlier run may temporarily
# fail the checkpoint-consistency check; defer without invalidating the newly
# completed evaluation because the second current evaluation will retry.
if [[ "$matrix_name" == "matrix_team_b_seeds" ]]; then
    team_b_comparison_root="runs/matrix_comparisons/$matrix_name"
    rho28_summary="$team_b_comparison_root/in_distribution_rho28/matrix_summary.json"
    changed_summary="$team_b_comparison_root/multirho_unseen_rho24_30/matrix_summary.json"
    combined_directory="$team_b_comparison_root/three_regime_summary"

    if [[ -f "$rho28_summary" && -f "$changed_summary" ]]; then
        if ! python -m lorenz_hackathon.compare_team_b_regimes \
            --matrix "$matrix_file" \
            --comparison-root "$team_b_comparison_root" \
            --output-dir "$combined_directory"
        then
            echo "Team B three-regime summary deferred until both current evaluations are complete." >&2
        fi
    else
        echo "Team B three-regime summary deferred until both evaluation labels are complete."
    fi
fi
