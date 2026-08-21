#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
starter_dir=$(cd -- "${script_dir}/.." && pwd)
runtime_file="${starter_dir}/.hackathon-runtime"
cd "$starter_dir"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash scripts/run.sh prepare
  bash scripts/run.sh status
  bash scripts/run.sh team-b
  bash scripts/run.sh train MATRIX_FILE [AFTER_JOB_ID]
  bash scripts/run.sh evaluate MATRIX_FILE DATA_FILE LABEL [AFTER_JOB_ID]

First choose a runtime with:
  bash scripts/select_runtime.sh {local|colab|jureca}
EOF
}

read_runtime() {
    if [[ -n "${LORENZ_RUNTIME:-}" ]]; then
        printf '%s\n' "$LORENZ_RUNTIME"
    elif [[ -r "$runtime_file" ]]; then
        tr -d '[:space:]' < "$runtime_file"
    else
        echo "No runtime selected." >&2
        echo "Run: bash scripts/select_runtime.sh {local|colab|jureca}" >&2
        return 2
    fi
}

runtime=$(read_runtime)
case "$runtime" in
    local|colab|jureca) ;;
    *) echo "Invalid selected runtime: $runtime" >&2; exit 2 ;;
esac

activate_portable() {
    if [[ "$runtime" == "local" ]]; then
        if [[ ! -f .venv/bin/activate ]]; then
            echo "Local environment is not prepared." >&2
            echo "Run: bash scripts/run.sh prepare" >&2
            return 2
        fi
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
    export PYTHONPATH="${starter_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
    export MPLBACKEND="${MPLBACKEND:-Agg}"
}

activate_jureca() {
    umask 0027
    # Activation messages belong in the terminal, while submission functions
    # reserve stdout for the parsable Slurm job ID.
    # shellcheck disable=SC1091
    source environment/activate.sh >&2
    if [[ -z "${HACKATHON_RESERVATION:-}" ]]; then
        echo "No active event reservation is selected." >&2
        return 2
    fi
}

prepare_portable() {
    if [[ "$runtime" == "local" ]]; then
        if [[ ! -f .venv/bin/activate ]]; then
            python3 -m venv .venv
        fi
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi

    python -m pip install --editable .
    export PYTHONPATH="${starter_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
    export MPLBACKEND="${MPLBACKEND:-Agg}"

    mkdir -p data runs
    if [[ ! -f data/standard_benchmark.npz ]]; then
        python -m lorenz_hackathon.data \
            --config configs/benchmark_standard.json \
            --output data/standard_benchmark.npz
    fi
    if [[ ! -f data/multirho_benchmark.npz ]]; then
        python -m lorenz_hackathon.data \
            --config configs/benchmark_multirho.json \
            --output data/multirho_benchmark.npz
    fi
    cp configs/frozen_data_SHA256SUMS data/SHA256SUMS
    (cd data && sha256sum -c SHA256SUMS)
    python -m unittest discover -s tests -v
}

show_status() {
    python - <<'PY'
import platform
import sys

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("Selected compute device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
PY
}

dependency_arguments() {
    local after_job=${1:-}
    if [[ -n "$after_job" ]]; then
        printf '%s\n' "--dependency=afterok:${after_job%%;*}"
    fi
}

submit_train() {
    local matrix_file=$1
    local after_job=${2:-}
    local matrix_name
    local -a dependency=()
    matrix_name=$(basename "$matrix_file")
    matrix_name=${matrix_name%.*}
    if [[ -n "$after_job" ]]; then
        mapfile -t dependency < <(dependency_arguments "$after_job")
    fi
    mkdir -p "$HACKATHON_RUN_ROOT/slurm"
    sbatch --parsable \
        --account="$HACKATHON_ACCOUNT" \
        --partition="$HACKATHON_PARTITION" \
        --reservation="$HACKATHON_RESERVATION" \
        "${dependency[@]}" \
        --job-name="lorenz-${matrix_name}" \
        --chdir="$starter_dir" \
        --output="$HACKATHON_RUN_ROOT/slurm/lorenz-${matrix_name}-%j.out" \
        --error="$HACKATHON_RUN_ROOT/slurm/lorenz-${matrix_name}-%j.err" \
        slurm/train_matrix.sbatch "$matrix_file"
}

submit_evaluation() {
    local matrix_file=$1
    local data_file=$2
    local label=$3
    local after_job=${4:-}
    local -a dependency=()
    if [[ -n "$after_job" ]]; then
        mapfile -t dependency < <(dependency_arguments "$after_job")
    fi
    mkdir -p "$HACKATHON_RUN_ROOT/slurm"
    sbatch --parsable \
        --account="$HACKATHON_ACCOUNT" \
        --partition="$HACKATHON_PARTITION" \
        --reservation="$HACKATHON_RESERVATION" \
        "${dependency[@]}" \
        --job-name="lorenz-${label}" \
        --chdir="$starter_dir" \
        --output="$HACKATHON_RUN_ROOT/slurm/lorenz-${label}-%j.out" \
        --error="$HACKATHON_RUN_ROOT/slurm/lorenz-${label}-%j.err" \
        slurm/evaluate_matrix.sbatch "$matrix_file" "$data_file" "$label"
}

run_train() {
    local matrix_file=$1
    local after_job=${2:-}
    if [[ "$runtime" == "jureca" ]]; then
        submit_train "$matrix_file" "$after_job"
    else
        if [[ -n "$after_job" ]]; then
            echo "AFTER_JOB_ID is only meaningful on JURECA." >&2
            return 2
        fi
        bash scripts/train_matrix_worker.sh "$matrix_file"
    fi
}

run_evaluation() {
    local matrix_file=$1
    local data_file=$2
    local label=$3
    local after_job=${4:-}
    if [[ "$runtime" == "jureca" ]]; then
        submit_evaluation "$matrix_file" "$data_file" "$label" "$after_job"
    else
        if [[ -n "$after_job" ]]; then
            echo "AFTER_JOB_ID is only meaningful on JURECA." >&2
            return 2
        fi
        bash scripts/evaluate_matrix.sh "$matrix_file" "$data_file" "$label"
    fi
}

run_team_b() {
    local matrix=configs/matrix_team_b_seeds.txt
    if [[ "$runtime" == "jureca" ]]; then
        local train_job rho28_job changed_job
        train_job=$(submit_train "$matrix")
        rho28_job=$(submit_evaluation \
            "$matrix" data/standard_benchmark.npz \
            in_distribution_rho28 "$train_job")
        changed_job=$(submit_evaluation \
            "$matrix" data/multirho_benchmark.npz \
            multirho_unseen_rho24_30 "$rho28_job")
        echo "Team B training job: ${train_job%%;*}"
        echo "rho=28 evaluation job: ${rho28_job%%;*}"
        echo "rho=30/24 evaluation job: ${changed_job%%;*}"
        echo "The jobs are serialized and use at most one node at a time."
    else
        bash scripts/train_matrix_worker.sh "$matrix"
        bash scripts/evaluate_matrix.sh \
            "$matrix" data/standard_benchmark.npz in_distribution_rho28
        bash scripts/evaluate_matrix.sh \
            "$matrix" data/multirho_benchmark.npz multirho_unseen_rho24_30
    fi
}

command=${1:-}
case "$command" in
    prepare)
        if (( $# != 1 )); then usage; exit 2; fi
        if [[ "$runtime" == "jureca" ]]; then
            activate_jureca
            bash scripts/preflight_login.sh
        else
            prepare_portable
            show_status
        fi
        ;;
    status)
        if (( $# != 1 )); then usage; exit 2; fi
        if [[ "$runtime" == "jureca" ]]; then activate_jureca; else activate_portable; fi
        echo "Selected runtime: $runtime"
        show_status
        ;;
    team-b)
        if (( $# != 1 )); then usage; exit 2; fi
        if [[ "$runtime" == "jureca" ]]; then activate_jureca; else activate_portable; fi
        run_team_b
        ;;
    train)
        if (( $# < 2 || $# > 3 )); then usage; exit 2; fi
        if [[ "$runtime" == "jureca" ]]; then activate_jureca; else activate_portable; fi
        run_train "$2" "${3:-}"
        ;;
    evaluate)
        if (( $# < 4 || $# > 5 )); then usage; exit 2; fi
        if [[ "$runtime" == "jureca" ]]; then activate_jureca; else activate_portable; fi
        run_evaluation "$2" "$3" "$4" "${5:-}"
        ;;
    *) usage; exit 2 ;;
esac
