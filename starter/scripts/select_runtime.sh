#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
starter_dir=$(cd -- "${script_dir}/.." && pwd)
runtime_file="${starter_dir}/.hackathon-runtime"
runtime=${1:-}

if [[ -z "$runtime" ]]; then
    if [[ ! -t 0 ]]; then
        echo "Usage: bash scripts/select_runtime.sh {local|colab|jureca}" >&2
        exit 2
    fi
    echo "Where will you run the Lorenz workflow?"
    echo "  1) Local computer (CPU or local GPU)"
    echo "  2) Google Colab"
    echo "  3) JURECA"
    read -r -p "Choose 1, 2, or 3: " selection
    case "$selection" in
        1) runtime=local ;;
        2) runtime=colab ;;
        3) runtime=jureca ;;
        *) echo "Invalid selection: $selection" >&2; exit 2 ;;
    esac
fi

case "$runtime" in
    local|colab|jureca) ;;
    *)
        echo "Unknown runtime '$runtime'; choose local, colab, or jureca." >&2
        exit 2
        ;;
esac

printf '%s\n' "$runtime" > "$runtime_file"
echo "Selected runtime: $runtime"
echo "Next: bash scripts/run.sh prepare"
