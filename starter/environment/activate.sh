#!/bin/bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced: source environment/activate.sh" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
starter_dir=$(cd -- "${script_dir}/.." && pwd)

source "${script_dir}/config.sh"

if declare -F deactivate >/dev/null 2>&1; then
    deactivate
fi

source "${script_dir}/modules.sh" || return 2

if [[ ! -f "${HACKATHON_ENV_DIR}/bin/activate" ]]; then
    echo "Shared environment not found: ${HACKATHON_ENV_DIR}" >&2
    echo "The organizer must first run: bash environment/setup.sh" >&2
    return 2
fi

source "${HACKATHON_ENV_DIR}/bin/activate"

# Resolve imports from this checkout, not from the organizer's checkout or an
# editable install stored in the shared environment. This preserves branch
# isolation for the two teams.
case ":${PYTHONPATH:-}:" in
    *":${starter_dir}/src:"*) ;;
    *) export PYTHONPATH="${starter_dir}/src${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# Use a non-interactive backend inside Slurm jobs.
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HACKATHON_ACTIVATE="${script_dir}/activate.sh"

echo "Activated ${HACKATHON_ENV_NAME}"
echo "Python source: ${starter_dir}/src"
echo "Run root: ${HACKATHON_RUN_ROOT}"
