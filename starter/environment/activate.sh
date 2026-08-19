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

# Resolve imports from this checkout, not from another participant checkout or
# an editable install stored in the shared environment. The current checkout
# must be first even when its path already appears later in PYTHONPATH after a
# user switches between clones or branches in the same shell.
pythonpath_entries=()
if [[ -n "${PYTHONPATH:-}" ]]; then
    IFS=: read -r -a pythonpath_entries <<< "${PYTHONPATH}"
fi
new_pythonpath="${starter_dir}/src"
for entry in "${pythonpath_entries[@]}"; do
    if [[ -z "${entry}" || "${entry}" == "${starter_dir}/src" ]]; then
        continue
    fi
    new_pythonpath="${new_pythonpath}:${entry}"
done
export PYTHONPATH="${new_pythonpath}"
unset entry new_pythonpath pythonpath_entries

# Use a non-interactive backend inside Slurm jobs.
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HACKATHON_ACTIVATE="${script_dir}/activate.sh"

echo "Activated ${HACKATHON_ENV_NAME}"
echo "Python source: ${starter_dir}/src"
echo "Run root: ${HACKATHON_RUN_ROOT}"
echo "Slurm account: ${HACKATHON_ACCOUNT}"
echo "Slurm partition: ${HACKATHON_PARTITION}"
if [[ -n "${HACKATHON_RESERVATION}" ]]; then
    echo "Slurm reservation: ${HACKATHON_RESERVATION}"
else
    echo "No event reservation selected for $(date +%F)." >&2
fi
