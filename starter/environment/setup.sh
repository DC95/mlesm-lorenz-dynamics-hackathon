#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

source "${script_dir}/config.sh"

if declare -F deactivate >/dev/null 2>&1; then
    deactivate
fi

source "${script_dir}/modules.sh"

if [[ -e "${HACKATHON_ENV_DIR}" ]]; then
    echo "Refusing to overwrite existing environment: ${HACKATHON_ENV_DIR}" >&2
    echo "Set HACKATHON_SHARED_ROOT to a new path or remove the old environment deliberately." >&2
    exit 2
fi

# The environment is shared read-only with the project group. The repository
# source is deliberately not installed here; activate.sh selects the source
# from the clone and branch from which it is sourced.
umask 0027
mkdir -p "${HACKATHON_SHARED_ROOT}"
python -m venv \
    --prompt "${HACKATHON_ENV_NAME}" \
    --system-site-packages \
    "${HACKATHON_ENV_DIR}"

source "${HACKATHON_ENV_DIR}/bin/activate"

if grep -Eq '^[[:space:]]*[^#[:space:]]' "${script_dir}/requirements.txt"; then
    python -m pip install --requirement "${script_dir}/requirements.txt"
fi

python - <<'PY'
import sys

import matplotlib
import numpy
import torch

if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10 or newer is required; found {sys.version.split()[0]}")

print("Environment created successfully")
print("Python:", sys.version.split()[0])
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("Matplotlib:", matplotlib.__version__)
PY

echo "Environment path: ${HACKATHON_ENV_DIR}"
echo "Activate from a starter checkout with: source environment/activate.sh"
