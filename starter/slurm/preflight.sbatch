#!/bin/bash -x
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ -z "${HACKATHON_ACTIVATE:-}" ]]; then
    echo "Set HACKATHON_ACTIVATE by sourcing environment/activate.sh before submission." >&2
    exit 2
fi

source "${HACKATHON_ACTIVATE}"

python - <<'PY'
import json
from pathlib import Path

import torch

from lorenz_hackathon.models import build_model

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the GPU allocation")

config = json.loads(Path("configs/train_a0_direct_onestep.json").read_text())
device = torch.device("cuda")
model = build_model(config["model"]).to(device)
inputs = torch.zeros((8, 3), dtype=torch.float32, device=device)
outputs = model(inputs)
loss = outputs.square().mean()
loss.backward()

if outputs.shape != inputs.shape:
    raise SystemExit(f"Unexpected output shape: {tuple(outputs.shape)}")
if not torch.isfinite(outputs).all():
    raise SystemExit("Model forward pass produced non-finite values")

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("GPU forward/backward pass: passed")
PY

echo "GPU preflight passed."
