# Local computer and Google Colab quick start

The benchmark does not require an HPC system. Training and evaluation use
PyTorch's automatic device selection: CUDA is used when it is available, and
otherwise the same code runs on the CPU. The configurations, seeds, frozen
data, metrics, and output paths are identical across runtimes.

## Local computer

Python 3.10 or newer and an internet connection for the initial package
installation are required. From a terminal:

```bash
git clone https://github.com/DC95/mlesm-lorenz-dynamics-hackathon.git
cd mlesm-lorenz-dynamics-hackathon
git switch team-b-changing-dynamics
cd starter

bash scripts/select_runtime.sh local
bash scripts/run.sh prepare
bash scripts/run.sh team-b
```

`prepare` creates `starter/.venv`, installs the package, generates the two
frozen deterministic datasets, verifies their released SHA-256 checksums, and
runs the contract tests. `team-b` then trains B1 and B2 at seeds 41--43 and
runs the `rho=28`, `rho=30`, and `rho=24` evaluations sequentially. Keep the
terminal open until it finishes.

## Google Colab

A GPU runtime is optional because the models are small. In a new Colab
notebook, run these cells:

```python
!git clone https://github.com/DC95/mlesm-lorenz-dynamics-hackathon.git
%cd /content/mlesm-lorenz-dynamics-hackathon
!git switch team-b-changing-dynamics
%cd starter
```

```python
!bash scripts/select_runtime.sh colab
!bash scripts/run.sh prepare
```

```python
!bash scripts/run.sh team-b
```

Colab may disconnect, so download the complete `starter/runs/` directory after
evaluation. If a session disconnects after training, rerun the clone and
prepare cells, restore `runs/`, and invoke the two `evaluate` commands shown
below instead of retraining.

## Run individual stages

The same wrapper works for local, Colab, and JURECA. Local and Colab execute
immediately; JURECA returns a Slurm job ID.

```bash
bash scripts/run.sh train configs/matrix_team_b_seeds.txt

bash scripts/run.sh evaluate \
    configs/matrix_team_b_seeds.txt \
    data/standard_benchmark.npz \
    in_distribution_rho28

bash scripts/run.sh evaluate \
    configs/matrix_team_b_seeds.txt \
    data/multirho_benchmark.npz \
    multirho_unseen_rho24_30
```

Inspect the selected device at any time:

```bash
bash scripts/run.sh status
```

Final presentation outputs are written under:

```text
runs/matrix_comparisons/matrix_team_b_seeds/three_regime_summary/
├── team_b_three_regime_summary.json
└── team_b_three_regime_summary.png
```
