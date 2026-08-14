# Lorenz Dynamics Hackathon Starter

This starter package implements the common scientific benchmark for the MLESM hackathon challenge **When Is an AI Weather Model Dynamically Trustworthy?**

It intentionally separates:

- the high-accuracy RK4 reference system;
- trajectory-disjoint data generation;
- direct, residual, one-step, multi-step, and parameter-conditioned emulators;
- a shared evaluation harness that measures forecast skill, stability, perturbation growth, and long-term statistics.

The supplied MLP is a baseline. The scientific objective is to explain when and why learned emulators reproduce or fail to reproduce the underlying dynamics.

## 1. Create the environment

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

On JURECA, use the organizer-provided environment and Slurm scripts instead of downloading packages inside a production job.

See [JURECA_QUICKSTART.md](JURECA_QUICKSTART.md) for the shared-environment and four-GPU matrix workflow.

## 2. Run the tests

```bash
python -m unittest discover -s tests -v
```

## 3. Generate the standard benchmark

```bash
python -m lorenz_hackathon.data \
  --config configs/benchmark_standard.json \
  --output data/standard_benchmark.npz
```

The generator uses independent trajectory seeds for training, validation, and public test data. It does not randomly mix individual time steps across splits.

## 4. Train the organizer baseline

```bash
python -m lorenz_hackathon.train --config configs/train_linear.json
python -m lorenz_hackathon.train --config configs/train_direct.json
```

## 5. Run the scientific evaluation

```bash
python -m lorenz_hackathon.evaluate \
  --checkpoint runs/direct_seed42/best_checkpoint.pt \
  --data data/standard_benchmark.npz \
  --output-dir runs/direct_seed42/evaluation
```

The evaluation writes:

- `benchmark_results.json`;
- one four-panel `model_autopsy_rho_*.png` per public-test rho value.

## 6. Team A starting comparison

Train the direct one-step baseline and the residual multi-step configuration:

```bash
python -m lorenz_hackathon.train --config configs/train_direct.json
python -m lorenz_hackathon.train --config configs/train_residual_multistep.json
```

This is a starting comparison, not a guaranteed fair final ablation. Team A must check whether parameter counts, training examples, optimization effort, and model selection remain comparable.

## 7. Team B starting comparison

Generate the multi-parameter dataset and train the conditioned model:

```bash
python -m lorenz_hackathon.data \
  --config configs/benchmark_conditioned.json \
  --output data/conditioned_benchmark.npz

python -m lorenz_hackathon.train --config configs/train_conditioned.json

python -m lorenz_hackathon.evaluate \
  --checkpoint runs/conditioned_seed42/best_checkpoint.pt \
  --data data/conditioned_benchmark.npz \
  --output-dir runs/conditioned_seed42/evaluation
```

The public test contains `rho = 24`, `30`, and `35`. The conditioned training set contains `rho = 26`, `28`, and `32`.

## 8. Reproducibility rules

- Never use public-test results to recompute normalization statistics.
- Never mix states from one physical trajectory between training and validation.
- Keep the configuration, seed, checkpoint, and result file together.
- Report any clipping, projection, early termination, or post-processing applied during rollout.
- Evaluate each principal configuration over at least three training seeds.
- Preserve failed or inconclusive experiments in the experiment ledger.
