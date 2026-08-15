# Lorenz Dynamics Hackathon Starter

This package implements the shared benchmark for **Beyond Predictive Skill: What Evidence Supports Claims of Learned Dynamics?**

The organizer supplies a high-accuracy RK4 reference, frozen trajectory-disjoint datasets, persistence and linear baselines, a direct one-step MLP, and one evaluation interface. Participants use these common foundations to test one controlled scientific intervention per team.

| Comparison | Fixed | Intended difference |
|---|---|---|
| Team A: A0 vs A1 | Data, normalization, direct MLP, optimizer settings, epochs, seeds | One-step loss vs closed-loop four-step loss |
| Team B: B1 vs B2 | Multi-`rho` data, normalization, direct MLP, one-step loss, optimizer, seeds | State only vs state plus `rho` |

The complete workflow has been rehearsed on JURECA-DC. Numerical settings and
dataset hashes are frozen in
[`../docs/frozen_benchmark_v1.0.md`](../docs/frozen_benchmark_v1.0.md); metric
definitions and interpretation are in the
[`evaluation evidence guide`](../docs/evaluation_evidence_guide.md).

Team-specific instructions are in the
[Team A](../docs/team_a_rollout_fidelity.md) and
[Team B](../docs/team_b_changing_dynamics.md) guides. Use the shared
[experiment and reporting templates](../docs/templates/) from the beginning of
the investigation.

## 1. Environment and tests

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
```

On JURECA, use the reproducible shared environment and Slurm scripts. The
environment shares pinned dependencies while resolving `lorenz_hackathon` from
the current team's checkout and branch. See
[JURECA_QUICKSTART.md](JURECA_QUICKSTART.md).

## 2. Link and verify the frozen datasets and run directory

```bash
test -e data || ln -s "$HACKATHON_SHARED_ROOT/data" data
mkdir -p "$HACKATHON_RUN_ROOT/runs"
test -e runs || ln -s "$HACKATHON_RUN_ROOT/runs" runs

sha256sum -c data/SHA256SUMS
```

Participants use the shared frozen copies. Dataset regeneration is an
organizer reproducibility operation, not a participant step.

The standard dataset uses `rho = 28` for trajectory-disjoint training, validation, and test splits. The multi-`rho` dataset uses `rho = 26, 28, 32` for training and validation, with different trajectories in each split, and `rho = 24, 30` for the public changed-dynamics test.

Team A uses only the standard dataset. B1 and B2 use exactly the same multi-`rho` training and validation data. Team B is evaluated on both the changed-dynamics test and the separate standard `rho = 28` test as an in-distribution reference.

Normalization is always computed from the training split of the selected dataset and saved in the checkpoint. Test data never changes it.

## 3. Reproduce the shared organizer baseline

The shared baseline matrix trains the direct one-step MLP at seeds 41–43 and one linear reference at seed 42:

```bash
bash scripts/train_matrix_worker.sh configs/matrix_shared_baseline_seeds.txt
```

Persistence requires no training. It is evaluated as a no-change rollout at every forecast lead and appears beside the learned model in each forecast-skill panel.

## 4. Run the mandatory comparisons

```bash
# Team A: A0 one-step vs A1 four-step, matched seeds 41–43
bash scripts/train_matrix_worker.sh configs/matrix_team_a_seeds.txt

# Team B: B1 state-only vs B2 state-plus-rho, matched seeds 41–43
bash scripts/train_matrix_worker.sh configs/matrix_team_b_seeds.txt
```

On a four-task JURECA job, a worker processes another row after its first experiment finishes, so the six-row team matrices do not require six GPUs.

## 5. Evaluate complete matrices

```bash
# Shared persistence, linear, and direct-MLP baselines at rho=28
bash scripts/evaluate_matrix.sh \
  configs/matrix_shared_baseline_seeds.txt \
  data/standard_benchmark.npz \
  standard_rho28

# Team A at rho=28
bash scripts/evaluate_matrix.sh \
  configs/matrix_team_a_seeds.txt \
  data/standard_benchmark.npz \
  standard_rho28

# Team B on unseen parameter values
bash scripts/evaluate_matrix.sh \
  configs/matrix_team_b_seeds.txt \
  data/multirho_benchmark.npz \
  multirho_unseen_rho24_30

# Team B in-distribution reference at rho=28
bash scripts/evaluate_matrix.sh \
  configs/matrix_team_b_seeds.txt \
  data/standard_benchmark.npz \
  in_distribution_rho28
```

Each checkpoint receives an evaluation directory containing:

- `benchmark_results.json`, including checkpoint and dataset hashes, frozen
  evaluation settings, sample counts, one-step error, learned and persistence
  rollout error, stability, perturbation growth, and long-term statistics;
- one four-panel `model_autopsy_rho_*.png` for every test value of `rho`.

Non-finite predictions are recorded as failures rather than silently omitted.
If any long emulator rollout is non-finite, its model climate metrics are
reported as `null`; finite and bounded fractions remain available.

## 6. Reproducibility rules

- Run the shared baseline end to end before changing a model or loss.
- Never mix time steps from one physical trajectory across data splits.
- Never recompute normalization from validation or test data.
- Keep the resolved configuration, seed, checkpoint, result, and Slurm job ID together.
- Report clipping, projection, early termination, or post-processing applied during rollout.
- Evaluate each mandatory neural configuration at seeds 41, 42, and 43.
- Preserve failed or inconclusive experiments in the experiment ledger.
- Change only the intended variable in the mandatory comparison; use the configuration-consistency tests to detect accidental confounders.
- Report seeds 41–43 individually and summarize them with the population
  standard deviation (`ddof=0`).
- Interpret the complete diagnostic set; finiteness and boundedness alone are
  weak evidence of dynamical fidelity.
