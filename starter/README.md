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
the investigation. Participants starting from a fresh JURECA login should use
the copy-paste [command and output reference](../docs/commands_and_outputs.md).
Participants running on a laptop, workstation, or Google Colab should use the
[local and Colab quick start](LOCAL_AND_COLAB_QUICKSTART.md).

## 1. Choose where to run

From `starter/`, select the runtime once:

```bash
bash scripts/select_runtime.sh local   # laptop or workstation
# bash scripts/select_runtime.sh colab # Google Colab
# bash scripts/select_runtime.sh jureca

bash scripts/run.sh prepare
bash scripts/run.sh status
```

With no argument, `select_runtime.sh` asks the same question interactively.
Local mode creates `.venv`; Colab uses its current Python environment; JURECA
uses the shared pinned environment and reservation. `prepare` generates and
checksum-verifies the frozen data only for local/Colab. JURECA continues to use
the organizer's immutable shared copies. The model configuration uses
`device=auto`, so local and Colab select CUDA when available and CPU otherwise.

For the complete Team B workflow on any selected runtime:

```bash
bash scripts/run.sh team-b
```

On local/Colab this runs directly and keeps the terminal or notebook cell
occupied. On JURECA it submits one serialized training/evaluation chain and
prints the three Slurm job IDs. See
[JURECA_QUICKSTART.md](JURECA_QUICKSTART.md) for the manual Slurm path.

## 2. Verify the frozen datasets

```bash
(cd data && sha256sum -c SHA256SUMS)
```

The preceding `bash scripts/run.sh prepare` command creates the appropriate
`data` and `runs` locations. On JURECA, participants use the shared frozen
copies. Local and Colab mode recreate byte-identical files from the frozen
generator configurations and verify the published SHA-256 checksums before
training.

The standard dataset uses `rho = 28` for trajectory-disjoint training, validation, and test splits. The multi-`rho` dataset uses `rho = 26, 28, 32` for training and validation, with different trajectories in each split, and `rho = 24, 30` for the public changed-dynamics test.

Team A uses only the standard dataset. B1 and B2 use exactly the same multi-`rho` training and validation data. Team B is evaluated on both the changed-dynamics test and the separate standard `rho = 28` test as an in-distribution reference.

Normalization is always computed from the training split of the selected dataset and saved in the checkpoint. Test data never changes it.

## 3. Reproduce the shared organizer baseline

The shared baseline matrix trains the direct one-step MLP at seeds 41–43 and one linear reference at seed 42:

```bash
bash scripts/run.sh train configs/matrix_shared_baseline_seeds.txt
```

Persistence requires no training. It is evaluated as a no-change rollout at every forecast lead and appears beside the learned model in each forecast-skill panel.

## 4. Run the mandatory comparisons

```bash
# Team A after the shared baseline: train A1 at matched seeds 41–43
bash scripts/run.sh train configs/matrix_team_a_a1_only_seeds.txt

# Team B: B1 state-only vs B2 state-plus-rho, matched seeds 41–43
bash scripts/run.sh train configs/matrix_team_b_seeds.txt
```

Local and Colab execute these commands directly. On JURECA each command
returns a job ID; add that ID as the final argument of a later `train` or
`evaluate` command to create an `afterok` dependency. On a four-task JURECA
job, a worker processes another row after its first experiment finishes, so
the six-row team matrices do not require six GPUs.

## 5. Evaluate complete matrices

```bash
# Shared persistence, linear, and direct-MLP baselines at rho=28
bash scripts/run.sh evaluate \
  configs/matrix_shared_baseline_seeds.txt \
  data/standard_benchmark.npz \
  shared_baseline_rho28

# Team A at rho=28
bash scripts/run.sh evaluate \
  configs/matrix_team_a_seeds.txt \
  data/standard_benchmark.npz \
  team_a_rho28

# Team B on unseen parameter values
bash scripts/run.sh evaluate \
  configs/matrix_team_b_seeds.txt \
  data/multirho_benchmark.npz \
  multirho_unseen_rho24_30

# Team B in-distribution reference at rho=28
bash scripts/run.sh evaluate \
  configs/matrix_team_b_seeds.txt \
  data/standard_benchmark.npz \
  in_distribution_rho28
```

Each checkpoint receives an evaluation directory containing:

- `benchmark_results.json`, including checkpoint and dataset hashes, frozen
  evaluation settings, sample counts, one-step error, learned and persistence
  rollout error, stability, perturbation growth, and long-term statistics;
- one four-panel `model_autopsy_rho_*.png` for every test value of `rho`.

After all matrix rows are evaluated, the same command automatically writes a
matched-seed comparison under
`runs/matrix_comparisons/<matrix-name>/<evaluation-label>/`:

- `forecast_comparison_rho_*.png` shows every seed plus the mean and population
  standard deviation of the forecast-NRMSE curve;
- `metric_comparison_rho_*.png` shows paired seeds, means, and population
  standard deviations for ten predictive and dynamical diagnostics; and
- `matrix_summary.json` preserves the plotted per-seed values, summaries,
  missing and censored counts, metric directions, provenance, and paired
  changes.

After both Team B evaluation labels are available, the workflow additionally
writes `three_regime_summary/team_b_three_regime_summary.png`. This
presentation-ready figure places the `rho=28` control, `rho=30` interpolation,
and `rho=24` extrapolation evidence in three columns without combining them
into one score.

An open triangle for useful horizon means that NRMSE did not reach one within
the frozen forecast window; it is not treated as an exact horizon or replaced
with zero in the JSON summary.

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
