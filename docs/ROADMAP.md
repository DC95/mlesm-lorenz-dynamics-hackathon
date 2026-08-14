# Organizer Roadmap

**Current state:** Scientific scope frozen; implementation aligned locally; JURECA rehearsal pending  
**Public release:** 18 August 2026  
**Hackathon start:** 19 August 2026

This page records what is complete, what remains provisional, and the order in which the repository should be developed.

## 1. Document authority

| Document | Role | Status |
|---|---|---|
| [`scientific_challenge_contract_v1.0.md`](scientific_challenge_contract_v1.0.md) | Scientific question, lean team scope, evidence categories, and required outputs | **Current authority** |
| [`lorenz63_primer.md`](lorenz63_primer.md) | Participant introduction to Lorenz-63 and its connection to the experiments | **Complete** |
| [`benchmark_spec_v0.1.md`](benchmark_spec_v0.1.md) | Earlier numerical and implementation proposal | **Superseded draft; retain temporarily for implementation history** |
| [`../starter/README.md`](../starter/README.md) | Commands and implementation overview | **Aligned with contract v1.0; numerical values provisional** |
| [`../original_materials/`](../original_materials/) | Earlier challenge brief and teaching notebook | **Preserved source material** |

If documents disagree about scientific scope, the contract v1.0 takes precedence.

## 2. Progress

| Stage | Status | Evidence or remaining gate |
|---|---|---|
| Preserve original materials | Complete | Original PDF and notebook are unchanged. |
| Freeze the scientific direction | Complete | Contract v1.0 defines the lean shared baseline and one intervention per team. |
| Introduce Lorenz-63 | Complete | The 10–15-minute participant primer is available. |
| Build the numerical foundation | Implemented locally | RK4 integration, trajectory-disjoint data generation, and numerical smoke tests pass locally. |
| Build the ML foundation | Implemented locally | Clean direct A0/A1 and state-only/conditioned B1/B2 configurations and matched seed matrices exist. Full PyTorch verification awaits JURECA. |
| Build the evaluation foundation | Implemented locally | Persistence rollout, forecast, stability, perturbation, distribution, and lobe diagnostics share one evaluator. End-to-end execution awaits JURECA. |
| Align mandatory experiments | Complete | Contract tests enforce that A0/A1 change only rollout horizon and B1/B2 change only `rho` conditioning. |
| Rehearse on JURECA | **Next** | Complete data generation, training, evaluation, plotting, and runtime measurement. |
| Freeze the benchmark | Pending | Fix numerical values, datasets, checksums, seeds, stability thresholds, and runtime budget. |
| Prepare participant release | Pending | Final commands, team pages, schedule, result templates, licence, and clean-account test. |
| Create team branches and publish | Pending | Branch from the frozen release and make the repository public. |

## 3. Mandatory configuration alignment

The starter configurations now implement the following controlled comparisons. Automated tests guard them against accidental confounders before JURECA results are interpreted.

### Team A

| Model | Architecture and target | Loss |
|---|---|---|
| A0 | Direct next-state MLP | One-step MSE |
| A1 | The same direct next-state MLP | Closed-loop multi-step rollout MSE |

Residual prediction remains optional and must not be mixed into the mandatory comparison.

### Team B

| Model | Training data | Input | Target and loss |
|---|---|---|---|
| B1 | Identical multi-$\rho$ data | State only | Common direct target and common loss |
| B2 | Identical multi-$\rho$ data | State plus $\rho$ | The same direct target and loss |

Conditioning must be the only intended difference between B1 and B2.

## 4. Numerical settings to freeze through rehearsal

- exposed ML forecast interval;
- internal RK4 time step and tolerances;
- trajectory counts and stored lengths;
- Team A rollout-loss horizon;
- Team B training, interpolation, and extrapolation values of $\rho$;
- long-run evaluation length and discarded initial period;
- stability thresholds;
- number of training seeds; and
- GPU, memory, and wall-time budget.

These settings must be fixed before teams see final test results.

## 5. Build sequence from here

1. Run numerical, configuration, and PyTorch tests in the shared JURECA environment.
2. Generate the standard and multi-$\rho$ datasets.
3. Train and evaluate all four mandatory configurations over seeds 41–43.
4. Inspect the four-panel model autopsies and record runtime and failure modes.
5. Simplify the code and commands based on the rehearsal.
6. Freeze datasets and publish checksums.
7. Write the participant team pages, schedule, experiment ledger, and presentation template.
8. Run the entire workflow from a clean participant-like account.
9. Merge the frozen release, create team branches, select the licence, and make the repository public.

## 6. Release gates

| Release state | Gate |
|---|---|
| **Organizer draft — current** | Scientific scope is frozen, but code and numerical settings remain provisional. |
| **Rehearsal candidate** | Mandatory configurations match v1.0 and the full JURECA workflow can run. |
| **Participant release** | Data, commands, timing, documentation, and licence are frozen and verified from a clean account. |
| **Event archive** | Team branches, configurations, presentations, conclusions, and reproducible results are preserved. |
