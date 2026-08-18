# Organizer Roadmap

**Current state:** Participant release validated and ready for the event
**Public release:** 18 August 2026
**Hackathon start:** 19 August 2026

This page records what is complete and the remaining participant-release gates.

## 1. Document authority

| Document | Role | Status |
|---|---|---|
| [`scientific_challenge_contract_v1.0.md`](scientific_challenge_contract_v1.0.md) | Scientific question, lean team scope, evidence categories, and required outputs | **Current authority** |
| [`frozen_benchmark_v1.0.md`](frozen_benchmark_v1.0.md) | Dataset hashes and frozen numerical, training, evaluation and failure settings | **Numerical authority** |
| [`evaluation_evidence_guide.md`](evaluation_evidence_guide.md) | Participant-facing diagnostic definitions, interpretation and reporting standard | **Current authority** |
| [`lorenz63_primer.md`](lorenz63_primer.md) | Participant introduction to Lorenz-63 and its connection to the experiments | **Complete** |
| [`benchmark_spec_v0.1.md`](benchmark_spec_v0.1.md) | Pointer to the superseded organizer draft retained in Git history | **Historical only** |
| [`../starter/README.md`](../starter/README.md) | Commands and implementation overview | **Aligned with frozen benchmark v1.0** |
| [`../original_materials/`](../original_materials/) | Earlier challenge brief and teaching notebook | **Preserved source material** |

If documents disagree about scientific scope, the contract v1.0 takes
precedence. If they disagree about numerical settings, Frozen Benchmark v1.0
takes precedence.

## 2. Progress

| Stage | Status | Evidence or remaining gate |
|---|---|---|
| Preserve original materials | Complete | Original PDF and notebook are unchanged. |
| Freeze the scientific direction | Complete | Contract v1.0 defines the lean shared baseline and one intervention per team. |
| Introduce Lorenz-63 | Complete | The 10–15-minute participant primer is available. |
| Build the numerical foundation | Complete | RK4 integration, frozen trajectory-disjoint datasets and checksums were verified on JURECA. |
| Build the ML foundation | Complete | A0/A1 and B1/B2 matched-seed matrices trained successfully on JURECA. |
| Build the evaluation foundation | Complete | The common evaluator ran end-to-end; non-finite behavior, provenance and frozen settings are protected by contract tests. |
| Align mandatory experiments | Complete | Contract tests enforce that A0/A1 change only rollout horizon and B1/B2 change only `rho` conditioning. |
| Rehearse on JURECA | Complete | Data generation, training, evaluation, plotting and artifact preservation completed. |
| Freeze the benchmark | Complete | Frozen Benchmark v1.0 records settings and dataset hashes. |
| Prepare participant release | Complete | A fresh public HTTPS clone passed both dataset checksums, all 22 tests, scratch isolation, and the GPU preflight on JURECA. |
| Create team branches and publish | Complete | The public Team A and Team B branches start from the same frozen release commit. |
| Automate matrix-level comparison figures | Complete | Every matrix evaluation produces per-`rho` forecast and ten-diagnostic matched-seed figures plus a JSON summary. |

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

## 4. Frozen numerical settings

The exposed forecast interval, RK4 step, trajectory counts, Team A four-step
loss horizon, Team B parameter values, long-run evaluation, stability screens,
non-finite policies, seeds and dataset hashes are fixed in
[Frozen Benchmark v1.0](frozen_benchmark_v1.0.md).

## 5. Event-start sequence

1. Assign each participant the appropriate team branch.
2. Ask participants to run the documented login-node preflight before training.
3. Record Slurm job IDs, hypotheses, failures, and results in the supplied
   experiment ledger from the beginning.

## 6. Release gates

| Release state | Gate |
|---|---|
| **Organizer draft** | Scientific scope is frozen, but code and numerical settings remain provisional. |
| **Rehearsal candidate** | Mandatory configurations match v1.0 and the full JURECA workflow can run. |
| **Release candidate** | Rehearsal and numerical freeze are complete; participant packaging is being finalized. |
| **Participant release — current** | Data, commands, timing, documentation and licence are frozen and verified from a fresh public clone. |
| **Event archive** | Team branches, configurations, presentations, conclusions, and reproducible results are preserved. |
