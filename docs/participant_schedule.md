# Participant Schedule and Scientific Gates

The hackathon is organized as a 2.5-day investigation. Clock times may be
adjusted to the venue schedule, but the gates and their order should not
change. Optional extensions begin only after the mandatory comparison has a
complete three-seed evaluation.

## Before the event

- Confirm JSC login and project membership.
- Clone the assigned team branch.
- Activate the shared environment.
- Verify both frozen dataset checksums.
- Run the unit tests and a CUDA smoke test in an allocation.
- Read the Lorenz primer, scientific contract, frozen benchmark and evaluation
  evidence guide.

## Day 1: Reproduce before changing

| Block | Activity | Required output |
|---|---|---|
| Opening | Scientific question, Lorenz primer and evidence hierarchy | Shared vocabulary; no model claims yet |
| Environment | Activate, verify checksums, run tests | Reproducible checkout and working GPU path |
| Baseline | Reproduce persistence, linear and A0 reference workflow | Baseline artifacts and job IDs |
| Hypothesis | Write the team hypothesis and falsification criterion | Ledger entry made before mandatory results |
| Mandatory training | Launch the matched-seed A0/A1 or B1/B2 matrix | Six neural runs tracked by seed and job ID |
| Checkpoint | Validate configs, checkpoints and permissions | Gate 1 passed: complete mandatory training |

## Day 2: Diagnose rather than rank

| Block | Activity | Required output |
|---|---|---|
| Mandatory evaluation | Run the frozen evaluator on every required regime | JSON and model-autopsy figures for all seeds |
| Predictive evidence | Compare one-step and lead-time errors | Full forecast curves and useful horizons |
| Dynamical evidence | Compare stability, sensitivity and climate-like statistics | Completed per-seed result table |
| Contradictions | Identify metrics that disagree and seed-sensitive findings | Written trade-off statement |
| Team checkpoint | Mentor review of evidence and provenance | Gate 2 passed: defensible mandatory comparison |
| Extension window | One focused extension, only if Gate 2 passed | Separate config and output names |

## Day 3: Reproduce and communicate

| Block | Activity | Required output |
|---|---|---|
| Internal reproduction | Another team member reruns the summary from preserved artifacts | Provenance and table values confirmed |
| Conclusion | Test the original hypothesis against all evidence | One principal conclusion and strongest limitation |
| Presentation | Build the common 15-minute presentation format | Final slides with no hidden metric selection |
| Final session | 12-minute presentation plus approximately 3 minutes of questions | Results, trade-offs and next experiment |
| Archive | Preserve configs, ledger, commit, job IDs and outputs | Gate 3 passed: reproducible team record |

## Scientific gates

1. **Baseline gate:** frozen data, tests and common baseline work end-to-end.
2. **Mandatory-comparison gate:** all three matched seeds and required regimes
   have complete artifacts.
3. **Evidence gate:** predictive, stability, sensitivity and long-term
   diagnostics have been considered together.
4. **Claim gate:** the conclusion states contradictory evidence and its
   strongest limitation.
5. **Reproducibility gate:** another participant can trace every reported value
   to a config, checkpoint, dataset hash and evaluation JSON.
