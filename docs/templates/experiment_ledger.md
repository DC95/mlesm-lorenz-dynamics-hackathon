# Experiment Ledger Template

Copy this section for every baseline, mandatory run and extension. Failed runs
remain in the ledger.

## Experiment: `<team>-<model>-seed<seed>-<short-purpose>`

### Question and design

- Team:
- Date and participant:
- Hypothesis tested:
- Evidence that would contradict the hypothesis:
- Mandatory comparison or optional extension:
- Intended change from the control:
- Possible confounders:

### Provenance

- Git branch:
- Git commit:
- Uncommitted diff present: yes/no
- Training config:
- Matrix file and row:
- Resolved config artifact:
- Dataset file:
- Dataset SHA-256:
- Training seed:
- Slurm training job ID:
- Slurm evaluation job ID:
- Checkpoint path:
- Checkpoint SHA-256:
- Evaluation label and result path:
- Result JSON SHA-256:

Useful commands:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
sha256sum data/*.npz
sacct -j JOB_ID --format=JobID,JobName%28,Partition,State,Elapsed,ExitCode
```

### Outcome

- Training completed: yes/no
- Evaluation completed: yes/no
- Best epoch and validation loss:
- One-step NRMSE:
- Useful forecast horizon:
- Finite / within-bound fractions:
- Perturbation-growth comparison:
- Variance-ratio comparison:
- Occupancy / switch / residence comparison:
- Wasserstein comparison:
- Unexpected behavior or failure:
- Interpretation:
- Strongest limitation:
- Next decision:

### Deviations

Record any clipping, projection, early termination, post-processing, changed
evaluation length, changed parameter value, rerun or manual artifact repair.
Write `none` when no deviation occurred.
