# Team A: Rollout Fidelity

## Scientific question

Does training through a short closed-loop rollout improve autoregressive
fidelity, and what does it sacrifice relative to one-step training?

## Mandatory controlled comparison

| Model | Objective | Frozen horizon |
|---|---|---:|
| A0 | Direct next-state MSE | 1 step |
| A1 | Closed-loop rollout MSE | 4 steps |

Both models use the standard `rho=28` dataset, the same direct MLP,
normalization, optimizer settings, 30 epochs and matched seeds 41-43. A1 is
not an equal-compute control: its loss unrolls and differentiates through four
successive model evaluations.

The four-step horizon was fixed by the organizer before release. It is not a
hint that A1 should win every diagnostic.

## Required hypothesis

Write one falsifiable statement before inspecting the evaluation. For example:

> Four-step training will increase useful forecast horizon without materially
> degrading one-step NRMSE, but may change perturbation growth or long-term
> variability.

Replace this example with the team's own hypothesis and define what evidence
would contradict it.

## Run the comparison

From the activated `starter/` directory:

```bash
(cd data && sha256sum -c SHA256SUMS)
```

First reproduce the shared baseline as described in the
[participant command reference](commands_and_outputs.md). That command has
already trained A0 at seeds 41--43, so the Team A continuation must train only
A1 rather than overwrite the A0 checkpoints.

```bash
team_a_job=$(sbatch --parsable \
  --dependency="afterok:$baseline_job" \
  --account="$HACKATHON_ACCOUNT" \
  --partition="$HACKATHON_PARTITION" \
  --reservation="$HACKATHON_RESERVATION" \
  --job-name=lorenz-team-a \
  --chdir="$(pwd)" \
  --output="$HACKATHON_RUN_ROOT/slurm/lorenz-team-a-%j.out" \
  --error="$HACKATHON_RUN_ROOT/slurm/lorenz-team-a-%j.err" \
  slurm/train_matrix.sbatch \
  configs/matrix_team_a_a1_only_seeds.txt)

echo "Team A training job: $team_a_job"
```

After successful training:

```bash
team_a_eval_job=$(sbatch --parsable \
  --dependency="afterok:$baseline_eval_job" \
  --account="$HACKATHON_ACCOUNT" \
  --partition="$HACKATHON_PARTITION" \
  --reservation="$HACKATHON_RESERVATION" \
  --job-name=lorenz-team-a-eval \
  --chdir="$(pwd)" \
  --output="$HACKATHON_RUN_ROOT/slurm/lorenz-team-a-eval-%j.out" \
  --error="$HACKATHON_RUN_ROOT/slurm/lorenz-team-a-eval-%j.err" \
  slurm/evaluate_matrix.sbatch \
  configs/matrix_team_a_seeds.txt \
  data/standard_benchmark.npz \
  team_a_rho28)

echo "Team A evaluation job: $team_a_eval_job"
```

This job automatically creates the A0/A1 matched-seed forecast and diagnostic
figures, together with `matrix_summary.json`, under
`runs/matrix_comparisons/matrix_team_a_seeds/team_a_rho28/`.

## Evidence checklist

Compare A0 and A1 at every matched seed, then summarize seeds with the
population standard deviation (`ddof=0`).

- one-step NRMSE and persistence NRMSE;
- complete forecast-NRMSE curves and useful forecast horizon;
- finite and within-reference-bound fractions;
- perturbation-growth curves and effective growth-rate error;
- component-wise variance ratios;
- positive-`x` occupancy, lobe-switch rate and mean residence time; and
- component-wise Wasserstein distances.

Use the [evaluation evidence guide](evaluation_evidence_guide.md) for exact
definitions. A horizon gain accompanied by implausible sensitivity or
long-term statistics is a trade-off, not an unqualified improvement.

## Required conclusion

State:

1. whether the useful horizon changed consistently across seeds;
2. whether one-step skill remained comparable;
3. which dynamical diagnostics improved or deteriorated;
4. whether any conclusion is seed-sensitive; and
5. the strongest limitation, including the unequal-compute caveat.

Only after completing this comparison may the team try another loss horizon,
residual prediction or another architecture. Extensions must use new config,
matrix, output and evaluation-label names.
