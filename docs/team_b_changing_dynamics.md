# Team B: Changing Dynamics

## Scientific question

Is exposure to several Lorenz regimes sufficient, or must the governing
parameter `rho` be supplied explicitly for an emulator to reproduce the
appropriate dynamics?

## Mandatory controlled comparison

| Model | Training values | Input |
|---|---|---|
| B1 | `rho=26, 28, 32` | State only |
| B2 | `rho=26, 28, 32` | State plus normalized `rho` |

B1 and B2 use identical trajectories, direct next-state targets, hidden
architecture, optimizer settings, 30 epochs and matched seeds 41-43. B2 adds
one normalized input to the first layer; no other difference is intended.

## Required hypothesis

Write one falsifiable statement before inspecting the evaluation. For example:

> Explicit `rho` conditioning will improve in-range interpolation because it
> disambiguates state-dependent tendencies, but this will not necessarily
> guarantee reliable out-of-range extrapolation.

Replace this example with the team's own hypothesis and define what evidence
would contradict it.

## Run the comparison

From the activated `starter/` directory:

```bash
(cd data && sha256sum -c SHA256SUMS)

team_b_job=$(sbatch --parsable \
  --account="$HACKATHON_ACCOUNT" \
  --job-name=lorenz-team-b \
  --chdir="$(pwd)" \
  --output="$HACKATHON_RUN_ROOT/slurm/lorenz-team-b-%j.out" \
  --error="$HACKATHON_RUN_ROOT/slurm/lorenz-team-b-%j.err" \
  slurm/train_matrix.sbatch \
  configs/matrix_team_b_seeds.txt)

echo "Team B training job: $team_b_job"
```

Evaluate the three regimes using two matrix jobs after successful training:

```bash
for specification in \
  "data/standard_benchmark.npz in_distribution_rho28" \
  "data/multirho_benchmark.npz multirho_unseen_rho24_30"
do
  read -r dataset label <<< "$specification"
  job=$(sbatch --parsable \
    --account="$HACKATHON_ACCOUNT" \
    --partition=dc-gpu-devel \
    --job-name="lorenz-team-b-${label}" \
    --chdir="$(pwd)" \
    --output="$HACKATHON_RUN_ROOT/slurm/lorenz-team-b-${label}-%j.out" \
    --error="$HACKATHON_RUN_ROOT/slurm/lorenz-team-b-${label}-%j.err" \
    slurm/evaluate_matrix.sbatch \
    configs/matrix_team_b_seeds.txt \
    "$dataset" \
    "$label")
  echo "$label evaluation job: $job"
done
```

The resulting evidence must be separated into:

- `rho=28`: in-distribution control on independent trajectories;
- `rho=30`: unseen interpolation within the training range; and
- `rho=24`: out-of-range extrapolation across changed dynamics.

## Evidence checklist

For B1 and B2 at each `rho`, compare:

- one-step NRMSE and complete forecast-error curves;
- useful forecast horizon;
- finite and within-reference-bound fractions;
- perturbation-growth-rate fidelity;
- component-wise variance ratios and Wasserstein distances; and
- lobe occupancy, switching and residence time.

Report every matched seed before the mean and population standard deviation.
Use the [evaluation evidence guide](evaluation_evidence_guide.md) for exact
definitions.

## Required conclusion

State separately:

1. whether conditioning helps on the in-distribution control;
2. whether it helps unseen interpolation;
3. whether it helps out-of-range extrapolation;
4. which diagnostics disagree or remain seed-sensitive; and
5. why success inside the training range does or does not justify an
   extrapolation claim.

Do not collapse the three regimes into one average. Only after completing this
comparison may the team add parameters, architectures or same-state tendency
experiments. Extensions require new config, matrix, output and evaluation-label
names.
