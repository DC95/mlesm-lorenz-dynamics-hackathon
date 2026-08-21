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

From `starter/`, choose the available runtime once. Use `local` for a laptop
or workstation, `colab` for Google Colab, or `jureca` when the HPC system is
available:

```bash
bash scripts/select_runtime.sh local
bash scripts/run.sh prepare
bash scripts/run.sh team-b
```

With no argument, `select_runtime.sh` displays an interactive choice. The
scientific workflow is identical in all three cases:

1. `prepare` establishes the Python environment, verifies the tests and makes
   the two checksum-verified frozen datasets available;
2. `team-b` trains B1 and B2 for matched seeds 41--43; and
3. the same command evaluates `rho=28`, `rho=30`, and `rho=24` and builds the
   comparison figures.

Local and Colab execute the stages sequentially in the current terminal or
notebook cell. The configurations use `device=auto`, so PyTorch selects an
available GPU and otherwise runs on the CPU. JURECA submits a training job and
two dependency-linked evaluation jobs, keeping Team B to one active node. See
[`starter/LOCAL_AND_COLAB_QUICKSTART.md`](../starter/LOCAL_AND_COLAB_QUICKSTART.md)
for copy-paste local and Colab cells, or
[`starter/JURECA_QUICKSTART.md`](../starter/JURECA_QUICKSTART.md) for the
manual Slurm commands.

The resulting evidence must be separated into:

- `rho=28`: in-distribution control on independent trajectories;
- `rho=30`: unseen interpolation within the training range; and
- `rho=24`: out-of-range extrapolation across changed dynamics.

Each evaluation job automatically creates separate B1/B2 matched-seed figures
for every `rho` in that dataset under
`runs/matrix_comparisons/matrix_team_b_seeds/<evaluation-label>/`, together
with a machine-readable `matrix_summary.json`.

After both Team B evaluation labels are complete, the workflow also creates a
presentation-ready overview in
`runs/matrix_comparisons/matrix_team_b_seeds/three_regime_summary/`. The figure
uses three columns in the required scientific order: `rho=28`
(in-distribution), `rho=30` (interpolation), and `rho=24` (extrapolation). Each
column combines the matched-seed forecast curves with a compact B1/B2 evidence
table. The directional seed count is a consistency aid, not an overall score
or significance test.

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
