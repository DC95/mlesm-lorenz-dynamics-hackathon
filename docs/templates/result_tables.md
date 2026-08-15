# Result Table Templates

Populate these tables from `benchmark_results.json`. Preserve per-seed results;
do not report only the best seed. Three-seed summaries use the population
standard deviation (`ddof=0`).

For compact error summaries:

- growth error = absolute emulator/reference effective-growth-rate difference;
- variance error = mean of `abs(variance_ratio - 1)` over `x`, `y`, `z`;
- occupancy error = absolute positive-`x` fraction difference;
- switching error = absolute switch-rate difference;
- residence error = absolute mean-residence-time difference; and
- mean Wasserstein = mean of the `x`, `y`, `z` Wasserstein distances.

Retain the component-wise and reference values in supplementary output. Never
replace a `null` metric with zero or omit the failed seed.

## Team A: per-seed A0 versus A1

| Seed | Model | One-step NRMSE | Useful horizon | Finite | Bounded | Growth error | Variance error | Occupancy error | Switch error | Residence error | Mean Wasserstein |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 41 | A0 H=1 | | | | | | | | | | |
| 41 | A1 H=4 | | | | | | | | | | |
| 42 | A0 H=1 | | | | | | | | | | |
| 42 | A1 H=4 | | | | | | | | | | |
| 43 | A0 H=1 | | | | | | | | | | |
| 43 | A1 H=4 | | | | | | | | | | |

## Team A: three-seed summary

| Diagnostic | Direction | A0 H=1 mean +/- SD | A1 H=4 mean +/- SD | Paired A1-A0 change | Interpretation |
|---|---|---:|---:|---:|---|
| One-step NRMSE | Lower | | | | |
| Useful horizon | Higher | | | | |
| Growth error | Lower | | | | |
| Variance error | Lower | | | | |
| Occupancy error | Lower | | | | |
| Switch error | Lower | | | | |
| Residence error | Lower | | | | |
| Mean Wasserstein | Lower | | | | |

## Team B: per-regime summary

Complete one table for each `rho`: 28 in-distribution, 30 interpolation and 24
extrapolation.

**Regime:** `rho = ____`
**Claim type:** in-distribution / interpolation / extrapolation

| Diagnostic | Direction | B1 state-only mean +/- SD | B2 conditioned mean +/- SD | Paired B2-B1 change | Seed consistency |
|---|---|---:|---:|---:|---|
| One-step NRMSE | Lower | | | | |
| Useful horizon | Higher | | | | |
| Finite fraction | Higher | | | | |
| Bounded fraction | Higher | | | | |
| Growth error | Lower | | | | |
| Variance error | Lower | | | | |
| Occupancy error | Lower | | | | |
| Switch error | Lower | | | | |
| Residence error | Lower | | | | |
| Mean Wasserstein | Lower | | | | |

## Evidence synthesis

- Principal result:
- Evidence supporting it:
- Evidence contradicting or qualifying it:
- Seed-sensitive findings:
- Numerical failures or `null` values:
- Strongest limitation:
- Smallest next experiment that could resolve the uncertainty:
