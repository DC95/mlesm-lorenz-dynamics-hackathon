# Evaluation Evidence Guide

## Why the benchmark uses several diagnostics

A small one-step error shows that a model approximates the next sampled state
near the data distribution. It does not establish that repeated application of
the model reproduces instability, lobe switching, long-term variability, or
the response to a changed control parameter. The evaluator therefore assembles
complementary evidence rather than producing one overall score.

The frozen numerical choices are listed in
[Frozen Benchmark v1.0](frozen_benchmark_v1.0.md). This page defines what each
reported quantity means, why it is useful, and what it cannot prove.

## Predictive skill

Let `s = (s_x, s_y, s_z)` be the training-split standard deviation and let
`e` be prediction minus reference. The normalized root-mean-square error is

$$
\mathrm{NRMSE}=\sqrt{\operatorname{mean}\left[(e/s)^2\right]},
$$

where division is component-wise and the mean is taken over trajectories and
state variables.

| Diagnostic | Exact meaning | Evidence supplied | Important limitation |
|---|---|---|---|
| One-step RMSE | Physical-coordinate RMSE for `x`, `y`, and `z` over all public-test transitions | Local next-state accuracy | Can be excellent for a map that fails when composed with itself |
| One-step NRMSE | Joint standardized error over all test transitions and variables | Comparable summary of local predictive skill | Does not measure autoregressive dynamics |
| Persistence NRMSE | Error from predicting that the state does not change | Checks that the learned map beats a trivial near-identity baseline | Beating persistence is necessary, not sufficient |
| NRMSE by lead | Error after repeatedly feeding each prediction back into the model | Shows how errors accumulate under closed-loop use | Pointwise trajectory agreement is inherently time-limited in a chaotic system |
| Useful forecast horizon | First lead at which forecast NRMSE reaches 1; a non-finite lead also fails immediately | Compact measure of the duration of useful trajectory information | Depends on the frozen normalization and threshold; the complete lead-time curve must also be shown |

If NRMSE never reaches 1 during the 200-step evaluation window, the JSON value
is `null`; this means “not crossed within the evaluated window,” not an infinite
horizon. Lead time is reported in Lorenz time units, with one model step equal
to 0.05.

## Numerical stability and variability

| Diagnostic | Exact meaning | Evidence supplied | Important limitation |
|---|---|---|---|
| Finite-trajectory fraction | Fraction of 32 long rollouts containing only finite values | Detects numerical divergence, overflow, and NaNs | A finite trajectory can converge to the wrong fixed point or attractor |
| Within-reference-bound fraction | Fraction of finite rollouts whose state-vector norm remains below five times the 99.9th percentile of the reference norm | Detects gross amplitude explosions | The bound is deliberately broad and is only a sanity screen |
| Variance ratio | Emulator variance divided by reference variance, separately for `x`, `y`, and `z`, after common burn-in | Detects collapse (`<1`) or excessive dispersion (`>1`) | Matching three marginal variances does not establish correct geometry or transitions |

Finiteness and boundedness are gates. Passing them is not strong evidence of
dynamical fidelity.

## Sensitivity to initial conditions

The evaluator starts paired trajectories with separation
`1e-5 * ||training_state_std||`, rolls both the RK4 system and emulator forward,
and records the median separation at each lead. It fits

$$
\log d(t) \approx a + \lambda_{\mathrm{eff}}t
$$

only in the frozen pre-saturation distance window.

| Diagnostic | Evidence supplied | Important limitation |
|---|---|---|
| Perturbation-growth curve | Whether nearby emulator trajectories separate on a similar time scale and with a similar shape to the reference | Sensitive to the chosen initial states, distance norm, and finite evaluation window |
| Effective growth rate | Compact comparison of early exponential-like separation | It is not a rigorous or global Lyapunov exponent; `null` means fewer than three valid fit points |
| Finite-pair fraction by lead | Reveals when perturbation comparisons are lost to numerical failure | Does not diagnose the cause of the failure |

A model can have a longer forecast horizon because it is overly dissipative.
That is why forecast horizon and perturbation growth must be interpreted
together.

## Long-term or climate-like behaviour

All long-term statistics use reference and emulator rollouts of equal duration
after the same 20-time-unit burn-in.

| Diagnostic | Exact meaning | Evidence supplied | Important limitation |
|---|---|---|---|
| Mean and standard deviation | Marginal first and second moments of `x`, `y`, and `z` | Bias and amplitude fidelity | Low-order moments do not define an attractor |
| Wasserstein distance | Mean absolute separation between 999 corresponding marginal quantiles for each variable | Distributional mismatch in physical units | One-dimensional marginals omit joint geometry and temporal ordering |
| Positive-`x` fraction | Fraction of samples with `x >= 0` | Relative occupancy of the two Lorenz lobes | Correct occupancy can coexist with incorrect switching |
| Lobe-switch rate | Mean number of sign changes of `x` per Lorenz time unit | Frequency of transitions between lobes | Does not describe the full residence-time distribution or transition path |
| Mean residence time | Mean duration of contiguous same-sign-`x` segments, including the first and last observed segments | Typical persistence within a lobe | Finite-window censoring affects boundary segments; report it as an observed-window summary |

The four-panel model autopsy combines the forecast curve, one representative
long-term phase-space projection, perturbation growth, and an `x`-distribution
comparison. The plot is diagnostic evidence, not a substitute for ensemble
statistics.

## Team-specific reasoning

### Team A: one-step versus four-step training

Ask whether the closed-loop four-step loss changes useful forecast horizon,
then test whether any gain is accompanied by realistic perturbation growth,
variance, lobe occupancy, switching, residence time, and state distributions.
A defensible conclusion must report trade-offs rather than declaring a winner
from horizon alone.

### Team B: state-only versus `rho`-conditioned training

Compare B1 and B2 at matched seeds for:

- `rho=28`, an in-distribution control using independent test trajectories;
- `rho=30`, unseen interpolation within the training range; and
- `rho=24`, out-of-range extrapolation across changed dynamics.

Interpolation and extrapolation are separate claims. Success at `rho=28` or
`rho=30` is not evidence that the model extrapolates correctly at `rho=24`.

## Minimum reporting standard

For every mandatory neural configuration:

1. report seeds 41, 42 and 43 individually;
2. report the three-seed mean and population standard deviation (`ddof=0`);
3. include one-step NRMSE and the full forecast-error curve;
4. report finite and bounded fractions before interpreting climate metrics;
5. compare perturbation growth, variance, lobe statistics and Wasserstein
   distance against the matching RK4 reference;
6. state which diagnostics improve, worsen or remain seed-sensitive; and
7. state the strongest limitation of the conclusion.

No individual metric proves that the emulator learned the dynamics. The claim
becomes stronger only when predictive, stability, sensitivity, long-term and
parameter-response evidence agree.
