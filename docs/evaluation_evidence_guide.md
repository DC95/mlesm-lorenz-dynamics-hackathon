# How to Read the Evaluation Results

This guide is a beginner-friendly map of the benchmark results. The central
idea is simple:

> A model can predict the next step well and still produce the wrong dynamics
> when it repeatedly predicts its own future.

For that reason, the evaluator reports several complementary diagnostics
instead of one overall score. Exact numerical settings are fixed in
[Frozen Benchmark v1.0](frozen_benchmark_v1.0.md).

## 1. Start with five questions

Read the results in this order:

| Question | Evidence to inspect | What you hope to see |
|---|---|---|
| 1. Can it predict the next state? | One-step NRMSE and persistence NRMSE | Low error that beats persistence |
| 2. What happens when it predicts repeatedly? | Forecast-NRMSE curve and useful forecast horizon | Error grows later or more realistically |
| 3. Does the rollout remain usable? | Finite and within-bound fractions | Values close to 1 |
| 4. Are the dynamics too sensitive or too smooth? | Perturbation-growth curve and effective growth rate | Growth resembles the RK4 reference |
| 5. Is the long-term Lorenz behaviour realistic? | Variance, distributions, lobe occupancy, switching, and residence time | Several statistics agree with the reference |

Team B asks one additional question: does the conclusion change when `rho`
changes?

## 2. Keep four rules in mind

1. **One-step skill is only the starting point.** During one-step evaluation,
   every input is a true state. During an autoregressive forecast, the model
   receives its own previous prediction, so errors can feed back and grow.
2. **Error growth is expected.** Lorenz-63 is chaotic, so an exact trajectory
   match cannot last forever. Compare how quickly error grows and what happens
   after trajectories separate.
3. **A longer forecast horizon is not automatically better.** An overly smooth
   model may suppress perturbation growth and appear predictable for longer.
4. **Look for agreement across metrics and seeds.** Conflicting diagnostics or
   seed-sensitive results are scientific findings, not inconveniences to hide.

## 3. Metrics in plain language

### Predictive skill

| Metric | Meaning | How to read it |
|---|---|---|
| One-step RMSE | Next-state error for `x`, `y`, and `z` in physical units | Smaller is better |
| One-step NRMSE | Combined next-state error after scaling the three variables fairly | `0` is perfect; smaller is better |
| Persistence NRMSE | Error from assuming that the state does not change | A learned model should beat this simple baseline |
| Forecast NRMSE | Error after repeatedly feeding predictions back into the model | Read the complete curve, not only one lead |
| Useful forecast horizon | First lead at which forecast NRMSE reaches `1` | Later is better only if other dynamics remain realistic |

One model step is `0.05` Lorenz time units. If the horizon is `null`, or is
shown as an open triangle, NRMSE did not reach `1` during the 200-step
evaluation window. This does not mean the horizon is infinite. A non-finite
forecast fails at its first non-finite lead.

### Stability and variability

| Metric | Meaning | How to read it |
|---|---|---|
| Finite-trajectory fraction | Fraction of 32 long rollouts without NaNs or infinities | `1` is desirable |
| Within-reference-bound fraction | Fraction of finite rollouts without gross amplitude explosions | `1` is desirable, but this is only a basic safety check |
| Variance ratio | Model variance divided by reference variance for `x`, `y`, and `z` | Near `1` is desirable; below `1` suggests collapse, above `1` excessive variability |

A finite and bounded trajectory can still converge to the wrong attractor.
Treat these metrics as gates, not proof of dynamical fidelity.

### Sensitivity to initial conditions

The evaluator begins with two almost identical states and follows how their
distance changes in the emulator and the RK4 reference.

| Metric | Meaning | How to read it |
|---|---|---|
| Perturbation-growth curve | How quickly nearby trajectories separate | Compare its shape and time scale with the reference |
| Effective growth rate | Compact summary of early, pre-saturation separation | Closer to the reference is better |
| Finite-pair fraction | Fraction of perturbation pairs still numerically valid at each lead | A drop indicates that numerical failure affects the comparison |

Much faster growth suggests excessive sensitivity. Much slower growth may
indicate overly damped dynamics. The effective growth rate is a finite-time
summary, not a rigorous global Lyapunov exponent. A `null` rate means too few
valid points were available for the fit.

### Long-term behaviour

These statistics compare reference and emulator rollouts of equal duration
after the same burn-in.

| Metric | Question it answers | How to read it |
|---|---|---|
| Mean and standard deviation | Are the centre and spread of `x`, `y`, and `z` similar? | Compare directly with the reference |
| Wasserstein distance | How different are the one-variable distributions? | Smaller is better |
| Positive-`x` fraction | Does the model spend a similar amount of time in the two lobes? | Compare with the matching reference |
| Lobe-switch rate | Does the model move between lobes at the correct frequency? | Compare switches per Lorenz time unit |
| Mean residence time | Does the model remain in a lobe for a realistic duration? | Compare typical same-lobe duration |

No row is sufficient alone. Correct lobe occupancy can coexist with incorrect
switching, and correct one-variable distributions can coexist with the wrong
attractor geometry or temporal ordering.

## 4. Read the automatic figures

Each completed evaluation matrix creates two figure types for every tested
`rho`.

### Forecast comparison figure

This shows every seed's forecast-NRMSE curve, the model mean and population
standard deviation, persistence, and the NRMSE threshold. First check whether
the seeds tell a consistent story; then compare the model means.

### Diagnostic comparison figure

This compares matched seeds for ten predictive and dynamical summaries.
Connected seed pairs show whether a change is consistent across seeds.

Use these directions:

- lower is better for error and mismatch metrics;
- higher is better for useful horizon and finite/bounded fractions;
- values derived from model-reference differences should approach zero;
- missing values are unavailable or failed evidence, not zeros; and
- no panel is an overall model ranking.

Exact values and missing counts are stored in `matrix_summary.json`. The
single-checkpoint model-autopsy figure is useful for diagnosis, but never choose
a conclusion from one visually attractive seed.

## 5. Team A reading order

Team A compares one-step training (A0) with four-step closed-loop training
(A1). Ask:

1. Did one-step skill remain comparable?
2. Did the full forecast curve and useful horizon improve consistently across
   seeds?
3. Is any horizon change supported by realistic perturbation growth?
4. Did stability, variability, distributions, and lobe behaviour remain
   realistic?
5. What improved, what worsened, and what remained seed-sensitive?

Describe the result as a trade-off rather than simply naming a winner. A1 also
performs more model evaluations during training, so A0 and A1 are not an
equal-compute comparison.

## 6. Team B reading order

Team B compares the state-only model (B1) with the `rho`-conditioned model
(B2). Analyse each regime separately:

1. `rho=28`: in-distribution control on independent trajectories;
2. `rho=30`: unseen interpolation inside the training range; and
3. `rho=24`: out-of-range extrapolation.

Apply the five-question sequence at every `rho`. Do not average the three
regimes into one result. Success at `rho=28` or `rho=30` does not prove
extrapolation at `rho=24`.

## 7. A simple conclusion template

> Compared with **[reference model]**, **[tested model]** improved/worsened
> **[forecast evidence]** across **[number]** of three seeds. This change
> was/was not supported by **[perturbation evidence]** and **[long-term
> evidence]**. The strongest disagreement or limitation was **[limitation]**.
> We therefore find evidence for **[narrow claim]**, but the results do not
> establish **[stronger unsupported claim]**.

For Team B, state whether the claim concerns in-distribution skill,
interpolation, or extrapolation.

## 8. Minimum reporting checklist

For every mandatory neural configuration:

- show seeds 41, 42, and 43 individually;
- report the mean and population standard deviation (`ddof=0`);
- include one-step NRMSE and the complete forecast-NRMSE curve;
- check finite and bounded fractions before interpreting long-term metrics;
- compare perturbation growth and long-term behaviour with the matching RK4
  reference;
- state which diagnostics agree, disagree, or remain seed-sensitive; and
- state the strongest limitation.

No individual metric proves that the emulator learned the dynamics. The claim
becomes stronger only when predictive skill, stability, sensitivity, long-term
behaviour, and - for Team B - parameter response tell a consistent story.

## 9. Exact NRMSE definition

Let `s = (s_x, s_y, s_z)` be the standard deviation calculated from the
training split, and let `e` be prediction minus reference. The evaluator uses

$$
\mathrm{NRMSE}=\sqrt{\operatorname{mean}\left[(e/s)^2\right]},
$$

where division is component-wise and the mean is taken over trajectories and
state variables. Frozen rollout lengths, burn-in, perturbation size, bounds,
fit windows, and quantiles are documented in
[Frozen Benchmark v1.0](frozen_benchmark_v1.0.md).
