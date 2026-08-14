# MLESM Hackathon Challenge: Scientific Benchmark Specification

**Working title:** *When Is an AI Weather Model Dynamically Trustworthy?*  
**Subtitle:** *Lorenz as a wind tunnel for prediction, chaos, climate stability, and regime change*  
**Status:** Organizer draft v0.1  
**Expected participation:** 6-8 participants in two teams of 3-4  
**Challenge duration:** Approximately 2.5 days

## 1. Scientific premise

Two learned emulators can have nearly identical one-step validation errors while behaving very differently when used autoregressively. One may reproduce useful short-range trajectories, perturbation growth, regime transitions, and the long-term state distribution. Another may drift, become excessively dissipative, remain artificially chaotic, collapse onto a spurious attractor, or become numerically unstable.

The challenge therefore asks:

> **What evidence is required before we can say that a neural emulator has learned trustworthy dynamics rather than only an accurate one-step mapping?**

The Lorenz-63 system provides a minimal, controlled setting in which the true equations, parameters, numerical reference solution, attractor, local instability, long-term statistics, and parameter interventions are all available. Participants will use it as a scientific test bench for questions that also arise in autoregressive AI weather and climate models.

## 2. Challenge objective

Participants will not be rewarded primarily for obtaining the smallest one-step mean-squared error. They will construct and diagnose learned emulators across five distinct capability dimensions:

1. short-range forecast skill;
2. autoregressive numerical stability;
3. local sensitivity and perturbation growth;
4. long-term or climate-like statistical fidelity;
5. response to changed system parameters.

The main outcome should be a defensible scientific conclusion about **why** a model succeeds or fails on these dimensions.

## 3. Scope and non-goals

### In scope

- Supervised neural emulation of Lorenz-63 evolution.
- Direct next-state, residual/increment, tendency, and multi-step formulations.
- State-only and parameter-conditioned emulators.
- Controlled ablations of training target, loss, rollout horizon, data coverage, or parameter conditioning.
- Dynamical diagnostics evaluated over ensembles of unseen initial conditions.
- Reproducible PyTorch experiments on JURECA-DC.

### Not the primary objective

- Training the largest possible model.
- Comparing many architectures without controlled hypotheses.
- Selecting a winner from one scalar leaderboard score.
- Judging long-term fidelity from one visually attractive trajectory.
- Treating one-step MSE as sufficient evidence that the model learned the dynamics.

Recurrent networks, neural ODEs, probabilistic models, symmetry-aware models, and physics-informed losses are optional extensions only after the team's required comparison is complete.

## 4. Reference dynamical system

The Lorenz-63 equations are

\[
\frac{dx}{dt}=\sigma(y-x),
\qquad
\frac{dy}{dt}=x(\rho-z)-y,
\qquad
\frac{dz}{dt}=xy-\beta z.
\]

The standard configuration is

\[
\sigma=10,
\qquad
\rho=28,
\qquad
\beta=\frac{8}{3}.
\]

### Numerical reference

- Reference solver: fourth-order Runge-Kutta (RK4), evaluated in `float64`.
- Internal reference time step: `dt_reference = 0.001`.
- Exposed ML forecast interval: `delta_t = 0.05`.
- Each ML transition therefore spans 50 internal RK4 steps.
- All reference datasets and evaluations must be generated with the same implementation and tolerances.

The larger exposed forecast interval is intentional. At `delta_t = 0.01`, the next state is so close to the current state that excellent one-step scores can be obtained by learning an almost-identity mapping. Persistence will remain a mandatory baseline at the selected interval.

## 5. Data protocol

### 5.1 Initial conditions and burn-in

- Candidate initial states are sampled independently from a documented broad box in state space, provisionally `[-20, 20]^3`.
- Each candidate is integrated for a burn-in period of 20 Lorenz time units.
- Stored samples begin only after burn-in.
- Initial conditions and random directions used in perturbation experiments are generated from fixed organizer-controlled seeds.

### 5.2 In-distribution dataset at rho = 28

Provisional dataset sizes:

| Split | Independent trajectories | Stored transitions per trajectory | Visibility |
|---|---:|---:|---|
| Training | 256 | 400 | Public |
| Validation | 64 | 400 | Public |
| Public test | 64 | 400 | Public |
| Hidden final test | 128 | Evaluation rollouts | Organizer only until results freeze |

Training, validation, public-test, and hidden-test trajectories must be disjoint. Splitting individual time steps from the same trajectory across datasets is prohibited.

### 5.3 Normalization

- Normalization statistics are computed from training inputs only.
- The same training statistics are applied to training targets, validation, tests, and autoregressive rollouts.
- Teams must not recompute normalization for a shifted parameter regime unless the experiment is explicitly labelled as recalibration or fine-tuning.
- All reported metrics are evaluated after returning predictions to physical Lorenz coordinates, except explicitly normalized forecast scores.

### 5.4 Parameter-shift datasets

Two different questions must not be conflated.

#### State-only model

A state-only emulator

\[
\hat{\mathbf{x}}_{t+\Delta t}=f_\theta(\mathbf{x}_t)
\]

trained at `rho = 28` has no explicit information that the governing parameter has changed. Evaluating it on trajectories generated at another rho is a **failure-under-system-shift test**, not proof that the model can reproduce parameter response.

#### Parameter-conditioned model

A parameter-conditioned emulator

\[
\hat{\mathbf{x}}_{t+\Delta t}
=f_\theta(\mathbf{x}_t,\sigma,\rho,\beta)
\]

can be tested for interpolation and extrapolation across parameter settings.

Provisional Team B protocol:

- Training rho values: `{26, 28, 32}`.
- Interpolation test: `rho = 30`.
- Extrapolation tests: `rho = 24` and `rho = 35`.
- Additional hidden parameter values may be used by the organizer.

The test at `rho = 24` is scientifically important. For the standard sigma and beta, the nonzero equilibria change stability near

\[
\rho_H=
\frac{\sigma(\sigma+\beta+3)}{\sigma-\beta-1}
\approx 24.74.
\]

Thus, the `rho = 24` experiment probes whether a learned emulator can reproduce a qualitative change in long-term behaviour rather than only a modest shift within the standard chaotic regime.

## 6. Organizer-supplied baselines

All teams receive the following common baselines:

1. **Persistence:**
   \[
   \hat{\mathbf{x}}_{t+\Delta t}=\mathbf{x}_t.
   \]

2. **Linear residual model:**
   \[
   \hat{\mathbf{x}}_{t+\Delta t}=\mathbf{x}_t+A\mathbf{x}_t+\mathbf{b}.
   \]

3. **Direct one-step MLP:** three hidden layers, `tanh` activations, and a normalized one-step MSE objective.

The baseline package must reproduce a complete training and evaluation run with one documented command. The supplied MLP is a reference, not an architecture that teams are required to preserve.

## 7. Team investigations

The teams answer complementary questions using the same data and evaluation harness.

### Team A: From one-step skill to trustworthy rollout

**Primary question:** Which learning formulation best converts one-step accuracy into useful, stable, and dynamically faithful autoregressive behaviour?

Required comparison:

1. direct next-state prediction;
2. residual or increment prediction;
3. multi-step rollout training applied to one of the above.

Suggested primary hypothesis:

> Multi-step training will extend useful forecast horizon and reduce attractor drift, but may suppress realistic perturbation growth if the loss rewards overly smooth trajectories.

Team A must report whether improvements in forecast horizon come with gains or losses in climate statistics and local instability.

### Team B: Generalization across changing dynamics

**Primary question:** Can parameter conditioning teach an emulator a family of dynamical systems, including behaviour near or beyond an unseen parameter transition?

Required comparison:

1. a state-only model trained at `rho = 28`;
2. a parameter-conditioned model trained at multiple rho values;
3. interpolation and extrapolation tests using the frozen evaluation harness.

Suggested primary hypothesis:

> Parameter conditioning will improve interpolation between training regimes, but correct extrapolation across the stability transition near `rho = 24.74` will remain substantially harder.

Team B must distinguish interpolation, extrapolation, and unidentifiable parameter response in its conclusions.

## 8. Required evaluation scorecard

No single metric is declared the overall truth. Every principal model must complete the same multidimensional scorecard.

### 8.1 One-step skill

- Physical-coordinate MSE and RMSE for each variable.
- Normalized aggregate RMSE.
- Improvement over persistence and the linear baseline.

One-step performance is a diagnostic, not the final ranking criterion.

### 8.2 Ensemble forecast skill

- Autoregressive rollouts from at least 128 unseen initial conditions.
- Normalized RMSE as a function of lead time.
- Median and interquartile range across initial conditions.
- Provisional useful forecast horizon: the first lead time at which aggregate normalized RMSE reaches 1.0, corresponding to a climatological-scale error in standardized state space.
- The complete lead-time curve must be shown even when a scalar horizon is reported.

### 8.3 Stability

- Fraction of rollouts containing NaN or infinite values.
- Fraction exceeding a documented broad reference-state bound.
- Long-run variance ratio relative to the RK4 reference, used to flag artificial collapse or excessive dispersion.

Any numerical clipping or state projection must be reported as part of the model rather than hidden inside evaluation.

### 8.4 Perturbation growth

- At least 128 unseen reference states.
- Random perturbation directions with documented magnitude relative to training standard deviations.
- Reference and emulator perturbation-growth curves.
- Median and interquartile range across initial states.
- Approximate effective early growth rate fitted only in a documented pre-saturation distance range.

The result must be described as an effective finite-time growth diagnostic, not as a rigorous global Lyapunov exponent.

### 8.5 Long-term statistical fidelity

Required:

- mean and standard deviation of `x`, `y`, and `z`;
- one-dimensional Wasserstein distance for each state variable;
- lobe occupancy, provisionally defined from the sign of `x`;
- lobe-switching frequency;
- residence-time distribution or its documented summary.

Recommended extension:

- autocorrelation of `x`;
- power spectral density;
- a multivariate distribution or attractor-geometry diagnostic.

These statistics must be computed after a common evaluation burn-in and over equal-duration reference and emulator rollouts.

### 8.6 Parameter generalization

For parameter-shift experiments, repeat at minimum:

- ensemble forecast RMSE versus lead time;
- stability rate;
- long-term mean, variance, and state distributions;
- qualitative long-term regime classification.

At `rho = 24`, report explicitly whether the emulator approaches the appropriate stable behaviour or remains spuriously chaotic.

## 9. Repetition and uncertainty

- Each principal learned configuration is trained with at least three random seeds.
- Model-selection decisions use validation data only.
- Public-test data may be used for development diagnostics but not repeated hyperparameter selection after results freeze.
- Final claims are verified on hidden initial conditions and, where applicable, hidden parameter settings.
- Report the central tendency and spread across training seeds.

## 10. Required challenge deliverables

Each team submits:

1. one reproducible training command for each principal model;
2. one reproducible evaluation command;
3. frozen configuration files and random seeds;
4. a machine-readable benchmark result file;
5. model checkpoints for the principal comparison;
6. a four-panel **model autopsy** containing:
   - forecast error versus lead time;
   - reference and learned long-term behaviour;
   - perturbation growth;
   - parameter-shift or primary-ablation result;
7. a concise README containing:
   - hypothesis;
   - experiment;
   - conclusion;
   - strongest limitation;
   - one failed or inconclusive attempt and what was learned;
8. a seven-minute final presentation.

## 11. Cross-team audit

Before the final results freeze:

- Each team provides its principal checkpoint and configuration to the other team.
- The receiving team runs the shared evaluation command independently.
- Differences between original and reproduced results must be resolved or documented.
- Evaluation code must not contain model-specific exceptions that silently advantage one method.

## 12. Scientific judging rubric

| Criterion | Weight |
|---|---:|
| Clear, falsifiable scientific hypothesis | 20% |
| Fair experimental design and controlled comparison | 20% |
| Quality of dynamical evaluation | 25% |
| Generalization and robustness analysis | 15% |
| Reproducibility | 10% |
| Interpretation and communication | 10% |

A correctly diagnosed negative result can score more highly than an unexplained model with a lower one-step error.

## 13. Compute policy

- The Lorenz problem is intentionally small; model size is not a success criterion.
- GPUs should be used for parallel seeds, controlled sweeps, multi-step backpropagation, parameter ensembles, and vectorized evaluation.
- Teams receive the same provisional compute budget.
- A provisional cap is 24 training runs per team, excluding organizer-provided baselines and failed infrastructure smoke tests.
- The final JURECA allocation policy will specify the number and duration of four-GPU node allocations.
- Production experiments must run without downloading packages, data, or pretrained weights from the internet.

## 14. Decisions still to freeze before code release

The following organizer decisions remain open in v0.1:

1. Confirm the broad initial-condition sampling box.
2. Confirm public and hidden trajectory counts against runtime measurements.
3. Confirm the exact long-rollout duration and reference-state stability bound.
4. Confirm training rho values and any hidden parameter settings.
5. Confirm whether `sigma` and `beta` remain fixed throughout Team B experiments.
6. Confirm the final compute allocation and Slurm account information.
7. Decide whether teams are formally competing or contributing complementary investigations to a joint challenge result.

These decisions must be frozen in v1.0 before participants begin. Teams should not be able to change the benchmark after viewing final-test results.
