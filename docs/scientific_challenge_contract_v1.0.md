# Scientific Challenge Contract v1.0

**Title:** *Beyond Predictive Skill: What Evidence Supports Claims of Learned Dynamics?*  
**Pitch hook:** *Same One-Step Error. Different Dynamics.*  
**Subtitle:** *Lorenz-63 as a controlled testbed for AI emulators*  
**Status:** Final scientific scope; numerical settings pending JURECA rehearsal  
**Supersedes:** Earlier organizer drafts  
**Public release:** 18 August 2026  
**Hackathon start:** 19 August 2026  
**Format:** Approximately 2.5 days; 6–8 participants in two teams of 3–4  
**Mentoring:** One challenge mentor

## 1. Central question

Two AI emulators can achieve similar one-step prediction errors while behaving very differently when used as dynamical systems. One may produce stable rollouts, realistic uncertainty growth, and credible long-term behaviour. Another may drift, collapse, become excessively smooth, or respond incorrectly when the governing system changes.

The challenge asks:

> **What evidence supports the claim that an AI emulator has learned the underlying dynamics rather than only an accurate one-step mapping?**

Lorenz-63 provides a controlled setting in which the equations, numerical reference solution, instability, attractor, and response to parameter changes are available. It is not a miniature atmospheric model. It is a minimal scientific laboratory for questions that also arise in AI weather and climate modelling.

## 2. What counts as evidence

The challenge will not assign a binary label stating that a model has or has not “learned the dynamics.” Instead, teams will evaluate dynamical fidelity through three evidence categories:

1. **Predictive skill** — one-step error and forecast error as a function of autoregressive lead time.
2. **Long-term behaviour** — numerical stability, attractor behaviour, and state or lobe distributions.
3. **Team-specific dynamical evidence** — perturbation growth for Team A and response to changed system parameters for Team B.

One-step accuracy is necessary but cannot establish the other two categories. Conclusions must state precisely which evidence improved, deteriorated, or remained inconclusive.

## 3. Participant preparation and shared baseline

No prior knowledge of Lorenz-63 is assumed. Before modelling, participants will receive a separate 10–15-minute primer derived from the earlier student workshop notebook. It will explain:

- Lorenz-63 as a simplified model of thermal convection;
- the physical interpretations of `x`, `y`, and `z`;
- the feedback between circulation and temperature structure;
- deterministic evolution, chaos, and sensitivity to initial conditions;
- the two-lobed attractor and regime switching;
- `rho` as a control parameter representing thermal forcing; and
- the difference between one-step prediction and autoregressive rollout.

Both teams will then reproduce the same organizer-supplied baseline:

- a high-accuracy numerical reference solution;
- trajectory-disjoint training, validation, and test data;
- training-data-only normalization;
- persistence and linear reference models;
- a direct one-step multilayer perceptron; and
- one common training and evaluation interface.

The baseline must run end-to-end before participants change a model or loss. Three training seeds will be launched through one organizer-supplied command so repetition does not become a participant programming task.

## 4. Team A — From one-step training to autoregressive behaviour

### Primary question

Does training through several autoregressive steps improve rollout fidelity, and what does it sacrifice?

### Mandatory comparison

Team A will compare two models with the same architecture, data, normalization, and approximate training effort:

| Model | Training objective |
|---|---|
| A0 | Direct next-state prediction with one-step MSE |
| A1 | Direct next-state prediction with closed-loop four-step rollout loss |

For A1, each predicted state is fed back as the input to the next training step. The frozen rollout horizon is four model steps, corresponding to 0.20 Lorenz time units at the benchmark interval of 0.05.

The organizer will aim to obtain models with sufficiently similar one-step validation skill to make the rollout comparison meaningful. Exact equality is not a participant requirement; any remaining one-step difference must be reported.

### Required conclusion

Team A must determine whether multi-step training changes:

- forecast-error growth and useful forecast horizon;
- rollout stability and long-term attractor behaviour; and
- the growth of small initial perturbations.

Residual prediction, alternative rollout horizons, and additional architectures are optional extensions only after this comparison is complete.

## 5. Team B — Learning changing dynamics

### Primary question

Is exposure to multiple regimes sufficient, or must the governing parameter be supplied explicitly for an emulator to reproduce changing dynamics?

### Mandatory comparison

The shared state-only model trained at `rho = 28` provides the single-regime reference. Team B will then train two models on identical multi-regime data:

| Model | Training data | Model input |
|---|---|---|
| B1 | Multiple `rho` values | State only |
| B2 | The same multiple `rho` values | State plus `rho` |

B1 controls for the benefit of seeing more diverse data. B2 tests whether explicit parameter conditioning allows the model to distinguish systems that can have different tendencies at the same state.

### Required tests

Team B will use:

- one interpolation value of `rho`, provisionally `rho = 30`; and
- one extrapolation or regime-transition value, provisionally `rho = 24`.

The evaluation harness will provide sufficiently long reference rollouts and a common burn-in so that transient behaviour near the transition is not mistaken for the final regime. Exact training and test values will be confirmed during the JURECA rehearsal.

### Required conclusion

Team B must determine whether explicit parameter conditioning changes:

- forecast error in an unseen regime;
- rollout stability and long-term state distributions; and
- the qualitative response across the changed parameter regime.

Same-state tendency tests, additional parameter values, and more advanced intervention experiments are optional extensions.

## 6. Required outputs

Each team will produce:

1. one explicit and falsifiable hypothesis;
2. one controlled mandatory comparison;
3. reproducible configurations and commands;
4. one compact result table;
5. one four-panel **model autopsy** containing:
   - one-step and rollout error;
   - rollout stability or attractor behaviour;
   - a long-term distribution diagnostic; and
   - the team-specific diagnostic;
6. one principal scientific conclusion;
7. the strongest limitation of that conclusion; and
8. one 15-minute final presentation.

There will be no single overall model score. A well-supported negative result is more valuable than a small error improvement without a dynamical interpretation.

## 7. Scope control

The challenge is not an architecture competition. The following are not sufficient final results:

- reporting only one-step MSE;
- showing one selected rollout or phase-space plot;
- comparing many architectures without a controlled hypothesis; or
- claiming internal “understanding” from predictive performance alone.

Residual models, recurrent networks, neural ODEs, Transformers, probabilistic methods, physics-informed losses, formal Lyapunov estimates, spectral diagnostics, and same-state response experiments are optional. They must not displace the mandatory comparison.

The two teams are scientifically complementary but operationally independent. Team B must not wait for Team A to identify a preferred training formulation.

## 8. Event, repository, and success criteria

- The opening pitch will last 3–5 minutes.
- The first day provides approximately eight working hours.
- Participants will arrive with active JSC accounts.
- Formal judging is not assumed.
- Team branches will be created from the same frozen participant release:
  - `team-a-rollout-fidelity`
  - `team-b-changing-dynamics`
- Materials and team solutions may be made public.

The challenge succeeds if both teams reproduce the baseline, complete their mandatory comparison, and produce an interesting, defensible conclusion that distinguishes predictive skill from stronger evidence of dynamical fidelity. The resulting repository should remain reusable for future courses and scientific extensions.

The code will use an open-source licence; the exact licence will be selected before public release.

## 9. Numerical settings still to freeze

The following are deliberately deferred until a complete rehearsal on JURECA using account `training2635`:

- the exposed ML forecast interval;
- final trajectory counts and rollout lengths;
- the multi-step training horizon;
- exact training and test values of `rho`;
- numerical stability thresholds; and
- the final GPU and runtime budget.

These values must be fixed before the public release on 18 August 2026 and must not be changed after teams inspect the final test results.
