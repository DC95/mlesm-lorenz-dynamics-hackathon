# Scientific Challenge Contract v0.2

**Working title:** *When Is an AI Weather Model Dynamically Trustworthy?*  
**Scientific question:** *Two AI emulators can have almost identical one-step errors. How do we determine which one has actually learned a trustworthy dynamical system?*  
**Status:** Organizer review draft  
**Public release:** 18 August 2026  
**Hackathon start:** 19 August 2026  
**Format:** Approximately 2.5 days; 6–8 participants in two teams of 3–4  
**Mentoring:** One challenge mentor

## 1. Purpose

This challenge investigates why predictive accuracy alone is insufficient evidence that a learned emulator reproduces the dynamics of a physical system.

The Lorenz-63 system will be used as a controlled scientific laboratory. Its equations, reference solutions, instability, long-term behaviour, and response to parameter changes are all known. This makes it possible to perform experiments that are difficult to isolate in a full weather or climate model while preserving a direct conceptual connection to forecasting, uncertainty growth, climate stability, and regime change.

The challenge is intended to produce:

1. an interesting and defensible scientific result during the hackathon; and
2. a reusable teaching and research challenge for future courses.

The results may also provide concrete examples and arguments for a broader perspective paper on predictive skill, dynamical understanding, and scientific evaluation of AI-based Earth-system models.

## 2. Meaning of “trustworthy dynamics” in this challenge

“Trustworthy” is not treated as a binary or universal property of a model. A model can be credible for one purpose and unreliable for another. The teams will therefore assess evidence along distinct capability dimensions:

1. **One-step prediction:** Does the emulator improve meaningfully over persistence and a linear baseline?
2. **Finite-horizon forecasting:** How does forecast error grow when predictions are fed back autoregressively?
3. **Numerical and dynamical stability:** Does the rollout remain finite and physically plausible, or does it explode, collapse, or drift to a spurious attractor?
4. **Local sensitivity:** Does the emulator reproduce the growth of small perturbations over many initial states?
5. **Long-term statistical fidelity:** Does it reproduce state distributions, lobe occupancy, switching behaviour, and residence times?
6. **Response to changed dynamics:** When the governing parameter changes, does the model reproduce the corresponding qualitative and quantitative response?

One-step error cannot, by itself, establish any of dimensions 2–6. Final claims must identify the specific dimensions for which evidence was obtained and must not claim that a model has “learned the dynamics” without qualification.

## 3. Shared scientific foundation

Both teams will begin from the same organizer-supplied benchmark and will first reproduce a common baseline result.

The common foundation will contain:

- an intuitive introduction to Lorenz-63 and its connection to physical prediction;
- a high-accuracy numerical reference solution;
- trajectory-disjoint training, validation, and test data;
- normalization computed from training data only;
- persistence and linear baselines;
- a direct one-step multilayer perceptron baseline;
- a common autoregressive rollout and evaluation harness; and
- fixed configurations and seeds for reproducibility.

No prior familiarity with Lorenz systems is assumed. Participant PyTorch experience is expected to range from low to moderate, so the baseline must run end-to-end before participants are asked to modify a model or loss.

The shared baseline is not the scientific result. It establishes a controlled starting point from which the two teams investigate complementary hypotheses.

## 4. Team A — From one-step skill to trustworthy rollout

### Primary question

Which learning formulation best converts similar one-step predictive skill into stable and dynamically faithful autoregressive behaviour?

### Minimum comparison

Team A will compare:

1. direct next-state prediction;
2. residual or increment prediction; and
3. multi-step rollout training applied to one of these formulations.

The comparison must control obvious confounding factors such as data, normalization, approximate model capacity, evaluation initial conditions, and training effort.

### Required scientific conclusion

Team A must determine whether improved forecast horizon is accompanied by improvement—or degradation—in:

- rollout stability;
- perturbation growth;
- long-term state statistics; and
- attractor or regime-switching behaviour.

A plausible result is that multi-step training improves stability and forecast horizon while making the emulator excessively smooth. This is a hypothesis to test, not an expected answer.

## 5. Team B — Learning changing dynamics

### Primary question

Can parameter conditioning teach an emulator a family of Lorenz systems, including behaviour near or across a qualitative dynamical transition?

### Minimum comparison

Team B will compare:

1. a state-only model trained at the standard parameter setting;
2. a parameter-conditioned model trained across multiple values of the Lorenz control parameter `rho`; and
3. interpolation and extrapolation to parameter values not used for training.

### Required conceptual distinction

A state-only model trained at `rho = 28` is not told that `rho` has changed. Applying it to another regime tests failure under a changed data-generating system; it does not test learned parameter response.

A parameter-conditioned model can be tested for interpolation and extrapolation because the changed control parameter is an explicit model input. Team B must keep these two experiments conceptually separate.

### Required scientific conclusion

Team B must determine whether parameter conditioning reproduces:

- short-range forecast changes;
- stable long-term behaviour;
- changes in state distributions and switching; and
- the qualitative transition near the critical regime around `rho ≈ 24.74`.

Correct interpolation must not be presented as proof of reliable extrapolation.

## 6. Common evidence and outputs

Every principal learned configuration will be evaluated from multiple unseen initial conditions. Principal comparisons should be repeated across multiple training seeds when runtime permits; the final minimum will be set after the JURECA rehearsal.

Each team will provide:

- one explicit, falsifiable hypothesis;
- reproducible training and evaluation commands;
- frozen configurations and seeds;
- forecast error as a function of lead time, including persistence;
- a documented rollout-stability assessment;
- long-term statistical diagnostics;
- the team-specific primary diagnostic;
- one four-panel “model autopsy” summarizing the main result;
- the strongest limitation of the experiment;
- one failed or inconclusive experiment and what was learned from it; and
- a 15-minute final presentation.

Team A’s primary diagnostic is perturbation growth and rollout formulation. Team B’s primary diagnostic is response to changed `rho`, including interpolation, extrapolation, and qualitative regime behaviour.

There will be no single score that defines the best model. A well-diagnosed negative result is scientifically more valuable than a marginally lower error without an explanation.

## 7. Scope control

The challenge is not primarily an architecture competition. Recurrent networks, neural ODEs, Transformers, probabilistic models, physics-informed losses, or symmetry-aware approaches are optional only after the required comparison is complete.

The following are explicitly insufficient as final results:

- reporting only one-step mean-squared error;
- showing one visually attractive rollout;
- comparing many architectures without a controlled hypothesis;
- declaring that a model “understands chaos” from a phase-space plot; or
- using larger models or more GPU time as evidence of scientific quality.

Given one mentor and mixed participant experience, the common workflow must remain simple enough that scientific diagnosis—not infrastructure debugging—occupies most of the event.

## 8. Event and collaboration assumptions

- The opening pitch will last 3–5 minutes.
- The first day provides approximately eight working hours.
- Participants will arrive with active JSC accounts.
- The challenge has one dedicated mentor.
- Formal judging or competition against other challenges is not assumed.
- The two teams contribute complementary parts of one scientific investigation.
- Team A and Team B will work on separate Git branches created from the same frozen participant release.

The intended branch names are:

- `team-a-rollout-fidelity`
- `team-b-changing-dynamics`

The teams’ solutions may be made public after the event.

## 9. Repository and release policy

The repository remains private during preparation and will become public on 18 August 2026. The participant release will contain the common baseline, public data definitions, shared evaluation code, team instructions, and JURECA workflow.

The original challenge PDF and teaching notebook may be publicly redistributed. The code will use an open-source licence; the exact licence must be selected before the public release.

Organizer-only material and any unreleased evaluation cases must not be placed on participant branches before the event.

## 10. Success criteria

The hackathon succeeds if:

1. both teams reproduce the common baseline;
2. each team completes at least one controlled comparison;
3. each team reaches a scientifically defensible conclusion, including uncertainty and limitations;
4. the conclusions distinguish predictive skill from stronger evidence of dynamical fidelity; and
5. the resulting repository can be reused and extended after the event.

The result does not need to confirm the initial hypothesis. A clear explanation of why an apparently successful emulator fails a dynamical test is a successful scientific outcome.

## 11. What this contract does not yet freeze

This contract freezes the scientific purpose, team structure, evidence hierarchy, outputs, and release intent. It does **not** yet freeze:

- the exposed ML forecast interval;
- final trajectory counts and rollout lengths;
- exact training and test values of `rho`;
- the minimum number of training seeds;
- numerical stability thresholds;
- the final GPU/run budget; or
- the exact open-source licence.

These settings will be selected only after a complete JURECA rehearsal using account `training2635`. They must be frozen before the participant release and must not be changed after teams inspect final test results.

## Approval gate

This document must be reviewed and approved by the organizer before numerical settings are rehearsed or starter code is revised. Approval of this contract authorizes the next stage—JURECA validation—but does not imply approval of every provisional value in `benchmark_spec_v0.1.md`.
