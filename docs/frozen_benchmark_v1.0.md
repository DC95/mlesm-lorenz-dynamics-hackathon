# Frozen Benchmark v1.0

**Status:** Participant release settings, rehearsed on JURECA-DC
**Applies to:** Team A and Team B mandatory comparisons
**Companion:** [Evaluation evidence guide](evaluation_evidence_guide.md)

This document is the numerical authority for the released benchmark. The
scientific questions and required conclusions are defined in the
[scientific challenge contract](scientific_challenge_contract_v1.0.md).

## 1. Reference system and sampling

| Setting | Frozen value |
|---|---:|
| Lorenz `sigma` | 10 |
| Lorenz `beta` | 8/3 |
| RK4 internal step | 0.001 Lorenz time units |
| Exposed ML interval | 0.05 Lorenz time units |
| Reference burn-in | 20 Lorenz time units |
| Initial-state box | `[-20, 20]^3` |
| Stored steps per trajectory | 400 |

Every stored transition spans 50 RK4 steps. Training, validation, and public
test trajectories are generated from independent initial states and seeds.
Normalization is fitted only to the training split selected by a model and is
stored in its checkpoint.

## 2. Frozen datasets

### Standard benchmark

| Split | `rho` | Trajectories | Generator seed | State shape |
|---|---:|---:|---:|---|
| Training | 28 | 256 | 1101 | `(256, 401, 3)` |
| Validation | 28 | 64 | 2202 | `(64, 401, 3)` |
| Public test | 28 | 64 | 3303 | `(64, 401, 3)` |

File: `standard_benchmark.npz`
SHA-256: `dd3c2072b2bf6780e84d824e001c5e32b8e8f5cb9388d206f848036294f3e138`

### Multi-`rho` benchmark

| Split | `rho` values | Trajectories per `rho` | Generator seed | State shape |
|---|---|---:|---:|---|
| Training | 26, 28, 32 | 128 | 4404 | `(384, 401, 3)` |
| Validation | 26, 28, 32 | 32 | 5505 | `(96, 401, 3)` |
| Public test | 24, 30 | 64 | 6606 | `(128, 401, 3)` |

File: `multirho_benchmark.npz`
SHA-256: `4a8c237c240384286ca4dc0392338d82f64644ce3e9c60f5428ed0b076a8a8fb`

For Team B, `rho=30` is unseen interpolation inside the training range and
`rho=24` is out-of-range extrapolation across a changed dynamical regime. A
separate standard-dataset evaluation at `rho=28` is the in-distribution
control.

## 3. Mandatory models and training

All neural models use direct next-state prediction, three hidden layers of 64
units, `tanh` activation, 30 epochs, batch size 512, Adam with learning rate
0.001 and no weight decay. Mandatory neural runs use matched seeds 41, 42 and
43.

| Model | Data | Input | Closed-loop loss horizon |
|---|---|---|---:|
| A0 | Standard | State | 1 step |
| A1 | Standard | State | 4 steps |
| B1 | Multi-`rho` | State | 1 step |
| B2 | Multi-`rho` | State plus normalized `rho` | 1 step |

A0 and A1 have the same architecture and optimizer settings, but they are not
an equal-compute comparison: A1 evaluates the model four times per training
window and backpropagates through the closed-loop unroll. B2 adds one input to
the first layer; no attempt is made to remove this small parameter-count
difference.

## 4. Frozen evaluation settings

| Setting | Frozen value |
|---|---:|
| Forecast steps | 200 (10 Lorenz time units) |
| Long-rollout trajectories | 32 |
| Long-rollout steps | 4000 (200 Lorenz time units) |
| Evaluation burn-in | 400 steps (20 Lorenz time units) |
| Perturbation pairs | Up to 64 |
| Perturbation steps | 200 |
| Relative perturbation distance | `1e-5 * ||training_state_std||` |
| Useful-horizon threshold | NRMSE >= 1 |
| Broad-bound reference quantile | 0.999 |
| Broad-bound multiplier | 5 |
| Wasserstein quantiles | 999, from 0.001 to 0.999 |

The effective perturbation-growth fit uses finite median distances between
five times the initial separation and 0.1 times the Euclidean norm of the
training state standard deviation. At least three fitted points are required.
This is a finite-time diagnostic, not a global Lyapunov exponent.

## 5. Failure and provenance policy

- The first non-finite forecast lead counts as useful-horizon failure.
- One-step metrics are `null` unless every one-step prediction is finite; the
  finite-prediction fraction remains available.
- Model climate metrics are reported only when every long rollout is finite.
  Otherwise they are `null`, preventing a biased comparison based only on
  surviving trajectories.
- Every result file records the checkpoint and dataset SHA-256 hashes, sample
  counts, time steps, and evaluation settings.
- Clipping, projection, early termination, or post-processing is part of the
  model and must be disclosed. It must not be hidden inside evaluation.

The contract tests in `starter/tests/` protect these settings and the intended
controlled comparisons.
