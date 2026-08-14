# Organizer roadmap

This page separates existing source material, current proposals, and work that is still required. It is the quickest way to understand the state of the project.

## What already existed

The `original_materials/` directory contains:

1. the earlier tutorial-led Lorenz workshop notebook; and
2. the original Bonn hackathon challenge description.

These files are source material. They have not been edited.

## What v0.1 adds

The benchmark specification proposes four changes to the earlier teaching workflow:

1. use trajectory-disjoint training, validation, and test splits;
2. compare neural models against persistence and linear baselines;
3. evaluate dynamics beyond one-step MSE; and
4. separate failure under a changed system from genuine parameter-conditioned generalization.

The `starter/` directory is a preliminary implementation of those proposals. Its existence does not mean every scientific choice is final.

## Decisions to review with the organizer

- Final challenge title and central pitch.
- Whether the exposed ML forecast interval should remain `0.05`.
- Final trajectory counts and runtime budget.
- Public and hidden values of the Lorenz parameter `rho`.
- Exact mandatory diagnostics and optional extensions.
- Whether teams are assigned or choose between the two investigations.
- What code and results remain hidden until the event.
- Final judging criteria and weights.

## Build sequence

1. Review and freeze the scientific benchmark.
2. Rehearse data generation, training, and evaluation on JURECA.
3. Simplify the starter code based on the rehearsal.
4. Create frozen public datasets and checksums.
5. Write the participant-facing one-page challenge brief.
6. Create the opening pitch and 2.5-day schedule.
7. Prepare the experiment ledger, submission template, and judging sheet.
8. Run a complete organizer rehearsal from a clean account.

## Release states

| State | Meaning |
|---|---|
| Organizer draft | Contains design discussion and provisional code. Current state. |
| Rehearsal candidate | Scientific choices frozen; full JURECA workflow ready to test. |
| Participant release | Only participant-facing instructions, starter code, and public data are exposed. |
| Event archive | Final presentations, winning conclusions, and reproducible results are preserved. |

