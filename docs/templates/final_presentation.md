# Final Presentation Template

Target: approximately 12 minutes speaking plus 3 minutes for questions. Use no
more than seven main slides. Put detailed per-seed tables in backup slides.

## Slide 1: Question and hypothesis - 1 minute

- Team and mandatory comparison.
- One falsifiable hypothesis written before evaluation.
- Why the question matters beyond Lorenz-63.

## Slide 2: Controlled design - 1.5 minutes

- What was held fixed.
- The one intended intervention.
- Datasets, regimes and matched seeds.
- Disclose unequal compute or parameter-count differences.

## Slide 3: Predictive evidence - 2 minutes

- One-step NRMSE with seed spread.
- Complete forecast-NRMSE curves.
- Useful horizon and persistence reference.

## Slide 4: Stability and sensitivity - 2 minutes

- Finite and broad-bound fractions.
- Perturbation-growth curves and effective-rate comparison.
- Explain whether longer skill reflects credible or overly damped dynamics.

## Slide 5: Long-term evidence - 2 minutes

- Variance ratios and state distributions.
- Occupancy, switching and residence behavior.
- One model-autopsy figure, annotated rather than merely displayed.

## Slide 6: Evidence synthesis - 2 minutes

- Which diagnostics agree.
- Which diagnostics contradict one another.
- Seed sensitivity and any failed or `null` metric.
- Team B must separate in-distribution, interpolation and extrapolation claims.
- Team B may use `team_b_three_regime_summary.png` here: read its columns in
  the order `rho=28`, `rho=30`, `rho=24` and do not turn the directional seed
  counts into an overall score.

## Slide 7: Conclusion and limitation - 1.5 minutes

- One precise principal conclusion.
- The strongest limitation.
- The smallest next experiment that would challenge or strengthen the claim.
- Avoid saying that the model “understands” or “learned the dynamics” without
  specifying the evidence category.

## Required backup material

- Per-seed result tables.
- Dataset, checkpoint and result hashes.
- Resolved configs and Slurm job IDs.
- Failed runs and deviations.
- Optional-extension results clearly separated from the mandatory comparison.
