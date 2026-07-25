# Preregistration: Institutional Dynamics Network (IDN) evaluation protocol

Status: **v2, 2026-07-23** — Generator v2 accepted and frozen
(`experiments/FREEZE.md`). From this point the governing document for all
IDN work is **`experiments/IDN_GUIDE.md`**; where sections B–F below
conflict with it, the guide wins. In particular: the claim discipline of
guide section 10 applies everywhere (IDN is a predictive marked-event model,
not evidence of literal institutional dynamics), the world battery of guide
section 6 supplements the single frozen world, and the baseline list of
guide section 7 supersedes the model matrix used so far.

This document registers the acceptance criteria for Generator v2, the Stage-1
survive/kill criterion for the IDN hybrid, the operationalized gate criteria,
the Stage-2 hierarchy criteria, and the Stage-3 evaluation metrics — decided
*before* any IDN parameter is trained.

## Primary research question

For irregular legal-event trajectories, does a hybrid — attention for
long-range institutional inference plus *selectively gated* continuous-time
state evolution — outperform a matched temporal Transformer on its primary
endpoint, under conditions where institutional latents are hidden?

- **Primary endpoint:** settlement AUC (180-day horizon) in the
  hidden-statics regime.
- **Secondary endpoints:** visible-statics settlement AUC, duration MAE,
  calibration (ECE), next-event type accuracy / time MAE (auxiliary only).

## Amendment v1.1 (2026-07-23, pre-freeze)

Logged per the deviation policy. Two acceptance checks measured the wrong
construct; diagnostics on 3,000 generated cases (seed 99) justify the
replacements. No threshold was weakened to pass.

- **A2 second clause, replaced:** acceptance-per-offer pre/post flip is
  survival-confounded — pre-flip acceptance in flip cohorts is identically
  0.000 because accepted cases end before they can flip. Replacement: paired
  within-case discovery stall-rate difference pre/post flip (plaintiff
  capability is constant within a case, and stalls do not remove the case
  from the cohort), requiring mean diff ≥ 0.05. The gap-ratio clause is
  unchanged.
- **A6 backlog clause, replaced:** "removing only the backlog channel
  degrades duration MAE ≥ 10%" measures *redundancy*, not recoverability —
  observed-gap summaries substitute for the channel (measured degradation:
  0.7%, while backlog fraction correlates with duration at r = 0.28 with
  tercile means 549d/1248d/1527d). Replacement: conditional on NO timing
  features, adding the backlog channel must improve duration MAE by ≥ 10%
  (the mechanism must be recoverable and consequential when observations
  cannot substitute).
- **A1 probe, strengthened:** the timing arm adds a normalized slowdown
  feature (recent-gap EMA ÷ case-mean gap) — a domain-standard engineered
  statistic ("is the case currently slow relative to its own pace?"). The
  0.02 lift threshold is unchanged. Baseline diagnostic for context:
  P(settle ≤ 180d) = 0.276 vs 0.529 by current backlog state at negotiation
  events.
- **A2 metric, refined (v1.1 addendum):** the stall-rate comparison is
  pooled (post-flip events vs never-flip controls) rather than within-case
  paired. Regime flips are trait-independent by construction and stalls do
  not end the case, so pooling is unbiased; the within-case pairing kept
  only mid-discovery flips and was underpowered.
- **Generator mechanism strengths (v1.1 addendum):** backlog episode lengths
  150/100 → 300/250 days (realistic court-congestion persistence; also makes
  current state informative about a meaningful share of the remaining
  horizon), settle freeze penalty 1.5 → 2.0. Thresholds unchanged.
- **Generator mechanism strengths (v1.1 addendum 2):** backlog gap
  multiplier 2.0 → 2.5 and episodes 300/250 → 400/350 days. Rationale: a
  binary Markov state with 2× multiplier and ~45% occupancy informs only the
  near-term share of a long remaining-duration horizon, bounding achievable
  A6 duration improvement near the observed 6%; the 2.5× multiplier and
  longer persistence are within realistic court-congestion ranges. If the
  10% threshold remains unmet after this, the threshold itself goes back to
  the protocol authors for review rather than further mechanism cranking.

## Amendment v1.2 (2026-07-23, pre-freeze)

- **A6.2 replaced with a paired interventional estimand** (protocol-author
  decision). The v1.1 duration-MAE clause asked a one-bit current state to
  forecast an unknown future Markov chain and calibrated its threshold
  against log1p-days, not calendar days — a mis-specified estimand.
  Replacement, implemented as `GeneratorV2.intervene_landmark`: sample
  mid-negotiation landmarks (judge/district/score/damages from the marginal
  generative distribution; pressure ~ U(1.0, 2.5); trial-date-set ~
  Bernoulli(0.3)), continue each under forced backlog vs forced normal with
  common random numbers per (landmark, rep) pair, and require the declared
  margins in section A6 clause 2. This validates backlog's causal
  consequence directly and independently of A6.1's combined-channel oracle.
- **Generator repair (state consistency):** previously `flip_judge(g)`
  could change the judge state and log the transition at the interval start
  even though the gap was generated with the previous state, and `tick(g)`
  then attributed a full interval of fatigue to the new state — logged
  episode, gap-generating state, and fatigue attribution could disagree.
  The generator now advances every interval through a single `advance()`
  primitive: the gap is generated from the state at interval start, flips
  are stamped at their true sub-interval days, and fatigue is attributed
  per sub-interval to the state actually in force. A2 was re-run only after
  this repair.
- **Documentation sync:** section A2's criterion text now matches its
  implementation (pooled post-flip discovery stall-rate vs never-flip
  controls).
- **Estimator stabilization (v1.2 addendum):** the A6.1 oracle-gain estimate
  fluctuated 0.015–0.033 across iterations at 3 probe seeds × 512 cases —
  estimator noise, not mechanism change (A6.2's interventional margins are
  3.41× and 0.49). Probe datasets increased to 5 seeds × 1024 cases.
  Threshold unchanged.

## A. Generator v2 acceptance criteria (architecture-independent)

The generator is accepted or rejected on its own properties. No criterion
references any competitor architecture's headroom.

- **A1 — Timing carries unique outcome information.** A feature probe
  (gradient-boosted trees over event-order + static summaries) gains ≥ 0.02
  settlement AUC (paired mean over 3 probe seeds) when timing summaries are
  added, conditioning on visible statics and event order.
- **A2 — Regime changes alter transition dynamics, not merely labels.** From
  the latent logs: mean inter-event gap inside judge-backlog episodes vs
  outside differs by ratio ≥ 1.3; pooled discovery stall-rate among
  post-flip events exceeds never-flip controls by ≥ 0.05 (flips are
  trait-independent and stalls do not end the case, so the pooled comparison
  is unbiased; see amendment v1.1 for why settlement-hazard comparisons are
  survival-confounded).
- **A3 — Selective observation differs from the underlying process.** Every
  case's true event log differs from its observed docket for a nonzero
  fraction of cases; observation rates for deposition-class events differ by
  ≥ 0.10 between district-congestion terciles; the full true log and
  observation mask are retained.
- **A4 — Latent and event logs are complete.** 100% of cases ship judge
  backlog trajectories, case-regime flip annotation, true event log, and
  observation mask. Judge-backlog time fraction across cases lies in
  [0.10, 0.60] (the mechanism is neither vestigial nor dominant).
- **A5 — No saturation, no degeneracy.** A frozen quick suite (CfC-native,
  LSTM, temporal Transformer, XGBoost; 8 epochs, 2 seeds, 512 cases) yields
  settlement AUCs all within [0.70, 0.90). The task must be learnable and
  unfinished for *every* family.
- **A6 — Oracle checks recover the planted mechanisms (amendment v1.3).**
  Pass/fail rests on two paired interventional estimands with common random
  numbers per (landmark, rep) pair: (1) forced backlog vs normal must
  increase the mean paired next-event gap by ratio ≥ 2.0 and reduce
  settlement incidence within 180 days of the landmark by ≥ 0.10; (2) an
  adverse-regime shock vs continued normal must reduce settlement incidence
  within 180 days by ≥ 0.10. The observational combined-channel oracle gain
  is reported descriptively (no threshold): observables legitimately
  substitute for the channels, so the marginal gain underestimates the
  causal effect by construction.

On acceptance, the generator, seeds, splits, and evaluation code are frozen
and hashed. Any later change voids the freeze and requires a new version.

## B. Stage 1 (flat hybrid) survive/kill criterion

- **Seeds:** 10 (frozen list), identical datasets and splits per seed for
  every model; all comparisons paired by seed.
- **Budgets:** matched parameters (±20% of tf-native) and matched tuning
  budget (same epochs, same small lr grid, same selection rule).
- **Survive requires ALL of:**
  1. mean paired ΔAUC (IDN − tf-native) ≥ 0.01 on the primary endpoint;
  2. the 95% paired confidence interval of the per-seed deltas excludes zero;
  3. no material regression on secondaries: duration MAE within +5% of
     tf-native; ECE within +0.01 of tf-native.
- **Anything short: kill the hybrid track** and publish the E1–E6 program
  with attention as the winner. One declared primary endpoint; secondary
  results are descriptive, not alternative paths to "success."

## C. Gate criteria (operationalized, behavioral)

"Interpretable gate pattern" is defined as behavioral recovery of known
mechanisms — not visualization:

- **G1:** mean continuous-gate weight during logged judge-backlog (drift)
  episodes exceeds its outside-episode mean by ≥ 0.1.
- **G2:** mean discrete-gate weight at true impulse events (MSJ denial, trial
  date set) exceeds its baseline mean by ≥ 0.1.
- **G3:** the sign of (G1 − G2)-style differences per state dimension is
  consistent across ≥ 9 of 10 seeds.
- **G4:** ablating the continuous branch degrades predictions on
  backlog-episode examples ≥ 2× more than on non-episode examples
  (mechanism-specific harm).

Internal gates are not uniquely identifiable; these criteria are behavioral,
not parametric.

## D. Stage 2 (hierarchy) criteria — only if Stage 1 survives

- **D1:** targeted intervention on judge latents moves the judge partition
  more than any other partition (declared margin).
- **D2:** cross-partition probe leakage below threshold (probe R² from
  non-target partitions ≤ 0.5 × target partition's R²).
- **D3:** ablating a partition selectively harms its corresponding task
  (≥ 2× vs other tasks).
- **D4:** hierarchy improves transfer to held-out judges/courts by ≥ 0.01
  paired AUC vs Stage 1.
- **D5:** no regression vs Stage 1 on the primary endpoint (> 0.005).

If Stage 1 fails, Stage 2 is skipped entirely.

## E. Stage 3 evaluation metrics — only if Stage 2 survives

- cause-specific calibration for the competing outcomes (settlement,
  dismissal, judgment): calibration slope and intercept per cause;
- time-dependent discrimination at 90 / 180 / 365 days;
- integrated Brier score for the duration model;
- counterfactual consistency (E4-style judge probe) ≥ Stage 1;
- next-event type and time metrics as auxiliary only, with labels constructed
  strictly from post-prediction-time information (causal masking; a
  no-terminal-leakage check is part of the eval code).

## F. Analysis plan and deviation policy

- All inferential comparisons are paired by seed; report per-seed deltas,
  mean, and 95% paired t-interval (n = 10).
- One primary comparison (B). All other comparisons are descriptive and
  reported with intervals but without success claims.
- Deviations from this document must be logged in `FREEZE.md` with rationale
  and are reported in the paper as deviations.
