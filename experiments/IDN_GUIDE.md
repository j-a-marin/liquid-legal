# IDN Builder Guide

> Governing document for IDN Stage 1 and beyond. Adopted 2026-07-23.
> Where this guide conflicts with earlier protocol documents
> (`PREREGISTRATION.md` sections B–F), this guide wins.

## Objective

Build IDN as a predictive model—not as proof of literal institutional dynamics.

Core hypothesis:

> Separating elapsed-time transitions from event-conditioned updates improves calibrated forecasts for irregular legal-event sequences.

Do not claim that latent dimensions are real institutional variables, gates identify mechanisms, or event jumps are causal effects.

## 1. Enforce correct chronology

For event \(e_k\) arriving after interval \(\Delta t_k\):

\[
c_{k-1}=\operatorname{HistoryEncoder}(e_{\le k-1},t_{\le k-1})
\]

\[
z_k^-=\Phi(z_{k-1},c_{k-1},\Delta t_k,u_{(t_{k-1},t_k)})
\]

\[
z_k=J(z_k^-,e_k,c_{k-1})
\]

Then recompute \(c_k\).

Requirements:

- The pre-event flow must never see \(e_k\).
- Interval covariates \(u\) must be available before \(e_k\).
- Add automated leakage tests.
- Call the encoder “causally masked” or “temporal,” not causally inferential.

## 2. Stage 1 architecture

Implement in `experiments/` only:

- Event encoder.
- Causally masked history encoder.
- Deterministic elapsed-time transition.
- Event-conditioned jump update.
- Explicit continuous/discrete gate.
- Shared latent state.
- Settlement, duration and recovery outputs.
- Auxiliary next-event type and next-event time objectives.

Treat \(\Phi\) as a predictive transition—not literal institutional motion.

## 3. Anchor the gate

Do not interpret individual latent coordinates. Partition state structurally:

\[
z=[z^{event},z^{clock},z^{context},z^{static}]
\]

Constrain updates:

- `event`: changes only at observed events.
- `clock`: evolves during intervals.
- `context`: receives history-encoder information.
- `static`: fixed descriptors.

Evaluate blocked groups through ablation and planted-mechanism tests. Gate plots alone are not evidence.

## 4. Model observation jointly

Legal events are marked arrivals, not passive samples. Add cause-specific event intensities:

\[
\lambda_m(t\mid\mathcal H_t,z_t)
\]

The model should jointly estimate:

- probability silence continues;
- next-event time;
- next-event type;
- terminal-event risk.

Longer term, derive survival, cumulative incidence and duration from this shared probabilistic object instead of unrelated heads.

## 5. Preserve uncertainty

Stage 1 may use deterministic flow, but:

- output predictive distributions or quantiles;
- track uncertainty across longer gaps;
- do not interpret the deterministic path literally;
- reserve stochastic flow/SDE experiments for a later ablation.

## 6. Generator requirements

Test IDN across hostile synthetic worlds:

1. Pure event-driven state.
2. Duration-only effects.
3. Semi-Markov phases.
4. Deterministic continuous drift.
5. Stochastic diffusion.
6. Unobserved external shocks.
7. Strategic policy changes.

IDN must exploit continuous flow only when present and avoid inventing it elsewhere.

Keep full latent logs for evaluation, but never expose future latent state to model inputs.

## 7. Required baselines

Compare against:

- XGBoost survival/hazard model;
- LSTM;
- GRU-D or time-decay recurrence;
- temporal Transformer with strong relative-time features;
- Transformer temporal point process;
- Mamba/compact SSM;
- Neural CDE;
- ODE-RNN or Latent ODE;
- semi-Markov competing-risk model;
- CfC.

Match or report parameters, tuning budget, epochs, compute, latency, early stopping and seeds.

## 8. Evaluation

Primary endpoint:

- Hidden-statics settlement AUC, paired against `tf-native`.

Also report:

- next-event likelihood;
- next-event time error;
- time-dependent discrimination;
- cumulative-incidence calibration;
- integrated Brier score;
- duration and recovery error;
- inference latency;
- long-gap and regime-transition performance;
- unseen-judge/court transfer.

Use strict as-of-time construction for every cross-matter feature.

## 9. Survival criterion

IDN survives only if it:

- beats the Transformer by mean paired AUC ≥ 0.01;
- has a paired 95% interval excluding zero;
- does not materially regress duration or calibration;
- concentrates its advantage in preregistered timing-sensitive conditions;
- behaves correctly in both pro-IDN and anti-IDN synthetic worlds;
- clears mechanism-group ablations.

Otherwise stop the hybrid track.

## 10. Claim discipline

Permitted:

> Observing an event changed the forecast.

Not permitted without causal identification:

> The event caused the latent state change.

Permitted:

> The clock partition improved prediction during long gaps.

Not permitted:

> The partition recovered the true institutional state.

Working description:

> **IDN is a hierarchical hybrid state-space marked-event model for litigation forecasting.**

Use “liquid” only when the implementation genuinely uses LTC-style learned time constants.
