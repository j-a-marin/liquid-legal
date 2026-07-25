# F2_PREREGISTRATION — interval-supervised marked temporal point process (F2 v1)

**Status: v1, 2026-07-25 — CONFIRMATORY once frozen. Frozen before any F2
candidate parameter is trained.** Governance: `experiments/IDN_GUIDE.md`
(claim discipline applies throughout: tf-tpp is a predictive marked-event
model, not evidence of institutional dynamics). Lineage: Stage-1 kill
(`STAGE1_RESULTS.md`, final); exploratory forensics (`F1_FORENSICS_RESULTS.md`)
which **retired the Stage-1 context-conditioned exponential-relaxation clock
topology**. F2 contains no continuous-flow component; per the F1 decision,
any future continuous-time proposal requires a materially different
transition mechanism or training objective — F2 changes the *objective*
(interval supervision), not the transition mechanism.

## 1. Scientific question

> Does jointly modeling event type and event occurrence time — supervising
> the interval itself through a marked temporal point-process likelihood —
> improve legal-event prediction beyond the auxiliary-matched temporal
> Transformer?

The Stage-1/F1 record motivates this: all Stage-1 losses fired at event
boundaries; F1 showed the killed clock never acquired interval semantics
(shuffle-invariant) and contributed only generic residual capacity
(matched by a 1,616-parameter MLP). F2 tests the *likelihood route*: every
day of continued silence becomes training information via the survival term.

## 2. Candidate: `tf-tpp` (F2 v1)

The F2 question is about the **objective**, so the candidate changes the
objective and nothing else. `tf-tpp` = the exact `tf-native-aux` Transformer
trunk (same widths, same causal masking, same time encodings) with the two
marked-event auxiliary heads (next-type CE logits, next-gap quantiles)
**replaced** by per-mark conditional intensity heads:

    λ_m(k) = softplus(w_m · h_k + b_m) + 1e-6      (events/day, per mark m)

where h_k is the causal Transformer output at position k. Hidden state is
constant between observed events (standard THP/RMTPP assumption), so the
conditional intensity is piecewise-constant over each observed interval and
the likelihood is **closed-form** — no numerical integration:

    per interval [t_k, t_{k+1}) with observed mark m_{k+1} and gap Δt (days):
        ℓ_k = log λ_{m_{k+1}}(k) − Δt · Σ_m λ_m(k)

This rewards the observed event at its observed time and penalizes all
unobserved marks throughout the interval; continued silence contributes
training signal. **Parameter cost:** 16 × 32 + 16 = 528 (intensity head) vs
the removed aux heads (next-type 16×32+16 = 528; next-gap 3×32+3 = 99) —
net −99 params. Well within the ±20% budget of tf-native-aux (23,129).

**Event-mark ontology (frozen):** the 16 `EventType` marks of the frozen
generator. Terminal marks {SETTLED, DISMISSED, VERDICT} act as competing
risks. FILED never occurs as a next mark: its log-intensity term therefore
never fires, the survival term drives λ_FILED toward the floor, and **no
masking is applied** — both models are scored over the full 16-mark
distribution, keeping J directly comparable with tf-native-aux's 16-way
softmax.

**Loss (frozen weights):** total = 1.0·settle BCE + 0.3·recovery log1p-MSE +
0.3·duration log1p-MSE + 0.2·duration pinball + 0.3·TPP NLL (mean masked
ℓ over intervals with a next observed event, negated). The main heads and
the duration-quantile head are identical to tf-native-aux; supervision is
equalized in structure, with the TPP NLL standing in for the removed
next-type CE (0.3) + next-gap pinball (0.2) marked-event terms.

**Derived predictions (closed form, from intensities at step k):**
- next-event type: p(m) = λ_m / Λ, Λ = Σ_m λ_m;
- next-gap distribution: exponential with rate Λ → quantile q:
  −ln(1−q)/Λ days (reported through log1p to match the scoring scale);
- settlement within 180 d (cause-specific): P = (λ_SETTLED/Λ)·(1 − e^{−Λ·180});
- remaining duration: expected time to any terminal mark, 1/Λ_term with
  Λ_term = λ_SETTLED + λ_DISMISSED + λ_VERDICT (descriptive only);
- recovery: unchanged conditional head p(R | terminal outcome, T, H_T)
  (log1p-MSE, weight 0.3).

The main settle-BCE head is retained so the primary guard (hidden-statics
settle AUC) is directly comparable to Stage 1; the TPP-derived settlement
probability is reported descriptively alongside it.

## 3. Opponents

- **Primary opponent:** `tf-native-aux`, the auxiliary-matched temporal
  Transformer, using the **archived Stage-1 weights and metrics of record**
  (`archive/stage1-killed/`): the data, splits, losses, and training
  protocol for that model are identical under this preregistration, so
  re-running would reproduce the same frozen artifacts (zero-drift
  reproduction already demonstrated). Declared, not re-trained.
- **Supporting (descriptive only, never part of the screen):**
  - `idn` (killed candidate, archived) — reference row only;
  - `xgb` (archived) — structured baseline;
  - `cr-discrete`: discrete-time cause-specific hazard baseline (monthly
    grid logistic hazards on the same engineered features as xgb), for the
    competing-risk comparison;
  - `sm-empirical`: duration-aware semi-Markov-flavored baseline scoring
    next-gap quantiles from per-(mark, phase-tercile) empirical quantiles
    fit on the training portion.

## 4. Primary endpoint and screen (frozen)

**Primary endpoint:** the joint event-process score

    J = mean masked next-event-type log-loss + mean masked next-gap
        pinball (levels {0.1, 0.5, 0.9}, log1p-day scale)

on holdout steps with a next observed event, hidden-statics regime. Both
models are scored on exactly the same steps; for tf-tpp the type
probabilities and gap quantiles are derived from the intensities
(section 2); for tf-native-aux they come from its auxiliary heads.

**Margin calibration (pre-candidate, exploratory):**
`experiments/results/f2_score_calibration.json` measured J for the two
frozen Stage-1 models from archived weights: tf-native-aux reference
J = 1.4409 (CE 1.1816 + pinball 0.2593); the paired difference between the
two strong equally-supervised models is −0.0114 ± 0.0210 (≈ ±1.5%
relative). The practical margin is frozen at **twice that noise band**.

**Survival requires ALL of:**
1. mean paired relative J reduction (J_tf-aux − J_tf-tpp)/J_tf-aux ≥ **3%**;
2. the 95% paired t-interval (df = 9) of the per-seed relative reductions
   excludes zero;
3. **guards — no material regression vs tf-native-aux (paired means):**
   hidden-statics settle AUC no worse than −0.005; ECE within +0.01;
   duration MAE within +5%; next-event-type accuracy (argmax p(m)) no worse
   than −0.01; training wall-clock ≤ 2× tf-native-aux's archived mean; zero
   NaN/Inf events and all 10 seeds completing (numerical stability).

**Anything short: kill the interval-supervision branch on Generator v2**
(charter default, adopted verbatim): if tf-tpp fails to exceed
tf-native-aux by the preregistered margin on the primary proper scoring
rule while preserving the guards, terminate the interval-supervision
branch. No continuous flow may be added to a TPP model that has failed to
improve the strong baseline. Verdict language: "killed candidate", not
"inconclusive".

## 5. Data, protocol, censoring, observation

- **Data:** frozen `gen_v2.py`, 1024 cases/seed, seeds [0–9], 20% holdout
  via `np.random.default_rng(seed).permutation`, identical to Stage 1.
  Primary regime: hidden statics; visible statics secondary/descriptive.
- **Training protocol:** identical to `STAGE1_SPEC.md` (Adam, batch 32,
  grad clip 1.0, 25 epochs, last-epoch model; lr grid {1e-3, 3e-3} selected
  on the inner split of seed-0's training portion only).
- **Censoring:** every Generator v2 case terminates in an observed terminal
  mark; there is no right-censoring in this world. The likelihood covers
  every observed interval plus the terminal event. (Real-docket censoring
  is a separate, later design problem — section VIII of the charter.)
- **Selective observation:** the candidate trains and is scored on the
  **observed docket**; the observed-process marked likelihood is the
  estimand. Analyses using the latent logs (e.g., scoring intervals that
  contain dropped true events) are **exploratory supplements**, labeled as
  such, and cannot contribute to the screen.

## 6. Generator validation relevant to F2 (architecture-independent)

Declared before training; run on the frozen generator without the
candidate:
1. **Re-affirm A1/A2** from the frozen acceptance (timing carries unique
   outcome information; regimes alter transition dynamics) — already
   frozen in `FREEZE.md`; quoted, not re-run.
2. **Non-degenerate per-mark gap distributions:** report per-mark-family
   gap quantiles on 3 probe seeds × 1024 cases (descriptive; guards against
   a world where interval supervision is vacuous because gaps are
   deterministic given the previous mark).
3. **Silence carries information:** a gap-length probe — predicting
   above/below-median next gap from (history features + elapsed context)
   must beat the history-only probe by a declared descriptive margin
   (reported with CI; no pass/fail threshold, mirroring A6's descriptive
   oracle clause). This documents that the interval itself holds learnable
   signal in this world.
4. **Acknowledged misspecification:** v2 gaps are multiplicative
   heavy-tailed, not exponential; the piecewise-constant intensity is
   deliberately simple. F2 tests whether interval supervision helps
   *despite* this misspecification, not whether the parametric family is
   true. A negative result therefore kills interval supervision *in this
   form* on this world — and is reported exactly that way.

## 7. Mechanism checks (only if the screen is survived; labeled MECHANISM CHECK)

Using Generator v2 latent logs on holdout:
- **M1:** mean predicted total intensity Λ on steps inside logged
  judge-backlog episodes is **lower** than outside (backlog stretches
  gaps), directional, reported with CI.
- **M2:** predicted λ_SETTLED responds in the correct direction to logged
  adverse-regime flips (lower after flip), directional.
These check whether the trained intensities align with the planted
mechanisms; they explain, they do not rescue.

## 8. Subgroup analyses (exploratory)

The F1 probe-7 strata (gap quartiles, long-gap >90 d, case-age quartiles,
procedural phase, sequence length, backlog episode, adverse regime,
selective vs always-observed classes, discovery-stall cases) applied to
the paired J comparison, with Holm within families, labeled exploratory.

## 9. Analysis plan and deviation policy

- All inferential comparisons paired by seed; per-seed values, mean, 95%
  paired t-interval (df = 9).
- One primary comparison (section 4). Everything else descriptive.
- Deviations from this document must be logged in `F2_FREEZE.md` with
  rationale and reported in `F2_RESULTS.md` as deviations.
- No rescue tuning, no seed selection, no metric shopping, no post-hoc
  redefinition of the endpoint. A second attempt at interval supervision
  requires F2 v2 (new preregistration).

## 10. Artifacts

- `experiments/F2_FREEZE.md` — SHA-256 of this preregistration, the F2
  model/training code, and the evaluation code, recorded before training.
- `experiments/F2_RESULTS.md` — the section-X reporting standard of the
  successor charter, all 14 questions answered.
- `experiments/archive/f2-tpp-v1/` — weights, raw predictions, logs, code
  snapshots, environment record, SHA-256 manifest (survive or kill).
