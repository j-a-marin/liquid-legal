# F2_RESULTS — F2 v1 `tf-tpp`: interval-supervised marked TPP (KILLED)

> **VERDICT: KILLED CANDIDATE.** Under the frozen F2 preregistration
> (`F2_PREREGISTRATION.md`, frozen 2026-07-25 before any candidate
> parameter was trained; freeze record `F2_FREEZE.md`), `tf-tpp` failed
> the preregistered primary screen against `tf-native-aux` on Generator
> v2, hidden-statics regime, 10 paired seeds. Per section 4 of the
> preregistration, **the interval-supervision branch on Generator v2 is
> terminated.** A second attempt at interval supervision requires F2 v2
> (new preregistration). No continuous flow may be added to a TPP model
> that has failed to improve the strong baseline.
>
> Claim discipline (`IDN_GUIDE.md`): tf-tpp is a predictive marked-event
> model. Nothing here is evidence about institutional dynamics.

Confirmatory assets: runner `run_f2_tpp.py` (frozen hash in
`F2_FREEZE.md`); machine-readable results `results/f2_tpp.json`; raw
archive `archive/f2-tpp-v1/` (59 files, SHA-256 manifest verified against
the directory contents after the run). Total wall-clock: **153 s** CPU
(python 3.14.6, torch 2.13.0, numpy 2.5.1).

## 1. What hypothesis

> Jointly modeling event type and event occurrence time — supervising the
> interval itself through a marked temporal point-process likelihood, so
> that every day of continued silence contributes training signal via the
> survival term — improves legal-event prediction beyond the
> auxiliary-matched temporal Transformer.

Preregistered margin: mean paired relative reduction of the joint
event-process score J of at least 3%, with the 95% paired t-interval
(df = 9) excluding zero and no guard regression.

## 2. What was added

Exactly one thing: the **objective**. `tf-tpp` is the exact
`tf-native-aux` Transformer trunk (d_model=32, nhead=4, num_layers=2,
dim_feedforward=64, max_len=128, time_mode="native", same causal +
padding masking) with the two marked-event auxiliary heads (next-type CE
logits, next-gap quantiles) replaced by per-mark conditional intensity
heads λ_m(k) = softplus(w_m·h_k + b_m) + 1e-6 (16 marks), trained with
the closed-form piecewise-constant marked TPP likelihood (weight 0.3) in
place of the next-type CE (0.3) + next-gap pinball (0.2) terms. Main
heads and the duration-quantile head are identical. Net parameter change
−99 (23,030 vs 23,129; within the declared ±20% budget). Next-type
probabilities, next-gap quantiles, and the TPP-derived
settlement-within-180d probability are derived in closed form from the
intensities.

## 3. What baseline already had the same supervision

`tf-native-aux`, the Stage-1 primary opponent, was constructed in the
Stage-1 fairness repair precisely to equalize supervision with the -event auxiliary losses (next-type CE 0.3, next-gap pinball 0.2, duration
pinball 0.2 on top of the shared main heads). F2 keeps the main-head and
duration supervision identical between the two models and swaps only the
marked-event objective — so the comparison isolates the interval
likelihood, not extra loss terms. The opponent was **declared, not
re-trained** (archived Stage-1 weights and metrics of record,
`archive/stage1-killed/`); its J was recomputed from the archived weights
under the identical scoring convention at evaluation time.

## 4. What was frozen

Recorded in `F2_FREEZE.md` before any training (SHA-256):

- `F2_PREREGISTRATION.md` `6fc9fea4…`
- `tpp_model.py` `a30f969a…`
- `run_f2_tpp.py` `28488154…`
- `f2_prefreeze_checks.py` `93c5e2de…`

Dependencies already frozen under `FREEZE.md` were re-hashed at the F2
freeze and confirmed unchanged (`gen_v2.py`, `baselines.py`, `train.py`).
Administrative note of record: no candidate parameter was trained before
the freeze — the only pre-freeze execution was the forward-only check
script (shapes, λ > 0, padded-position NaN behavior, quantile monotonicity,
finite masked TPP NLL, parameter budget; all seven checks PASS). Learning
rates {1e-3, 3e-3} were selected afterward on an inner split of seed-0's
training portion only (selected: 1e-3 hidden, 3e-3 visible).

## 5. Primary endpoint

J = mean masked next-event-type log-loss + mean masked next-gap pinball
(levels {0.1, 0.5, 0.9}, log1p-day scale), on holdout steps with a next
observed event, hidden-statics regime — the identical scoring convention
as `f2_score_calibration.py` (same masking, same quantile scale, full
16-mark distribution for both models). Both models scored on exactly the
same steps per seed.

## 6. Required improvement

ALL of: (1) mean paired relative J reduction ≥ 3%; (2) 95% paired
t-interval (df = 9) of the per-seed relative reductions excludes zero;
(3) guards: settle AUC ≥ −0.005, ECE ≤ +0.01, duration MAE ≤ +5%,
next-type accuracy ≥ −0.01, training wall-clock ≤ 2× the archived
tf-native-aux mean, zero NaN/Inf events with all 10 seeds × 2 regimes
completing.

## 7. Paired results

Hidden statics (primary), 10 paired seeds:

| model | J | next-type log-loss | next-gap pinball |
|---|---|---|---|
| tf-native-aux (archived, re-scored) | **1.4409** | 1.1816 | 0.2593 |
| tf-tpp | **1.6605** | 1.3965 | 0.2640 |

Per-seed relative J reductions (J_aux − J_tpp)/J_aux: −0.107, −0.157,
−0.164, −0.204, −0.203, −0.162, −0.111, −0.135, −0.148, −0.135 —
**all 10 seeds negative**.

- **Mean paired relative J reduction: −0.1526 (required ≥ +0.03): FAIL.**
- **95% paired t-interval (df = 9): [−0.1762, −0.1290] — excludes zero in
  the wrong direction: FAIL.**

The recomputed opponent matches the records of note exactly: J = 1.4409
(CE 1.1816 + pinball 0.2593) reproduces the frozen calibration reference
digit-for-digit, and the recomputed hidden-statics settle AUC matches
`archive/stage1-killed/stage1_baselines.json` of record with max
|diff| = 0.00e+00 across all 10 seeds.

Visible statics (secondary/descriptive): J 1.5044 (tf-tpp) vs 1.4003
(tf-native-aux), relative reduction −7.4%; settle AUC 0.8545 vs 0.8551.
Same direction, smaller magnitude — with statics visible, less of the
signal has to come through the timing channel.

## 8. Guards

Hidden statics, paired means:

| guard | tf-tpp | tf-native-aux | delta | threshold | result |
|---|---|---|---|---|---|
| settle AUC (main BCE head) | 0.8480 | 0.8492 | −0.0012 | ≥ −0.005 | PASS |
| ECE (15 bins) | 0.0464 | 0.0395 | +0.0069 | ≤ +0.01 | PASS |
| duration MAE (days) | 308.6 | 303.8 | +1.61% | ≤ +5% | PASS |
| next-type accuracy | 0.4908 | 0.5206 | −0.0297 | ≥ −0.01 | **FAIL** |
| wall-clock per run | 6.26 s | 25.68 s (archived ref) | — | ≤ 2× | PASS |
| stability | 0 NaN/Inf events, 20/20 runs | — | — | zero events, all complete | PASS |

Descriptive alongside: the TPP-derived settle-within-180d probability
scores AUC 0.8054 vs the main BCE head's 0.8480. Supporting rows (never
part of the screen): `sm-empirical` next-gap pinball 0.2667; `cr-discrete`
(monthly-grid cause-specific logistic hazards) settle AUC(180d) 0.7818;
`xgb` (archived, hidden) settle AUC 0.8532 / duration MAE 315.8 d / ECE
0.0321; `idn` (archived reference) paired ΔJ vs tf-native-aux −0.0114
± 0.0210 from the frozen calibration.

## 9. What it rejects

The preregistered kill criterion fired on both inferential criteria and
one guard: on Generator v2, replacing the marked-event auxiliary losses
with a closed-form piecewise-constant marked TPP likelihood — the
interval-supervision objective — does not merely fail to improve the
strong baseline; it **degrades** the joint event-process score by 15.3%
± 2.4% on every seed, concentrated in the type component (CE 1.397 vs
1.182) with next-type accuracy down 0.030 (guard failure). Per section 4:
**the interval-supervision branch on Generator v2 is terminated**, in
this form — the preregistration's acknowledged-misspecification clause
applies verbatim: v2 gaps are multiplicative heavy-tailed, not
exponential, and a negative result kills interval supervision *in this
form* on this world. The hybrid question is dead twice over: neither the
Stage-1 flow–jump mechanism (Stage 1/F1) nor the likelihood route (F2)
moves the auxiliary-matched Transformer.

## 10. What it does NOT reject

- Interval supervision in any other form: a different parametric family
  (the exponential gap law is the deliberate simplicity being tested, not
  a finding), history-dependent/self-exciting intensities, discrete-time
  hazard objectives, or a different trunk. Each requires a new F2 v2
  preregistration.
- The generator. Section-6 validation (architecture-independent, run
  before the candidate): per-mark gap quantiles on 3 probe seeds × 1024
  cases are non-degenerate (13 non-terminal marks with a following
  observed event; n = 458–5,250 gaps per mark; minimum q10–q90 spread
  9.4 days — interval supervision is not vacuous in this world), and the
  silence probe finds a positive, small elapsed-context lift in
  above/below-median next-gap prediction: +0.0055 AUC ± 0.0068 (3 paired
  probe seeds, df = 2; descriptive, no pass/fail).
- The Stage-1/F1 record. Nothing here reopens the Stage-1 kill or the F1
  topology retirement; F2 tested the objective route precisely because
  the mechanism route was already dead.
- Anything about real dockets. This world has no right-censoring and a
  selective-observation process the candidate saw only through the
  observed docket; external validity remains an F5 question.

## 11. Confirmatory vs exploratory

- **Confirmatory:** the primary screen (section 4) and its guards, exactly
  as frozen. Verdict: KILLED.
- **Descriptive (architecture-independent):** generator validation
  (per-mark gap quantiles, silence probe) — no pass/fail by design.
- **Exploratory:** probe-7 strata applied to the paired J comparison
  (Holm within families). The degradation is uniform: all 22 strata are
  negative (Holm-adjusted p < 2.1e-06 in every stratum of every family),
  from −0.081 ± 0.017 (selectively observed event classes) to −0.256
  ± 0.044 (always-observed classes) and −0.249 ± 0.047 (youngest
  case-age quartile). No stratum localizes a benefit; the loss is
  broadest where observation is complete and histories are short.
- **Gated, not run:** mechanism checks M1/M2 (predicted Λ inside logged
  backlog episodes; λ_SETTLED after adverse-regime flips) were not run —
  they are gated on screen survival (section 7) and the screen failed.
  Recorded as NOT RUN in `results/f2_tpp.json`.

## 12. Artifact paths

- Preregistration and freeze: `experiments/F2_PREREGISTRATION.md`,
  `experiments/F2_FREEZE.md` (hashes, administrative note, deviation log).
- Code (frozen): `experiments/tpp_model.py`,
  `experiments/run_f2_tpp.py`, `experiments/f2_prefreeze_checks.py`.
- Machine-readable results: `experiments/results/f2_tpp.json`.
- Archive: `experiments/archive/f2-tpp-v1/` — `weights/` (20 tf-tpp runs:
  10 seeds × 2 regimes), `predictions/` (30 npz: per-run raw holdout
  settle logits + labels, duration/recovery predictions + targets,
  per-step lambdas and derived next-type probabilities + gap quantiles
  for tf-tpp; the paired tf-native-aux hidden-statics arrays),
  `run_f2_tpp.log`, `config.json`, `environment.json`,
  `generator_validation.json`, `f2_results.json`, `code/` (snapshots of
  the frozen files), `hashes.json` (59 files; verified against the
  directory contents after the run, both in-script and independently).
- Nothing under `archive/stage1-killed/` was modified.

## 13. Cheapest decisive next test

The branch is terminated; any second attempt is F2 v2 (new
preregistration). If one is written, the cheapest decisive test is a
misspecification-controlled retry on the same trunk: keep the intensity
head but train it with a discrete-time (monthly-grid) cause-specific
hazard likelihood — which matches the world's actual gap family no worse
than the exponential piecewise-constant law while retaining interval
supervision — scored under the identical J screen. If that also fails to
beat tf-native-aux by the preregistered margin, the interval-supervision
question is dead independent of the parametric family, for ≈ the same
153-second compute cost. Note the descriptive evidence already points at
the objective rather than the capacity: `sm-empirical` quantiles (0.2667
pinball) and the TPP closed-form quantiles (0.2640) both sit at the
opponent's gap score (0.2593) within noise, while the type channel is
where the damage concentrates.

## 14. Deviations

None against the preregistration. One post-run measurement note is logged
in `F2_FREEZE.md`: the frozen wall-clock reference (25.68 s/run, from
consecutive log-line deltas in the archived Stage-1 log) overestimates
tf-native-aux's true per-run time because other models trained between
those log lines; the guard passes under the corrected estimate as well
(tf-tpp 6.26 s/run ≈ 1.0–1.3× the corrected ≈ 4–6 s/run, limit 2×).
Verdict unaffected.

---

**Bottom line:** `tf-tpp` is a killed candidate. Interval supervision via
a closed-form marked TPP likelihood, on the exact trunk that defines the
strong baseline, made the marked-event score reliably worse on every seed
(−15.3% ± 2.4% relative J) while leaving the main-head metrics intact.
The interval-supervision branch on Generator v2 is terminated.
