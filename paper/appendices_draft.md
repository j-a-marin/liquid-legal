# Appendices (draft for review/merge)

Sources: `experiments/RESULTS.md`, `experiments/STAGE1_SPEC.md`,
`experiments/PREREGISTRATION.md`, `experiments/gen_v2.py`,
`src/liquid_legal/synthetic.py`, `src/liquid_legal/baselines.py`,
`src/liquid_legal/train.py` (archived copy:
`experiments/archive/stage1-killed/code/train.py`),
`experiments/xgb_baseline.py`, `experiments/results/*.json`.
All numbers are taken from those files.

---

## Appendix A — Model and parameter matrix

Time-handling ablation codes (`time_mode`, symmetric across the neural
families): `native` = sinusoidal time encodings + Δt input feature;
`timespans_only` = time encodings only; `feature` = Δt input feature only;
`none` = no timing. The LSTM has no time encodings, so only `feature` /
`none` apply. XGBoost variants differ only in whether the temporal columns
(`day`, `last_delta`, `mean_delta`, `max_delta`) are included.

Supervision: *main* = settle BCE (1.0) + recovery log1p-MSE (0.3) +
duration log1p-MSE (0.3); *aux* = next-event-type CE (0.3) + next-gap
pinball (0.2) + duration pinball (0.2), quantile levels {0.1, 0.5, 0.9},
nonnegative and non-crossing by construction. Loss weights are part of the
freeze.

| model | family | time handling | supervision | params |
|---|---|---|---|---|
| cfc-native / -timespans_only / -feature / -none | CfC (liquid), AutoNCP wiring; ODE flow between events | per `time_mode` (4 variants) | main | 15,898 |
| lstm-feature / -none | LSTM (64 units) | Δt as input feature, or none | main | 24,259 |
| tf-native / -timespans_only / -feature / -none | causal-masked Transformer encoder, position + time encodings | per `time_mode` (4 variants) | main | 22,403 |
| xgb / xgb-no-time | XGBoost on engineered per-prefix features (counts, amounts, statics, event one-hots) | temporal columns included / dropped | settle classifier (logloss) + duration/recovery regressors | — (trees; 8-config grid frozen separately) |
| **IDN (Stage-1 candidate)** | hybrid flow–jump state-space: GRU event partition + gated continuous-time clock partition + context partition over causal-masked Transformer history encoder | continuous-time flow between events (rate/target/gate from context) | main + aux | 22,536 (+0.6% vs tf-native) |
| **tf-native-aux (primary opponent)** | tf-native with IDN's auxiliary heads added | `native` | main + aux (identical heads and weights as IDN) | 23,129 (+3.2% vs tf-native) |

Parameter budgets were frozen at ±20% of tf-native (22,403); both Stage-1
models land within +3.2%.

---

## Appendix B — Generator pseudocode

Faithful, compact renderings of the main loops. `gap(base, mult)` samples a
multiplicative heavy-tailed inter-arrival: `max(1, base · mult · judge.speed
· exp(volatility·N(0,1)) · regime)`, with `regime ∈ U(0.15, 3)` on 6% of
draws (sudden acceleration/stall). `σ` is the logistic function.

### Algorithm B1 — Generator v1 (`SyntheticLitigationGenerator.sample`)

```
fix judge pool (12): speed ~ LogN(0,.45), volatility ~ U(.15,1), defense_tilt ~ N(0,.6)
fix district pool (8): congestion ~ LogN(0,.35)

sample_case():
  judge, cong ← random pool draws
  capability ~ Beta(2.2, 2.0);  score ~ N(0,1);  damages ~ LogN(log 2e6, 1.1)
  day ← 0; pressure ← 0; leverage ← 1; events ← [FILED(damages)]
  maybe_settle(): if not settled and U < σ(−2 + pressure):
                  settle at day; recovery = damages·(0.03+0.42·σ(0.9·score))·leverage·LogN(0,.2)

  pleadings:   day += gap(25); ANSWER
               w.p. σ(0.5 + tilt − 0.3·score): day += gap(45); MTD(granted w.p. σ(−0.9 + tilt − 0.9·score));
               if granted → DISMISSED, return
  discovery:   day += gap(30); DISCOVERY_OPEN
               budget = 240·cong·speed·LogN(0,.25); n_steps ~ U{4..9}
               stall_hazard = 0.04 + 0.5·(1−capability)²·min(cong,2.5)/2.5
               repeat n_steps times or until settled:
                 w.p. stall_hazard:  MOTION_TO_COMPEL (granted w.p. 0.6);
                    day += gap(35, mult=1.5+2.5·(1−capability));
                    leverage ×= 0.95 − 0.18·(1−σ(score));  pressure −= 0.15
                 else: DEPOSITION (.5) / EXPERT_DISCLOSURE (.3) / SETTLEMENT_OFFER (.2);
                    offer: pressure += 0.35; maybe_settle()
  if not settled:
    day += gap(20); DISCOVERY_CLOSE
    w.p. σ(0.4 − 0.2·score + 0.3·tilt): day += gap(75); MSJ(granted w.p. σ(−1.3 + tilt − 1.1·score));
        if granted → DISMISSED, return; else pressure += 1.2
    negotiation (≤6 rounds, until settled):
        SETTLEMENT_OFFER (.45) / MEDIATION (.25) / TRIAL_DATE_SET (.30, once)
        day += gap(40, mult=0.5 if trial date set)
        trial date: pressure += 0.9;  mediation: pressure += 0.55, maybe_settle();
        offer: pressure += 0.4, maybe_settle()
    if not settled: day += gap(30); TRIAL_START; day += U(3,15);
        VERDICT(win w.p. σ(0.9·score − 0.7·tilt); recovery = damages·U(.15,.9) if win else 0)
  return timeline(events, statics={score, judge traits, congestion, capability, log damages}, outcome)
```

### Algorithm B2 — Generator v2 (`GeneratorV2.sample_with_latents`)

v2 wraps the v1 skeleton with time-varying latents and selective observation;
only the differences are spelled out.

```
hidden state: backlogged ~ Bern(0.5);  adverse ← False;  regime_flip_day ← none
config: mean_normal 400d, mean_backlog 350d, backlog_gap_mult 2.5,
        fatigue 0.0015/day, backlog settle penalty 2.0,
        flip hazard 0.04/event, adverse stall bonus +0.15,
        accept penalty 0.8, decay 0.002/day, pressure mult 0.5

gap(base, mult): multiplier ×= 2.5 while backlogged
advance(g):  evolve the judge chain WITHIN the interval:
             n_flips ~ Poisson(g / mean_episode); flips at sorted U(0,1)·g;
             each sub-interval ticks fatigue at the state in force;
             log every flip (day, state) to backlog_log
tick(g):     day += g;  if backlogged: leverage ×= exp(−0.0015·g);
             if adverse: leverage ×= exp(−0.002·g)
maybe_flip_regime(): after each event, w.p. 0.04 (if Normal):
             adverse ← True; flip day logged; pressure ← 0   (regime shock resets negotiations)
add_pressure(x):  pressure += x · (0.5 if adverse else 1)
maybe_settle():   logit = −2 + pressure − (0.8 if adverse) − (2.0 if backlogged)
stall_hazard:     v1 hazard + (0.15 if adverse)

run v1 phases (pleadings → discovery → MSJ → negotiation → trial),
calling advance()/maybe_flip_regime() at every interval/event boundary

finish(true_events):
  outcome (settled, recovery, duration) computed from the TRUE process
  observation mask: keep FILED and the terminal event always;
      deposition-class events kept w.p. base_rate − 0.35·min(cong,2.5)/2.5
      (base rates: DEPOSITION .95, EXPERT_DISCLOSURE .95,
       MOTION_TO_COMPEL .90, SETTLEMENT_OFFER .85; all others 1.0)
  observed docket = kept events;  latent log = {backlog_log,
      backlog time fraction, regime_flip_day, true_events, observed_mask}
  return (observed timeline, latent log)
```

---

## Appendix C — Training and evaluation pseudocode

Faithful to `STAGE1_SPEC.md` (third freeze) and the shared training loop
(`src/liquid_legal/train.py`, archived with the kill).

### Algorithm C1 — Configuration selection (leakage-free)

```
for each model m in {tf-native-aux, tf-native, lstm-feature, cfc-native, idn} × {visible, hidden}:
  timelines ← GeneratorV2(seed=0).generate_with_latents(1024)   # observed docket only
  train₀, holdout₀ ← split(timelines, 20%, permutation rng(0))  # identical for every model
  inner_train, inner_val ← split(train₀, 20%, same permutation rule)
  for lr in {1e-3, 3e-3}: train m on inner_train (C2); record inner-val settle AUC
  select best lr per model; fix for all 10 seeds; holdout₀ is never touched by selection
XGBoost: same inner-split rule over its frozen 8-config grid
  (n_estimators {200,400} × max_depth {3,5} × lr {0.05,0.1};
   subsample 0.9, colsample_bytree 0.8, logloss), refit per seed; subprocess-isolated
```

### Algorithm C2 — Training (all neural models, identical)

```
torch.manual_seed(seed); dataset GeneratorV2Config(seed=seed)     # paired across models
val, train ← 20% holdout split (np.random.default_rng(seed).permutation)
initialize head biases to label base statistics
  (settle bias = base-rate log-odds; recovery/duration biases = target means)
optimizer Adam(β=(0.9,0.999), lr=selected); grad clip 1.0; batch 32
for epoch in 1..25:                       # fixed; last-epoch model, no early stopping
  for batch in shuffled train:
    out = model(event_ids, event_feats, static, timespans=Δt, lengths=mask.sum(1))
    loss = 1.0·masked_mean(BCE(settle_logit, y_settle))
         + 0.3·masked_mean((log_recovery − y_recovery)²)
         + 0.3·masked_mean((log_remaining − y_remaining)²)
         + aux                                             # IDN and tf-native-aux only
    aux = 0.3·masked CE(next_type_logit[:, :-1], event_ids[:, 1:])
        + 0.2·masked pinball(next_gap_q[:, :-1], log1p(Δt[:, 1:]))
        + 0.2·masked pinball(duration_q, y_remaining)
backprop, clip, step
masking: all losses masked to valid (non-padded) steps; next-event targets
masked at each sequence's final valid step; state frozen at padded steps
```

### Algorithm C3 — Evaluation and paired comparison

```
metrics on holdout, masked steps only:
  settle AUC (primary: hidden statics; secondary: visible),
  BCE, duration MAE (days), recovery log-MAE (dollars),
  next-event-type accuracy, next-gap pinball, ECE (15 equal-width bins)
primary endpoint: per seed s, ΔAUC_s = AUC_idn(s) − AUC_tf-native-aux(s)   (hidden statics)
report: mean ΔAUC ± 95% t-interval (df = 9)
timing-sensitive strata: preceding-gap > 90d; steps within 30d after a
  logged judge-state or case-regime flip (from latent logs)
provisional survival requires ALL: mean ΔAUC ≥ +0.01; CI excludes 0;
  duration MAE within +5% of tf-native-aux; ECE within +0.01
anything short → kill the hybrid track, report as-is, archive everything
recorded: wall-clock per epoch; inference latency (ms/case, batch 1,
  mean over 100 cases after 20-case warmup); params per model
```

---

## Appendix D — Hardware, environment, dependencies

All runs (training and evaluation) were executed on CPU.

| item | value |
|---|---|
| Machine | Apple M3 Max (arm64), 36 GiB RAM |
| OS | macOS 15.7.3 (Darwin 24.6.0, `RELEASE_ARM64_T6031`) |
| Python | 3.14.6 (project venv `.venv/`) |
| torch | 2.13.0 |
| numpy | 2.5.1 |
| ncps | 0.0.2 |
| xgboost | 3.3.0 |
| scikit-learn | 1.9.0 |
| matplotlib (figures only) | 3.11.1 |

Determinism: generation is deterministic per seed
(`np.random.default_rng(seed)`); model init via `torch.manual_seed(seed)`
before construction; splits use the same seeded permutation for every
model. Frozen artifacts (generator, spec, protocol, models, training loop)
are sha256-hashed in `experiments/results/freeze_hashes.txt` and archived
under `experiments/archive/stage1-killed/` (weights, predictions, code,
hashes).
