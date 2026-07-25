# STAGE1_SPEC — frozen Stage-1 training and evaluation specification

Frozen 2026-07-23 (third freeze; see discrepancy/fairness notes in
`FREEZE.md`). Governing document: `experiments/IDN_GUIDE.md`. No IDN or
baseline training run occurred before this freeze.

## Data

- Generator: frozen `experiments/gen_v2.py` (hash in `FREEZE.md`).
- Cases per seed: 1024. Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
- Split: 20% holdout, `np.random.default_rng(seed).permutation`, identical
  for every model.
- Regimes: visible-statics (secondary) and hidden-statics (statics zeroed;
  primary). The same timelines are used for both regimes — only the static
  features differ.

## Models in the frozen core comparison

- `idn` — `experiments/idn_model.py`, recorded hash at this freeze.
- `tf-native-aux` — **primary opponent**: temporal Transformer with the
  identical auxiliary heads and auxiliary loss weights as IDN. Supervision is
  equalized so that an IDN win is attributable to the flow–jump architecture,
  not to extra loss terms.
- `tf-native`, `lstm-feature`, `cfc-native` — supporting neural baselines.
- `xgb` — gradient-boosted trees (separate frozen procedure below).
- Extended baselines (guide section 7) are a later battery, not part of the
  Stage-1 survival screen.

Parameter budgets: ±20% of tf-native (22,403). At this freeze: IDN 22,536
(+0.6%); tf-native-aux 23,129 (+3.2%).

## Training protocol (all neural models, identical)

- Optimizer: Adam (β = (0.9, 0.999)), grad clip 1.0, batch size 32.
- Learning-rate grid: {1e-3, 3e-3}. **Selection without leakage:** seed 0's
  TRAIN portion is split 80/20 (inner split, same permutation rule); each
  candidate lr is trained on inner-train, selected on inner-val settle AUC
  per model, then fixed for all seeds. Seed 0's holdout is never touched by
  selection and remains in the reported paired comparison.
- Epochs: 25, fixed; last-epoch model (no early stopping).
- Seed handling: `torch.manual_seed(seed)` before construction; dataset
  `GeneratorV2Config(seed=seed)`; identical timelines for every model in a
  seed (paired).
- Device: CPU. Record wall-clock per epoch, and inference latency (ms/case,
  batch 1, mean over 100 cases after 20-case warmup).

## XGBoost procedure (frozen separately — Adam/LR does not apply)

- Grid (8 configs): n_estimators {200, 400} × max_depth {3, 5} ×
  learning_rate {0.05, 0.1}; fixed subsample 0.9, colsample_bytree 0.8,
  eval_metric logloss, random_state = seed.
- Selection: same inner-split rule as neural models (seed-0 inner validation
  settle AUC), then refit on the full training portion per seed.
- Execution: subprocess-isolated (macOS OpenMP conflict).

## Losses (weights are part of the freeze)

Main heads (all models): settle BCE 1.0, recovery log1p-MSE 0.3,
duration log1p-MSE 0.3.

Auxiliary heads (IDN and tf-native-aux, identical): next-event-type
cross-entropy 0.3; next-gap pinball loss 0.2; duration pinball loss 0.2.
Quantile levels {0.1, 0.5, 0.9}; nonnegative and non-crossing by construction
(softplus base, cumulative softplus increments).

## Masking rules

- All losses are masked by the batch mask (no padded-step loss).
- Auxiliary next-event targets exist only where a next event exists (masked
  at each sequence's final valid step).
- Models accepting `lengths` receive per-row valid counts; state is frozen
  at padded steps; `hx` is the last valid state.

## Metrics and comparisons

- **Primary endpoint:** hidden-statics settlement AUC, paired
  IDN − tf-native-aux per seed.
- **Secondary:** visible-statics settle AUC; duration MAE; recovery log-MAE;
  next-event-type accuracy; next-gap pinball; ECE.
- **ECE:** 15 equal-width bins on masked settle probabilities.
- **CI:** mean of per-seed paired deltas ± 95% t-interval (df = 9).
- **Timing-sensitive strata:** long-gap steps (preceding Δt > 90 days);
  regime-transition steps (within 30 days after a logged judge-state or
  case-regime flip, from latent logs). Report primary metric per stratum.

## Decision rules

- **Provisional survival (primary screen):** mean paired ΔAUC ≥ 0.01 AND
  paired 95% CI excluding zero AND no material regression (duration MAE
  within +5% of tf-native-aux; ECE within +0.01).
- **Final survival:** provisional survival AND mechanism-group ablations
  (event/clock/context/static) AND hostile-world battery (guide section 6)
  AND concentration of advantage in the preregistered timing-sensitive
  strata.
- Anything short: stop the hybrid track; report results as-is.

## Execution order (frozen)

1. Frozen core baselines (tf-native-aux, tf-native, lstm-feature,
   cfc-native, xgb), 10 seeds, both regimes; outputs preserved.
2. IDN trained with the identical protocol; outputs preserved.
3. Paired primary comparison → provisional screen.
4. If provisionally surviving: mechanism-group ablations, then hostile-world
   battery.
5. Final survival declaration only after step 4.
