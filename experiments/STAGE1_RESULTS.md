# Stage-1 result: IDN vs auxiliary-matched Transformer (frozen protocol)

**Verdict: FAIL — provisional survival denied; hybrid track stopped per
protocol.** 2026-07-23. Protocol: `STAGE1_SPEC.md` (third freeze);
governance: `IDN_GUIDE.md`; data: frozen Generator v2 (`FREEZE.md`).
Raw outputs: `results/stage1_baselines.json`, `results/stage1_idn.json`.

## What was run

- Frozen Generator v2, 1024 cases/seed, seeds [0–9], 20% holdout, paired per
  seed, two regimes (visible / hidden statics).
- Config selection on an inner split of seed-0's training portion only
  (no holdout contact). Selected lr: 3e-3 for all neural models; XGBoost
  8-config grid per its frozen procedure.
- Identical training for all: Adam, batch 32, 25 epochs, grad clip 1.0.
- Equal supervision: primary opponent `tf-native-aux` carries the same
  auxiliary heads and loss weights as IDN. Params: IDN 22,536,
  tf-native-aux 23,129, tf-native 22,403.

## Reference distribution (baselines, mean ± std over 10 seeds)

| model | hidden-statics settle AUC | visible-statics settle AUC |
|---|---|---|
| tf-native-aux | 0.849 ± 0.014 | 0.855 ± 0.018 |
| tf-native | 0.850 ± 0.013 | 0.856 ± 0.017 |
| lstm-feature | 0.850 ± 0.013 | 0.849 ± 0.015 |
| xgb | 0.853 ± 0.018 | 0.854 ± 0.017 |
| cfc-native | 0.790 ± 0.018 | 0.784 ± 0.042 |

## Primary screen (hidden statics, paired per seed)

Per-seed ΔAUC (IDN − tf-native-aux):
−0.003, −0.006, −0.011, −0.010, +0.004, −0.002, −0.003, −0.013, +0.005, −0.001

- **mean ΔAUC = −0.0039** (required ≥ +0.01)
- **95% paired CI = [−0.0082, +0.0004]** (required to exclude zero)
- duration MAE: IDN 308d vs 304d (within +5% ✓)
- ECE: IDN 0.043 vs 0.039 (within +0.01 ✓)

The screen requires ALL of: mean Δ ≥ 0.01, CI excludes zero, no material
regressions. The first two fail; the regression checks pass.

## Reading

- IDN is statistically indistinguishable from the auxiliary-matched
  Transformer on this world, with the point estimate slightly negative. The
  CI's upper bound (+0.0004) rules out the preregistered minimum practical
  effect (+0.01): this is an informative negative, not an underpowered
  ambiguity.
- Duration and calibration show no regression — the hybrid is a competent
  model, merely not a better one under equal supervision and matched budgets.
- Per the frozen execution order, mechanism-group ablations and the
  hostile-world battery are gated on provisional survival and were **not**
  run. No rescue tuning, no metric shopping, no post-hoc re-analysis was
  performed.

## What survives the verdict

- The evaluation program itself: preregistration, architecture-independent
  generator acceptance, frozen protocols, paired interventional estimands,
  leakage tests, equal-supervision controls, and a clean kill executed
  exactly as written.
- The E1–E6 record on Generator v1 (`experiments/RESULTS.md`) and the
  Generator v2 world with its latent logs as research infrastructure.
- The open question, unchanged and now sharper: real docket data is the only
  remaining arbiter for whether continuous-time or hybrid temporal state
  earns anything that attention and feature engineering don't already
  provide.
