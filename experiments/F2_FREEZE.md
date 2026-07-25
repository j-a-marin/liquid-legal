# F2_FREEZE — F2 v1 candidate (`tf-tpp`), training runner, and evaluation code

**Freeze: 2026-07-25, before any F2 candidate parameter was trained.**
Governance: `IDN_GUIDE.md`; protocol: `F2_PREREGISTRATION.md` (v1); training
specification reused verbatim: `STAGE1_SPEC.md` (third freeze, recorded in
`FREEZE.md`).

## Administrative note (required)

- **No candidate parameter was trained before this freeze.** The only
  execution of the candidate prior to this record was
  `f2_prefreeze_checks.py` — FORWARD PASS ONLY (shapes, positivity of
  intensities, NaN behavior at padded positions, monotonicity of the
  closed-form gap quantiles, finiteness of the masked TPP NLL on a real
  batch, parameter budget). No `optimizer.step()` was executed. All seven
  checks passed (23,030 params; budget 23,129 ± 20%).
- The generator-validation analyses (F2 preregistration section 6) are
  architecture-independent and run inside the frozen runner before any
  candidate training; they touch no candidate parameter.
- The primary opponent (`tf-native-aux`) is **declared, not re-trained**:
  archived weights and metrics of record under `archive/stage1-killed/`
  (preregistration section 3). J for the opponent is recomputed from the
  archived weights at evaluation time under the identical scoring
  convention as `f2_score_calibration.py`; the calibration file itself is
  not used for the comparison.

## Frozen artifacts (SHA-256)

Recorded with `shasum -a 256` at freeze time, working directory
`experiments/`:

```
6fc9fea484c275054c16397b71e6e74fcf38ed9c5eb53054f1909c1cabf605bf  F2_PREREGISTRATION.md
a30f969a74b59022eaa3b499f5f2e0ecbb30b44b119a69062ab8acff695013c0  tpp_model.py
28488154ba964ea659c796d654c8fd66d05a2236933ec4c2359040c868226879  run_f2_tpp.py
93c5e2decadf564b8c6de3c02b1d8905d034616b3ca858d9d07589260b0107df  f2_prefreeze_checks.py
```

Dependencies already frozen under `FREEZE.md` — re-hashed at this freeze
and confirmed **unchanged** (hashes match `results/freeze_hashes.txt`):

```
f33b87db952728fc2bdb9d5c77a52fba393f860e163f60e2a096d552ecdebaa4  gen_v2.py
1a1f8e3b8a162496a87cef8d3080e5efe42fbcf0c5f4e44d2acce460714ec2eb  ../src/liquid_legal/baselines.py
2763416915745af42c896a727bf41a7905a02b872aa2823672f7ff7e262e6aa5  ../src/liquid_legal/train.py
```

Any change to the files in the first table after this record voids the
freeze; deviations must be logged below with rationale and reported in
`F2_RESULTS.md`.

## Implementation notes (not deviations)

- **Engineered features for the descriptive probes are re-implemented
  inside `run_f2_tpp.py`** (function `prefix_features`, identical columns
  to `xgb_baseline.prefix_features`) because importing `xgboost` in the
  same process as torch segfaults on macOS (two OpenMP runtimes — the
  reason Stage 1 ran XGBoost subprocess-isolated). This affects only the
  architecture-independent silence probe (section 6) and the descriptive
  `cr-discrete` supporting baseline (section 3) — never the candidate, the
  opponent, or the screen.
- `cr-discrete` per section 3 is "monthly grid logistic hazards": the
  logistic fits use the runner's deterministic full-batch torch logistic
  probe (30-day grid, landmark CIF with covariates fixed at the step).
  The preregistration specifies the model class, not the optimizer.
- Generator-validation probe seeds are [0, 1, 2], following the house
  convention of `validate_generator_v2.py` (`PROBE_SEEDS`); the
  preregistration fixes the count (3 probe seeds × 1024 cases), not the
  seed values.
- The archived tf-native-aux mean per-run training wall-clock (guard
  reference) is computed from the per-run timestamps in
  `archive/stage1-killed/stage1_run.log` (mean of consecutive deltas):
  25.68 s/run.

## Deviation log

- (none against the preregistration)
- **Post-run measurement note (2026-07-25, not a protocol change):** the
  wall-clock guard was executed exactly as frozen above (reference 25.68
  s/run from consecutive log-line deltas in `stage1_run.log`). That
  reference overestimates tf-native-aux's true per-run training time,
  because consecutive `tf-native-aux` lines in the archived log have four
  other models and XGBoost training between them. A corrected estimate
  from the same archive (selection line: two full 25-epoch runs on the
  inner-train portion completed within 9 s) is ≈ 4–6 s/run. The guard
  passes under either reference: tf-tpp's measured mean is 6.26 s/run,
  i.e. ≈ 1.0–1.3× the corrected estimate and 0.24× the frozen reference
  (limit 2×). Verdict unaffected; recorded for transparency per the
  deviation policy.
