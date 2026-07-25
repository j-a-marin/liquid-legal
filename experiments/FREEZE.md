# FREEZE — Generator v2, evaluation protocol, and Stage-1 training spec

**Third freeze: 2026-07-23** (fairness repair, pre-training). Supersedes the
second freeze of the same date.

## Administrative log

- **Freeze 1:** recorded `PREREGISTRATION.md` as `fc7504fc…`; the document was
  subsequently edited (status header), invalidating the record.
- **Freeze 2:** re-recorded all hashes (`d3a31544…` etc.), acceptance re-run
  (identical PASS).
- **Freeze 3 (this):** two experimental confounds repaired before any
  training — (a) learning-rate selection moved from seed-0 holdout to an
  inner split of seed-0's training portion (hyperparameter leakage);
  (b) auxiliary supervision equalized by making the primary opponent
  `tf-native-aux`, a Transformer with identical auxiliary heads and loss
  weights as IDN (unequal supervision). Also: XGBoost's non-Adam tuning
  procedure frozen separately; quantile heads made nonnegative (not merely
  non-crossing); encoder mask warnings resolved. No training of any kind had
  occurred before this freeze.

## Frozen artifacts (sha256, see results/freeze_hashes.txt)

- `gen_v2.py` (generator)
- `validate_generator_v2.py`, `validate_worker.py` (acceptance)
- `PREREGISTRATION.md` (protocol)
- `IDN_GUIDE.md` (governing document)
- `STAGE1_SPEC.md` (frozen training/evaluation specification)
- `idn_model.py` (IDN Stage-1 candidate, 22,536 params)
- `../src/liquid_legal/baselines.py` (tf-native-aux, 23,129 params)
- `../src/liquid_legal/train.py` (shared training loop incl. aux losses)

## Acceptance result (frozen generator, re-confirmed at freeze 2)

A1 lift 0.026 (≥0.02); A2 ratios 2.44 / 0.111; A3 spread 0.13; A4 fraction
0.46; A5 AUCs 0.81–0.84; A6 backlog 3.41/0.490, adverse 0.502; descriptive
oracle gain 0.024.

## Frozen protocol

- **Seeds:** [0–9] (10, paired). **Split:** 20% holdout via
  `np.random.default_rng(seed).permutation`, identical for every model.
- **Primary endpoint:** hidden-statics settlement AUC (180-day horizon),
  paired IDN − tf-native-aux per seed.
- **Selection:** lr/config chosen on an inner split of seed-0's TRAIN
  portion only; seed-0 holdout never touched by selection.
- **Training specification:** `STAGE1_SPEC.md`.
- **Survival:** provisional screen per spec; final survival per
  `IDN_GUIDE.md` section 9 after mechanism ablations and hostile worlds.

Any change to the frozen files voids the freeze; bump versions and
re-validate. Deviations must be logged here with rationale.

## Deviation log

- (none)
