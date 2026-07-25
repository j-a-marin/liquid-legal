# ARCHIVE: stage1-killed — IDN Stage-1 candidate (killed 2026-07-23)

**Status: KILLED CANDIDATE. Not the flagship model. Do not resurrect without
a new versioned preregistration.**

## Verdict of record

Under the frozen Stage-1 protocol (`../STAGE1_SPEC.md`, third freeze;
governance `../IDN_GUIDE.md`), the IDN Stage-1 candidate failed its
preregistered primary screen against the auxiliary-matched temporal
Transformer (`tf-native-aux`) on frozen Generator v2, hidden-statics regime,
10 paired seeds:

- mean paired ΔAUC = **−0.0039** (preregistered requirement: ≥ +0.010)
- 95% paired CI = **[−0.0082, +0.0004]** (preregistered requirement: exclude
  zero; the interval instead excludes the required benefit)
- duration MAE 308d vs 304d (no regression); ECE 0.043 vs 0.039 (no
  regression)
- seed splits: IDN won 2/10, Transformer won 8/10

Per the frozen execution order, gated mechanism ablations and the
hostile-world battery were not run. No rescue tuning, metric shopping, or
post-hoc re-analysis was performed. The defensible conclusion:

> Under matched parameters, supervision, tuning, and data, Stage-1 IDN does
> not outperform a temporal Transformer. Its estimated effect is slightly
> negative, and the preregistered practical improvement is incompatible with
> the observed interval.

Why it failed is unknowable from this experiment alone (the diagnostic
ablations were correctly gated off): attention already capturing the usable
timing signal, redundancy of the flow, or sequence-length limits of the
world remain hypotheses, not findings.

## Contents

- `weights/` — final state_dicts, 2 models × 2 regimes × 10 seeds (40 runs),
  reproduced deterministically under the frozen protocol after the kill.
- `predictions/` — raw holdout prediction arrays (settle logits + labels,
  duration and recovery predictions + targets) per run.
- `stage1_baselines.json`, `stage1_idn.json` — metrics of record (these, not
  the reproduced weights, are the authoritative numbers).
- `stage1_run.log`, `stage1_idn_run.log` — original run logs.
- `reproduction_check.json` — per-run reproduced vs recorded holdout AUC
  with drift; any nonzero drift reflects machine-state effects on floating
  point, not protocol changes (code, seeds, splits identical).
- `code/` — snapshots of the exact model and training code at kill time.
- `hashes.json` — sha256 of every file in this archive.

## Lineage

Frozen generator + acceptance: `../FREEZE.md`. Protocol:
`../PREREGISTRATION.md` + `../STAGE1_SPEC.md`. Result write-up:
`../STAGE1_RESULTS.md`. Prior program (Generator v1, E1–E6):
`../RESULTS.md`.
