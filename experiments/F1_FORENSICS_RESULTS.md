# F1_FORENSICS_RESULTS — exploratory forensics on the killed Stage-1 IDN

> **EVIDENTIARY STATUS: EXPLORATORY — every analysis, table, and
> interpretation below is exploratory.**
>
> **The Stage-1 kill stands.** The preregistered confirmatory screen failed
> (paired ΔAUC IDN − tf-native-aux = −0.0039, 95% CI [−0.0082, +0.0004];
> `PROVISIONAL SURVIVAL: FAIL — stop the hybrid track`). Nothing in this
> document reopens, softens, or alters that verdict. F1's sole decision
> scope, per the frozen `F1_FORENSICS_SPEC.md`, is whether to **retain or
> retire the Stage-1 context-conditioned exponential-relaxation clock
> topology** in future designs.

Spec: `experiments/F1_FORENSICS_SPEC.md` (frozen 2026-07-25, before any F1
probe was run). Runner: `experiments/f1_forensics.py`. Machine-readable
results: `experiments/results/f1_forensics.json`; raw archive:
`experiments/archive/f1-forensics/` (config, environment record, code
snapshot, raw per-seed metrics, per-step outputs, SHA-256 manifest).

## 1. What was tested

- **Assets:** archived checkpoints `archive/stage1-killed/weights/idn_hidden_seed{0..9}.pt`
  (primary hidden-statics regime); paired reference
  `tf-native-aux_hidden_seed{0..9}.pt`. No archived weight was fine-tuned in
  any ablation; the one probe that fits parameters (probe 5) is defined and
  reported as a **new exploratory model**.
- **Data:** frozen `gen_v2.py`, 1024 cases/seed, seeds 0–9, 20% holdout via
  `np.random.default_rng(seed).permutation` (identical to Stage 1). One
  dataset regeneration per seed, reused across all probes. All metrics are
  masked, per-step, on the same holdout steps as the archived evaluation.
- **Code-snapshot check (spec mandate):** `experiments/idn_model.py`,
  `src/liquid_legal/train.py`, and `src/liquid_legal/baselines.py` are
  **byte-identical** (`diff` empty) to the archived snapshots under
  `archive/stage1-killed/code/`. The live copies were therefore used; the
  archived IDN file was never modified — probes subclass it (`ProbeIDN`).
- **Parity checks (both pass):** (a) `ProbeIDN` with default options is
  bitwise identical to the archived `IDNModel` forward on every head output;
  (b) the seed-0 baseline metrics reproduce
  `archive/stage1-killed/stage1_idn.json` (`idn/hidden`) to < 1e-6 on all
  shared metrics.
- **Metrics (every probe):** settle AUC (180-day), duration MAE (days),
  recovery log-MAE, ECE (15 equal-width bins), next-event-type accuracy,
  next-gap pinball (0.1/0.5/0.9), duration pinball. All deltas are paired
  per seed against the unmodified archived IDN on identical holdout steps,
  reported as mean ± 95% t-interval (df = 9, t = 2.262).
- **Runtime:** 64 s CPU (python 3.14.6, torch 2.13.0, numpy 2.5.1).

**Archived IDN baseline on the F1 holdout (mean ± 95% CI over 10 seeds):**

| settle AUC | duration MAE (d) | recovery log-MAE | ECE | next-type acc | next-gap pinball | duration pinball |
|---|---|---|---|---|---|---|
| 0.8453 ± 0.0119 | 307.6 ± 36.6 | 4.069 ± 0.253 | 0.0433 ± 0.0149 | 0.5272 ± 0.0082 | 0.2691 ± 0.0090 | 0.3124 ± 0.0135 |

## 2. Probe 1 — partition zeroing (inference-time)

Paired deltas vs the archived IDN. The `*_hard` variant additionally zeroes
the partition inside the recurrence.

| condition | ΔAUC | Δdur MAE (d) | Δrec log-MAE | ΔECE | Δnext-type acc | Δnext-gap pinball | Δdur pinball |
|---|---|---|---|---|---|---|---|
| zero_event_head | −0.0063 ± 0.0015 | +38.4 ± 27.9 | −0.092 ± 0.066 | +0.012 ± 0.019 | −0.0624 ± 0.0180 | +0.023 ± 0.011 | +0.063 ± 0.031 |
| zero_event_hard | identical to head variant | — | — | — | — | — | — |
| zero_clock_head | −0.0069 ± 0.0045 | +282.4 ± 162.9 | −0.177 ± 0.054 | +0.022 ± 0.029 | −0.0131 ± 0.0109 | +0.024 ± 0.015 | +0.018 ± 0.017 |
| zero_clock_hard | identical to head variant | — | — | — | — | — | — |
| zero_context_head | −0.0778 ± 0.0252 | +144.8 ± 20.8 | +1.411 ± 0.069 | +0.036 ± 0.015 | −0.1397 ± 0.0355 | +0.524 ± 0.059 | +1.173 ± 0.091 |
| zero_context_hard | identical to head variant | — | — | — | — | — | — |

**Structural observation (exploratory):** the head and hard variants are
numerically identical for *every* partition. In this architecture the state
partitions never cross-couple — each partition feeds only the heads, so
zeroing a partition in the recurrence cannot change any prediction beyond
what head-input zeroing already does. The "hard" variant is vacuous for the
Stage-1 topology; the partitions are three independent channels concatenated
at the head.

All three partitions are load-bearing at the head: context dominates
(ΔAUC −0.078), event and clock each contribute ≈ 0.006–0.007 AUC. The
clock's contribution is largest for duration prediction (+282 d MAE when
zeroed), i.e. it carries time-to-resolution signal.

## 3. Probe 2 — gate clamps and the effective coefficient β_k = g_k·α_k

| condition | ΔAUC | Δdur MAE (d) | ΔECE | Δnext-type acc | Δnext-gap pinball | Δdur pinball |
|---|---|---|---|---|---|---|
| g ≡ 0 | −0.0069 ± 0.0045 | +282.4 ± 162.9 | +0.022 ± 0.029 | −0.0131 ± 0.0109 | +0.024 ± 0.015 | +0.018 ± 0.017 |
| g ≡ 1 | −0.0028 ± 0.0028 | +3.8 ± 4.3 | +0.006 ± 0.009 | −0.0063 ± 0.0057 | +0.003 ± 0.002 | +0.003 ± 0.003 |
| observed g (baseline) | 0 (reference) | 0 | 0 | 0 | 0 | 0 |

g ≡ 0 freezes the clock state at its zero initialization, so it is
numerically identical to clock zeroing (verified to 1e-12) — an internal
consistency check, not an independent result.

Empirical distributions over holdout steps (per-seed statistics averaged
over seeds; only β is identified, g and r are reported for completeness and
are **not** interpreted separately):

| quantity | mean | q05 | q25 | q50 | q75 | q95 |
|---|---|---|---|---|---|---|
| β = g·α | 0.359 | 0.000 | 0.073 | 0.298 | 0.599 | 0.930 |
| g | 0.572 | 0.137 | 0.398 | 0.560 | 0.776 | 0.965 |
| r | 1.350 | 0.151 | 0.533 | 0.925 | 1.727 | 4.090 |

The gate did not collapse (broad β and g distributions spanning [0, 1]).
Clamping g ≡ 1 costs ≈ 0.003 AUC (CI barely includes zero), so the gate's
modulation contributes at most marginally beyond always-flowing.

## 4. Probe 3 — clock identity and removal

| condition | ΔAUC | Δdur MAE (d) | Δrec log-MAE | ΔECE | Δnext-gap pinball | Δdur pinball |
|---|---|---|---|---|---|---|
| no-flow (α ≡ 0) | identical to zero_clock_head | — | — | — | — | — |
| heads-only | identical to zero_clock_head | — | — | — | — | — |
| const (train-mean z_clock) | −0.0069 ± 0.0045 | +21.4 ± 6.7 | −0.010 ± 0.023 | +0.000 ± 0.005 | +0.009 ± 0.003 | +0.005 ± 0.003 |

`no-flow` freezes the clock state at its zero initialization, so it too is
numerically identical to clock zeroing (verified to 1e-12). The `const`
condition replaces the clock at the head input with its per-seed
training-portion mean: because any constant replacement shifts every head's
output by a step-invariant constant, its AUC delta equals the zeroing delta
exactly (rank-preserving), while its regression-metric deltas are far
smaller (+21 d vs +282 d duration MAE). The clock's *constant level* thus
carries most of its duration-MAE contribution; its *variation over time*
contributes ≈ 0.007 AUC of ranking signal.

## 5. Probe 4 — time perturbations (input-side, inference only)

| condition | ΔAUC | Δdur MAE (d) | ΔECE | Δnext-type acc | Δnext-gap pinball | Δdur pinball |
|---|---|---|---|---|---|---|
| shuffle within phase | +0.0016 ± 0.0015 | +7.3 ± 8.3 | +0.001 ± 0.002 | +0.000 ± 0.002 | −0.002 ± 0.001 | −0.001 ± 0.002 |
| per-phase median | +0.0020 ± 0.0022 | −3.3 ± 2.0 | +0.001 ± 0.005 | +0.001 ± 0.002 | −0.004 ± 0.001 | −0.002 ± 0.001 |
| global median (train) | −0.0072 ± 0.0050 | +7.6 ± 5.0 | +0.005 ± 0.008 | −0.001 ± 0.002 | +0.006 ± 0.002 | +0.004 ± 0.003 |
| Δt = 0 everywhere | −0.0160 ± 0.0107 | +164.8 ± 134.6 | +0.055 ± 0.043 | −0.026 ± 0.011 | +0.070 ± 0.033 | +0.045 ± 0.031 |
| log-scaled Δt | −0.0096 ± 0.0034 | +22.2 ± 6.7 | +0.038 ± 0.030 | −0.016 ± 0.011 | +0.036 ± 0.019 | +0.021 ± 0.008 |

Shuffling the intervals within each procedural phase — destroying the true
interval sequence while preserving its per-phase multiset — has **no
effect** (ΔAUC CI includes zero, slightly positive). Replacing intervals
with their per-phase median is likewise neutral. But Δt = 0, log-scaling,
and even a global-median replacement do degrade outputs. Read together:
the model's output depends on the Δt **magnitude/scale** reaching the flow
and the history encoder's time features, but **not on the ordering of true
intervals** — the signature of generic magnitude sensitivity, not of
interval-compositional temporal reasoning. Per the frozen spec, the
log-scale probe is interpreted only as "does the output depend on the Δt
scale at all" (it does) — never as mechanism evidence.

## 6. Probe 5 — `idn-ffres` (**new exploratory model**)

All archived weights frozen; the clock partition's head input replaced by
MLP(c_{k−1}, log1p Δt_k) = Linear(33→32)–ReLU–Linear(32→16), **1,616
params** (target 1,600 ± 20% = [1280, 1920] ✓). Only the MLP trained, on
each seed's training portion, with the frozen multi-task loss, 25 epochs,
lr = 0.001 (the archived Stage-1 selection for idn/hidden), batch 32, grad
clip 1.0. Mean final train loss 8.80 (from 9.29 at epoch 1).

| condition | ΔAUC | Δdur MAE (d) | Δrec log-MAE | ΔECE | Δnext-type acc | Δnext-gap pinball | Δdur pinball |
|---|---|---|---|---|---|---|---|
| idn-ffres | −0.0007 ± 0.0015 | +0.9 ± 4.4 | +0.026 ± 0.082 | −0.009 ± 0.010 | −0.004 ± 0.005 | +0.001 ± 0.002 | −0.003 ± 0.004 |

**A parameter-matched generic MLP residual, given 25 epochs on the same
training portion, fully matches the clock's contribution on every metric**
(all CIs include zero). The clock machinery was not providing anything a
generic nonlinear residual of the same capacity does not provide.

## 7. Probe 6 — head dependence

(a) Input-weight column norms by partition (mean over seeds; means per input
dimension):

| head | event | clock | context | static |
|---|---|---|---|---|
| settle | 0.113 | 0.095 | 0.095 | 0.060 |
| recovery | 0.157 | 0.123 | 0.140 | 0.059 |
| remaining | 0.136 | 0.118 | 0.122 | 0.063 |
| next-type | 0.731 | 0.630 | 0.569 | 0.274 |
| next-gap-q | 0.150 | 0.143 | 0.155 | 0.110 |
| duration-q | 0.189 | 0.159 | 0.201 | 0.114 |

(b) Gradient norms of the masked holdout loss w.r.t. each state partition
(RMS per step per dimension, averaged over seeds): event 0.00126, clock
0.00100, context 0.00118.

(c) Cross-reference with probe 1: the weight and gradient profiles are
roughly flat across partitions (clock norms ≈ 80–90% of event/context
norms), yet zeroing context costs 11× more AUC than zeroing clock. Head
*dependence* on the clock is therefore modest and diffuse — consistent with
the clock channel carrying a small, generic, replaceable signal (probe 5)
rather than a dedicated temporal readout. The duration heads show no
privileged clock connectivity (duration-q clock norm 0.159 < context 0.201).

## 8. Probe 7 — stratified paired evaluation (IDN − tf-native-aux, settle AUC)

Per stratum: total holdout steps/cases/positives over 10 seeds, mean per-seed
AUC for each model, paired ΔAUC ± 95% CI, and Holm-adjusted p within each
stratum family (**descriptive aid only — no confirmatory claim is possible
post-verdict**). Eligibility per seed-stratum: ≥ 100 steps and ≥ 2 label
classes (all listed strata qualified on all 10 seeds).

| family | stratum | steps | cases | pos | AUC IDN | AUC tf-aux | ΔAUC ± CI | Holm-adj p |
|---|---|---|---|---|---|---|---|---|
| delta_quartile | q1 | 5736 | 2040 | 1717 | 0.920 | 0.923 | −0.0027 ± 0.0053 | 0.549 |
| | q2 | 5730 | 1615 | 1716 | 0.782 | 0.787 | −0.0058 ± 0.0057 | 0.195 |
| | q3 | 5728 | 1694 | 1297 | 0.813 | 0.818 | −0.0053 ± 0.0074 | 0.416 |
| | q4 | 5732 | 1521 | 917 | 0.817 | 0.816 | +0.0009 ± 0.0078 | 0.808 |
| long_gap | ≤ 90 d | 17863 | 2040 | 4857 | 0.849 | 0.852 | −0.0038 ± 0.0050 | 0.243 |
| | > 90 d | 5063 | 1449 | 790 | 0.814 | 0.815 | −0.0010 ± 0.0086 | 0.794 |
| age_quartile | q1 | 5738 | 2040 | 613 | 0.786 | 0.790 | −0.0042 ± 0.0083 | 1.000 |
| | q2 | 5728 | 1716 | 1341 | 0.850 | 0.853 | −0.0032 ± 0.0092 | 1.000 |
| | q3 | 5728 | 1334 | 1857 | 0.834 | 0.837 | −0.0029 ± 0.0077 | 1.000 |
| | q4 | 5732 | 862 | 1836 | 0.828 | 0.825 | +0.0032 ± 0.0097 | 1.000 |
| phase | early | 8478 | 2040 | 411 | 0.607 | 0.626 | −0.0191 ± 0.0196 | 0.167 |
| | mid | 7499 | 2040 | 1528 | 0.622 | 0.634 | −0.0117 ± 0.0127 | 0.167 |
| | late | 6949 | 2040 | 3708 | 0.841 | 0.835 | +0.0066 ± 0.0118 | 0.241 |
| seq_length | short | 12434 | 1414 | 3701 | 0.886 | 0.892 | −0.0055 ± 0.0026 | **0.0019** |
| | long | 10492 | 626 | 1946 | 0.844 | 0.845 | −0.0009 ± 0.0128 | 0.883 |
| backlog | in episode | 8580 | 1529 | 979 | 0.868 | 0.871 | −0.0024 ± 0.0071 | 0.460 |
| | outside | 14346 | 1832 | 4668 | 0.840 | 0.844 | −0.0033 ± 0.0046 | 0.280 |
| adverse_regime | adverse | 5621 | 636 | 1238 | 0.799 | 0.797 | +0.0024 ± 0.0127 | 0.679 |
| | not adverse | 17305 | 2040 | 4409 | 0.875 | 0.879 | −0.0038 ± 0.0031 | 0.044 |
| selective_obs | selective | 9528 | 1570 | 2324 | 0.752 | 0.758 | −0.0069 ± 0.0091 | 0.243 |
| | always | 13398 | 2040 | 3323 | 0.895 | 0.897 | −0.0021 ± 0.0041 | 0.281 |
| discovery_stall | stall case | 11246 | 789 | 2463 | 0.864 | 0.870 | −0.0057 ± 0.0070 | 0.194 |
| | no stall | 11680 | 1251 | 3184 | 0.837 | 0.839 | −0.0020 ± 0.0034 | 0.220 |

Only one stratum survives Holm within its family (descriptive): **short
sequences**, where the IDN trails tf-native-aux by −0.0055 ± 0.0026 AUC —
the opposite of a temporal-mechanism success story. The nominally largest
IDN advantage (late phase, +0.0066) has a CI spanning zero. The
latent-log strata (judge-backlog episodes, adverse regimes, selectively
observed event classes, discovery-stall cases) show no stratum in which the
IDN — and by extension its clock — is reliably superior: backlog-episode
steps Δ = −0.0024 ± 0.0071, adverse-regime steps Δ = +0.0024 ± 0.0127,
selectively observed classes Δ = −0.0069 ± 0.0091, stall cases Δ = −0.0057
± 0.0070. There is no identifiable temporal stratum where the clock is
load-bearing in the IDN's favor.

## 9. Interpretation matrix applied (frozen matrix, actual observations)

| Frozen row | Fired? | Observation |
|---|---|---|
| Clock zeroing has no effect → unused/redundant | **No** | Zeroing costs −0.0069 ± 0.0045 AUC, +282 d duration MAE |
| Clock zeroing improves → harmful/misspecified | **No** | Zeroing strictly hurts |
| Clock zeroing hurts, but `idn-ffres` matches IDN → **generic residual capacity, not temporal mechanism** | **YES** | ffres ΔAUC −0.0007 ± 0.0015, ΔdurMAE +0.9 ± 4.4 d, all metrics' CIs include zero (probe 5) |
| Time shuffling has no effect → **no real interval semantics** | **YES** | shuffle-within-phase ΔAUC +0.0016 ± 0.0015; phase-median +0.0020 ± 0.0022 (probe 4). Nuance: output still depends on Δt *scale* (dt=0, log, global-median all hurt) |
| Gate clamps have no effect → gate collapsed / heads ignored clock | **No** | Gate did not collapse (β spans [0, 0.93]; g median 0.56); g ≡ 1 costs ≈ 0.003 AUC; g ≡ 0 ≡ full clock removal |
| Clock helps only in late/long strata → preregisterable temporal stratum | **No** | Late phase Δ(IDN−tf) = +0.0066 ± 0.0118 (CI spans 0); long sequences −0.0009 ± 0.0128; the only Holm-surviving stratum is *short* sequences, where the IDN is *worse* |
| Clock helps broadly but IDN still loses → benefit offset elsewhere | **Consistent (descriptive)** | Clock worth ≈ +0.007 AUC broadly, yet IDN − tf-aux = −0.0039 overall; probe 6 shows no privileged clock connectivity |

## 10. Topology decision (frozen stopping rule)

The stopping rule: retire the topology if clock ablation (probes 1/3) is
neutral or beneficial; if the clock proves load-bearing in identifiable
strata, that informs — but does not guarantee — a successor preregistration.

Clock ablation is *not* neutral (it costs ≈ 0.007 AUC), so the literal
retirement trigger does not fire — but the clock also fails the only
condition under which the rule allows the topology to inform a successor:
it is **not load-bearing in any identifiable stratum** (probe 7: every
pro-clock stratum CI spans zero; the single Holm survivor runs against the
IDN). And the two interpretation-matrix rows that *did* fire strip the
topology of its mechanism claim:

1. **Clock zeroing hurts, but `idn-ffres` matches IDN** → the clock's
   contribution is generic residual capacity, not temporal mechanism.
2. **Time shuffling has no effect** → the flow did not use real interval
   semantics.

A context-conditioned exponential-relaxation clock whose exponential
relaxation is replaceable by a 1,616-parameter MLP on (context, log1p Δt),
and which is indifferent to the ordering of the intervals it integrates,
was never functioning as a clock. Retaining the topology would retain
*capacity*, not *mechanism* — and that capacity is available more cheaply
and with fewer moving parts elsewhere.

**Decision (exploratory, within F1's sole decision scope): RETIRE the
Stage-1 context-conditioned exponential-relaxation clock topology.** Any
future continuous-time proposal must introduce a materially different
transition mechanism or training objective (e.g., F2's interval
supervision); resubmitting this topology with retuned gates, widths, or
schedules is not a materially different proposal.

**The Stage-1 kill stands.** Nothing above revisits the confirmatory
verdict; these findings may only inform a separately preregistered
successor (F2+).

## 11. Reproducibility

- Runner: `experiments/f1_forensics.py` (CPU, deterministic seeds; runtime 64 s).
  Run: `cd experiments && ../.venv/bin/python f1_forensics.py`.
- Results JSON: `experiments/results/f1_forensics.json` (all numbers above
  are taken from `summary.*` in that file).
- Archive: `experiments/archive/f1-forensics/` containing `config.json`,
  `environment.json`, `raw_metrics.json` (per-seed per-condition metrics),
  `per_step_outputs.npz` (per-step settle logits/labels for baseline, ffres,
  tf-native-aux, plus step metadata), the code snapshot `f1_forensics.py`,
  and `hashes.json` (SHA-256 manifest; verified against the directory
  contents after the run).
- No file under `archive/stage1-killed/` or listed in `FREEZE.md` was
  modified. The archived IDN forward was never edited; probes subclass it.
- Environment: python 3.14.6, torch 2.13.0, numpy 2.5.1, scipy 1.18.0
  (descriptive t-tests only), macOS (see `environment.json` for `uname`).
