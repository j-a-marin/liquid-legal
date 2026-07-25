# Results: ablations and stress tests on synthetic litigation trajectories

> **Stage-1 epilogue (2026-07-23):** the IDN hybrid, evaluated against an
> auxiliary-matched Transformer under a frozen protocol on Generator v2,
> failed its preregistered primary screen (mean paired ΔAUC −0.0039, 95% CI
> [−0.0082, +0.0004]) and the hybrid track was stopped per protocol.
> Full record: [`STAGE1_RESULTS.md`](STAGE1_RESULTS.md).

Reproduce everything below with:

```bash
python examples/benchmark.py            # headline CfC vs LSTM (~12 min CPU)
python experiments/run_all.py           # E2–E5 full matrix (~15 min CPU)
python experiments/run_hidden_statics.py # E6 hidden-latent ablation (~8 min CPU)
```

Config: 1024 generated cases per run, 25 epochs, seeds {0, 1, 2} (fresh
dataset, init, and split per seed), 20% holdout, identical splits, heads,
losses, and metrics for every model. Raw numbers: `experiments/results/`.

Model zoo and parameter budgets:

| family | variants | params |
|---|---|---|
| CfC (liquid), AutoNCP wiring | native / timespans_only / feature / none | 15,898 |
| LSTM | feature / none | 24,259 |
| Transformer encoder (causal-masked, time encodings) | native / timespans_only / feature / none | 22,403 |
| XGBoost on engineered features | full / no-time | — |

The question: *for irregular legal-event trajectories, does predictive
performance come primarily from recurrence, continuous-time dynamics,
attention, or conventional feature engineering?*

## E2 — the full model matrix (holdout, mean ± std)

| model | settle AUC ↑ | duration MAE (days) ↓ |
|---|---|---|
| xgb-no-time | 0.917 ± 0.012 | 144 ± 20 |
| xgb | 0.915 ± 0.013 | 146 ± 20 |
| lstm-feature | 0.914 ± 0.015 | 146 ± 19 |
| lstm-none | 0.912 ± 0.015 | 153 ± 29 |
| tf-feature | 0.911 ± 0.018 | 149 ± 23 |
| tf-none | 0.911 ± 0.022 | 153 ± 25 |
| tf-native | 0.909 ± 0.019 | 154 ± 24 |
| tf-timespans_only | 0.908 ± 0.023 | 158 ± 30 |
| cfc-none | 0.899 ± 0.017 | 151 ± 24 |
| cfc-feature | 0.897 ± 0.007 | 149 ± 25 |
| cfc-timespans_only | 0.886 ± 0.010 | 154 ± 21 |
| cfc-native | 0.871 ± 0.018 | 155 ± 22 |

At 50 epochs the neural ordering is unchanged (CfC 0.878 vs LSTM 0.913), so
the gap is not a convergence artifact.

**Finding 1: on this generator, conventional feature engineering sets the
ceiling, neural sequence models merely reach it, and liquid dynamics sit
below.** Time handling adds nothing anywhere: the no-time variant of every
family matches or beats its time-aware variants.

**Finding 2 (why): the synthetic world over-discloses its latents.** Judge
volatility, district congestion, plaintiff capability — the quantities that
*cause* the irregular timing — are handed to the model as static features.
When the cause is observable, the effect (the timestamps) is redundant. This
motivates E6.

## E3 — irregularity stress (clean-trained models, corrupted holdout)

Event dropout simulates incomplete dockets; timestamp jitter simulates noisy
metadata. Settle AUC, mean over seeds:

| model | 0% drop | 10% | 20% | 30% | 50% |
|---|---|---|---|---|---|
| cfc-native | 0.871 | 0.867 | 0.867 | 0.869 | 0.872 |
| cfc-timespans_only | 0.886 | 0.883 | 0.881 | 0.878 | 0.873 |
| cfc-feature | 0.897 | 0.896 | 0.896 | 0.894 | 0.893 |
| lstm-feature | 0.914 | 0.914 | 0.915 | 0.912 | 0.913 |
| tf-native | 0.909 | 0.911 | 0.909 | 0.907 | 0.901 |
| tf-feature | 0.911 | 0.912 | 0.912 | 0.912 | 0.910 |
| xgb | 0.915 | 0.916 | 0.916 | 0.914 | 0.913 |
| xgb-no-time | 0.917 | 0.917 | 0.918 | 0.915 | 0.916 |

| model | 0% jitter | 10% | 25% |
|---|---|---|---|
| cfc-native | 0.871 | 0.860 | 0.835 |
| cfc-timespans_only | 0.886 | 0.872 | 0.853 |
| cfc-feature | 0.897 | 0.891 | 0.877 |
| lstm-feature | 0.914 | 0.908 | 0.896 |
| tf-native | 0.909 | 0.904 | 0.891 |
| tf-feature | 0.911 | 0.906 | 0.894 |
| xgb | 0.915 | 0.912 | 0.904 |
| xgb-no-time | 0.917 | 0.915 | 0.907 |

**Finding: dropout barely degrades anyone; jitter degrades everyone, and the
models that consume raw timestamps degrade most** (25% jitter: −0.036/−0.033
for the timespan variants, −0.018 for LSTM/Transformer, −0.011 for XGBoost).
The "liquid models degrade more gracefully" hypothesis is not supported on
synthetic data — when timing is redundant, corrupting it mostly adds noise to
the models that consume it. Reported as-is.

## E4 — counterfactual judge probe

Same mid-case docket prefix (truncated at 50% of events, n=100 per seed),
judge traits swapped across the 12-judge pool. Mean Spearman correlation
between judge speed (higher = slower) and predicted remaining duration:

| model | r (mean ± std) |
|---|---|
| lstm-feature | 0.903 ± 0.069 |
| tf-native | 0.869 ± 0.126 |
| cfc-native | 0.564 ± 0.090 |

All three families internalize "who the judge is" in the correct direction.
The probe itself — *same docket, different judge* — is the query a funder
actually wants to run, and it works on a trained model with no re-simulation.

## E5 — ground-truthed saliency (the positive result)

Mean input-gradient saliency of the settlement head per event occurrence
(cfc-native, seed 0 holdout; consistent across seeds):

| event | saliency | n |
|---|---|---|
| FILED | 3.377 | 204 |
| MOTION_SUMMARY_JUDGMENT | 3.143 | 65 |
| MOTION_TO_COMPEL | 2.192 | 100 |
| DEPOSITION | 1.640 | 426 |
| MOTION_TO_DISMISS | 1.543 | 121 |
| SETTLEMENT_OFFER | 1.335 | 323 |
| … | | |
| MEDIATION | 0.912 | 93 |
| SETTLED | 0.068 | 148 |
| DISMISSED | 0.022 | 55 |

Three checks against the known generative process:

- **Causal events rank highest.** FILED carries claimed damages; MSJ and
  motions to compel are the leverage-changing events by construction.
- **Terminal events get ≈0 saliency** — the model correctly treats them as
  outcome, not predictor.
- **Stall saliency splits by plaintiff capability: 3.50 vs 1.27 (2.8×)** —
  matching the generator's mechanism, where stalls are both more frequent and
  more damaging for under-equipped plaintiffs.

Because the ground-truth process is known, explanation *correctness* is
measurable here — a property most interpretability benchmarks lack.

## E6 — hidden-latent ablation: zeroing the static covariates

Same suite, but all static features zeroed, so event semantics and the
timestamps themselves must carry the signal — the condition continuous-time
models are built for.

| model | AUC (statics hidden) | Δ vs full statics |
|---|---|---|
| tf-native | 0.902 ± 0.010 | −0.007 |
| tf-feature | 0.898 ± 0.009 | −0.013 |
| lstm-feature | 0.898 ± 0.005 | −0.016 |
| xgb | 0.894 ± 0.011 | −0.021 |
| cfc-feature | 0.876 ± 0.015 | −0.021 |
| cfc-native | 0.872 ± 0.026 | **+0.001** |
| cfc-timespans_only | 0.868 ± 0.019 | −0.018 |
| xgb-no-time | 0.847 ± 0.008 | **−0.070** |

**Finding 3: when latents are hidden, timing carries real signal — but
attention, not liquid dynamics, exploits it best.** The evidence that timing
matters: XGBoost's time features go from useless (+0.002) to decisive
(+0.047 over its no-time variant). The evidence that liquid dynamics don't
differentiate: cfc-native is the only model *indifferent* to losing the
statics (it never learned to use them), yet it still trails the field in
absolute terms; the time-aware Transformer handles the condition best.

## What this supports in the paper

1. A reproducible formulation, generator, and evaluation harness; every
   number above regenerates from three commands.
2. A rigorous negative result: on synthetic dockets with observable latents,
   feature engineering suffices, neural sequence models add nothing, and
   liquid continuous-time dynamics add nothing over discrete ones.
3. A sharp conditional: timing information is decisive exactly when the
   latent drivers are *not* observed — and in that regime attention
   outperforms recurrence, liquid or not. This is the hypothesis real
   docket data must adjudicate.
4. A positive auditability result: 15.9k parameters, 86% structural
   sparsity (NCP wiring), and saliency that recovered several planted mechanisms under
   controlled synthetic conditions — stable directional alignment with the
   causal events, variable magnitude across seeds — plus a counterfactual
   probe that works on the trained model.
5. Process finding: the E5 saliency infrastructure caught a semantic bug in
   the generator (enum-identity collapse in event sampling) that all
   predictive benchmarks had silently tolerated — explanation-driven
   debugging found what accuracy could not.

## Threats to validity

Synthetic data only; the default generator over-discloses latents relative to
any real underwriting setting; settle AUC uses a 180-day horizon label;
matched parameter budget but no exhaustive hyperparameter search; the XGBoost
baseline receives hand-engineered features a real deployment would need to
re-derive; single taxonomy of 16 event types. Real dockets may differ in
exactly the direction that favors continuous-time models — or not.
