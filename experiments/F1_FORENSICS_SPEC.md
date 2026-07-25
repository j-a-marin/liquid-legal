# F1_FORENSICS_SPEC — exploratory checkpoint forensics on the killed Stage-1 IDN

**Status: frozen 2026-07-25, before any F1 probe is run.**
**Evidentiary status: EXPLORATORY.** The confirmatory Stage-1 analysis ended
when the preregistered primary screen failed (`STAGE1_RESULTS.md`). F1 runs
on the archived killed candidate (`archive/stage1-killed/`) and cannot alter,
soften, or reopen that verdict. Its sole decision scope is whether to retain
or retire the Stage-1 **context-conditioned exponential-relaxation clock
topology** in future designs. Findings may only inform a separately
preregistered successor (F2+).

Governing charter: successor handoff prompt, section IV. Governance:
`IDN_GUIDE.md` claim discipline (F1 analyzes model behavior, not latent
institutional structure; gate values and partition norms are model internals,
not mechanisms, unless anchored to Generator v2 latent logs).

## Assets under test

- Archived checkpoints: `archive/stage1-killed/weights/idn_hidden_seed{0..9}.pt`
  (primary regime, hidden statics). Opponent for paired reference:
  `tf-native-aux_hidden_seed{0..9}.pt`. Visible-statics checkpoints are
  descriptive only.
- Data: frozen `gen_v2.py`, 1024 cases/seed, seeds [0–9], 20% holdout via
  `np.random.default_rng(seed).permutation`, identical to Stage 1. All F1
  metrics are computed on the same holdout steps as the archived evaluation.
- No archived weight is ever fine-tuned within an ablation. Any probe that
  fits parameters is defined as a **new exploratory model** (section 5 below)
  and labeled as such.

## Metrics recorded by every probe

On holdout, masked per-step, per seed: settle AUC (180-day), duration MAE
(days), recovery log-MAE, ECE (15 equal-width bins), next-event-type
accuracy, next-gap pinball loss (quantiles 0.1/0.5/0.9), duration pinball
loss. Deltas are reported against the unmodified archived IDN on identical
steps, paired per seed, mean ± 95% t-interval (df = 9).

## Probe battery

### 1. Partition zeroing (inference-time)

Separately replace `z_event`, `z_clock`, `z_context` with zeros at every
step before the prediction heads (state recurrence left untouched — zeroing
applies to the head input only; a second variant zeroes the partition in the
recurrence as well, labeled `*_hard`). Record the full metric set per
condition.

### 2. Gate clamps and effective-coefficient distribution

The composed clock update is z_k = (1 − g_k α_k) z_{k−1} + g_k α_k T(c_{k−1});
only the product β_k = g_k α_k is identified. Conditions: g ≡ 0, g ≡ 1,
observed g (baseline). Record the empirical distribution of β_k over holdout
steps (quantiles 0.05–0.95, mean), plus the observed distributions of g and
r for completeness. g and r are NOT interpreted separately: the analysis
addresses their non-identifiability by reporting only β as the effective
quantity.

### 3. Clock identity and removal

- `no-flow`: clock state copied unchanged (α ≡ 0).
- `heads-only`: clock partition removed from the head input (recurrence
  untouched).
- `const`: clock partition replaced at the head input by a constant vector =
  the per-seed empirical mean of `z_clock` over that seed's TRAINING portion
  (no gradient updates; computed once, then frozen for evaluation).

### 4. Time perturbations (input-side)

Applied to `timespans` at inference: (a) Δt shuffled within procedural phase
(phase = position tercile of the case, per case, seeded permutation,
seed = run seed); (b) Δt replaced by its per-phase median; (c) Δt replaced
by the global median; (d) Δt = 0 everywhere; (e) log-scaled Δt
(inference-only sensitivity probe — changes the flow's semantics, so it is
interpreted only as "does the model's output depend on the Δt scale at
all", never as mechanism evidence).

### 5. Generic residual comparison (new exploratory model `idn-ffres`)

Question: was the clock useful as elapsed-time dynamics, or as generic
nonlinear residual capacity? Construction: all archived IDN weights frozen;
the clock partition's head input is replaced by MLP(c_{k−1}, log1p Δt_k)
with parameter count matched (±20%) to the clock machinery it replaces
(flow_rate 528 + flow_target 528 + flow_gate 544 = 1,600 params at the
archived widths d_context = 32, d_clock = 16). Only the MLP is
trained, on the seed's training portion, with the identical frozen losses
and 25-epoch schedule as Stage 1. Compared against the archived IDN's
holdout metrics, paired per seed. This is a new exploratory model, not an
ablation, and is reported as such.

### 6. Head dependence (static analysis + gradients)

For each head (settle, recovery, remaining, next-type, next-gap-q,
duration-q): (a) input-weight column norms grouped by partition
[event, clock, context, static]; (b) gradient norms of the masked holdout
loss w.r.t. each state partition, accumulated over evaluation batches;
(c) per-head sensitivity = metric delta under that partition's zeroing
(cross-reference probe 1). Any mutual-information or probing estimates, if
added later, are exploratory supplements and labeled as such.

### 7. Stratified evaluations

Retains the existing H8 strata (preceding-gap quartiles, gap > 90d, case-age
quartiles, procedural-phase terciles) and adds, where sample size permits
(≥ 100 steps and ≥ 2 label classes per seed-stratum): short vs long
sequences (median split on valid length); backlog-regime steps (from latent
logs: step occurs inside a logged judge-backlog episode); adverse-regime
steps (inside a logged adverse case-regime episode); steps whose event class
is selectively observed vs always observed (from the observation mask);
steps in cases with ≥ 1 discovery stall vs none. Per stratum report: step
count, case count, positive-label count, per-seed metric, paired delta vs
tf-native-aux, 95% CI, and multiple-comparison status (Holm within each
stratum family, applied as a descriptive aid only — no confirmatory claim is
possible post-verdict).

## Interpretation matrix (frozen)

| Observation | Interpretation |
|---|---|
| Clock zeroing has no effect | Clock was unused or fully redundant |
| Clock zeroing improves results | Flow was harmful or misspecified |
| Clock zeroing hurts, but `idn-ffres` matches IDN | Generic residual capacity, not temporal mechanism |
| Time shuffling has no effect | Clock did not use real interval semantics |
| Gate clamps have no effect | Gate collapsed or heads ignored the clock |
| Clock helps only in late/long strata | Successor may target a preregistered temporal stratum |
| Clock helps broadly but IDN still loses | Benefit offset by budget fragmentation or another partition |

## Stopping rule (narrow by design)

F1 does not decide whether continuous-time modeling is globally valid. If
clock ablation (probes 1/3) is neutral or beneficial, **retire the Stage-1
context-conditioned exponential-relaxation topology**. Any future
continuous-time proposal must then introduce a materially different
transition mechanism or training objective (F2's interval supervision is
such a change). If the clock proves load-bearing in identifiable strata,
that informs — but does not guarantee — a successor preregistration.

## Reproducibility

- One runner: `experiments/f1_forensics.py`; outputs to
  `experiments/results/f1_forensics.json` and
  `experiments/archive/f1-forensics/` (raw per-step outputs, config, code
  snapshot, environment record, SHA-256 manifest).
- CPU, deterministic seeds; no changes to any file under
  `archive/stage1-killed/` or to frozen files listed in `FREEZE.md`.
- Write-up: `experiments/F1_FORENSICS_RESULTS.md`, following the section-X
  reporting standard of the successor charter (all analyses exploratory).
