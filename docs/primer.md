# Litigation Physics: Learning Institutional Dynamics

*A primer on irregular legal-event modeling, continuous-time neural networks,
and what our experiments actually show. Revised July 22, 2026; the prior
version is preserved in [`docs/archive/`](https://github.com/j-a-marin/liquid-legal/tree/main/docs/archive).*

---

## The object is not the document

Most legal AI reads documents. It extracts facts, retrieves precedent,
classifies filings, or generates text. Those are useful tasks, but litigation
is not a pile of documents. It is a process unfolding inside institutions.

A matter occupies a partially observed state: merits, leverage, resources,
procedure, accumulated cost, court capacity, doctrine, and party strategy.
Filings and rulings reveal pieces of that state, but can also change it.
Meanwhile, the matter continues to evolve when nothing appears on the docket.
Deadlines approach, evidence ages, budgets tighten, courts accumulate backlog,
and bargaining positions move.

That is the legitimate sense in which litigation has a kind of physics:

> **Litigation has no literal laws of motion, but it has learnable empirical
> transition dynamics.**

The research question is larger than “can we predict settlement from docket
features?” It is:

> Can we learn a latent institutional state that explains how matters drift
> between observable events, jump when actors intervene, and cross into
> settlement, dismissal, judgment, or another procedural regime?

## A state-space view of litigation

Let a latent state at time *t* be

```text
z_t = (merits, leverage, resources, procedure,
       court capacity, doctrine, strategy)
```

Institutional conditions and procedural rules shape how that state evolves:

```text
dz/dt = F(z_t, I_t, R_t)
```

where \(I_t\) represents the institution—judge, court, congestion, political
environment—and \(R_t\) the governing procedural regime. An observed event
can create a discontinuity:

```text
z_(t+) = J(z_(t-), e_t)
```

A ruling can change leverage; a trial date can change the settlement hazard;
a substitution of counsel can change resources and strategy. Settlement,
dismissal, and judgment behave like absorbing boundaries for one phase of the
matter, although appeal and enforcement may begin another.

This analogy has limits. Legal systems are strategic, jurisdiction-dependent,
mutable, and reflexive. Actors can observe a forecast and change their
behavior. These are empirical, local laws of motion—not universal laws like
gravity.

> **[DIAGRAM 1 — institutional state.]** *A matter trajectory moving through
> a field shaped by judge, court, procedural regime, party resources, and
> doctrine. Filings and rulings appear as impulses; settlement, dismissal,
> and judgment as boundaries; a period of silence still contains drift.*

## Why irregular time matters

A docket is an irregular observation process. Three events may arrive in a
week, followed by eight months of silence, followed by a ruling and rapid
settlement. Event order alone does not distinguish eight days from eight
months.

But irregular timestamps do not automatically justify a continuous-time
model. Timing helps only when it contains information that event identity,
event order, and observed covariates do not already disclose. If a model is
explicitly told the judge's speed and the court's congestion, a long gap may
be redundant. If those institutional causes are hidden, the gap may become
an important measurement of them.

This distinction—between *having timestamps* and *timestamps carrying unique
signal*—became the central result of our experiments.

> **[DIAGRAM 2 — irregular observation.]** *The same event sequence shown
> twice with short and long gaps. The outcomes differ because elapsed time
> changes latent leverage, cost, deadline pressure, or inferred institutional
> state—not because an LSTM requires a fixed weekly grid.*

## Four competing explanations

We compare four model families because each embodies a different account of
where predictive structure comes from.

### Engineered features and gradient-boosted trees

XGBoost receives a table of static covariates, event counts, event types,
amounts, and optionally timing summaries. It asks whether domain-informed
feature engineering already exposes enough state that sequence modeling is
unnecessary.

### Recurrent state

An LSTM updates a compressed memory after every event. It naturally represents
path dependence:

```text
z_k = F(z_(k-1), e_k)
```

An event-step LSTM does **not** require a regularly sampled grid. It can also
receive elapsed time as an input feature. What it lacks is an intrinsic rule
for evolving its memory during the interval between events.

### Attention over history

A temporal Transformer reconstructs the current representation by attending
to relevant events across the observed history. Transformers can consume
absolute timestamps, time gaps, continuous-time embeddings, or relative
temporal biases; they are not limited to token indices. Their distinctive
assumption is that temporal context conditions attention rather than directly
governing autonomous state evolution.

### Continuous-time latent dynamics

Liquid Time-Constant networks (LTCs) and Closed-form Continuous-time networks
(CfCs) model a hidden state that evolves as a function of elapsed time. CfC is
a computationally efficient closed-form approximation to LTC-style dynamics,
avoiding a numerical ODE solver. With sparse Neural Circuit Policy (NCP)
wiring, the model can also be compact and structurally inspectable.

The appeal is intuitive: some components of legal state may decay quickly,
others may persist, and the rates may depend on context. The empirical
question is whether this inductive bias actually improves learning.

> **[DIAGRAM 3 — four mechanisms.]** *XGBoost: an engineered snapshot table.
> LSTM: recurrent state updated per event. Transformer: attention across
> historical observations with temporal encodings. CfC: event updates joined
> by continuous-time latent evolution.*

## Where the liquid lineage comes from

The liquid models used here descend from work by Ramin Hasani, Mathias
Lechner, Alexander Amini, Daniela Rus, and collaborators:

- **Neural Circuit Policies (2020)** introduced sparse, structured neural
  wiring inspired by the compact nervous system of *C. elegans* and emphasized
  auditable autonomy.
- **Liquid Time-Constant Networks (2021)** gave neurons input-dependent
  continuous-time dynamics governed by differential equations.
- **Closed-form Continuous-time Networks (2022)** approximated those dynamics
  in closed form, making training and inference more practical.

The open-source `ncps` library implements these research architectures. In
this project, CfC is neither a foregone conclusion nor a product slogan. It
is one hypothesis about how institutional trajectories should be represented.

## The modeling stack

Raw legal text and institutional dynamics are different problems. A practical
system may contain several layers:

1. A language model or encoder extracts structured events, entities, rulings,
   amounts, and claims from documents.
2. A trajectory model—tree ensemble, LSTM, Transformer, CfC, or hybrid—updates
   estimates of latent legal and institutional state.
3. Calibrated prediction heads estimate competing outcomes, duration, and
   uncertainty.
4. Decision tools expose counterfactuals, explanations, portfolio risk, and
   monitoring alerts to human experts.

ModernBERT, T5, and Gemma belong primarily to the language layer: respectively
encoder, encoder-decoder, and decoder-oriented model families. They may help
read a docket, but scale alone does not answer how a legal process evolves.

No architecture has earned permanent ownership of the temporal core. Our
results make that architecture-neutral stance necessary.

## Hypotheses and experiments

The project separates hypotheses (**H**) from evaluations (**E**):

- **H1: native continuous time adds predictive value.** E1–E2 compare CfC,
  LSTM, Transformer, and engineered XGBoost variants under matched data and
  neural parameter budgets.
- **H2: continuous-time models degrade gracefully under irregularity.** E3
  corrupts held-out event streams with event dropout and timestamp jitter.
- **H3: learned state supports institutional counterfactuals.** E4 swaps
  judge traits while holding the docket prefix fixed.
- **H4: explanations recover known causal mechanisms.** E5 compares
  input-gradient saliency with the synthetic generator's ground truth.
- **H5: timing becomes valuable when institutional causes are hidden.** E6
  removes static covariates and retrains the model suite.

The benchmark uses 1,024 synthetic cases per seed, three seeds, a 20% holdout,
25 training epochs, a 16-event taxonomy, and a 180-day settlement-horizon
label. Synthetic data lets us know the causal process, but it is not evidence
about real-world accuracy.

## What the experiments found

### E1–E2: feature engineering sets the ceiling when latents are visible

With static institutional covariates available, the leading results are:

| model | settlement AUC | duration MAE |
|---|---:|---:|
| XGBoost, no timing features | 0.917 | 144 days |
| XGBoost, full features | 0.915 | 146 days |
| LSTM, time feature | 0.914 | 146 days |
| Transformer, time feature | 0.911 | 149 days |
| Transformer, native time | 0.909 | 154 days |
| CfC, no time | 0.899 | 151 days |
| CfC, native time | 0.871 | 155 days |

On this generator, conventional engineered features match or exceed the
neural models. Native continuous-time handling does not help: the no-time
variant of every family matches or beats its time-aware alternatives. The
neural ordering persists with a larger training budget, so the CfC gap is not
merely early convergence.

The explanation is structural. The default generator reveals judge
volatility, district congestion, plaintiff capability, and other covariates
that cause both timing and outcomes. Once those causes are observable, timing
adds little.

### E3: missing events are benign; noisy time is not

Dropping as many as half the held-out events barely changes settlement AUC for
any model. Timestamp jitter hurts every model that uses time, and the raw
timespan liquid variants degrade most at the highest tested corruption.

The hypothesis that liquid dynamics are intrinsically robust to irregular or
messy timestamps is therefore not supported here. Irregular sampling and
timestamp corruption are different problems: a model designed for the first
does not automatically solve the second.

### E4: neural models support a judge counterfactual

Holding a mid-case docket prefix fixed while changing judge traits produces
predictions in the correct direction for all three neural families. The mean
Spearman correlations between judge speed and predicted remaining duration
are 0.903 for the LSTM, 0.869 for the Transformer, and 0.564 for native CfC.

This does not prove causal identification on real data. It demonstrates that
the trained synthetic models can answer a useful structural query: *same
docket, different judge*.

### E5: explanations recover mechanisms built into the world

CfC input-gradient saliency ranks leverage-changing events such as dispositive
motions and discovery stalls highly, while terminal events receive nearly
zero saliency. In seed 0, motion-to-compel saliency is 2.8 times higher for
low-capability than high-capability plaintiffs, matching the generator's
mechanism. The direction persists across seeds, although its magnitude varies.

This is a controlled validation of explanation correctness—not proof that
the same saliency map is causally correct in real litigation. Its practical
value is still substantial: the explanation infrastructure exposed a semantic
bug in the generator that predictive scores had silently tolerated.

### E6: hidden institutions make timing useful—but attention uses it best

When static covariates are removed and every model is retrained, performance
compresses into a narrower range. Native temporal Transformer leads at 0.902
AUC. Full-feature XGBoost reaches 0.894, while its no-time version falls to
0.847: a 0.047 gap that directly demonstrates unique timing signal.

Timing therefore matters when the institutional causes of timing are hidden.
But native CfC does not take the lead. Its AUC remains about 0.872 with or
without statics, suggesting that it failed to exploit the visible covariates
effectively rather than proving special robustness.

The central result is conditional:

> **Temporal information becomes valuable when institutional state is latent,
> but native continuous-time dynamics are not automatically the best way to
> extract it. In this controlled setting, attention reconstructs hidden state
> more effectively.**

## What the results support—and what they do not

They support:

- a reproducible benchmark for irregular legal-event trajectories;
- a rigorous negative result for liquid dynamics when timing is redundant;
- evidence that timing becomes informative when institutional drivers are
  hidden;
- attention as the strongest tested neural mechanism in that hidden-latent
  condition;
- compact, structurally sparse CfC models with useful controlled
  interpretability results;
- counterfactual evaluation as a promising product and research interface.

They do not support:

- predictive superiority of CfC;
- universal “laws of litigation”;
- causal claims about real judges, parties, or courts;
- the assumption that irregular timestamps alone require liquid networks;
- deployment for legal or capital-allocation decisions without real-data
  validation, calibration, subgroup analysis, and human oversight.

The CfC model has 15.9k parameters versus 22.4k for the Transformer and 24.3k
for the LSTM. Its NCP wiring uses substantially fewer recurrent connections
than a dense alternative; this is structural sparsity, not post-training
pruning. XGBoost complexity must be reported separately through tree count,
depth, serialized size, and inference cost rather than omitted from efficiency
comparisons.

## Where this goes

The next empirical test is real docket data, where institutional conditions
are incomplete, timestamps are noisy for reasons the generator cannot fully
capture, and observation itself is selective. CourtListener/RECAP is a
natural starting point, but credible evaluation requires outcome definitions,
leakage controls, temporal splits, jurisdictional transfer tests, missingness
analysis, calibration, and careful treatment of docket text.

The synthetic program has since narrowed the hypothesis space under frozen,
preregistered criteria: the Stage-1 attention-plus-continuous-state hybrid
failed its screen (killed); forensics showed its continuous clock behaved as
generic residual capacity, not a temporal mechanism (topology retired); and
an interval-supervised marked point-process objective also failed against the
strong temporal-Transformer baseline (killed). Remaining candidate families —
semi-Markov duration models, selective state-space models, time-integrated
attention — each require their own preregistration, and each must beat the
Transformer and conventional baselines to justify its complexity. Every
negative result is preserved with specifications, reports, and
reproducibility code in the package's `experiments/` line; the complete
research archive (checkpoints, raw outputs, hash manifests) lives with the
source repository.

The broader program includes hierarchical state at the matter, judge, court,
jurisdiction, and system levels; changes in doctrine and capacity over calendar
time; competing outcome hazards; strategic interventions; and feedback when
forecasts alter behavior.

Most legal AI reads what happened. This program asks whether we can learn the
empirical laws of motion governing what happens next—and determine, rather
than assume, which models are capable of doing so.

---

## Sources and further reading

- Lechner, Hasani, Amini, Henzinger, Rus, Grosu. *Neural Circuit Policies
  Enabling Auditable Autonomy.* Nature Machine Intelligence, 2020.
- Hasani, Lechner, Amini, Rus, Grosu. *Liquid Time-constant Networks.* AAAI,
  2021.
- Hasani, Lechner, Amini, et al. *Closed-form Continuous-time Neural
  Networks.* Nature Machine Intelligence, 2022.
- `ncps` reference implementation: <https://github.com/mlech26l/ncps>
- Complete experimental results: [`../experiments/RESULTS.md`](../experiments/RESULTS.md)
- Raw result files: [`../experiments/results/`](../experiments/results/)
