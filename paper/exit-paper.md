# When Does Continuous Time Help? A Preregistered Evaluation of Models for Irregular Legal-Event Streams

*Exit paper for the liquid-legal / IDN program, written so a successor can
improve on it. Every number below is backed by a frozen artifact; every
artifact is hashed; every hash is in this repository. Nothing in this paper
is a claim about real litigation outcomes — all evidence is synthetic,
and the limits of that evidence are stated in section 10.*

**Status of the program:** complete for this iteration. The candidate hybrid
(IDN Stage 1) was killed by its own preregistered criterion on 2026-07-23.
This document is simultaneously a methods paper, a negative-result report,
and a handoff brief.

---

## 1. Abstract

Legal dockets are irregular, regime-shifting event streams — a natural
habitat, in principle, for continuous-time ("liquid") neural networks and
hybrid state-space architectures. We built a complete, preregistered
evaluation program to test that intuition: two synthetic litigation worlds
with known ground truth and logged latents, a frozen training/evaluation
protocol with equal-supervision controls, and paired statistical comparisons
against XGBoost, LSTM, causal Transformer, and CfC baselines. We report two
negative results and one positive. (i) On Generator v1, liquid continuous-time
models do not beat an LSTM, and no time-aware architecture beats engineered
features, because the world over-discloses its latents (E1–E6). (ii) On the
harder Generator v2 — where timing carries unique outcome information by an
architecture-independent acceptance probe — a chronology-correct hybrid (causal attention + gated
continuous-time flow + event jumps) fails its preregistered primary screen
against an auxiliary-matched Transformer: mean paired ΔAUC −0.0039, 95% CI
[−0.0082, +0.0004], ruling out the required +0.01 improvement. (iii) The positive
result is auditability: saliency recovered several planted mechanisms under
controlled synthetic conditions, with stable directional alignment but
variable magnitude across seeds, and explanation-driven debugging caught a
semantic generator bug that every predictive benchmark had missed. The
program's main contribution is methodological: a template for killing
architecture claims quickly and honestly, and — in section 8 — a
prioritized diagnostic map of where the next candidate should look.

## 2. The problem and the bet

A case is not a document; it is a process. Filings, rulings, discovery
events, and settlement talks arrive at uneven intervals; a matter can drift
for months and then accelerate without warning. Litigation funders make
mid-course decisions from exactly this stream: settlement probability,
expected recovery, remaining duration — re-estimated at every docket event.

Three intuitions motivated the program:

1. **Irregular sampling is native territory for continuous-time models.**
   LTC/CfC cells take elapsed time as an argument of the dynamics rather
   than as a bolted-on feature.
2. **Volatility is institutional.** Judge, district, and plaintiff
   characteristics drive non-linear timing effects; discovery stalls
   disproportionately harm under-equipped plaintiffs.
3. **Regulated capital needs auditability.** Sparse Neural Circuit Policy
   wirings offer a model small enough to inspect.

The bet was that these three would compound into a predictive advantage.
They did not — at least not in any world we built — and this paper is the
record of finding that out cleanly.

## 3. Related work

The field relevant to this program is best organized by three questions, which
also structure each subsection below: **(i) what temporal assumption does the
model make** (discrete ticks, exponential decay, arbitrary continuous
dynamics, or an explicit event intensity); **(ii) how does it consume
irregular time** (as a feature, as an argument of the dynamics, or as the
likelihood itself); and **(iii) does it model event occurrence jointly with
event content** — because the successor program (section 9, F2) is a move
toward objectives that do. Each family is additionally compared against the
killed Stage-1 candidate — a causal-Transformer context encoder feeding a
gated continuous-time clock partition (zₖ = (1 − gₖαₖ)zₖ₋₁ + gₖαₖT(cₖ₋₁)),
a GRU event partition, and a blended context partition, with endpoint-only
supervision — and against the question that actually killed it: whether a
small post-attention continuous-time adapter earns parameters that attention
could have spent on itself.

### 3.1 Neural Circuit Policies (NCP)

Neural Circuit Policies (Lechner et al., 2020) couple liquid time-constant
neurons with sparse wirings distilled from C. elegans connectome structure,
motivated by auditability: a 19-neuron policy for end-to-end driving that a
human can, in principle, inspect. **(1) Temporal assumption:** continuous-time
ODE dynamics (a leaky-integrator with synaptic gating), inherited from the LTC
neuron model. **(2) Irregular time:** time enters as the ODE independent
variable; an ODE solver handles arbitrary step sizes, so elapsed time is
native rather than featurized. **(3) Occurrence vs. content:** neither — NCP
is a control policy over continuous observations, with no notion of discrete
event occurrence or marks. **(4) Identifiability:** the wiring is auditable by
design, but the neuron-level dynamics are not identifiable in any statistical
sense; interpretability is structural, not inferential. **(5) vs. Stage 1:**
NCP contributed this program's third motivating intuition (sparse,
inspectable wiring) and its CfC baseline lineage, but it is an architecture
philosophy, not a trajectory model; Stage 1's auditability came from small
partitioned state and saliency checks (F6), not from connectome-inspired
topology. **(6) Benchmark matrix:** no — its time semantics are subsumed by
the LTC/CfC entries below, which are already represented; the auditability
lesson is carried by the program's explanation-correctness checks, not by
adding another architecture.

### 3.2 Liquid Time-Constant Networks (LTC)

Liquid Time-Constant Networks (Hasani et al., 2021) replace the RNN's discrete
recurrence with a continuous-time neuron whose effective time constant is
itself a learned function of state and input, integrated by a numerical ODE
solver. **(1) Temporal assumption:** the hidden state obeys an ODE with
input-dependent time constants — the "how fast" of forgetting is modulated
online. **(2) Irregular time:** natively, through solver step size; this was
the program's first motivating intuition. **(3) Occurrence vs. content:**
content only; no event-intensity term, so silence between observations
carries no likelihood. **(4) Identifiability:** weak. The time constant and
the target computation are entangled in the same ODE right-hand side, so
comparable trajectories admit many (τ, f) decompositions — the same disease
this program later diagnosed as Stage 1's gate/rate collapse (H2), where only
the effective coefficient βₖ = gₖαₖ was ever identified.
**(5) vs. Stage 1:** LTC is a *substrate* — a recurrent cell with liquid
dynamics — whereas Stage 1 was a *post-attention adapter*: a small liquid
flow bolted onto an already-optimized Transformer context. The v1 experiments
(E1–E6) showed the substrate alone does not beat an LSTM when statics are
visible; Stage 1 then showed the adapter form does not beat its attention
host either. **(6) Benchmark matrix:** no (represented) — the solver-based
cell is dominated in practice by its closed-form successor, which already
serves as the program's liquid control.

### 3.3 Closed-form Continuous-time Networks (CfC)

Closed-form Continuous-time Networks (Hasani et al., 2022) approximate the LTC
ODE in closed form, replacing solver iterations with a gated exponential
interpolation between decay-to-baseline and decay-to-target terms. **(1)
Temporal assumption:** per-interval exponential relaxation with
input-conditioned rates — exactly the family Stage 1's clock partition
belongs to. **(2) Irregular time:** elapsed time is an argument of the
closed-form update, so irregular gaps are consumed without resampling.
**(3) Occurrence vs. content:** content only; the interval between events is
modeled dynamically but never supervised. **(4) Identifiability:** the update
exposes rates and gates separately, but — as this program's H2 analysis made
concrete — a gated product of rate and acceptance is identifiable only in
the product; CfC does not escape that. **(5) vs. Stage 1:** CfC is the
nearest architectural ancestor and the direct baseline in both experiment
waves. Stage 1 differed in *placement* (adapter after causal attention,
consuming the Transformer's compression cₖ₋₁ rather than raw inputs) and in
*partitioning* (structurally separate event/clock/context state). Both
modifications failed to move the auxiliary-matched Transformer (mean paired
ΔAUC −0.0039), and CfC itself trailed the field on Generator v2 (0.790 ±
0.018), suggesting the limitation lies in the endpoint-supervised
exponential-relaxation family, not in any one cell. **(6) Benchmark matrix:**
yes — it remains the liquid control in the frozen harness, and F1's
retirement criterion is stated against this exact topology.

### 3.4 GRU-D

GRU-D (Che et al., 2018) was the reference point for irregular multivariate
clinical time series: missingness and elapsed time are injected into a GRU
through trainable decays of both inputs and hidden state toward empirical
means, plus explicit mask and time-lag inputs. **(1) Temporal assumption:**
discrete event sequence with exponentially decaying evidence — decay is a
heuristic applied to observations, not a dynamical law. **(2) Irregular
time:** via per-feature time-lag vectors multiplying learned decay rates;
simple, robust, and notoriously strong as a baseline. **(3) Occurrence vs.
content:** content only; the mask encodes *that* something was observed, not
a model of *when* the next event will occur. **(4) Identifiability:** decay
rates are per-feature regression coefficients, loosely interpretable but not
tied to any generative semantics. **(5) vs. Stage 1:** GRU-D puts its decay
*inside* the recurrence and drives it with raw time gaps; Stage 1 put its
decay *after* attention and drove it with context. The v2 result — and the
H1 redundancy hypothesis — suggests the former placement at least guarantees
the decay sees distinct evidence, while the latter risks recomputing what
attention already encoded. **(6) Benchmark matrix:** yes — it is the cheapest
strong discrete-time baseline the next matrix lacks, and its missingness-mask
machinery maps naturally onto Generator v2's selective observation channel.

### 3.5 Neural ODEs

Neural ODEs (Chen et al., 2018) define the hidden state as the solution of a
learned vector field, trained by adjoint sensitivity — continuous depth and,
incidentally, a continuous-time primitive. **(1) Temporal assumption:**
deterministic smooth flow; the state evolves everywhere, including through
silence. **(2) Irregular time:** evaluation times are arbitrary solver
queries — the cleanest possible answer to irregular sampling, at the price
of assuming a single autonomous-ish vector field. **(3) Occurrence vs.
content:** neither natively; events must be injected externally (which is
what ODE-RNN does). **(4) Identifiability:** poor in general — vector fields
are identifiable only up to trajectory-equivalent reparameterizations, and
the program's H3 hypothesis (a restrictive flow shape silently constraining
what can be expressed) applies here with full force. **(5) vs. Stage 1:**
Stage 1's clock was a hand-picked first-order flow (monotone relaxation to a
context target), deliberately weaker than a learned vector field but cheaper
and chronology-testable; its failure modes H2/H3 are the specific instances
of the general Neural-ODE identifiability problem. **(6) Benchmark matrix:**
no (subsumed) — as a standalone it is superseded for this task by its
event-aware descendants below; as a design space it returns only via F3
(SDE variants, multi-phase flows) after interval supervision is shown to
matter.

### 3.6 ODE-RNN and Latent ODE

Rubanova et al. (2019) give the canonical event-aware continuous-time
recurrence: an ODE defines state evolution *between* observations and an RNN
cell applies jumps *at* observations; Latent ODE wraps this in a
recognition-encoder/generative-decoder VAE. **(1) Temporal assumption:**
piecewise — continuous drift between events, discrete jumps at events. This
is precisely the two-mechanism structure Stage 1 re-implemented with a
closed-form drift (gated exponential relaxation) and a GRU jump. **(2)
Irregular time:** native; timestamps parameterize the drift segments.
**(3) Occurrence vs. content:** content (observation reconstruction) only;
event times are conditioned on, not modeled. **(4) Identifiability:** the
latent trajectory is weakly identified — the drift/jump decomposition is
constrained by reconstruction but not by any interval likelihood, so
equivalent (drift, jump) pairs can fit the same endpoints. Stage 1's H4 is
this observation stated for supervised heads: a continuous-time mechanism
trained only at event boundaries is a weakly identified one. **(5) vs.
Stage 1:** near-isomorphic in state structure; the differences are placement
(Stage 1 drifts on attention context rather than raw inputs) and supervision
(Stage 1 added marked-event auxiliary heads — a step toward occurrence
modeling — but still fired every loss at endpoints). The paired screen says
these differences were worth −0.0039 AUC against matched attention.
**(6) Benchmark matrix:** yes — ODE-RNN is the canonical continuous-time RNN
baseline the F2 matrix should include, both as a competitor and as the drift
half of any honest ablation of interval supervision.

### 3.7 Neural Controlled Differential Equations

Neural CDEs (Kidger et al., 2020) drive a learned vector field with the
natural cubic spline of the observations: the data itself becomes the
control path, making the model a continuous-time function of the whole
observed trajectory. **(1) Temporal assumption:** rough-path/driven-system
semantics — state responds continuously to the interpolation of everything
seen so far. **(2) Irregular time:** native and elegant; irregularity and
even missing channels are handled by the spline, and irregular sampling is
where NCDEs report their largest gains. **(3) Occurrence vs. content:**
content only, like the rest of the ODE lineage. **(4) Identifiability:**
limited; the vector field is regularized by the control structure but not
identified, and the spline is a modeling choice (it smooths away exactly the
jump discontinuities that event models treat as information). **(5) vs.
Stage 1:** NCDE consumes the entire interpolated history as its driver;
Stage 1 consumed a causal attention summary and enforced — by executable
test, not by assumption — that contexts are pre-event and the flow never
sees its own current event. NCDE makes no such chronology guarantee
(its spline is defined over the observed prefix but is a smoother, not a
causal filter), which matters for this program's mid-course prediction
setting. **(6) Benchmark matrix:** yes — the strongest continuous-time
sequence encoder in the literature and the most serious omission from the
current baseline set; it belongs in the F2 matrix.

### 3.8 Marked temporal point processes: RMTPP and the Neural Hawkes Process

This family is the most consequential for the successor program, because F2
proposes to rebuild the training objective around it. A marked temporal point
process specifies a conditional intensity λₘ(t | Hₜ) per mark m: the
likelihood of an event sequence factorizes into per-event intensities and a
survival term S(t) = exp(−∫λ du) over every interval of silence. RMTPP (Du
et al., 2016) first parameterized this with an RNN whose hidden state feeds
an exponential intensity, jointly predicting next-event time and type. The
Neural Hawkes Process (Mei and Eisner, 2017) replaced the exponential
self-excitation of classical Hawkes processes with a continuous-time LSTM
whose state decays toward event-dependent baselines between events, allowing
excitation *and inhibition* and non-monotone intensity evolution.
**(1) Temporal assumption:** events are generated by a stochastic intensity;
time between events is the random variable, not a feature. **(2) Irregular
time:** it *is* the model — intervals enter through the survival integral,
so every day of silence is training signal. This is the formal version of
this program's H4 diagnosis: Stage 1's continuous-time clock was rewarded
only at event boundaries, and a TPP objective supervises the trajectory
through silence exactly where Stage 1 was unsupervised. **(3) Occurrence vs.
content:** jointly, by construction — marks and times share one likelihood.
Stage 1's marked-event auxiliaries (next type, next-time quantiles) were a
half-step in this direction, but quantile regression on endpoints does not
impose the survival consistency that makes the intensity identifiable.
**(4) Identifiability:** better than any family above — the intensity is
constrained by both the events that happened and the stretches where nothing
did — though the *latent decompositions* inside the neural parameterization
remain unidentifiable, and Neural Hawkes intensities are notoriously
sensitive to numerical integration of the survival term. **(5) vs. Stage 1:**
Stage 1 had the right auxiliaries and the wrong objective; the TPP families
have the right objective and (in their 2016–2017 form) weaker sequence
encoders than modern attention. F2 is precisely the synthesis: a
TPP-augmented Transformer, killed or crowned on the frozen screen.
**(6) Benchmark matrix:** yes — central. RMTPP and Neural Hawkes are the
objective's reference implementations and belong in the F2 comparison even
though their encoders are dated.

### 3.9 Transformer Hawkes processes

Zuo et al. (2020) replaced the Neural Hawkes LSTM with self-attention over
the event history, using time-shift positional encodings to make attention
time-aware; Zhang et al. (2020) independently proposed the Self-Attentive
Hawkes Process with aligned event embeddings. (A note on names: the
literature is sometimes miscited here — the two verified 2020 ICML papers
are Zuo et al. and Zhang, Lipani, Kirnap, and Yilmaz.) **(1) Temporal
assumption:** stochastic intensity as in §3.8, with history dependence
modeled by attention rather than recurrence — long-range excitation without
a decaying bottleneck. **(2) Irregular time:** through time-shifted
positional encodings inside attention and through the survival integral in
the loss; events arrive at arbitrary times. **(3) Occurrence vs. content:**
joint, via per-type intensities — the full marked-TPP likelihood.
**(4) Identifiability:** same favorable likelihood-level identification as
any TPP; internally, attention weights over event history are no more
identifiable than any other attention map, which tempers interpretability
claims but not predictive ones. **(5) vs. Stage 1:** this is the sharpest
contrast available. Stage 1 *was* attention plus continuous-time machinery
plus marked-event heads — but with endpoint losses and a hand-built
exponential clock instead of a learned intensity. Transformer Hawkes keeps
the attention, discards the bespoke clock, and gets its continuous-time
semantics from the likelihood rather than from the architecture. The Stage-1
negative result (ΔAUC −0.0039, CI excluding +0.01) is direct evidence that
the architectural route did not pay; F2 tests whether the likelihood route
does, on the same frozen world, against the same auxiliary-matched opponent.
**(6) Benchmark matrix:** yes — this is F2's stated first candidate and its
natural baseline simultaneously; the kill criterion (TPP-augmented
Transformer must beat tf-native-aux by the preregistered margin) is written
against exactly this family.

### 3.10 Semi-Markov models

Classical semi-Markov processes (Limnios and Oprişan, 2001) generalize
Markov jump processes by letting sojourn-time distributions be arbitrary
rather than exponential — the transition mechanism depends on time-since-entry
into the current regime. **(1) Temporal assumption:** duration-dependent
regime dynamics; the process is Markovian in (state, age), not in state
alone. **(2) Irregular time:** natively — sojourn distributions are defined
on continuous time. **(3) Occurrence vs. content:** occurrence (regime
durations and transitions) is the object; arbitrary marks require extension.
**(4) Identifiability:** the best of any family here — sojourn distributions
and the embedded jump chain are directly estimable, which is exactly the
property the neural families traded away for expressiveness. **(5) vs.
Stage 1:** Generator v2's judge-backlog episodes and hidden adverse regimes
are semi-Markov in spirit (regimes with dwell-time structure and
regime-dependent dynamics), and Stage 1's fixed-shape exponential clock
(H3) could not express duration-dependent hazard — a deadline-activated or
rising-then-falling pressure profile is a semi-Markov object, not a
first-order relaxation. **(6) Benchmark matrix:** deferred — the concept is
the right prior if F2's exponential-family intervals prove misspecified, but
honestly assessed neural semi-Markov variants are sparse: no neural
semi-Markov event-sequence model with a verifiable citation and mainstream
uptake was found in preparing this section, so none is cited. If F2 fails on
interval-shape grounds, the repair is more likely a nonparametric
sojourn/hazard head than an off-the-shelf neural semi-Markov model.

### 3.11 Competing-risk survival models

Fine and Gray (1999) model the subdistribution hazard of a cumulative
incidence function, giving direct regression on event probabilities when
multiple terminal causes compete; DeepHit (Lee et al., 2018) learned the
joint discrete-time distribution over (cause, time) with a shared neural
encoder and cause-specific softmax heads, trained under right-censoring.
**(1) Temporal assumption:** survival time is the outcome; causes are
mutually exclusive absorbing states. **(2) Irregular time:** Fine–Gray is
continuous-time but observation-static; DeepHit discretizes the horizon —
neither consumes an *event stream*, both consume a covariate snapshot.
**(3) Occurrence vs. content:** terminal occurrence and cause jointly, but
with no model of the intermediate event process that leads there. **(4)
Identifiability:** cause-specific quantities are identified under standard
censoring assumptions; the (well-known) caveat is that joint-cause
dependence is not identifiable from competing-risks data at all.
**(5) vs. Stage 1:** Stage 1 predicted settlement-within-180-days at every
prefix — a repeatedly re-estimated terminal-risk query over a streaming
history, which neither snapshot model does; but the *evaluation* of that
query (calibration, cause structure, horizon-conditional risk) is borrowed
survival machinery, and this program's ECE guards are a weak form of it.
**(6) Benchmark matrix:** yes, as evaluation machinery — the F2 matrix
should report cumulative-incidence-style calibration for its terminal-risk
heads; DeepHit itself is a reasonable additional baseline only if the
benchmark admits a snapshot-mode comparison, which the streaming protocol
makes awkward.

### 3.12 Mamba and selective state-space models

S4 (Gu et al., 2022) structured the SSM transition matrix (HiPPO
initialization, diagonal-plus-low-rank structure) to make long-range
sequence modeling trainable; Mamba (Gu and Dao, 2023) made the SSM
parameters input-dependent ("selective"), recovering attention-like content
addressing at linear cost. **(1) Temporal assumption:** a continuous-time
linear SSM *discretized at a fixed step* — the continuous-time object is a
design device, not the deployed semantics. **(2) Irregular time:** poorly,
and this is the honest crux. S4 assumes a uniform Δt baked into its
discretization; Mamba's input-dependent Δ modulates effective step size per
token, which is a soft form of time-awareness but consumes no timestamps —
wall-clock gaps between docket events would have to be featurized, recreating
the bolted-on time this program set out to avoid. **(3) Occurrence vs.
content:** content only. **(4) Identifiability:** the SSM state is a
compression device; individual state coordinates carry no estimable meaning.
**(5) vs. Stage 1:** Mamba solves the *efficiency* half of persistent state
(linear-time long context), which this program's H7 judged irrelevant at
docket sequence lengths — causal attention already revisits the whole
history at trivial cost, so state compression has no efficiency niche here,
and the statistical niche must come from time semantics Mamba does not have.
**(6) Benchmark matrix:** deferred — a Mamba baseline is cheap to add via
the frozen harness and would close the "was it just any recurrence?"
loophole, but nothing in H1–H8 predicts it will beat matched attention on
short irregular sequences; it is a completeness entry, not a hypothesis.

### 3.13 Physically time-aware SSM variants

The natural desideratum — an SSM whose discretization step is the actual
elapsed time, making Δt a first-class argument the way CfC's closed form
does — is, to our knowledge, an open gap rather than an established family.
The closest verified relatives are Liquid Structural State-Space Models
(Hasani et al., 2022b), which make the SSM transition *input-dependent* in
the liquid spirit but still operate on discrete steps, and Mamba's selective
Δ (§3.12), which is content-dependent rather than clock-dependent. A careful
search for SSM variants that consume arbitrary wall-clock intervals as
dynamical arguments (in the way ODE-RNN or CfC do) did not surface a
peer-reviewed, widely adopted model worth citing; preprints exist in
adjacent spaces but none met this paper's verification bar. **(1)–(4):** not
applicable to a family we decline to characterize from unverified sources.
**(5) vs. Stage 1:** the gap is itself informative — Stage 1's clock
partition was, in effect, a hand-built instance of a physically time-aware
state update, and its failure is one data point about the difficulty of the
whole design space, not just of one cell. **(6) Benchmark matrix:** no —
nothing established to benchmark; flagged here so a successor can revisit if
the literature fills in.

### 3.14 Continuous-time Transformers: ContiFormer

ContiFormer (Chen et al., 2024) interleaves attention with Neural-ODE
evolution, extending the query–key–value relation modeling of the
Transformer into continuous time so that representations evolve between
observations under learned dynamics. **(1) Temporal assumption:** continuous
latent trajectory with attention-defined interactions — the ODE lineage's
drift married to the Transformer lineage's relational model. **(2) Irregular
time:** native, via the ODE segments between event times. **(3) Occurrence
vs. content:** the published model handles next-event type and time
prediction among its tasks, but as prediction heads over the continuous
representation, not as an intensity-based likelihood — it sits between the
ODE family and the TPP family on this axis. **(4) Identifiability:** that of
its parents — weak at the latent level, with the added caveat that
interleaving solvers with attention multiplies the ways equivalent
trajectories can be produced. **(5) vs. Stage 1:** ContiFormer is the
closest *published* relative of the killed candidate: both put continuous
dynamics adjacent to attention for irregular streams. The architectural
difference is placement and provenance — ContiFormer weaves ODE drift
through the attention stack itself, while Stage 1 attached a closed-form
clock downstream of a frozen-position causal encoder — and Stage 1's H1
hypothesis (the adapter saw only attention's own compression) does not apply
to ContiFormer's interleaved design. That makes ContiFormer the single most
informative external sanity check on whether the post-attention placement,
rather than continuous-time machinery per se, was Stage 1's mistake.
**(6) Benchmark matrix:** yes — it is the direct test of the placement
hypothesis and belongs in the F2 matrix alongside, not instead of, the TPP
objective variants.

### 3.15 Longitudinal legal-outcome prediction and legal NLP

Public legal NLP is overwhelmingly *text* modeling. LexGLUE (Chalkidis et
al., 2022) standardized legal language understanding — classification and
multiple-choice over statutes, cases, and contracts — and the ECtHR line
(Aletras et al., 2016; Chalkidis et al., 2019) predicts judgments from
case *descriptions*, i.e., from documents at a decision point, not from the
event stream that produced them. **(1) Temporal assumption:** none — the
document is the unit; chronology inside a matter is typically discarded.
**(2) Irregular time:** not consumed. **(3) Occurrence vs. content:**
content only (outcome labels), occurrence never. **(4) Identifiability:**
not a latent-trajectory literature. **(5) vs. Stage 1:** complementary, not
competitive — text models *produce* events and covariates; trajectory models
like Stage 1 and its successors *consume* them. Meanwhile the models that do
consume litigation trajectories — litigation funders' and insurers'
underwriting models — are proprietary, unpublished, and unbenchmarked; no
public benchmark of case-trajectory prediction exists. That absence is half
the justification for this program building its own synthetic worlds, and it
caps every external-validity claim until the CourtListener/RECAP adapter
(F5) exists. **(6) Benchmark matrix:** no — different task; the family's
role in this program is as the upstream event extractor for F5 and as the
reminder that "legal AI" evidence almost never concerns trajectories.

### 3.16 Calibration under censoring and competing risks

Haider et al. (2020) systematized the evaluation of individual survival
distributions — concordance, integrated Brier score, and, most relevantly,
distribution calibration (D-calibration): whether a model's predicted
distributions are *meaningful*, not merely rank-ordered. **(1) Temporal
assumption:** survival-time outcomes under right-censoring. **(2) Irregular
time:** censoring times are arbitrary, but covariates are snapshots, as in
§3.11. **(3) Occurrence vs. content:** occurrence of the terminal event.
**(4) Identifiability:** this family's contribution is epistemic hygiene —
it names what survival predictions can and cannot support, which is the
discipline this program's frozen protocol tries to enforce for streaming
prediction. **(5) vs. Stage 1:** Stage 1's regression guards (duration MAE,
ECE) are scalar shadows of distribution calibration; the marked-event
quantile heads were exactly the kind of distributional output D-calibration
exists to police, and they were checked only coarsely. **(6) Benchmark
matrix:** not a model family — but its metrics belong in the F2 evaluation
harness: an interval-supervised TPP objective makes distributional claims
(survival curves, quantiles) that ECE alone cannot validate, and
D-calibration (or a streaming analogue) is the standard a successor should
be held to.

### 3.17 Counterfactual prediction from longitudinal observational data

Robins' marginal structural models (Robins, Hernán, and Brumback, 2000)
established the classical machinery for estimating treatment effects from
longitudinal observational data under time-varying confounding
(inverse-probability weighting of treatment and censoring); Bica et al.
(2020) brought the problem into representation learning, building
adversarially balanced recurrent representations (the Counterfactual
Recurrent Network) to estimate treatment-outcome trajectories over time.
**(1) Temporal assumption:** discrete clinical time with time-varying
treatment and confounders; dynamics serve causal identification, not
forecasting. **(2) Irregular time:** typically regularized away — visit
grids are assumed; irregular observation is a missing-data problem, not a
dynamics problem. **(3) Occurrence vs. content:** content (outcomes under
interventions); occurrence is ancillary. **(4) Identifiability:** the
family's entire point — but identification is *assumption-driven*
(exchangeability, positivity, consistency), achieved by balancing and
weighting, not by architecture. No neural architecture confers it.
**(5) vs. Stage 1:** this program's counterfactual judge probes (E5; the
paired interventional acceptance checks A6) are interventional by
construction — the generator *is* the intervention — so they sidestep
observational identification entirely. The moment the program touches real
dockets (F5), that protection vanishes: "what if this judge were faster?"
becomes a marginal-structural-model question, and saliency-recovered
mechanisms (the program's one positive result) become hypotheses requiring
exactly this family's machinery to test. **(6) Benchmark matrix:** deferred
— irrelevant to the synthetic F2 screen (ground truth is logged), essential
to the F5 protocol; cited here so the successor does not mistake synthetic
interventional access for a property of the world.

### 3.18 Synthesis

Read against the killed candidate, the literature sorts into three verdicts.
First, the *architectural* route to continuous time — liquid cells (§3.2–3.3),
ODE hybrids (§3.5–3.7, 3.14), and Stage 1's own post-attention clock — has
produced no verified win in this program's worlds, and the diagnostic map
(H1, H2, H4) explains why: adapters on attention's output see no new
evidence, gated rates are unidentifiable, and endpoint-only losses leave
intervals unsupervised. Second, the *likelihood* route — marked temporal
point processes (§3.8–3.9) — is the one family that supervises exactly the
quantity Stage 1 left unsupervised, and it is therefore the successor's
stated next step (F2), with Transformer Hawkes as template and kill
criterion both. Third, the surrounding disciplines — competing risks
(§3.11), survival calibration (§3.16), and longitudinal causal inference
(§3.17) — contribute not architectures but the evaluation and identification
standards without which any future positive result on real dockets would be
uninterpretable. The legal-NLP literature (§3.15) stands apart as the
upstream producer of the events all of these models consume, and the
proprietary status of actual litigation-trajectory modeling remains the
program's largest external-validity debt.

### 3.19 Bibliography

All identifiers verified 2026-07-25 against Crossref, the ACL Anthology,
JMLR, or arXiv.

- Aletras, N., Tsarapatsanis, D., Preoţiuc-Pietro, D., and Lampos, V. (2016).
  Predicting judicial decisions of the European Court of Human Rights: a
  Natural Language Processing perspective. *PeerJ Computer Science* 2:e93.
  DOI: 10.7717/peerj-cs.93
- Bica, I., Alaa, A. M., Jordon, J., and van der Schaar, M. (2020).
  Estimating counterfactual treatment outcomes over time through
  adversarially balanced representations. *ICLR 2020*. arXiv: 2002.04083
- Chalkidis, I., Androutsopoulos, I., and Aletras, N. (2019). Neural legal
  judgment prediction in English. *ACL 2019*. DOI: 10.18653/v1/P19-1424;
  arXiv: 1906.02059
- Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I.,
  Katz, D. M., and Aletras, N. (2022). LexGLUE: A benchmark dataset for
  legal language understanding in English. *ACL 2022*.
  DOI: 10.18653/v1/2022.acl-long.297; arXiv: 2110.00976
- Che, Z., Purushotham, S., Cho, K., Sontag, D., and Liu, Y. (2018).
  Recurrent neural networks for multivariate time series with missing
  values. *Scientific Reports* 8:6085. DOI: 10.1038/s41598-018-24271-9;
  arXiv: 1606.01865
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. (2018).
  Neural ordinary differential equations. *NeurIPS 2018*. arXiv: 1806.07366
- Chen, Y., Ren, K., Wang, Y., Fang, Y., Sun, W., and Li, D. (2024).
  ContiFormer: Continuous-time transformer for irregular time series
  modeling. *NeurIPS 2023*. arXiv: 2402.10635
- Du, N., Dai, H., Trivedi, R., Upadhyay, U., Gomez-Rodriguez, M., and
  Song, L. (2016). Recurrent marked temporal point processes: Embedding
  event history to vector. *KDD 2016*. DOI: 10.1145/2939672.2939875
- Fine, J. P. and Gray, R. J. (1999). A proportional hazards model for the
  subdistribution of a competing risk. *Journal of the American Statistical
  Association* 94(446):496–509. DOI: 10.1080/01621459.1999.10474144
- Gu, A., Goel, K., and Ré, C. (2022). Efficiently modeling long sequences
  with structured state spaces. *ICLR 2022*. arXiv: 2111.00396
- Gu, A. and Dao, T. (2023). Mamba: Linear-time sequence modeling with
  selective state spaces. arXiv: 2312.00752
- Haider, H., Hoehn, B., Davis, S., and Greiner, R. (2020). Effective ways
  to build and evaluate individual survival distributions. *Journal of
  Machine Learning Research* 21(85):1–63.
  https://jmlr.org/papers/v21/18-772.html
- Hasani, R., Lechner, M., Amini, A., Rus, D., and Grosu, R. (2021). Liquid
  time-constant networks. *AAAI 2021*. DOI: 10.1609/aaai.v35i9.16936;
  arXiv: 2006.04439
- Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A.,
  Tschaikowski, M., Teschl, G., and Rus, D. (2022). Closed-form
  continuous-time neural networks. *Nature Machine Intelligence* 4:992–1003.
  DOI: 10.1038/s42256-022-00556-7; arXiv: 2106.13898
- Hasani, R., Lechner, M., Wang, T.-H., Chahine, M., Amini, A., and Rus, D.
  (2023). Liquid structural state-space models. *ICLR 2023*.
  arXiv: 2209.12951
- Kidger, P., Morrill, J., Foster, J., and Lyons, T. (2020). Neural
  controlled differential equations for irregular time series.
  *NeurIPS 2020*. arXiv: 2005.08926
- Lechner, M., Hasani, R., Amini, A., Henzinger, T. A., Rus, D., and
  Grosu, R. (2020). Neural circuit policies enabling auditable autonomy.
  *Nature Machine Intelligence* 2:642–652. DOI: 10.1038/s42256-020-00237-3
- Lee, C., Zame, W., Yoon, J., and van der Schaar, M. (2018). DeepHit: A
  deep learning approach to survival analysis with competing risks.
  *AAAI 2018*. DOI: 10.1609/aaai.v32i1.11842
- Limnios, N. and Oprişan, G. (2001). *Semi-Markov Processes and
  Reliability*. Birkhäuser. DOI: 10.1007/978-1-4612-0161-8
- Mei, H. and Eisner, J. (2017). The neural Hawkes process: A neurally
  self-modulating multivariate point process. *NeurIPS 2017*.
  arXiv: 1612.09328
- Robins, J. M., Hernán, M. Á., and Brumback, B. (2000). Marginal structural
  models and causal inference in epidemiology. *Epidemiology* 11(5):550–560.
  DOI: 10.1097/00001648-200009000-00011
- Rubanova, Y., Chen, R. T. Q., and Duvenaud, D. (2019). Latent ordinary
  differential equations for irregularly-sampled time series.
  *NeurIPS 2019*. arXiv: 1907.03907
- Zhang, Q., Lipani, A., Kirnap, O., and Yilmaz, E. (2020). Self-attentive
  Hawkes processes. *ICML 2020*. arXiv: 1907.07561
- Zuo, S., Jiang, H., Li, Z., Zhao, T., and Zha, H. (2020). Transformer
  Hawkes process. *ICML 2020*. arXiv: 2002.09291

## 4. Formulation and worlds

**Task.** Given a prefix of a docket — event types, amounts, flags, exact
timestamps, static covariates — predict at every prefix: settlement within
180 days (classification), expected final recovery (log1p regression), and
remaining duration (log1p regression). Marked-event auxiliaries (next-event
type and time, quantiles) were added for Stage 1.

**Generator v1** (Figure 2, `paper/figures/fig_gen_v1_mechanisms.pdf`)
encodes the domain hypothesis: judge speed/erraticness/
defense-tilt and district congestion act multiplicatively; discovery stall
hazard is quadratic in plaintiff incapability; stalls erode leverage
non-linearly; settlement pressure accumulates through procedural events. Its
mechanisms are known by construction (they are programmed), and the
generation variables it logs are observable quantities of the run; it does
not ship the complete latent-state logs that v2 provides.

**Generator v2** (Figure 3, `paper/figures/fig_gen_v2_latents.pdf`)
adds what v1 lacked: time-varying judge backlog episodes
(Markov chain, 2.5× gap multiplier, litigation-fatigue leverage erosion),
hidden adverse case regimes (pressure reset, halved gains, decay), and
selective observation (the docket is an observation of the true process, not
the process itself). Crucially, v2 ships **complete latent logs** — judge
backlog trajectories, case-regime flip annotation, the full true event log,
and the observation mask for every case — providing the latent observability
required for the declared mechanism and intervention checks (sections C and
A6 of the preregistration). The two worlds must not be treated as having
identical latent observability: mechanism checks that require latent ground
truth are v2-only.

**Generator acceptance was architecture-independent** (A1–A6): timing must
carry unique outcome information conditioned on statics and order (probe
lift ≥ 0.02); regimes must alter transition dynamics (gap ratio ≥ 1.3, stall
diff ≥ 0.05); observation must differ from process; latent logs must be
complete; no model family may saturate or fail the task; and planted
mechanisms must be recoverable by **paired interventional probes** — forcing
backlog vs normal (or adverse vs normal) at landmarks with common random
numbers must change next-event time and settlement incidence by declared
margins. Acceptance was failed twice and repaired before passing — including
a state-consistency repair making logged episodes, gap-generating states,
and fatigue attribution agree by construction.

## 5. The evaluation program

The methodology, in invariant order (Figure 4,
`paper/figures/fig_protocol_flowchart.pdf`):

1. **Preregistration before training.** Endpoints, thresholds, kill
   criteria, and analysis plans written first
   (`experiments/PREREGISTRATION.md`).
2. **Frozen world and protocol.** Generator, seeds [0–9], splits, training
   spec, and evaluation code hashed (`experiments/FREEZE.md`).
3. **Fair comparison.** Matched parameter budgets (IDN 22,536 vs
   tf-native-aux 23,129 vs tf-native 22,403), identical training (Adam,
   25 epochs, batch 32), leakage-free config selection (inner split of the
   training portion only), and **equal supervision** — the primary opponent
   carries the identical auxiliary heads and loss weights as the hybrid.
4. **Paired statistics.** One declared primary endpoint (hidden-statics
   settlement AUC); per-seed paired deltas; mean ± 95% paired t-interval;
   minimum practical effect Δ ≥ 0.01; regression guards on duration MAE and
   ECE.
5. **Chronology by test, not by assertion.** Automated tests verify the
   intended chronology and numerical invariants: contexts are causal
   (pre-event only), the flow never sees its own current event, padding
   cannot contaminate state, Δt = 0 is an exact identity, and the flow
   satisfies interval composition. These are executable checks of the
   implementation, not mathematical proofs.
6. **Fail fast, in public.** A kill is executed exactly as written, archived
   with weights, raw predictions, logs, code snapshots, and hashes, and
   labeled a killed candidate.

## 6. Results on Generator v1 (E1–E6)

Full matrix, mean ± std over seeds {0,1,2}, 1024 cases:

| model | params | settle AUC (visible) | duration MAE |
|---|---|---|---|
| XGBoost (no timing) | — | 0.917 ± 0.012 | 144 ± 20 |
| XGBoost | — | 0.915 ± 0.013 | 146 ± 20 |
| LSTM | 24.3k | 0.914 ± 0.015 | 146 ± 19 |
| Transformer (time-aware) | 22.4k | 0.909 ± 0.019 | 154 ± 24 |
| CfC native (liquid) | 15.9k | 0.871 ± 0.018 | 155 ± 22 |

Findings: (a) engineered features set the ceiling; (b) time handling adds
nothing in any family (the no-time variant of every family matches or beats
its time-aware variants); (c) under timestamp jitter the time-consuming
models degrade *most*; (d) hiding statics compresses the field and makes
timing decisive (+0.047 for XGBoost's time features) — but attention, not
liquid dynamics, exploits it best; (e) the counterfactual judge probe works
on all neural families (Spearman judge-speed ↔ predicted remaining duration:
LSTM 0.903, TF 0.869, CfC 0.564); (f) the positive: CfC saliency ranks
causal events highest, assigns ≈0 to terminal events, and splits 2.8× on
discovery stalls by plaintiff capability — matching the planted mechanism.

## 7. Results on Generator v2 (Stage-1 IDN)

**The candidate** (Figure 1, `paper/figures/fig_stage1_architecture.pdf`).
A chronology-correct hybrid: causal-masked history
encoder producing context cₖ; a gated continuous-time clock partition with
rate r = softplus(R(c)) and relaxation α = 1 − exp(−r·Δt) toward a context
target; an event-conditioned GRU jump; a context blend; structurally
partitioned state [event, clock, context]; main heads plus marked-event
auxiliaries; nonnegative, non-crossing quantile heads.

**The reference distribution** (hidden statics, 10 seeds): tf-native-aux
0.849 ± 0.014, tf-native 0.850 ± 0.013, lstm 0.850 ± 0.013, xgb 0.853 ±
0.018, cfc 0.790 ± 0.018.

**The screen** (paired per seed, IDN − tf-native-aux; Figure 5,
`paper/figures/fig_paired_dauc.pdf`):

```
ΔAUC: −0.003 −0.006 −0.011 −0.010 +0.004 −0.002 −0.003 −0.013 +0.005 −0.001
mean ΔAUC = −0.0039   95% CI = [−0.0082, +0.0004]   wins: IDN 2/10, TF 8/10
duration MAE 308d vs 304d (ok)   ECE 0.043 vs 0.039 (ok)
```

**Verdict: FAIL.** The CI excludes the preregistered practical improvement,
so this is an informative negative, not an underpowered ambiguity. Duration
and calibration show no regression — the hybrid is competent, merely not
better. Per protocol, gated ablations and the hostile-world battery were not
run; no rescue tuning was performed. The candidate is archived as
`experiments/archive/stage1-killed/` with a reproduction check showing zero
drift on all 40 retrained runs.

**Reproduction terminology.** We distinguish three levels, and report each
under its own name. *Independent calculation:* recomputing the reported
statistic from the same archived outputs. *Deterministic reproduction:*
rerunning the frozen pipeline and obtaining matching outputs within the
declared tolerance. *Independent replication:* a separate implementation or
independently generated dataset reproducing the conclusion. Under these
definitions: (i) the archive's `reproduction_check.json` is a deterministic
reproduction — 40 retrained runs, zero drift; (ii) the project owner's
recomputation of the primary statistic from the archived outputs is an
independent calculation — mean −0.00391, CI [−0.00821, +0.00040], matching
the values of record exactly (repeated on archive intake by the successor
team, 2026-07-25, with the same result); (iii) no independent replication
has been performed, and the paper claims none.

## 8. Why it (probably) failed: a prioritized diagnostic map

**Evidentiary status of this section.** The confirmatory Stage-1 analysis
ended when the preregistered primary screen failed. Everything in sections
8–8.1 — all checkpoint forensics and stratified analyses — is exploratory:
it cannot alter the kill decision, and it may only inform a separately
preregistered successor. We cannot know why the hybrid failed from this
experiment alone — the diagnostic ablations were correctly gated off. The
hypotheses below are therefore *labeled as hypotheses*, each with a cheap
decisive test. They are ordered by expected information per unit of effort
(qualitative judgment, not costed estimates), and section 9 turns them into
the successor's work plan.

### H1 — No new information stream (redundancy; strongest prior)

The adapters consumed c_{k−1} — the Transformer's own optimized compression
of the same history. The experiment asked: *after attention has already
decided how history and time matter, does a small second network benefit
from recomputing a persistent state from that answer?* Usually not, unless
it gets a distinct evidence stream or a distinct training signal.
**Test:** replace the clock partition with a parameter-matched feed-forward
residual on the archived checkpoint. If it matches IDN's metrics, the flow
was contributing generic residual capacity, nothing more.

### H2 — Gate/rate collapse (a mathematical non-identifiability)

The composed update is z_k = (1 − gₖαₖ)z_{k−1} + gₖαₖT(c_{k−1}): the model
only ever sees the **effective coefficient βₖ = gₖαₖ**. Large rate with small
acceptance, and small rate with large acceptance, are indistinguishable.
The intended semantic decomposition — r as *how fast*, g as *whether* — was
never identifiable from the predictive objective, so gate inspection could
not have revealed institutional structure regardless of outcome.
**Test:** clamp g to 0 and to 1 at inference on archived checkpoints. If
metrics are invariant to clamping, the mechanism is unused (collapse);
reparametrize or delete it next time.

### H3 — The flow shape was too restrictive

The implemented flow is a context-conditioned leaky integrator: monotone
exponential relaxation toward one fixed attractor per interval. It cannot
express direction changes, deadline activation, periodic calendars, multiple
phases, shocks during silence, or rising-then-falling pressure. It tested
"first-order decay toward a fixed target," not "continuous dynamics."
**Test:** clock ablation on archived checkpoints. If ablation *improves*
metrics, the flow was actively moving state in the wrong direction
(misspecification), not merely unused.

### H4 — The interval was never supervised (strongest candidate fix)

All losses fired at event boundaries. The model was rewarded for useful
endpoints, never for a correct trajectory through silence. A continuous-time
mechanism trained only at endpoints is a weakly identified one. A marked
temporal point-process objective supervises the entire interval via the
survival term S(t) = exp(−∫λ du): every day of continued silence becomes
training information.
**Test:** this is a training change, not a forensic one — see F2 in
section 9. It is the single most principled repair available.

### H5 — Partitions were named, not specialized

Nothing forced [event, clock, context] to specialize: heads could ignore
the clock, context could duplicate event, information could migrate across
seeds. The easiest optimization path is to rely on the already-good
attention representation and treat adapters as residual noise.
**Test:** zero each partition at inference on archived checkpoints and
inspect head weights/gradient norms per partition. If zeroing the clock
changes nothing, H5 and H1 are confounded and both point at removal.

### H6 — Budget fragmentation at 22k

IDN split ~22.5k parameters across attention, rate net, target net, gate,
GRU, blend, and heads; the opponent concentrated on one mechanism. Parameter
matching is fair in one sense and severe in another: the inductive bias had
to be worth fragmenting a very small model. It was not.
**Test:** implied by H1/H5 outcomes; a successor should either shrink the
adapter surface radically or earn a larger budget with a stronger objective.

### H7 — Short sequences give persistent state no niche

When most matters have a handful of informative events, causal attention
revisits the whole history at trivial cost; state compression has no
efficiency niche, and the statistical niche must come from the time
semantics themselves (H3/H4).

### H8 — Aggregate AUC may mask a narrow timing-sensitive effect

Hidden-statics AUC is dominated by event identity and procedural markers.
Continuous dynamics should earn whatever they earn in long gaps, phase
transitions, and sparse-information stretches. If those are a small share of
steps, a real but narrow effect is diluted to zero.
**Test (cheap, run for this paper — see section 8.1):** stratify the paired
comparison by Δt quartile, case age, and procedural phase on the archived
predictions. This does not reopen the verdict — the aggregate screen stands
— but it indicates whether any successor should aim at the narrow stratum
instead of the aggregate.

### 8.1 Stratified forensic check (run on archived predictions)

(Figure 6, `paper/figures/fig_stratified_dauc.pdf`.)

Paired ΔAUC (IDN − tf-native-aux, hidden statics, 10 seeds, mean ± 95% CI)
by stratum, from `experiments/forensic_stratified.py` on the archived
checkpoints:

| stratum | Δ AUC |
|---|---|
| preceding-gap quartiles (short→long) | −0.0027, −0.0058, −0.0053, **+0.0009 ± 0.0078** |
| long gap > 90d | −0.0010 ± 0.0086 |
| case-age quartiles (young→old) | −0.0042, −0.0032, −0.0029, **+0.0032 ± 0.0097** |
| procedural phase (early/mid/late) | −0.0191, −0.0117, **+0.0066 ± 0.0118** |

Reading: the aggregate verdict stands — no stratum shows a significant IDN
advantage, and early-phase steps are where IDN loses most (−0.019). But the
gradient is consistent across all three temporal axes: the most
time-intensive strata (longest gaps, oldest cases, latest phase) are the
only non-negative ones. That is weak but directional evidence for H8 — the
hybrid's handicap is worst exactly where persistent state has least to
compress, and least bad where it has most. A successor should treat "aim at
the late/long regime" as a *prioritized hypothesis worth F1-style
confirmation*, not as a discovered effect.

## 9. The successor's program (fast fail, in order)

Rules of engagement: the frozen protocol stays frozen; any new architecture
is a **new versioned preregistration**, not a continuation of this one; the
killed candidate stays killed; every stage below has its own kill criterion
and stops when met.

- **F1 — Forensics before training (highest value per effort).** Run the
  inference-time battery on the archived checkpoints: partition zeroing;
  gate clamps (0 and 1); Δt shuffles within procedural phase; Δt → median;
  parameter-matched feed-forward residual in place of the clock; head-weight
  and gradient-norm inspection per partition; longest-gap-quartile-only
  evaluation. *Retirement criterion:* if clock ablation is neutral or
  beneficial, retire this context-conditioned exponential-relaxation
  topology. This retires one topology, not continuous-time modeling: any
  future continuous-time proposal must introduce a materially different
  transition mechanism or training objective (F2 is exactly that).
  *Status note (post-draft):* F1 has since been executed under a frozen spec
  (`experiments/F1_FORENSICS_SPEC.md`, frozen 2026-07-25 before any probe ran)
  and is reported separately in `experiments/F1_FORENSICS_RESULTS.md` — all
  findings exploratory; the Stage-1 kill is unchanged. Its within-scope
  decision: **retire** the Stage-1 context-conditioned exponential-relaxation
  clock topology (the clock behaved as generic residual capacity, not a
  temporal mechanism).
- **F2 — Supervise the interval (the principled repair).** Rebuild the
  objective around a marked temporal point process: cause-specific
  intensities λ_m(t | H_t, z_t), training against the survival term over
  every interval, jointly with next-type, terminal-risk, and quantile heads.
  *Kill criterion:* on the same frozen world and screen, TPP-augmented
  Transformer must itself beat tf-native-aux by the preregistered margin;
  if the objective alone doesn't move the strong baseline, the hybrid
  question is dead twice over.
  *Status note (post-draft):* F2 has since been executed under its own frozen
  preregistration (`experiments/F2_PREREGISTRATION.md`, margin calibrated from
  the inter-model noise floor and frozen with all code hashes in
  `F2_FREEZE.md` before any candidate training). **Verdict: killed candidate**
  — `tf-tpp` worsened the primary joint score J by 15.3% (paired relative
  reduction −15.3%, CI [−17.6%, −12.9%]) against a required ≥ +3%, all 10
  seeds negative; settle-AUC/ECE/duration
  guards passed, making this an informative negative. The interval-supervision
  branch on Generator v2 is terminated (`experiments/F2_RESULTS.md`).
- **F3 — If flow is kept, make it earn its keep.** *(Closed by F2: per the
  frozen criterion, no continuous flow may be added to a TPP objective that
  failed to improve the strong baseline.)* Multi-phase or
  deadline-activated flows; SDE variants only after F2 shows interval
  supervision matters; explicit gate reparameterization so that β is
  identified by construction (or no gate at all).
- **F4 — Stratified primary endpoint.** *(Closed by F1: no stratum showed a
  real pro-IDN pocket, so there is no timing-sensitive stratum to promote.)*
  If F1 had shown the narrow effect was real (H8), the path was to preregister the timing-sensitive stratum as the primary
  endpoint for the next candidate — never as a post-hoc rescue of this one.
- **F5 — Real dockets.** The CourtListener/RECAP adapter remains the only
  external-validity test that can overturn the synthetic verdicts. Build it
  with the same harness: outcome definitions, leakage controls, temporal
  splits, jurisdiction transfer, calibration.
- **F6 — Keep the auditability line.** The E5 result is the program's one
  positive: saliency that recovered several planted mechanisms under
  controlled synthetic conditions (stable directional alignment, variable
  magnitude across seeds), and explanation-driven debugging. Any successor
  model should ship with explanation-correctness checks from day one.

### 9.1 Successor postscript (2026-07-25): F1 and F2 have been run

Both preregistered successor steps are complete; the Stage-1 narrative and
verdict above are unchanged.

- **F1 (exploratory, `experiments/F1_FORENSICS_RESULTS.md`):** the archived
  checkpoints answered *why* the hybrid failed. The clock partition was
  load-bearing at the head (zeroing costs ≈ 0.007 AUC) but contributed only
  **generic residual capacity** — a parameter-matched MLP on (context,
  log1p Δt) reproduced its full contribution on every metric — and it was
  **insensitive to interval ordering** (shuffling Δt within procedural
  phase is neutral), so it never functioned as a clock. No stratum,
  including the latent-log strata, showed a reliable pro-IDN pocket; the
  one Holm-surviving stratum (short sequences) runs *against* the IDN.
  Decision per the frozen rule: **the context-conditioned
  exponential-relaxation clock topology is retired.**
- **F2 (confirmatory, `experiments/F2_PREREGISTRATION.md`,
  `experiments/F2_RESULTS.md`):** the principled repair — supervising the
  interval through a marked temporal point-process likelihood on the
  identical Transformer trunk (`tf-tpp`, closed-form piecewise-constant
  intensities) — was killed by its own frozen screen. On the primary joint
  event-process score J (next-type log-loss + next-gap pinball), tf-tpp did
  not improve tf-native-aux; it was decisively worse: mean paired relative
  J reduction **−15.3%**, 95% CI [−17.6%, −12.9%], against the required
  ≥ +3%, with the next-type-accuracy guard also failing (settle AUC, ECE,
  and duration guards passed). Per the frozen criterion, **the
  interval-supervision branch on Generator v2 is terminated**: in this
  world, endpoint supervision with matched auxiliary heads already extracts
  what the interval likelihood was expected to add.
- **What remains open:** the non-continuous-time successors (time-integrated
  attention, selective SSMs, neural semi-Markov regimes) under their own
  preregistrations, and the real-docket bridge — still the only arbiter of
  external validity. F2's kill narrows the hypothesis space exactly as
  designed: the failure is not specific to the killed flow topology; the
  interval itself, at least under piecewise-constant intensities on
  Generator v2, carries no exploitable signal beyond endpoint supervision.

## 10. Threats to validity

All evidence is synthetic; the worlds encode stated mechanisms, not measured
litigation. Generator v1 over-discloses latents relative to any real
underwriting setting. The primary metric is a 180-day settlement-horizon
AUC. Parameter matching at ~22k is one of infinitely many budget choices.
The negative results rule out *these* architectures at *these* budgets on
*these* worlds — not continuous-time modeling of litigation in general. The
archive and frozen protocol exist precisely so a successor can challenge
each of those "these" variables cheaply.

## 11. Reproducibility

- Package: `liquid-legal` 0.1.0. Verified install (clean venv, macOS,
  Python 3.14, 2026-07-25): `pip install -e ".[dev,experiments]"`, then
  `pytest` (41 passed) and `examples/`.
- E1–E6: `experiments/RESULTS.md` + `experiments/run_all.py`,
  `run_hidden_statics.py`, `results/results*.json`.
- Generator v2 + acceptance: `experiments/gen_v2.py`,
  `validate_generator_v2.py`, `FREEZE.md` with sha256s.
- Stage-1: `STAGE1_SPEC.md`, `STAGE1_RESULTS.md`, run logs, results JSONs.
- Killed-candidate archive: `experiments/archive/stage1-killed/` —
  MANIFEST, 40 weight files, 40 prediction files, code snapshots,
  zero-drift `reproduction_check.json`, `hashes.json` over everything.
- Upstream contribution: batched-timespans fix for `ncps`
  ([mlech26l/ncps#85](https://github.com/mlech26l/ncps/pull/85), closing
  upstream issues #81–#83).

## 12. Conclusion

The evaluation program succeeded even though the models did not — which was
the point of building it that way. Two plausible-sounding architecture bets
died quickly, cheaply, and reproducibly, with the machinery of their deaths
preserved for inspection. The successor inherits: two calibrated worlds with
latent logs, a frozen paired-comparison harness, a prioritized diagnostic
map, forty checkpoints with raw predictions, and one positive result —
explanations you can verify — to build on. Since this paper was drafted, that
program has run: F1 retired the Stage-1 clock topology (load-bearing but
generic — a parameter-matched MLP residual reproduced it; it never functioned
as a clock), and F2 killed the interval-supervised marked-TPP objective under
its own frozen preregistration. Two mechanisms are now dead on Generator v2
by frozen criteria, each with full artifacts. The best next move is not a
bigger model, and it is no longer another synthetic architecture — further
search there risks optimizing against one generator's ontology. It is the
real-docket bridge (CourtListener/RECAP, with frozen outcome definitions and
leakage controls): the only remaining external-validity arbiter of whether
timing, silence, and selective observation create a genuine modeling niche.

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
| ncps | 1.0.1 (via local batched-timespan subclass `liquid_legal.rnn`; fix upstreamed as [mlech26l/ncps#85](https://github.com/mlech26l/ncps/pull/85)) |
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

---

## Appendix E — Ethical and deployment limitations

This work is a methods and negative-results study on fully synthetic data.
No real docket, party, judge, or litigation outcome was used, and no claim
in this paper transfers to real litigation. Three deployment-relevant
concerns nevertheless deserve explicit statement, because the motivating
application — litigation finance — is one where model errors are
consequential and asymmetrically distributed.

1. **Synthetic-to-real gap.** The generators encode stated, stylized
   mechanisms (judge backlog episodes, adverse case regimes, discovery
   stalls). Real dockets add selection effects, strategic behavior, sealing,
   missing and duplicated entries, and outcome-recording biases that no
   current component of this program measures. Any real-data successor must
   re-derive outcome and censoring definitions under audit before any
   predictive claim is made, per the real-docket bridge requirements.
2. **Fairness and institutional encoding.** Models trained on docket
   histories can encode jurisdiction-, judge-, and party-level proxies for
   protected or legally irrelevant characteristics. The counterfactual judge
   probe demonstrated here is a diagnostic of exactly such encoding; in a
   deployment context it would be an audit obligation, not a feature. The
   program's explanation-correctness checks are a minimum bar, not a
   sufficient one, for regulated use.
3. **Decision-context calibration.** A settlement-probability estimate is
   used to price capital. Miscalibration at decision-relevant horizons has
   direct financial and access-to-justice externalities. This is why the
   evaluation protocol treats calibration (ECE) and duration error as kill
   guards rather than secondary curiosities, and why any successor must
   report calibration by horizon and institution.

Nothing in this repository is legal, financial, or investment advice, and
the killed candidate must not be deployed under any label.

---

## Appendix F — Artifact availability and licensing

All artifacts needed to inspect or recompute every number in this paper are
in the repository:

- Source code: `src/liquid_legal/` (MIT license, see `LICENSE`; `NOTICE`
  covers third-party attributions).
- Program record: `experiments/RESULTS.md`, `PREREGISTRATION.md`,
  `FREEZE.md`, `STAGE1_SPEC.md`, `STAGE1_RESULTS.md`, with SHA-256 hashes of
  the frozen files in `experiments/results/freeze_hashes.txt`.
- Killed-candidate archive: `experiments/archive/stage1-killed/` — 40 weight
  files, 40 prediction files, run logs, code snapshots, deterministic
  reproduction check, and `hashes.json` over the full archive (verified
  91/91 on successor intake, 2026-07-25).
- Exploratory forensics: `experiments/archive/f1-forensics/` with its own
  manifest.
- Figures regenerate via `paper/figures/make_figures.py`.

Verified installation (clean environment): `pip install -e
".[dev,experiments]"`, then `pytest`. Synthetic generators produce all data;
no external dataset is required. There is no proprietary component.

---

## Appendix G — Author contributions and acknowledgments

[Author names and affiliations to be inserted at submission. CRediT roles:
conceptualization, methodology, software, validation, formal analysis,
investigation, data curation, writing — original draft, writing — review &
editing. The program was designed, executed, and archived by the project
team; the successor team performed archive verification, the independent
calculation of the primary statistic, the exploratory F1 forensics, and the
preparation of this manuscript for publication.]

The authors thank the contributors to the `ncps` library; a
batched-timespans fix developed during this program was contributed back
upstream ([mlech26l/ncps#85](https://github.com/mlech26l/ncps/pull/85)). No external funding supported
this work [confirm at submission]. The authors declare no competing
interests [confirm at submission].

---

## Appendix H — Figure index

All figures regenerate with `.venv/bin/python paper/figures/make_figures.py`
(PDF + PNG in `paper/figures/`).

1. `fig_stage1_architecture` — the killed Stage-1 IDN hybrid: history
   encoder, clock/event/context partitions, heads, chronology contract.
2. `fig_gen_v1_mechanisms` — Generator v1 causal/mechanism DAG.
3. `fig_gen_v2_latents` — Generator v2 latent processes and selective
   observation (true process vs observed docket).
4. `fig_protocol_flowchart` — acceptance → freeze → train → screen → kill;
   gated ablation path shown as not run.
5. `fig_paired_dauc` — per-seed paired ΔAUC, primary screen (confirmatory).
6. `fig_stratified_dauc` — stratified ΔAUC (exploratory; cannot reopen the
   verdict).

---

*Artifacts verified 2026-07-23; archive re-verified on successor intake
2026-07-25 (91/91 hashes). If you are the successor: read
`experiments/archive/stage1-killed/MANIFEST.md` first, then section 9 here.*
