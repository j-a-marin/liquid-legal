# 3. Related work (expanded draft — for merge into `exit-paper.md`)

*Draft status: replacement for section 3. Every bibliography entry carries an
identifier verified against Crossref, the ACL Anthology, JMLR, or arXiv on
2026-07-25. Where a citation could not be verified, the text says so rather
than citing from memory. Nothing in this section asserts real-world litigation
claims; all results referenced from this repository are synthetic.*

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

## 3.1 Neural Circuit Policies (NCP)

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

## 3.2 Liquid Time-Constant Networks (LTC)

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

## 3.3 Closed-form Continuous-time Networks (CfC)

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

## 3.4 GRU-D

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

## 3.5 Neural ODEs

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

## 3.6 ODE-RNN and Latent ODE

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

## 3.7 Neural Controlled Differential Equations

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

## 3.8 Marked temporal point processes: RMTPP and the Neural Hawkes Process

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

## 3.9 Transformer Hawkes processes

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

## 3.10 Semi-Markov models

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

## 3.11 Competing-risk survival models

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

## 3.12 Mamba and selective state-space models

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

## 3.13 Physically time-aware SSM variants

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

## 3.14 Continuous-time Transformers: ContiFormer

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

## 3.15 Longitudinal legal-outcome prediction and legal NLP

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

## 3.16 Calibration under censoring and competing risks

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

## 3.17 Counterfactual prediction from longitudinal observational data

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

## Synthesis

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

## Bibliography

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
