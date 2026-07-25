# liquid-legal

**An experimental toolkit for modeling and benchmarking irregular
legal-event trajectories with liquid, recurrent and attention-based neural
models** — settlement-within-horizon, expected recovery, and remaining
duration at every docket prefix, with models small enough to audit.

> **Status: experimental research software.** All validation to date is on
> synthetic litigation worlds. There is no pretrained production checkpoint
> and no real-docket validation yet. The benchmark section below reports the
> honest state of play — including where conventional baselines win.

![A matter trajectory crossing the institutional field: filings and rulings as impulses, silence as drift, the settlement basin as terminal boundary](https://raw.githubusercontent.com/j-a-marin/liquid-legal/main/article/fig_institutional_field_3d.png)

During this work we found and fixed an upstream `ncps` bug — batched
irregular-timespan broadcasting in torch CfC/LTC
([mlech26l/ncps#85](https://github.com/mlech26l/ncps/pull/85)); the packaged
`liquid_legal.rnn` subclasses carry the fix locally until it merges. The full
preregistered evaluation program — the Stage-1 exit paper
(`paper/exit-paper.md`), frozen preregistrations, and forensics reports —
ships in the source distribution. The Stage-1 IDN hybrid studied there was an
experimental candidate that failed its preregistered screen and is archived;
it is **not** part of the package's model offering. Subsequent frozen-criteria
follow-ups retired its clock topology (F1) and killed an interval-supervised
marked-TPP objective (F2) on the synthetic world — both preserved as
informative negatives in `experiments/`.

Built on [`ncps`](https://github.com/mlech26l/ncps), the maintained
implementation of Liquid Time-Constant (LTC) and Closed-form Continuous-time
(CfC) networks (Hasani, Lechner, Rus et al.).

## Why liquid networks for legal

Docket data is a pathological time series for standard sequence models:

- **Irregular sampling is the norm.** Filings, motions, discovery events, and
  settlement talks arrive at uneven intervals. LTC/CfC cells take the elapsed
  time between events as an explicit input (`timespans`), so the model's state
  evolves with *calendar time*, not event count. Discrete RNNs and
  Transformers consume elapsed time as an input feature (bucketed, embedded,
  or as a bias term) rather than as the integration variable of the state
  dynamics.
- **Regime shifts are the signal.** A case drifts for months, then a trial date
  is set or summary judgment is denied and everything accelerates. Liquid time
  constants let individual neurons speed up or slow down based on input —
  which maps cleanly onto *case velocity*.
- **Small, causal, and auditable.** Litigation finance is regulated and
  high-stakes. A 64-neuron model on a sparse Neural Circuit Policy (NCP)
  wiring has a fraction of the synapses of a dense RNN, and its sparse
  structure makes per-neuron inspection tractable in a way dense models of
  equal capacity are not.

## The domain hypothesis

Legal-case volatility is **non-linear** and driven by *who* is involved:

- **Judges** differ in speed, erraticness, and defense lean — multiplicative,
  heavy-tailed effects on timing, not additive noise.
- **Districts** differ in calendar congestion, stretching every phase.
- **Plaintiffs** differ in case-management capability. Discovery is always a
  potential blocker: under-equipped plaintiffs face a *quadratically* higher
  stall hazard, each stall stretches the calendar and erodes settlement
  leverage — and weak cases suffer disproportionately from each stall.

The synthetic generator in `liquid_legal.synthetic` encodes exactly these
assumptions, so models can be developed and benchmarked before real PACER-style
data is wired in.

## Install

```bash
pip install liquid-legal   # available once the 0.1.0 release is live on PyPI
```

For development:

```bash
git clone https://github.com/j-a-marin/liquid-legal && cd liquid-legal
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Quickstart

```python
import torch
from liquid_legal import (
    CaseTrajectoryEngine, GeneratorConfig, SyntheticLitigationGenerator,
    TrainConfig, snapshot, train_model,
)

# 1. Synthetic litigation timelines (irregular event streams)
timelines = SyntheticLitigationGenerator(GeneratorConfig(seed=0)).generate(512)

# 2. Liquid engine: 64 CfC neurons on a sparse, auditable NCP wiring.
#    Seed before construction: torch entropy-seeds its default generator
#    at process start, so unseeded construction differs run to run.
torch.manual_seed(0)
engine = CaseTrajectoryEngine(units=64, wiring="ncp", ncp_output_size=16)

# 3. Train to predict settlement / recovery / remaining duration from any prefix
train_model(engine, timelines, TrainConfig(epochs=25))

# 4. Snapshot a case mid-flight — the object an agent would consume
snap = snapshot(engine, timelines[0])
print(snap.p_settle_within_horizon, snap.expected_recovery, snap.velocity)
```

End-to-end demo with an LSTM baseline comparison:

```bash
liquid-legal demo --cases 1024 --epochs 25 --baseline
liquid-legal generate --n 1000 --out cases.jsonl
```

## How it fits together

| Module | What it does |
|---|---|
| `events` | `EventType` taxonomy, `CaseEvent` / `CaseTimeline` dataclasses |
| `synthetic` | Hazard-based generator encoding the domain hypothesis above |
| `featurize` | Padded batches + per-timestep targets (settle ≤ horizon, recovery, remaining) |
| `models` | `CaseTrajectoryEngine`: CfC or LTC cell, NCP or dense wiring, 3 heads |
| `baselines` | `LSTMTrajectoryModel` and `TemporalTransformerModel` (causal-masked, time-aware), same interface, for benchmarking |
| `train` | Masked multi-task training loop + evaluation |
| `interpret` | Event saliency (input gradients) and wiring-sparsity reports |
| `agents` | `CaseSnapshot` — predictions + liquid hidden state for agent frameworks |

The key call is `engine(..., timespans=deltas)`: inter-arrival times feed the
continuous-time dynamics directly.

## Benchmarks

`python examples/benchmark.py` — 1024 synthetic cases, 25 epochs, seeds
{0, 1, 2} (each seed draws a fresh dataset, model init, and train/val split),
mean ± std on the holdout. Single-run CPU figures are not bit-stable across
processes, so these are the numbers that reproduce:

| model | settle AUC ↑ | duration MAE (days) ↓ | recovery MAE (log$) ↓ |
|---|---|---|---|
| CfC, NCP wiring (64 units, 788 synapses) | 0.871 ± 0.018 | 155 ± 22 | 2.74 ± 0.35 |
| LSTM (64 units, dense) | 0.914 ± 0.015 | 146 ± 19 | 2.64 ± 0.33 |

Read these honestly: **on synthetic dockets, conventional feature engineering
sets the ceiling** — XGBoost on counts and statics (0.917) matches or beats
every neural model, including the LSTM (0.914) and a time-aware Transformer
(0.909) at comparable parameter budgets, with the liquid CfC behind (0.871).
The full model matrix, stress curves, and the hidden-latent ablation are in
`experiments/RESULTS.md`; the short version is that the synthetic world
over-discloses its latent traits, making the timing channel redundant — and
when those latents are hidden (E6), timing becomes decisive but attention
exploits it better than liquid dynamics. The case for the liquid model
therefore rests on what it uniquely delivers — 15.9k parameters, ~86% structural
sparsity (NCP wiring), an inspectable hidden state, and saliency that aligned
with planted synthetic mechanisms — plus one explicitly open question:
whether native continuous-time handling pays off on real dockets, where
latent traits are never directly observed.

## Interpretability & agents

- `event_saliency(engine, batch)` attributes each settlement prediction to the
  docket events that drove it.
- `wiring_report(engine.wiring)` reports synapse counts and sparsity — NCP
  wirings are typically >50% sparser than dense equivalents.
- `snapshot(engine, prefix)` returns a `CaseSnapshot` (predictions, velocity
  classification, hidden state) designed to be read by an LLM orchestrator —
  the liquid hidden state acts as a compact, elapsed-time-aware memory of
  the case that higher-level agents can condition on.

## Roadmap (research targets, not shipped features)

- **Real-docket adapter** — PACER/CourtListener event ingestion producing
  `CaseTimeline`s, with frozen outcome/censoring definitions and leakage
  controls. External validity is untested until this exists.
- **Portfolio risk monitoring** — a continuous risk surface over a book of
  matters; detecting adverse regime shifts is a research question, not a
  current capability.
- **Settlement timing** — continuous-time negotiation dynamics for push/wait
  analysis.
- **Matter velocity** — progress vs. comparable historical matters.
- Counterfactual probes: same case, different judge/district draw.

## References

- [Litigation Physics: Learning Institutional Dynamics](https://github.com/j-a-marin/liquid-legal/blob/main/docs/primer.md) — the
  accessible, diagram-ready version of this project's motivation.
- Hasani, Lechner, Amini, Rus, Grosu. *Liquid Time-constant Networks.* AAAI 2021.
- Hasani, Lechner, Amini, et al. *Closed-form Continuous-time Neural Networks.* Nature Machine Intelligence, 2022.
- Lechner, Hasani, Amini, et al. *Neural Circuit Policies Enabling Auditable Autonomy.* Nature Machine Intelligence, 2020.
- `ncps` library: <https://github.com/mlech26l/ncps>

## Intended Use

liquid-legal is intended for research, education, and benchmarking of models
for irregular legal-event streams, presently using synthetic data. It has not
been validated for predicting outcomes, valuing claims, selecting counsel or
jurors, evaluating judges or litigants, allocating capital, or making
decisions in real legal matters. Outputs are experimental model estimates,
not legal, financial, or investment advice, and must not substitute for
qualified professional judgment or independent validation.

## Disclaimer

Research software. Not legal advice, not investment advice. Synthetic data
encodes stylized assumptions; validate against real docket data before any
capital allocation decision.

## License

MIT — see [LICENSE](https://github.com/j-a-marin/liquid-legal/blob/main/LICENSE).
