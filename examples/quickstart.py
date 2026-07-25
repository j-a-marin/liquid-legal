"""liquid-legal quickstart: synthetic dockets -> CfC engine -> predictions.

Run with: python examples/quickstart.py
"""

from liquid_legal import (
    CaseTrajectoryEngine,
    GeneratorConfig,
    SyntheticLitigationGenerator,
    TrainConfig,
    snapshot,
    train_model,
    wiring_report,
)

# 1. Generate synthetic litigation timelines (irregular event streams).
gen = SyntheticLitigationGenerator(GeneratorConfig(seed=0))
timelines = gen.generate(256)
t0 = timelines[0]
print(f"case {t0.case_id}: {t0.n_events} events over {t0.duration_days:.0f} days")
print(f"  settled={t0.outcome['settled']:.0f} recovery=${t0.outcome['recovery']:,.0f}")

# 2. Build the liquid engine: 64 CfC neurons on a sparse NCP wiring.
# Seed before construction: torch entropy-seeds its default generator at
# process start, so unseeded construction differs run to run.
import torch

torch.manual_seed(0)
engine = CaseTrajectoryEngine(units=64, wiring="ncp", ncp_output_size=16, cell="cfc")
print(f"  wiring: {wiring_report(engine.wiring)}")

# 3. Train: predict settlement / recovery / remaining duration from any prefix.
cfg = TrainConfig(epochs=10, batch_size=32)
train_model(engine, timelines, cfg)

# 4. Snapshot a case mid-flight — the object an agent would consume.
snap = snapshot(engine, timelines[1])
print(f"\nsnapshot of {snap.case_id} at day {snap.day:.0f} ({snap.last_event}):")
print(f"  p(settle within 180d) = {snap.p_settle_within_horizon:.2f}")
print(f"  expected recovery     = ${snap.expected_recovery:,.0f}")
print(f"  expected remaining    = {snap.expected_remaining_days:.0f} days")
print(f"  velocity              = {snap.velocity}")
print(f"  hidden state (dim {len(snap.hidden_state)}) = {[round(v, 3) for v in snap.hidden_state[:6]]} ...")
