import numpy as np

from liquid_legal import EventType, GeneratorConfig, SyntheticLitigationGenerator
from liquid_legal.events import TERMINAL_EVENTS


def test_deterministic_per_seed():
    g1 = SyntheticLitigationGenerator(GeneratorConfig(seed=7))
    g2 = SyntheticLitigationGenerator(GeneratorConfig(seed=7))
    a = g1.generate(8)
    b = g2.generate(8)
    assert [t.duration_days for t in a] == [t.duration_days for t in b]
    assert [t.judge_id for t in a] == [t.judge_id for t in b]


def test_timeline_structure_and_outcome_consistency():
    gen = SyntheticLitigationGenerator(GeneratorConfig(seed=1))
    for t in gen.generate(25):
        assert t.events[0].event_type is EventType.FILED
        assert t.events[-1].event_type in TERMINAL_EVENTS
        days = [e.day for e in t.events]
        assert all(b >= a for a, b in zip(days, days[1:]))
        out = t.outcome
        assert out["duration_days"] == t.events[-1].day
        assert out["recovery"] >= 0.0
        if out["settled"]:
            assert t.events[-1].event_type is EventType.SETTLED
            assert out["settle_day"] == t.events[-1].day
            assert out["recovery"] > 0.0
        if t.events[-1].event_type is EventType.DISMISSED:
            assert out["recovery"] == 0.0


def test_under_equipped_plaintiffs_stall_more_in_discovery():
    """The domain claim: discovery is a blocker for plaintiffs not equipped
    for case-management tasks. Low-capability plaintiffs must see
    systematically more discovery stalls."""
    gen = SyntheticLitigationGenerator(GeneratorConfig(seed=3))
    timelines = gen.generate(400)
    low = [t.outcome["n_stalls"] for t in timelines if t.static["plaintiff_capability"] < 0.35]
    high = [t.outcome["n_stalls"] for t in timelines if t.static["plaintiff_capability"] > 0.65]
    assert len(low) > 30 and len(high) > 30
    assert np.mean(low) > np.mean(high)


def test_judge_pool_has_heterogeneous_traits():
    gen = SyntheticLitigationGenerator(GeneratorConfig(seed=0, n_judges=12))
    vols = [j["volatility"] for j in gen.judges_]
    speeds = [j["speed"] for j in gen.judges_]
    assert max(vols) - min(vols) > 0.3
    assert max(speeds) / min(speeds) > 1.5


def test_event_types_are_enum_instances():
    """rng.choice over enums returns np.int64, breaking identity checks —
    every stored event_type must be a genuine EventType."""
    gen = SyntheticLitigationGenerator(GeneratorConfig(seed=5))
    for t in gen.generate(30):
        for ev in t.events:
            assert type(ev.event_type) is EventType


def test_negotiation_events_carry_intended_semantics():
    """Settlement offers must carry dollar amounts (a prior bug silently
    recorded them as plain events), and mediation/trial-date events exist."""
    gen = SyntheticLitigationGenerator(GeneratorConfig(seed=6))
    timelines = gen.generate(150)
    offers = [
        ev
        for t in timelines
        for ev in t.events
        if ev.event_type is EventType.SETTLEMENT_OFFER
    ]
    assert len(offers) > 50
    assert all(ev.amount > 0.0 for ev in offers)
    types = {ev.event_type for t in timelines for ev in t.events}
    assert EventType.MEDIATION in types
    assert EventType.TRIAL_DATE_SET in types
