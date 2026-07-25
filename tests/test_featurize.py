import numpy as np

from liquid_legal import CaseEvent, CaseTimeline, EventType
from liquid_legal.featurize import (
    EVENT_FEATURE_DIM,
    collate_timelines,
    featurize_timeline,
)


def _timeline(duration=100.0, settled=True):
    events = [
        CaseEvent(0.0, EventType.FILED, amount=2_000_000.0),
        CaseEvent(30.0, EventType.ANSWER),
        CaseEvent(100.0, EventType.SETTLED, amount=1_000_000.0),
    ]
    return CaseTimeline(
        case_id="t-0",
        events=events,
        static={},
        outcome={
            "settled": float(settled),
            "recovery": 1_000_000.0 if settled else 0.0,
            "duration_days": duration,
            "settle_day": duration if settled else -1.0,
            "n_stalls": 0.0,
        },
    )


def test_featurize_shapes_and_deltas():
    f = featurize_timeline(_timeline())
    assert f["event_ids"].shape == (3,)
    assert f["event_feats"].shape == (3, EVENT_FEATURE_DIM)
    np.testing.assert_allclose(f["deltas"], [0.0, 30.0, 70.0])
    assert np.isclose(f["event_feats"][1, 0], np.log1p(30.0), rtol=1e-6)


def test_collate_padding_and_labels():
    short = _timeline()
    long_gen_events = [CaseEvent(0.0, EventType.FILED)] + [
        CaseEvent(10.0 * i, EventType.DEPOSITION) for i in range(1, 6)
    ] + [CaseEvent(80.0, EventType.SETTLED, amount=5e5)]
    long = CaseTimeline(
        case_id="t-1",
        events=long_gen_events,
        static={},
        outcome={
            "settled": 1.0,
            "recovery": 5e5,
            "duration_days": 80.0,
            "settle_day": 80.0,
            "n_stalls": 0.0,
        },
    )
    batch = collate_timelines([short, long], horizon_days=180.0)
    assert batch.event_ids.shape == (2, 7)
    assert batch.mask[0].sum().item() == 3.0
    assert batch.mask[1].sum().item() == 7.0
    # settle-within-horizon label is 1 everywhere for both cases
    assert batch.y_settle[0, :3].sum().item() == 3.0
    # remaining time decreases along the case
    rem = batch.y_remaining[1, :7]
    assert all(rem[i] > rem[i + 1] for i in range(6))
    # log1p recovery target is broadcast to every step
    assert np.isclose(batch.y_recovery[1, 0].item(), np.log1p(5e5))
