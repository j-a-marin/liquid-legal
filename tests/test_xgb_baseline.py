import multiprocessing as mp

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")

from liquid_legal import CaseEvent, CaseTimeline, EventType, GeneratorConfig, SyntheticLitigationGenerator
from experiments.xgb_baseline import N_FEATURES, prefix_features


def _timeline():
    events = [
        CaseEvent(0.0, EventType.FILED, amount=2_000_000.0),
        CaseEvent(30.0, EventType.ANSWER),
        CaseEvent(100.0, EventType.SETTLEMENT_OFFER, amount=500_000.0),
        CaseEvent(200.0, EventType.SETTLED, amount=800_000.0),
    ]
    return CaseTimeline(
        case_id="x-0",
        events=events,
        static={"plaintiff_capability": 0.5},
        outcome={"settled": 1.0, "recovery": 800_000.0,
                 "duration_days": 200.0, "settle_day": 200.0, "n_stalls": 0.0},
    )


def test_prefix_features_shape_and_content():
    X = prefix_features(_timeline())
    assert X.shape == (4, N_FEATURES)
    # day column is the absolute calendar day
    np.testing.assert_allclose(X[:, 8], [0.0, 30.0, 100.0, 200.0])
    # last_delta column
    np.testing.assert_allclose(X[:, 9], [0.0, 30.0, 70.0, 100.0])
    # offer count becomes 1 after the offer event
    assert X[2, 12] == 1.0 and X[1, 12] == 0.0


def _worker(train_tls, eval_tls, connection):
    from experiments.xgb_baseline import train_and_eval_isolated

    connection.send(train_and_eval_isolated(train_tls, eval_tls, seed=0))
    connection.close()


def test_xgb_trains_and_evaluates():
    """xgboost must run in a spawned subprocess: on macOS its OpenMP runtime
    can segfault when torch is already loaded in this process."""
    timelines = SyntheticLitigationGenerator(GeneratorConfig(seed=2)).generate(48)
    train, ev = timelines[:40], timelines[40:]
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(train, ev, child))
    proc.start()
    child.close()
    proc.join()
    assert proc.exitcode == 0
    assert parent.poll(), "xgboost worker exited without returning results"
    out = parent.recv()
    for variant in ("aware", "no_time"):
        assert 0.0 <= out[variant]["settle_auc"] <= 1.0
        assert out[variant]["duration_mae_days"] >= 0.0
