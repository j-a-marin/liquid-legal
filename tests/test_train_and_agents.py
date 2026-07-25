import numpy as np
import torch

from liquid_legal import (
    CaseTrajectoryEngine,
    GeneratorConfig,
    SyntheticLitigationGenerator,
    TrainConfig,
    collate_timelines,
    event_saliency,
    evaluate,
    snapshot,
    train_model,
    wiring_report,
)


def _timelines(n=96, seed=11):
    return SyntheticLitigationGenerator(GeneratorConfig(seed=seed)).generate(n)


def test_train_model_reduces_loss_and_evaluates():
    torch.manual_seed(0)
    timelines = _timelines()
    engine = CaseTrajectoryEngine(units=32, wiring="ncp", ncp_output_size=8)
    cfg = TrainConfig(epochs=8, batch_size=32, verbose=False, seed=0)
    history = train_model(engine, timelines, cfg)
    assert len(history["train"]) == 8
    assert len(history["val"]) == 8
    first, last = history["train"][0]["loss"], history["train"][-1]["loss"]
    assert np.isfinite(first) and np.isfinite(last)
    assert last < first
    metrics = evaluate(engine, timelines[:20], cfg)
    assert 0.0 <= metrics["settle_auc"] <= 1.0
    assert metrics["duration_mae_days"] >= 0.0


def test_snapshot_fields_and_ranges():
    timelines = _timelines(n=8)
    engine = CaseTrajectoryEngine(units=24, wiring="fully_connected")
    snap = snapshot(engine, timelines[0])
    assert 0.0 <= snap.p_settle_within_horizon <= 1.0
    assert snap.expected_recovery >= 0.0
    assert snap.expected_remaining_days >= 0.0
    assert snap.velocity in {"accelerating", "stalled", "steady"}
    assert len(snap.hidden_state) == 24
    assert snap.n_events == timelines[0].n_events


def test_event_saliency_shape_and_padding_zero():
    timelines = _timelines(n=4)
    engine = CaseTrajectoryEngine(units=16, wiring="fully_connected")
    batch = collate_timelines(timelines)
    sal = event_saliency(engine, batch)
    assert sal.shape == batch.mask.shape
    assert np.all(sal[batch.mask.numpy() == 0.0] == 0.0)
    assert np.any(sal[batch.mask.numpy() == 1.0] > 0.0)


def test_wiring_report_ncp_is_sparser_than_dense():
    ncp = CaseTrajectoryEngine(units=48, wiring="ncp", ncp_output_size=8)
    fc = CaseTrajectoryEngine(units=48, wiring="fully_connected")
    rep_ncp = wiring_report(ncp.wiring)
    rep_fc = wiring_report(fc.wiring)
    assert rep_ncp["sparsity"] > 0.0
    assert rep_fc["sparsity"] == 0.0
    assert rep_ncp["recurrent_synapses"] < rep_fc["recurrent_synapses"]
