import torch

from liquid_legal import CaseTrajectoryEngine, LSTMTrajectoryModel
from liquid_legal.events import STATIC_DIM
from liquid_legal.featurize import EVENT_FEATURE_DIM

B, T = 2, 5


def _inputs():
    return (
        torch.randint(0, 16, (B, T)),
        torch.randn(B, T, EVENT_FEATURE_DIM),
        torch.randn(B, STATIC_DIM),
        torch.rand(B, T) * 30.0,
    )


def test_engine_fully_connected_forward():
    engine = CaseTrajectoryEngine(units=24, wiring="fully_connected")
    ids, feats, static, deltas = _inputs()
    out = engine(ids, feats, static, timespans=deltas)
    assert out["settle_logit"].shape == (B, T)
    assert out["log_recovery"].shape == (B, T)
    assert out["log_remaining"].shape == (B, T)
    assert out["hx"].shape == (B, 24)


def test_engine_ncp_wiring_forward():
    engine = CaseTrajectoryEngine(units=32, wiring="ncp", ncp_output_size=8)
    ids, feats, static, deltas = _inputs()
    out = engine(ids, feats, static, timespans=deltas)
    assert out["settle_logit"].shape == (B, T)
    assert out["hx"].shape == (B, 32)


def test_engine_accepts_3d_timespans_and_2d():
    engine = CaseTrajectoryEngine(units=16, wiring="fully_connected")
    ids, feats, static, deltas = _inputs()
    out2 = engine(ids, feats, static, timespans=deltas)
    out3 = engine(ids, feats, static, timespans=deltas.unsqueeze(-1))
    assert torch.allclose(out2["settle_logit"], out3["settle_logit"])


def test_engine_without_timespans():
    engine = CaseTrajectoryEngine(units=16, wiring="fully_connected")
    ids, feats, static, _ = _inputs()
    out = engine(ids, feats, static)
    assert out["settle_logit"].shape == (B, T)


def test_ltc_cell_forward():
    engine = CaseTrajectoryEngine(units=16, wiring="fully_connected", cell="ltc")
    ids, feats, static, deltas = _inputs()
    out = engine(ids, feats, static, timespans=deltas)
    assert out["settle_logit"].shape == (B, T)


def test_hidden_state_threading():
    engine = CaseTrajectoryEngine(units=16, wiring="fully_connected")
    ids, feats, static, deltas = _inputs()
    first = engine(ids, feats, static, timespans=deltas)
    cont = engine(ids, feats, static, timespans=deltas, hx=first["hx"])
    fresh = engine(ids, feats, static, timespans=deltas)
    assert not torch.allclose(cont["settle_logit"], fresh["settle_logit"])


def test_lstm_baseline_interface():
    model = LSTMTrajectoryModel(units=24)
    ids, feats, static, deltas = _inputs()
    out = model(ids, feats, static, timespans=deltas)
    assert out["settle_logit"].shape == (B, T)
    assert out["hx"].shape == (B, 24)


def test_gradients_flow_to_all_heads():
    engine = CaseTrajectoryEngine(units=16, wiring="ncp", ncp_output_size=8)
    ids, feats, static, deltas = _inputs()
    out = engine(ids, feats, static, timespans=deltas)
    loss = (
        out["settle_logit"].mean()
        + out["log_recovery"].mean()
        + out["log_remaining"].mean()
    )
    loss.backward()
    grads = [p.grad for p in engine.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_time_modes_run_and_differ():
    ids, feats, static, deltas = _inputs()
    outs = {}
    for mode in ("native", "timespans_only", "feature", "none"):
        torch.manual_seed(0)
        engine = CaseTrajectoryEngine(units=16, wiring="fully_connected", time_mode=mode)
        outs[mode] = engine(ids, feats, static, timespans=deltas)["settle_logit"]
        assert outs[mode].shape == (B, T)
    # each ablation removes real information, so outputs should differ
    assert not torch.allclose(outs["native"], outs["feature"])
    assert not torch.allclose(outs["native"], outs["none"])


def test_time_mode_does_not_mutate_batch():
    engine = CaseTrajectoryEngine(units=16, wiring="fully_connected", time_mode="timespans_only")
    ids, feats, static, deltas = _inputs()
    before = feats.clone()
    engine(ids, feats, static, timespans=deltas)
    assert torch.equal(feats, before)
