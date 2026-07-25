import torch

from liquid_legal.baselines import LSTMTrajectoryModel, TemporalTransformerModel
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


def test_transformer_forward_shapes():
    model = TemporalTransformerModel()
    ids, feats, static, deltas = _inputs()
    out = model(ids, feats, static, timespans=deltas)
    assert out["settle_logit"].shape == (B, T)
    assert out["log_recovery"].shape == (B, T)
    assert out["log_remaining"].shape == (B, T)
    assert out["hx"].shape == (B, model.state_size)


def test_transformer_time_modes_run_and_differ():
    ids, feats, static, deltas = _inputs()
    outs = {}
    for mode in TemporalTransformerModel.TIME_MODES:
        torch.manual_seed(0)
        model = TemporalTransformerModel(time_mode=mode)
        outs[mode] = model(ids, feats, static, timespans=deltas)["settle_logit"]
    assert not torch.allclose(outs["native"], outs["none"])
    assert not torch.allclose(outs["feature"], outs["none"])


def test_transformer_causal_mask_blocks_padding():
    """Right-padding must not change predictions at real positions."""
    torch.manual_seed(0)
    model = TemporalTransformerModel().eval()
    ids, feats, static, deltas = _inputs()
    base = model(ids, feats, static, timespans=deltas)["settle_logit"]

    pad = 4
    ids_p = torch.cat([ids, torch.zeros(B, pad, dtype=ids.dtype)], dim=1)
    feats_p = torch.cat([feats, torch.zeros(B, pad, EVENT_FEATURE_DIM)], dim=1)
    deltas_p = torch.cat([deltas, torch.zeros(B, pad)], dim=1)
    padded = model(ids_p, feats_p, static, timespans=deltas_p)["settle_logit"]
    assert torch.allclose(base, padded[:, :T], atol=1e-5)


def test_transformer_does_not_mutate_batch():
    model = TemporalTransformerModel(time_mode="timespans_only")
    ids, feats, static, deltas = _inputs()
    before = feats.clone()
    model(ids, feats, static, timespans=deltas)
    assert torch.equal(feats, before)


def test_lstm_time_modes():
    ids, feats, static, deltas = _inputs()
    torch.manual_seed(0)
    m_feat = LSTMTrajectoryModel(units=16, time_mode="feature")
    torch.manual_seed(0)
    m_none = LSTMTrajectoryModel(units=16, time_mode="none")
    out_feat = m_feat(ids, feats, static, timespans=deltas)["settle_logit"]
    out_none = m_none(ids, feats, static, timespans=deltas)["settle_logit"]
    assert out_feat.shape == (B, T)
    assert not torch.allclose(out_feat, out_none)


def test_auxiliary_transformer_heads():
    ids, feats, static, deltas = _inputs()
    plain = TemporalTransformerModel(d_model=16)
    assert "next_type_logit" not in plain(ids, feats, static, timespans=deltas)
    aux = TemporalTransformerModel(d_model=16, auxiliary=True).eval()
    out = aux(ids, feats, static, timespans=deltas)
    assert out["next_type_logit"].shape == (B, T, 16)
    for key in ("next_gap_q", "duration_q"):
        q = out[key]
        assert q.shape == (B, T, 3)
        assert (q >= 0).all()
        assert (q[..., 2] >= q[..., 1]).all() and (q[..., 1] >= q[..., 0]).all()


def test_auxiliary_loss_flows():
    """The aux-supervised models must train through the shared loss path."""
    from liquid_legal import (
        CaseTrajectoryEngine,
        GeneratorConfig,
        SyntheticLitigationGenerator,
        TrainConfig,
        train_model,
    )
    timelines = SyntheticLitigationGenerator(GeneratorConfig(seed=4)).generate(48)
    torch.manual_seed(0)
    aux = TemporalTransformerModel(d_model=16, num_layers=1, max_len=64,
                                   time_mode="native", auxiliary=True)
    cfg = TrainConfig(epochs=2, verbose=False, seed=0)
    history = train_model(aux, timelines, cfg)
    first, last = history["train"][0]["loss"], history["train"][-1]["loss"]
    assert last < first
