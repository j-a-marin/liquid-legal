"""Leakage, chronology, clock-semantics, and padding tests for IDN
(IDN_GUIDE.md sections 1, 3, 6 and the Stage-1 repair blockers).

These tests are the automated guarantee that the pre-event flow never sees
e_k, that the clock flow has proper continuous-time semantics, and that
padding cannot contaminate outputs or state.
"""

import torch

from experiments.idn_model import IDNModel
from liquid_legal.events import N_EVENT_TYPES, STATIC_DIM
from liquid_legal.featurize import EVENT_FEATURE_DIM

B, T = 2, 6


def _inputs():
    return (
        torch.randint(0, N_EVENT_TYPES, (B, T)),
        torch.randn(B, T, EVENT_FEATURE_DIM),
        torch.randn(B, STATIC_DIM),
        torch.rand(B, T) * 30.0,
    )


# --------------------------------------------------------------------- #
# shapes and chronology
# --------------------------------------------------------------------- #

def test_forward_shapes():
    model = IDNModel()
    ids, feats, static, deltas = _inputs()
    out = model(ids, feats, static, timespans=deltas)
    assert out["settle_logit"].shape == (B, T)
    assert out["log_recovery"].shape == (B, T)
    assert out["log_remaining"].shape == (B, T)
    assert out["next_type_logit"].shape == (B, T, N_EVENT_TYPES)
    assert out["next_gap_q"].shape == (B, T, 3)
    assert out["duration_q"].shape == (B, T, 3)
    assert out["gate_clock"].shape == (B, T, model.d_clock)
    assert out["hx"].shape == (B, model.state_size)


def test_history_encoder_is_causal():
    """Mutating event j must not change any prediction at steps < j."""
    torch.manual_seed(0)
    model = IDNModel().eval()
    ids, feats, static, deltas = _inputs()
    j = 4
    ids_mut = ids.clone()
    ids_mut[:, j] = (ids[:, j] + 1) % N_EVENT_TYPES
    with torch.no_grad():
        base = model(ids, feats, static, timespans=deltas)
        mut = model(ids_mut, feats, static, timespans=deltas)
    for key in ("settle_logit", "log_recovery", "log_remaining", "next_type_logit"):
        assert torch.allclose(base[key][:, :j], mut[key][:, :j], atol=1e-6), key
    assert not torch.allclose(base["settle_logit"][:, j], mut["settle_logit"][:, j])


def test_pre_event_flow_never_sees_event():
    """z_k^- must be invariant to mutations of e_k itself and of later events."""
    torch.manual_seed(0)
    model = IDNModel().eval()
    ids, feats, static, deltas = _inputs()
    with torch.no_grad():
        base = model(ids, feats, static, timespans=deltas, return_intermediates=True)
        for k in range(T):
            ids_mut = ids.clone()
            ids_mut[:, k] = (ids[:, k] + 1) % N_EVENT_TYPES
            mut = model(ids_mut, feats, static, timespans=deltas,
                        return_intermediates=True)
            assert torch.allclose(base["z_minus"][:, k], mut["z_minus"][:, k], atol=1e-6), (
                f"flow at step {k} saw its own event")
            assert torch.allclose(base["z_minus"][:, :k], mut["z_minus"][:, :k], atol=1e-6)


def test_contexts_are_causal():
    torch.manual_seed(0)
    model = IDNModel().eval()
    ids, feats, static, deltas = _inputs()
    ids_mut = ids.clone()
    ids_mut[:, 3] = (ids[:, 3] + 1) % N_EVENT_TYPES
    with torch.no_grad():
        base = model(ids, feats, static, timespans=deltas, return_intermediates=True)
        mut = model(ids_mut, feats, static, timespans=deltas, return_intermediates=True)
    assert torch.allclose(base["contexts"][:, :3], mut["contexts"][:, :3], atol=1e-6)
    assert not torch.allclose(base["contexts"][:, 3], mut["contexts"][:, 3])


def test_gate_bounds_and_partition_update_rules():
    """Gate in [0, 1]; event partition unchanged by the interval."""
    model = IDNModel().eval()
    ids, feats, static, deltas = _inputs()
    with torch.no_grad():
        out = model(ids, feats, static, timespans=deltas, return_intermediates=True)
    g = out["gate_clock"]
    assert (g >= 0).all() and (g <= 1).all()
    z_minus = out["z_minus"]
    d_e = model.d_event
    assert torch.allclose(z_minus[:, 0, :d_e], torch.zeros_like(z_minus[:, 0, :d_e]))
    z_post = out["rnn_out"]
    assert torch.allclose(z_minus[:, 1:, :d_e], z_post[:, :-1, :d_e], atol=1e-6)


# --------------------------------------------------------------------- #
# clock semantics (Blocker 2)
# --------------------------------------------------------------------- #

def test_zero_elapsed_time_is_exact_identity():
    """Δt = 0 must be exactly a no-op on the clock partition."""
    torch.manual_seed(0)
    model = IDNModel().eval()
    z = torch.randn(3, model.d_clock)
    c = torch.randn(3, model.d_context)
    with torch.no_grad():
        flowed = model._clock_flow(z, c, torch.zeros(3, 1))
    assert torch.equal(flowed, z)

    # end-to-end: with all-zero intervals the clock never changes from init
    ids, feats, static, _ = _inputs()
    with torch.no_grad():
        out = model(ids, feats, static,
                    timespans=torch.zeros(B, T), return_intermediates=True)
    d_e, d_c = model.d_event, model.d_clock
    clock_minus = out["z_minus"][:, :, d_e : d_e + d_c]
    assert torch.all(clock_minus == 0)


def test_flow_is_time_monotone_and_compositional():
    """Larger Δt moves the state more (toward the target); constant-context
    flow satisfies Φ(t1+t2) = Φ(t2)∘Φ(t1) exactly."""
    torch.manual_seed(0)
    model = IDNModel().eval()
    z = torch.zeros(4, model.d_clock)
    c = torch.randn(4, model.d_context)
    with torch.no_grad():
        f1 = model._clock_flow(z, c, torch.full((4, 1), 0.5))
        f2 = model._clock_flow(z, c, torch.full((4, 1), 5.0))
        assert (f2 - z).norm(dim=-1).ge((f1 - z).norm(dim=-1) - 1e-6).all()
        t1 = torch.full((4, 1), 0.7)
        t2 = torch.full((4, 1), 1.3)
        two_step = model._clock_flow(model._clock_flow(z, c, t1), c, t2)
        one_step = model._clock_flow(z, c, t1 + t2)
    assert torch.allclose(two_step, one_step, atol=1e-5)


# --------------------------------------------------------------------- #
# padding (Blocker 3)
# --------------------------------------------------------------------- #

def test_right_padding_is_invisible():
    torch.manual_seed(0)
    model = IDNModel().eval()
    ids, feats, static, deltas = _inputs()
    base = model(ids, feats, static, timespans=deltas)["settle_logit"]
    pad = 3
    ids_p = torch.cat([ids, torch.zeros(B, pad, dtype=ids.dtype)], dim=1)
    feats_p = torch.cat([feats, torch.zeros(B, pad, EVENT_FEATURE_DIM)], dim=1)
    deltas_p = torch.cat([deltas, torch.zeros(B, pad)], dim=1)
    with torch.no_grad():
        padded = model(ids_p, feats_p, static, timespans=deltas_p,
                       lengths=torch.tensor([T, T]))["settle_logit"]
    assert torch.allclose(base, padded[:, :T], atol=1e-5)


def test_padded_batch_matches_unpadded_cases():
    """A padded batch and the same cases run individually must produce
    identical valid outputs AND identical final valid states."""
    torch.manual_seed(0)
    model = IDNModel().eval()
    ids, feats, static, deltas = _inputs()
    lengths = torch.tensor([3, 6])
    with torch.no_grad():
        batched = model(ids, feats, static, timespans=deltas, lengths=lengths)
        singles = [
            model(ids[i : i + 1, : L], feats[i : i + 1, : L], static[i : i + 1],
                  timespans=deltas[i : i + 1, : L])
            for i, L in enumerate(lengths.tolist())
        ]
    for i, L in enumerate(lengths.tolist()):
        for key in ("settle_logit", "log_recovery", "log_remaining"):
            assert torch.allclose(batched[key][i, :L], singles[i][key][0], atol=1e-5), key
        assert torch.allclose(batched["hx"][i], singles[i]["hx"][0], atol=1e-5)


# --------------------------------------------------------------------- #
# quantiles (Blocker 4 / guide section 5)
# --------------------------------------------------------------------- #

def test_quantiles_nonnegative_and_ordered():
    model = IDNModel().eval()
    ids, feats, static, deltas = _inputs()
    with torch.no_grad():
        out = model(ids, feats, static, timespans=deltas)
    for key in ("next_gap_q", "duration_q"):
        q = out[key]
        assert (q >= 0).all(), key
        assert (q[..., 1] >= q[..., 0]).all(), key
        assert (q[..., 2] >= q[..., 1]).all(), key
