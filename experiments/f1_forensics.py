"""F1 — EXPLORATORY checkpoint forensics on the killed Stage-1 IDN.

Implements F1_FORENSICS_SPEC.md (frozen 2026-07-25) exactly: seven probe
groups over the archived checkpoints in archive/stage1-killed/ (hidden-statics
regime, seeds 0-9), all deltas paired per seed against the unmodified
archived IDN on identical holdout steps.

EVIDENTIARY STATUS: EXPLORATORY. The confirmatory Stage-1 screen FAILED
(PROVISIONAL SURVIVAL: FAIL — stop the hybrid track); nothing here reopens
that verdict. Sole decision scope: retain or retire the context-conditioned
exponential-relaxation clock topology in future designs.

Code-snapshot check (run 2026-07-25): experiments/idn_model.py,
src/liquid_legal/train.py and src/liquid_legal/baselines.py are
byte-identical to the archived snapshots under archive/stage1-killed/code/
(verified with diff). The live copies are therefore the authoritative
frozen implementations and are imported directly; the archived IDN forward
is never modified — probes subclass it (ProbeIDN below).

Outputs:
  experiments/results/f1_forensics.json   (machine-readable results)
  experiments/archive/f1-forensics/       (raw outputs, config, code
                                           snapshot, environment, hashes)

Run with: cd experiments && ../.venv/bin/python f1_forensics.py
Runtime: ~15 min on CPU.
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from liquid_legal import TrainConfig
from liquid_legal import train as train_mod  # frozen losses (_loss, _pinball)
from liquid_legal.baselines import TemporalTransformerModel
from liquid_legal.events import STATIC_DIM, EventType
from liquid_legal.featurize import collate_timelines, featurize_timeline
from liquid_legal.metrics import auc_score, mae_days, mae_log_dollars
from gen_v2 import GeneratorV2, GeneratorV2Config
from idn_model import IDNModel
from run_all import split
from run_hidden_statics import strip_statics

ARCHIVE = Path(__file__).parent / "archive" / "stage1-killed"
OUT_JSON = Path(__file__).parent / "results" / "f1_forensics.json"
OUT_DIR = Path(__file__).parent / "archive" / "f1-forensics"
SEEDS = list(range(10))
CASES = 1024
T_CRIT = 2.262  # t_{0.975, 9} for the 10-seed paired interval (df = 9)

# Stage-1 IDN hidden-statics lr, as archived in
# archive/stage1-killed/stage1_idn.json ("selected": {"idn/hidden": lr}).
FFRES_LR = 0.001
FFRES_EPOCHS = 25
FFRES_HIDDEN = 32  # MLP(33->32->16) = 1,616 params, within 1,600 +/- 20%

METRICS = ("settle_auc", "duration_mae_days", "recovery_mae_log", "ece",
           "next_type_acc", "next_gap_pinball", "duration_pinball")

# State partition slices in the head input [z_event|z_clock|z_context|static].
PARTITIONS = {
    "event": (0, 16),
    "clock": (16, 32),
    "context": (32, 64),
    "static": (64, 64 + STATIC_DIM),
}

#: Event classes under selective observation in Generator v2 (gen_v2.py
#: _OBSERVATION_RATES); every other class is always docketed.
SELECTIVE_EVENT_IDS = {
    int(EventType.DEPOSITION),
    int(EventType.EXPERT_DISCLOSURE),
    int(EventType.MOTION_TO_COMPEL),
    int(EventType.SETTLEMENT_OFFER),
}

_QUANTILE_LEVELS = (0.1, 0.5, 0.9)


# --------------------------------------------------------------------- #
# Probe vehicle: subclass of the archived IDN (archived file untouched)
# --------------------------------------------------------------------- #

class ProbeIDN(IDNModel):
    """IDN with inference-time probe hooks. With default options the forward
    is operation-for-operation identical to the archived IDNModel (verified
    by an exact-parity assertion in main)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.probe_zero_head: frozenset = frozenset()   # zero at head input
        self.probe_zero_rec: frozenset = frozenset()    # zero in recurrence too
        self.probe_gate_clamp: float | None = None      # g := clamp
        self.probe_no_flow: bool = False                # alpha := 0
        self.probe_const_clock: torch.Tensor | None = None  # (d_clock,)
        self.ffres_mlp: nn.Module | None = None         # probe-5 replacement

    def forward(self, event_ids, event_feats, static, timespans=None, hx=None,
                lengths=None, return_intermediates=False):
        B, T = event_ids.shape
        device = event_ids.device
        if timespans is None:
            timespans = torch.ones(B, T, device=device, dtype=event_feats.dtype)
        if timespans.dim() == 3:
            timespans = timespans.squeeze(-1)
        deltas = timespans.to(event_feats.dtype) / self.time_scale
        if lengths is None:
            lengths = torch.full((B,), T, device=device, dtype=torch.long)
        lengths = lengths.to(device).clamp(min=0, max=T)

        contexts = self._contexts(event_ids, event_feats, static, deltas, lengths)

        z = torch.zeros(B, self.state_size, device=device, dtype=event_feats.dtype) if hx is None else hx
        z_event, z_clock, z_context = (
            z[:, : self.d_event],
            z[:, self.d_event : self.d_event + self.d_clock],
            z[:, self.d_event + self.d_clock :],
        )

        outs, gates, alphas, rates = [], [], [], []
        for k in range(T):
            c_prev = contexts[:, k - 1] if k > 0 else self.c_null.expand(B, -1)
            dtk = deltas[:, k].clamp(min=0)

            # ---- elapsed-time flow (pre-event only), with probe hooks
            rate = self.softplus(self.flow_rate(c_prev))
            if self.probe_no_flow:
                alpha = torch.zeros_like(rate)
                flowed = z_clock
            else:
                alpha = 1.0 - torch.exp(-rate * dtk.unsqueeze(-1))
                target = torch.tanh(self.flow_target(c_prev))
                flowed = (1.0 - alpha) * z_clock + alpha * target
            gate_in = torch.cat([c_prev, torch.log1p(dtk).unsqueeze(-1)], dim=-1)
            g = torch.sigmoid(self.flow_gate(gate_in))
            if self.probe_gate_clamp is not None:
                g = torch.full_like(g, float(self.probe_gate_clamp))
            z_clock_minus = g * flowed + (1.0 - g) * z_clock

            # ---- event-conditioned jump (unchanged)
            emb_k = self.event_embedding(event_ids[:, k])
            z_event_new = self.jump_event(torch.cat([emb_k, c_prev], dim=-1), z_event)
            blend = torch.sigmoid(self.context_blend(contexts[:, k]))
            z_context_new = self.context_norm(
                (1.0 - blend) * z_context + blend * contexts[:, k]
            )

            # ---- padding freeze
            active = (k < lengths).unsqueeze(-1)
            z_event = torch.where(active, z_event_new, z_event)
            z_clock = torch.where(active, z_clock_minus, z_clock)
            z_context = torch.where(active, z_context_new, z_context)

            # ---- hard recurrence zeroing (state itself is zeroed)
            if "event" in self.probe_zero_rec:
                z_event = torch.zeros_like(z_event)
            if "clock" in self.probe_zero_rec:
                z_clock = torch.zeros_like(z_clock)
            if "context" in self.probe_zero_rec:
                z_context = torch.zeros_like(z_context)

            # ---- head-input partition selection
            h_event = (torch.zeros_like(z_event)
                       if "event" in self.probe_zero_head else z_event)
            if self.ffres_mlp is not None:
                ffres_in = torch.cat([c_prev, torch.log1p(dtk).unsqueeze(-1)], dim=-1)
                h_clock = self.ffres_mlp(ffres_in)
            elif self.probe_const_clock is not None:
                h_clock = self.probe_const_clock.to(device, z_clock.dtype).expand(B, -1)
            elif "clock" in self.probe_zero_head:
                h_clock = torch.zeros_like(z_clock)
            else:
                h_clock = z_clock
            h_context = (torch.zeros_like(z_context)
                         if "context" in self.probe_zero_head else z_context)

            outs.append(torch.cat([h_event, h_clock, h_context], dim=-1))
            gates.append(g)
            alphas.append(alpha)
            rates.append(rate)

        Z = torch.stack(outs, dim=1)
        head_in = torch.cat(
            [Z, static.unsqueeze(1).expand(B, T, static.shape[-1])], dim=-1
        )
        result = {
            "settle_logit": self.head_settle(head_in).squeeze(-1),
            "log_recovery": self.head_recovery(head_in).squeeze(-1),
            "log_remaining": self.head_remaining(head_in).squeeze(-1),
            "next_type_logit": self.head_next_type(head_in),
            "next_gap_q": self._non_crossing(self.head_next_gap_q(head_in)),
            "duration_q": self._non_crossing(self.head_duration_q(head_in)),
            "gate_clock": torch.stack(gates, dim=1),
            "hx": torch.cat([z_event, z_clock, z_context], dim=-1),
            "rnn_out": Z,
        }
        if return_intermediates:
            result["alpha_clock"] = torch.stack(alphas, dim=1)
            result["rate_clock"] = torch.stack(rates, dim=1)
            result["contexts"] = contexts
        return result


def make_ffres_mlp() -> nn.Module:
    """MLP(c_{k-1}, log1p dt) -> d_clock; 1,616 params (target 1,600 +/- 20%)."""
    return nn.Sequential(
        nn.Linear(33, FFRES_HIDDEN), nn.ReLU(), nn.Linear(FFRES_HIDDEN, 16)
    )


# --------------------------------------------------------------------- #
# Metrics (frozen definitions; ece copied verbatim from run_stage1.py)
# --------------------------------------------------------------------- #

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bucket = np.digitize(probs, edges[1:-1])
    val = 0.0
    for b in range(n_bins):
        m = bucket == b
        if m.any():
            val += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(val)


def pinball_np(pred_q: np.ndarray, target: np.ndarray) -> float:
    """Mean pinball loss over quantile levels, mirroring train._pinball."""
    e = target[..., None] - pred_q
    vals = []
    for i, q in enumerate(_QUANTILE_LEVELS):
        vals.append(np.maximum(q * e[..., i], (q - 1.0) * e[..., i]))
    return float(np.mean(vals))


def paired_ci(deltas: list[float]) -> dict:
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n < 2:
        return {"mean": float(d.mean()) if n else None, "ci95": None, "se": None, "n": n}
    se = float(d.std(ddof=1) / np.sqrt(n))
    return {"mean": float(d.mean()), "ci95": T_CRIT * se, "se": se, "n": n}


# --------------------------------------------------------------------- #
# Per-step collection
# --------------------------------------------------------------------- #

def _accepts(model, name: str) -> bool:
    return name in inspect.signature(model.forward).parameters


@torch.no_grad()
def collect(model, val, cfg, transform=None, intermediates=False):
    """Batched masked per-step outputs on the holdout.

    ``transform``: optional fn(deltas_np (B,T), lengths_np (B,)) -> deltas_np,
    applied to the model INPUT only (labels/meta stay on true times).

    Returns {"main": {...arrays over valid steps...},
             "next": {...arrays over steps with a successor...},
             "meta": {...}, "inter": {...optional gate/alpha/rate...}}.
    """
    model.eval()
    main = defaultdict(list)
    nxt = defaultdict(list)
    meta = defaultdict(list)
    inter = defaultdict(list)
    for start in range(0, len(val), 256):
        chunk = val[start : start + 256]
        batch = collate_timelines(chunk, horizon_days=cfg.horizon_days)
        deltas_in = batch.deltas
        if transform is not None:
            lengths_np = batch.mask.sum(1).numpy().astype(np.int64)
            deltas_in = torch.from_numpy(
                transform(batch.deltas.numpy().copy(), lengths_np)
            ).to(batch.deltas.dtype)
        kwargs = {}
        if _accepts(model, "lengths"):
            kwargs["lengths"] = batch.mask.sum(1).long()
        if _accepts(model, "return_intermediates"):
            kwargs["return_intermediates"] = intermediates
        out = model(batch.event_ids, batch.event_feats, batch.static,
                    timespans=deltas_in, **kwargs)

        m = batch.mask.bool()
        nm = batch.mask[:, 1:].bool()
        days = batch.deltas.cumsum(dim=1)
        B, T = batch.event_ids.shape
        lengths = batch.mask.sum(1)
        pos = torch.arange(T).float().unsqueeze(0) / lengths.clamp(min=1).unsqueeze(1)
        case_idx = torch.arange(start, start + B).unsqueeze(1).expand(B, T)

        main["settle_logit"].append(out["settle_logit"][m].numpy())
        main["y_settle"].append(batch.y_settle[m].numpy())
        main["log_recovery"].append(out["log_recovery"][m].numpy())
        main["y_recovery"].append(batch.y_recovery[m].numpy())
        main["log_remaining"].append(out["log_remaining"][m].numpy())
        main["y_remaining"].append(batch.y_remaining[m].numpy())
        main["duration_q"].append(out["duration_q"][m].numpy())
        meta["case_idx"].append(case_idx[m].numpy())
        meta["delta"].append(batch.deltas[m].numpy())
        meta["day"].append(days[m].numpy())
        meta["pos_frac"].append(pos[m].numpy())
        meta["length"].append(lengths.unsqueeze(1).expand(B, T)[m].numpy())
        meta["event_id"].append(batch.event_ids[m].numpy())

        nxt["next_type_pred"].append(out["next_type_logit"][:, :-1].argmax(-1)[nm].numpy())
        nxt["next_type_true"].append(batch.event_ids[:, 1:][nm].numpy())
        nxt["next_gap_q"].append(out["next_gap_q"][:, :-1][nm].numpy())
        nxt["next_gap_target"].append(
            torch.log1p(batch.deltas[:, 1:].clamp(min=0))[nm].numpy())

        if intermediates:
            for key in ("gate_clock", "alpha_clock", "rate_clock"):
                flat = out[key][m]  # (n_steps, d_clock)
                inter[key].append(flat.numpy())

    pack = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    return {"main": pack(main), "next": pack(nxt), "meta": pack(meta),
            "inter": pack(inter) if inter else {}}


def metric_set(coll: dict) -> dict:
    main, nxt = coll["main"], coll["next"]
    probs = 1.0 / (1.0 + np.exp(-main["settle_logit"]))
    return {
        "settle_auc": auc_score(main["y_settle"], main["settle_logit"]),
        "duration_mae_days": mae_days(main["log_remaining"], main["y_remaining"]),
        "recovery_mae_log": mae_log_dollars(main["log_recovery"], main["y_recovery"]),
        "ece": ece(probs, main["y_settle"]),
        "next_type_acc": float((nxt["next_type_pred"] == nxt["next_type_true"]).mean()),
        "next_gap_pinball": pinball_np(nxt["next_gap_q"], nxt["next_gap_target"]),
        "duration_pinball": pinball_np(main["duration_q"], main["y_remaining"]),
    }


# --------------------------------------------------------------------- #
# Probe-4 timespan transforms (input-side, inference only)
# --------------------------------------------------------------------- #

def make_transforms(train_deltas: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)  # seeded per spec (seed = run seed)
    global_median = float(np.median(train_deltas))

    def _tercile_groups(length: int) -> list[np.ndarray]:
        idx = np.arange(length)
        frac = idx / max(length, 1)
        return [idx[frac <= 1 / 3], idx[(frac > 1 / 3) & (frac <= 2 / 3)],
                idx[frac > 2 / 3]]

    def shuffle_within_phase(deltas: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        out = deltas.copy()
        for b, L in enumerate(lengths):
            for grp in _tercile_groups(int(L)):
                if len(grp) > 1:
                    out[b, grp] = deltas[b, grp[rng.permutation(len(grp))]]
        return out

    def phase_median(deltas: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        out = deltas.copy()
        for b, L in enumerate(lengths):
            for grp in _tercile_groups(int(L)):
                if len(grp):
                    out[b, grp] = float(np.median(deltas[b, grp]))
        return out

    def global_median_tf(deltas: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        return np.full_like(deltas, global_median)

    def dt_zero(deltas: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        return np.zeros_like(deltas)

    def dt_log(deltas: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        return np.log1p(np.clip(deltas, 0.0, None))

    return {
        "dt_shuffle_phase": shuffle_within_phase,
        "dt_phase_median": phase_median,
        "dt_global_median": global_median_tf,
        "dt_zero": dt_zero,
        "dt_log": dt_log,
    }


# --------------------------------------------------------------------- #
# Probe 5: idn-ffres (new exploratory model)
# --------------------------------------------------------------------- #

def train_ffres(model: ProbeIDN, train_set, cfg: TrainConfig) -> list[float]:
    """Freeze every archived weight; train ONLY the ffres MLP with the frozen
    multi-task loss and the Stage-1 25-epoch schedule (no validation-based
    selection; head biases are archived parameters and are not touched)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.ffres_mlp.parameters():
        p.requires_grad_(True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.Adam(model.ffres_mlp.parameters(), lr=cfg.lr)
    losses = []
    for _epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(train_set))
        total, nb = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            chunk = [train_set[i] for i in order[start : start + cfg.batch_size]]
            batch = collate_timelines(chunk, horizon_days=cfg.horizon_days)
            loss = train_mod._loss(model, batch, cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.ffres_mlp.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss.detach())
            nb += 1
        losses.append(total / max(nb, 1))
    model.eval()
    return losses


# --------------------------------------------------------------------- #
# Probe 6: head dependence
# --------------------------------------------------------------------- #

HEADS = ("head_settle", "head_recovery", "head_remaining",
         "head_next_type", "head_next_gap_q", "head_duration_q")


def head_weight_norms(model: ProbeIDN) -> dict:
    """Per-head input-weight column norms grouped by partition."""
    out = {}
    for hname in HEADS:
        W = getattr(model, hname).weight.detach()  # (out, in)
        norms = W.norm(dim=0)  # column norms (in,)
        out[hname] = {
            part: {
                "column_norm_sum": float(norms[a:b].sum()),
                "column_norm_mean": float(norms[a:b].mean()),
            }
            for part, (a, b) in PARTITIONS.items()
        }
    return out


def state_gradient_norms(model: ProbeIDN, val, cfg: TrainConfig) -> dict:
    """Gradient norms of the frozen masked holdout loss w.r.t. each state
    partition of rnn_out, accumulated over evaluation batches."""
    model.eval()
    ss = {p: 0.0 for p in ("event", "clock", "context")}
    n_steps = 0
    for start in range(0, len(val), 64):
        batch = collate_timelines(val[start : start + 64],
                                  horizon_days=cfg.horizon_days)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = model(batch.event_ids, batch.event_feats, batch.static,
                        timespans=batch.deltas,
                        lengths=batch.mask.sum(1).long())
            bce = nn.functional.binary_cross_entropy_with_logits(
                out["settle_logit"], batch.y_settle, reduction="none")
            mse_rec = (out["log_recovery"] - batch.y_recovery) ** 2
            mse_rem = (out["log_remaining"] - batch.y_remaining) ** 2
            loss = (
                cfg.weight_settle * train_mod._masked_mean(bce, batch.mask)
                + cfg.weight_recovery * train_mod._masked_mean(mse_rec, batch.mask)
                + cfg.weight_duration * train_mod._masked_mean(mse_rem, batch.mask)
                + train_mod._aux_losses(out, batch, cfg)
            )
            grad = torch.autograd.grad(loss, out["rnn_out"])[0]
        m = batch.mask.bool()
        for part, (a, b) in PARTITIONS.items():
            if part == "static":
                continue
            ss[part] += float((grad[..., a:b][m] ** 2).sum())
        n_steps += int(m.sum())
    return {
        part: {
            "accumulated_l2": float(np.sqrt(ss[part])),
            "rms_per_step_dim": float(np.sqrt(ss[part] / (max(n_steps, 1) * (b - a)))),
        }
        for part, (a, b) in PARTITIONS.items() if part != "static"
    }


# --------------------------------------------------------------------- #
# Probe 7: stratified paired evaluation
# --------------------------------------------------------------------- #

def backlog_state_at(log: list, day: float) -> bool:
    """True if ``day`` falls inside a logged judge-backlog episode."""
    state = log[0][1]
    for d, s in log:
        if d <= day + 1e-9:
            state = s
        else:
            break
    return state == "backlog"


def build_strata(coll: dict, val, latents: dict) -> dict:
    """Stratum masks (bool arrays over holdout steps) per family."""
    meta = coll["meta"]
    d, age, pos = meta["delta"], meta["day"], meta["pos_frac"]
    lengths = meta["length"].astype(int)
    q = np.quantile(d, [0.25, 0.5, 0.75])
    qa = np.quantile(age, [0.25, 0.5, 0.75])
    med_len = float(np.median(lengths))

    case_idx = meta["case_idx"].astype(int)
    event_id = meta["event_id"].astype(int)
    backlog = np.zeros(len(d), dtype=bool)
    adverse = np.zeros(len(d), dtype=bool)
    stall = np.zeros(len(d), dtype=bool)
    for i, (ci, day) in enumerate(zip(case_idx, meta["day"])):
        tl = val[ci]
        lat = latents[tl.case_id]
        backlog[i] = backlog_state_at(lat["judge_backlog"], float(day))
        flip = lat["regime_flip_day"]
        adverse[i] = flip is not None and day >= flip - 1e-9
        stall[i] = float(tl.outcome.get("n_stalls", 0.0)) >= 1.0
    selective = np.isin(event_id, list(SELECTIVE_EVENT_IDS))

    return {
        "delta_quartile": {
            "q1": d <= q[0], "q2": (d > q[0]) & (d <= q[1]),
            "q3": (d > q[1]) & (d <= q[2]), "q4": d > q[2]},
        "long_gap": {"le_90d": d <= 90.0, "gt_90d": d > 90.0},
        "age_quartile": {
            "q1": age <= qa[0], "q2": (age > qa[0]) & (age <= qa[1]),
            "q3": (age > qa[1]) & (age <= qa[2]), "q4": age > qa[2]},
        "phase": {"early": pos <= 0.33,
                  "mid": (pos > 0.33) & (pos <= 0.66),
                  "late": pos > 0.66},
        "seq_length": {"short": lengths <= med_len, "long": lengths > med_len},
        "backlog": {"in_episode": backlog, "outside": ~backlog},
        "adverse_regime": {"adverse": adverse, "not_adverse": ~adverse},
        "selective_obs": {"selective": selective, "always": ~selective},
        "discovery_stall": {"stall_case": stall, "no_stall": ~stall},
    }


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm-adjusted p-values (descriptive aid only)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def ttest_rel_p(deltas: list[float]) -> float | None:
    """Two-sided paired t-test p-value for deltas vs zero (descriptive)."""
    from scipy import stats
    d = np.asarray(deltas, dtype=float)
    if len(d) < 2 or float(d.std(ddof=1)) == 0.0:
        return None
    p = float(stats.ttest_1samp(d, 0.0).pvalue)
    return None if np.isnan(p) else p


# --------------------------------------------------------------------- #
# Main battery
# --------------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()
    torch.manual_seed(0)
    cfg = TrainConfig(verbose=False)

    results: dict = {
        "status": "EXPLORATORY — Stage-1 kill stands regardless of any F1 finding",
        "spec": "F1_FORENSICS_SPEC.md (frozen 2026-07-25)",
        "code_snapshot_diff": {
            "idn_model.py": "identical (diff experiments/idn_model.py vs "
                            "archive/stage1-killed/code/idn_model.py: no output)",
            "train.py": "identical (src/liquid_legal/train.py vs archived snapshot)",
            "baselines.py": "identical (src/liquid_legal/baselines.py vs archived snapshot)",
        },
        "config": {
            "seeds": SEEDS, "cases": CASES, "regime": "hidden statics",
            "holdout": "20% via np.random.default_rng(seed).permutation (Stage-1 split)",
            "metrics": list(METRICS),
            "ece_bins": 15, "quantile_levels": list(_QUANTILE_LEVELS),
            "paired_interval": "mean +/- 95% t-interval, df=9 (t=2.262)",
            "ffres": {"hidden": FFRES_HIDDEN, "params": None,
                      "param_target": [1280, 1920], "lr": FFRES_LR,
                      "epochs": FFRES_EPOCHS,
                      "note": "new exploratory model; archived weights frozen"},
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "uname": subprocess.run(["uname", "-a"], capture_output=True,
                                    text=True).stdout.strip(),
            "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "parity_check": {},
        "baseline": {},
        "probes": {},
    }

    raw: dict = {"per_seed": {}}
    npz_store: dict = {}

    for seed in SEEDS:
        gen = GeneratorV2(GeneratorV2Config(seed=seed))
        tls, latents = gen.generate_with_latents(CASES)
        data = strip_statics(tls)
        train, val = split(data, seed)
        scfg = TrainConfig(verbose=False, seed=seed)
        seed_raw: dict = {"conditions": {}}

        # ---- archived IDN (probe vehicle) ---- #
        model = ProbeIDN()
        model.load_state_dict(torch.load(
            ARCHIVE / "weights" / f"idn_hidden_seed{seed}.pt", weights_only=True))
        model.eval()

        # ---- exact-parity assertion: ProbeIDN == archived IDNModel ---- #
        ref = IDNModel()
        ref.load_state_dict(torch.load(
            ARCHIVE / "weights" / f"idn_hidden_seed{seed}.pt", weights_only=True))
        ref.eval()
        b = collate_timelines(val[:64], horizon_days=cfg.horizon_days)
        kw = {"lengths": b.mask.sum(1).long()}
        with torch.no_grad():
            o1 = ref(b.event_ids, b.event_feats, b.static, timespans=b.deltas, **kw)
            o2 = model(b.event_ids, b.event_feats, b.static, timespans=b.deltas, **kw)
        for key in ("settle_logit", "log_recovery", "log_remaining",
                    "next_type_logit", "next_gap_q", "duration_q", "gate_clock"):
            assert torch.equal(o1[key], o2[key]), f"parity failure on {key} seed {seed}"
        if seed == 0:
            results["parity_check"]["probe_idn_vs_archived_forward"] = \
                "bitwise identical on seed-0 holdout batch (all head outputs)"

        # ---- baseline (probe 2 'observed' condition + beta logging) ---- #
        base = collect(model, val, scfg, intermediates=True)
        base_metrics = metric_set(base)
        seed_raw["conditions"]["baseline"] = base_metrics
        results["baseline"][str(seed)] = base_metrics

        # archived-eval parity (seed 0): reproduction_check style guard
        if seed == 0:
            arch = json.loads((ARCHIVE / "stage1_idn.json").read_text())
            ref_m = arch["runs"]["idn/hidden"][0]
            for k in ("settle_auc", "duration_mae_days", "recovery_mae_log",
                      "ece", "next_type_acc"):
                assert abs(base_metrics[k] - ref_m[k]) < 1e-6, \
                    f"archived eval parity failed on {k}: {base_metrics[k]} vs {ref_m[k]}"
            results["parity_check"]["archived_stage1_eval_seed0"] = \
                "all shared metrics match archive/stage1-killed/stage1_idn.json to <1e-6"

        # ---- probe 1: partition zeroing ---- #
        p1 = {}
        for part in ("event", "clock", "context"):
            for variant in ("head", "hard"):
                model.probe_zero_head = frozenset({part})
                model.probe_zero_rec = frozenset({part}) if variant == "hard" else frozenset()
                name = f"zero_{part}_{variant}"
                m = metric_set(collect(model, val, scfg))
                p1[name] = m
                seed_raw["conditions"][name] = m
        model.probe_zero_head = frozenset()
        model.probe_zero_rec = frozenset()
        results["probes"].setdefault("probe1_partition_zeroing", {})[str(seed)] = p1

        # ---- probe 2: gate clamps + effective-coefficient distribution ---- #
        p2 = {}
        for clamp in (0.0, 1.0):
            model.probe_gate_clamp = clamp
            name = f"gate_clamp_{clamp:.0f}"
            m = metric_set(collect(model, val, scfg))
            p2[name] = m
            seed_raw["conditions"][name] = m
        model.probe_gate_clamp = None
        p2["observed"] = base_metrics
        results["probes"].setdefault("probe2_gate_clamps", {})[str(seed)] = p2

        beta = base["inter"]["gate_clock"] * base["inter"]["alpha_clock"]
        dist = {}
        for label, arr in (("beta", beta), ("g", base["inter"]["gate_clock"]),
                           ("r", base["inter"]["rate_clock"])):
            flat = arr.ravel()
            dist[label] = {
                "mean": float(flat.mean()),
                "q05": float(np.quantile(flat, 0.05)),
                "q25": float(np.quantile(flat, 0.25)),
                "q50": float(np.quantile(flat, 0.50)),
                "q75": float(np.quantile(flat, 0.75)),
                "q95": float(np.quantile(flat, 0.95)),
            }
        results["probes"].setdefault("probe2_distributions", {})[str(seed)] = dist

        # ---- probe 3: clock identity and removal ---- #
        p3 = {}
        model.probe_no_flow = True
        p3["no_flow"] = metric_set(collect(model, val, scfg))
        model.probe_no_flow = False
        model.probe_zero_head = frozenset({"clock"})
        p3["heads_only"] = metric_set(collect(model, val, scfg))
        model.probe_zero_head = frozenset()
        # const: per-seed TRAINING-portion mean of z_clock (frozen once)
        zs, zn = np.zeros(16), 0
        with torch.no_grad():
            for start in range(0, len(train), 256):
                bt = collate_timelines(train[start : start + 256],
                                       horizon_days=cfg.horizon_days)
                out = model(bt.event_ids, bt.event_feats, bt.static,
                            timespans=bt.deltas, lengths=bt.mask.sum(1).long())
                mt = bt.mask.bool()
                zs += out["rnn_out"][mt][:, 16:32].sum(0).numpy()
                zn += int(mt.sum())
        const_vec = torch.tensor(zs / max(zn, 1), dtype=torch.float32)
        model.probe_const_clock = const_vec
        p3["const"] = metric_set(collect(model, val, scfg))
        model.probe_const_clock = None
        for name, m in p3.items():
            seed_raw["conditions"][f"clock_{name}"] = m
        results["probes"].setdefault("probe3_clock_identity", {})[str(seed)] = p3
        seed_raw["const_clock_vector"] = const_vec.tolist()

        # ---- probe 4: time perturbations ---- #
        train_deltas = np.concatenate(
            [featurize_timeline(t)["deltas"] for t in train])
        p4 = {}
        for name, tf_fn in make_transforms(train_deltas, seed).items():
            m = metric_set(collect(model, val, scfg, transform=tf_fn))
            p4[name] = m
            seed_raw["conditions"][name] = m
        results["probes"].setdefault("probe4_time_perturbations", {})[str(seed)] = p4

        # ---- probe 5: idn-ffres (new exploratory model) ---- #
        torch.manual_seed(seed)
        ffres = ProbeIDN()
        ffres.load_state_dict(torch.load(
            ARCHIVE / "weights" / f"idn_hidden_seed{seed}.pt", weights_only=True))
        ffres.ffres_mlp = make_ffres_mlp()
        n_mlp = sum(p.numel() for p in ffres.ffres_mlp.parameters())
        assert 1280 <= n_mlp <= 1920, f"ffres param count {n_mlp} outside band"
        fcfg = TrainConfig(verbose=False, seed=seed, lr=FFRES_LR,
                           epochs=FFRES_EPOCHS)
        losses = train_ffres(ffres, train, fcfg)
        ffres_coll = collect(ffres, val, scfg)
        ffres_metrics = metric_set(ffres_coll)
        results["probes"].setdefault("probe5_idn_ffres", {})[str(seed)] = {
            "metrics": ffres_metrics,
            "mlp_params": n_mlp,
            "train_loss_first": losses[0],
            "train_loss_last": losses[-1],
        }
        seed_raw["conditions"]["ffres"] = ffres_metrics
        results["config"]["ffres"]["params"] = n_mlp

        # ---- probe 6: head dependence ---- #
        p6 = {
            "head_weight_column_norms": head_weight_norms(model),
            "state_gradient_norms": state_gradient_norms(model, val, scfg),
        }
        results["probes"].setdefault("probe6_head_dependence", {})[str(seed)] = p6

        # ---- probe 7: stratified paired evaluation ---- #
        tf = TemporalTransformerModel(d_model=32, nhead=4, num_layers=2,
                                      dim_feedforward=64, max_len=128,
                                      time_mode="native", auxiliary=True)
        tf.load_state_dict(torch.load(
            ARCHIVE / "weights" / f"tf-native-aux_hidden_seed{seed}.pt",
            weights_only=True))
        tf_coll = collect(tf, val, scfg)
        seed_raw["conditions"]["tf_native_aux"] = metric_set(tf_coll)

        strata = build_strata(base, val, latents)
        p7 = results["probes"].setdefault("probe7_stratified", {})
        for family, masks in strata.items():
            fam = p7.setdefault(family, {})
            for sname, mask in masks.items():
                y = base["main"]["y_settle"][mask]
                entry = fam.setdefault(sname, {
                    "steps": [], "cases": [], "positives": [],
                    "auc_idn": [], "auc_tf": [], "delta": []})
                entry["steps"].append(int(mask.sum()))
                entry["cases"].append(int(len(np.unique(base["meta"]["case_idx"][mask]))))
                entry["positives"].append(int(y.sum()))
                if mask.sum() >= 100 and len(np.unique(y)) >= 2:
                    a_idn = auc_score(y, base["main"]["settle_logit"][mask])
                    a_tf = auc_score(y, tf_coll["main"]["settle_logit"][mask])
                    entry["auc_idn"].append(a_idn)
                    entry["auc_tf"].append(a_tf)
                    entry["delta"].append(a_idn - a_tf)

        # per-step raw outputs (settle logits) for the archive
        for cname, coll in (("baseline", base), ("ffres", ffres_coll),
                            ("tf_native_aux", tf_coll)):
            npz_store[f"seed{seed}/{cname}/settle_logit"] = coll["main"]["settle_logit"]
            npz_store[f"seed{seed}/{cname}/y_settle"] = coll["main"]["y_settle"]
        for key, arr in base["meta"].items():
            npz_store[f"seed{seed}/meta/{key}"] = arr

        raw["per_seed"][str(seed)] = seed_raw
        print(f"seed={seed} done ({time.time() - t0:.0f}s)", flush=True)

    # ----------------------------------------------------------------- #
    # aggregation: paired deltas vs archived IDN baseline
    # ----------------------------------------------------------------- #
    def _agg_condition(get_metrics) -> dict:
        """get_metrics(seed) -> metric dict; returns value + paired delta."""
        deltas = {k: [] for k in METRICS}
        vals = {k: [] for k in METRICS}
        for seed in SEEDS:
            m = get_metrics(str(seed))
            b = results["baseline"][str(seed)]
            for k in METRICS:
                vals[k].append(m[k])
                deltas[k].append(m[k] - b[k])
        return {
            "value": {k: paired_ci(vals[k]) for k in METRICS},
            "delta_vs_archived_idn": {k: paired_ci(deltas[k]) for k in METRICS},
        }

    def aggregate(probe_key: str) -> dict:
        per_seed = results["probes"][probe_key]
        conds = sorted(next(iter(per_seed.values())).keys())
        return {cond: _agg_condition(lambda s, c=cond: per_seed[s][c])
                for cond in conds}

    summary_probes = {}
    for key in ("probe1_partition_zeroing", "probe2_gate_clamps",
                "probe3_clock_identity", "probe4_time_perturbations"):
        summary_probes[key] = aggregate(key)
    p5 = results["probes"]["probe5_idn_ffres"]
    summary_probes["probe5_idn_ffres"] = {
        "ffres": _agg_condition(lambda s: p5[s]["metrics"]),
        "mlp_params": p5["0"]["mlp_params"],
        "train_loss_last_mean": float(np.mean(
            [p5[str(s)]["train_loss_last"] for s in SEEDS])),
    }

    # probe 2 distributions pooled over seeds
    dists = results["probes"]["probe2_distributions"]
    summary_probes["probe2_distributions"] = {
        label: {stat: float(np.mean([dists[str(s)][label][stat] for s in SEEDS]))
                for stat in ("mean", "q05", "q25", "q50", "q75", "q95")}
        for label in ("beta", "g", "r")
    }

    # probe 6 aggregated over seeds
    p6s = results["probes"]["probe6_head_dependence"]
    agg6 = {"head_weight_column_norms": {}, "state_gradient_norms": {}}
    for hname in HEADS:
        agg6["head_weight_column_norms"][hname] = {
            part: {
                stat: float(np.mean([
                    p6s[str(s)]["head_weight_column_norms"][hname][part][stat]
                    for s in SEEDS]))
                for stat in ("column_norm_sum", "column_norm_mean")
            } for part in PARTITIONS
        }
    for part in ("event", "clock", "context"):
        agg6["state_gradient_norms"][part] = {
            stat: float(np.mean([
                p6s[str(s)]["state_gradient_norms"][part][stat] for s in SEEDS]))
            for stat in ("accumulated_l2", "rms_per_step_dim")
        }
    summary_probes["probe6_head_dependence"] = agg6

    # probe 7 aggregated with Holm within each stratum family
    p7s = results["probes"]["probe7_stratified"]
    agg7 = {}
    for family, strata_d in p7s.items():
        fam_out = {}
        for sname, e in strata_d.items():
            ci = paired_ci(e["delta"])
            p = ttest_rel_p(e["delta"])
            fam_out[sname] = {
                "steps_total": int(np.sum(e["steps"])),
                "cases_total": int(np.sum(e["cases"])),
                "positives_total": int(np.sum(e["positives"])),
                "seeds_used": len(e["delta"]),
                "mean_auc_idn": float(np.mean(e["auc_idn"])) if e["auc_idn"] else None,
                "mean_auc_tf": float(np.mean(e["auc_tf"])) if e["auc_tf"] else None,
                "delta": ci,
                "p_value": p,
            }
        pvals = [fam_out[s]["p_value"] for s in fam_out]
        ok = [not (p is None or (isinstance(p, float) and np.isnan(p))) for p in pvals]
        names = list(fam_out)
        adj = holm_adjust([p for p, k in zip(pvals, ok) if k]) if any(ok) else []
        it = iter(adj)
        for sname, p, k in zip(names, pvals, ok):
            fam_out[sname]["holm_adj_p_descriptive"] = next(it) if k else None
            fam_out[sname]["holm_reject_05_descriptive"] = (
                bool(fam_out[sname]["holm_adj_p_descriptive"] < 0.05) if k else None)
        agg7[family] = fam_out
    summary_probes["probe7_stratified"] = agg7

    results["summary"] = summary_probes
    results["baseline_mean"] = {
        k: paired_ci([results["baseline"][str(s)][k] for s in SEEDS])
        for k in METRICS
    }
    results["runtime_seconds"] = time.time() - t0

    # ----------------------------------------------------------------- #
    # outputs
    # ----------------------------------------------------------------- #
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT_JSON} ({time.time() - t0:.0f}s)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "config.json").write_text(json.dumps(results["config"], indent=2))
    (OUT_DIR / "environment.json").write_text(json.dumps(results["environment"], indent=2))
    (OUT_DIR / "raw_metrics.json").write_text(json.dumps(raw, indent=2))
    np.savez_compressed(OUT_DIR / "per_step_outputs.npz", **npz_store)
    shutil.copy2(Path(__file__), OUT_DIR / "f1_forensics.py")

    manifest = {}
    for f in sorted(OUT_DIR.iterdir()):
        if f.name == "hashes.json":
            continue
        manifest[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    (OUT_DIR / "hashes.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT_DIR} ({time.time() - t0:.0f}s)", flush=True)

    # ----------------------------------------------------------------- #
    # console digest
    # ----------------------------------------------------------------- #
    print("\n=== F1 digest (EXPLORATORY; Stage-1 kill stands) ===")
    for probe, conds in summary_probes.items():
        if probe in ("probe2_distributions", "probe6_head_dependence",
                     "probe7_stratified"):
            continue
        print(f"\n{probe}")
        for cond, agg in conds.items():
            if not isinstance(agg, dict) or "delta_vs_archived_idn" not in agg:
                continue
            d = agg["delta_vs_archived_idn"]["settle_auc"]
            dd = agg["delta_vs_archived_idn"]["duration_mae_days"]
            if d["ci95"] is None:
                continue
            print(f"  {cond:22s} dAUC={d['mean']:+.4f}+/-{d['ci95']:.4f} "
                  f"dDurMAE={dd['mean']:+.1f}+/-{dd['ci95']:.1f}d")
    b = summary_probes["probe2_distributions"]["beta"]
    print(f"\nbeta_k = g*alpha: mean={b['mean']:.3f} "
          f"q05={b['q05']:.3f} q50={b['q50']:.3f} q95={b['q95']:.3f}")


if __name__ == "__main__":
    main()
