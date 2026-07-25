"""F2 v1 — train and score `tf-tpp` under the frozen F2 preregistration.

Implements experiments/F2_PREREGISTRATION.md exactly:

* Candidate: `tf-tpp` (tpp_model.TppTransformerModel) — the exact
  tf-native-aux trunk with per-mark intensity heads in place of the
  next-type / next-gap aux heads. Loss (frozen): 1.0*settle BCE +
  0.3*recovery log1p-MSE + 0.3*duration log1p-MSE + 0.2*duration pinball +
  0.3*TPP NLL (closed-form marked likelihood over intervals with a next
  observed event).
* Protocol: identical to STAGE1_SPEC.md — 1024 cases/seed, seeds 0-9, 20%
  holdout via np.random.default_rng(seed).permutation, Adam (0.9, 0.999),
  batch 32, grad clip 1.0, 25 epochs, last-epoch model, lr grid
  {1e-3, 3e-3} selected on an inner split of seed-0's TRAIN portion only
  (per regime), torch.manual_seed(seed) before construction, CPU.
  Primary regime: hidden statics; visible statics secondary/descriptive.
* Opponent: archived tf-native-aux weights (archive/stage1-killed/),
  re-scored here under the identical J convention as
  f2_score_calibration.py (masked steps with a next valid event; log1p-day
  quantile scale) — the calibration file is not reused for the comparison;
  recomputed settle AUC is cross-checked against stage1_baselines.json of
  record.
* Screen (section 4): mean paired relative J reduction >= 3%, 95% paired
  t-interval (df=9) excludes zero, and all guards (settle AUC >= -0.005,
  ECE <= +0.01, duration MAE <= +5%, next-type acc >= -0.01, wall-clock
  <= 2x archived tf-native-aux mean, zero NaN/Inf events and all seeds
  completing). Anything short: killed candidate.
* Generator validation (section 6, architecture-independent, run first):
  per-mark gap quantiles on 3 probe seeds x 1024 cases; silence probe
  (elapsed-context lift, paired over the 3 probe seeds, CI, no pass/fail).
* Subgroups (section 8, exploratory): F1 probe-7 strata applied to the
  paired J comparison, Holm within families.
* Mechanism checks M1/M2 (section 7): ONLY if the screen is survived.

NaN/Inf policy: any non-finite loss aborts that run and is recorded as a
stability event (no silent restart, no tweaks); it fails the stability
guard.

Note: the engineered feature builder mirrors xgb_baseline.prefix_features
(the same columns) but is re-implemented here because importing xgboost in
the same process as torch segfaults on macOS (two OpenMP runtimes) — see
run_all.py. No xgboost model is used anywhere in F2.

Outputs:
  experiments/results/f2_tpp.json        (machine-readable results)
  experiments/archive/f2-tpp-v1/         (weights, raw predictions, log,
                                          code snapshot, environment,
                                          SHA-256 manifest)

Run with: cd experiments && ../.venv/bin/python run_f2_tpp.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from liquid_legal import TrainConfig, evaluate
from liquid_legal import train as train_mod  # frozen helpers (_masked_mean, _pinball)
from liquid_legal.baselines import TemporalTransformerModel
from liquid_legal.events import (
    N_EVENT_TYPES,
    STATIC_FIELDS,
    CaseTimeline,
    EventType,
)
from liquid_legal.featurize import collate_timelines
from liquid_legal.metrics import auc_score, mae_days
from gen_v2 import GeneratorV2, GeneratorV2Config
from run_all import split
from run_hidden_statics import strip_statics
from tpp_model import QUANTILE_LEVELS, TppTransformerModel

SEEDS = list(range(10))
CASES = 1024
EPOCHS = 25
LR_GRID = [1e-3, 3e-3]
REGIMES = ("hidden", "visible")  # hidden statics primary (listed first)
PROBE_SEEDS = [0, 1, 2]  # generator validation (section 6), probe convention
QS = QUANTILE_LEVELS
T_CRIT = 2.262  # t_{0.975, 9}
T_CRIT_DF2 = 4.303  # t_{0.975, 2} (3 paired probe seeds)
TPP_NLL_WEIGHT = 0.3  # frozen loss weight (prereg section 2)
J_MARGIN = 0.03  # frozen practical margin (twice the noise band)

ARCHIVE_S1 = Path(__file__).parent / "archive" / "stage1-killed"
OUT_JSON = Path(__file__).parent / "results" / "f2_tpp.json"
OUT_DIR = Path(__file__).parent / "archive" / "f2-tpp-v1"

#: Event classes under selective observation in Generator v2 (gen_v2.py).
SELECTIVE_EVENT_IDS = {
    int(EventType.DEPOSITION),
    int(EventType.EXPERT_DISCLOSURE),
    int(EventType.MOTION_TO_COMPEL),
    int(EventType.SETTLEMENT_OFFER),
}
TERMINAL_IDS = (int(EventType.SETTLED), int(EventType.DISMISSED), int(EventType.VERDICT))

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


# --------------------------------------------------------------------- #
# Engineered features — mirrors xgb_baseline.prefix_features (same
# columns) without importing xgboost (macOS OpenMP conflict with torch).
# --------------------------------------------------------------------- #

ENGINEERED = [
    "n_events", "day", "last_delta", "mean_delta", "max_delta",
    "n_offers", "n_compels", "n_mtd", "n_msj", "n_mediations",
    "n_trial_dates", "log_total_offer_amount",
]
TIME_FEATURES = {"day", "last_delta", "mean_delta", "max_delta"}
_TRACKED = {
    "n_offers": EventType.SETTLEMENT_OFFER,
    "n_compels": EventType.MOTION_TO_COMPEL,
    "n_mtd": EventType.MOTION_TO_DISMISS,
    "n_msj": EventType.MOTION_SUMMARY_JUDGMENT,
    "n_mediations": EventType.MEDIATION,
    "n_trial_dates": EventType.TRIAL_DATE_SET,
}
N_FEATURES = len(STATIC_FIELDS) + len(ENGINEERED) + N_EVENT_TYPES
_TIME_COL_IDX = [len(STATIC_FIELDS) + ENGINEERED.index(f) for f in TIME_FEATURES]


def prefix_features(tl: CaseTimeline) -> np.ndarray:
    """One feature row per timestep, built from the prefix up to that step
    (identical columns to xgb_baseline.prefix_features)."""
    static = [tl.static.get(k, 0.0) for k in STATIC_FIELDS]
    counts = {k: 0 for k in _TRACKED}
    rows = np.zeros((len(tl.events), N_FEATURES), dtype=np.float32)
    prev_day, max_delta, offer_sum = 0.0, 0.0, 0.0
    for i, ev in enumerate(tl.events):
        delta = max(0.0, ev.day - prev_day)
        prev_day = ev.day
        max_delta = max(max_delta, delta)
        if ev.event_type is EventType.SETTLEMENT_OFFER:
            offer_sum += ev.amount
        for name, et in _TRACKED.items():
            if ev.event_type is et:
                counts[name] += 1
        onehot = np.zeros(N_EVENT_TYPES, dtype=np.float32)
        onehot[int(ev.event_type)] = 1.0
        rows[i] = np.array(
            static
            + [
                i + 1,
                ev.day,
                delta,
                ev.day / (i + 1),
                max_delta,
                counts["n_offers"],
                counts["n_compels"],
                counts["n_mtd"],
                counts["n_msj"],
                counts["n_mediations"],
                counts["n_trial_dates"],
                math.log1p(offer_sum),
            ]
            + onehot.tolist(),
            dtype=np.float32,
        )
    return rows


# --------------------------------------------------------------------- #
# Frozen-convention helpers (ece verbatim from run_stage1.py; CI/Holm
# verbatim from f1_forensics.py)
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


def paired_ci(deltas: list[float], t_crit: float = T_CRIT) -> dict:
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n < 2:
        return {"mean": float(d.mean()) if n else None, "ci95": None, "se": None, "n": n}
    se = float(d.std(ddof=1) / np.sqrt(n))
    return {"mean": float(d.mean()), "ci95": t_crit * se, "se": se, "n": n}


def holm_adjust(pvals: list[float]) -> list[float]:
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
    from scipy import stats
    d = np.asarray(deltas, dtype=float)
    if len(d) < 2 or float(d.std(ddof=1)) == 0.0:
        return None
    p = float(stats.ttest_1samp(d, 0.0).pvalue)
    return None if np.isnan(p) else p


def backlog_state_at(log_: list, day: float) -> bool:
    state = log_[0][1]
    for d, s in log_:
        if d <= day + 1e-9:
            state = s
        else:
            break
    return state == "backlog"


# --------------------------------------------------------------------- #
# Candidate construction, frozen loss, training loop (mirrors train.py)
# --------------------------------------------------------------------- #

def make_tpp(seed: int) -> TppTransformerModel:
    torch.manual_seed(seed)
    return TppTransformerModel(d_model=32, nhead=4, num_layers=2,
                               dim_feedforward=64, max_len=128,
                               time_mode="native")


def make_aux(seed: int) -> TemporalTransformerModel:
    torch.manual_seed(seed)
    return TemporalTransformerModel(d_model=32, nhead=4, num_layers=2,
                                    dim_feedforward=64, max_len=128,
                                    time_mode="native", auxiliary=True)


def f2_loss(model: nn.Module, batch, cfg: TrainConfig) -> torch.Tensor:
    """Frozen F2 loss: main heads (as train.py) + duration pinball +
    0.3 * closed-form marked TPP NLL over intervals with a next event."""
    out = model(batch.event_ids, batch.event_feats, batch.static,
                timespans=batch.deltas)
    bce = nn.functional.binary_cross_entropy_with_logits(
        out["settle_logit"], batch.y_settle, reduction="none")
    mse_rec = (out["log_recovery"] - batch.y_recovery) ** 2
    mse_rem = (out["log_remaining"] - batch.y_remaining) ** 2
    pb_dur = train_mod._pinball(out["duration_q"], batch.y_remaining)

    lam = out["lambdas"][:, :-1]                     # (B, T-1, 16)
    next_mask = batch.mask[:, 1:]                    # intervals with a next event
    tgt = batch.event_ids[:, 1:]
    log_lam_tgt = torch.log(lam).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    dt = batch.deltas[:, 1:].clamp(min=0)
    ell = log_lam_tgt - dt * lam.sum(-1)             # per-interval log-likelihood
    tpp_nll = -train_mod._masked_mean(ell, next_mask)

    return (
        cfg.weight_settle * train_mod._masked_mean(bce, batch.mask)
        + cfg.weight_recovery * train_mod._masked_mean(mse_rec, batch.mask)
        + cfg.weight_duration * train_mod._masked_mean(mse_rem, batch.mask)
        + cfg.weight_duration_q * train_mod._masked_mean(pb_dur, batch.mask)
        + TPP_NLL_WEIGHT * tpp_nll
    )


def train_tpp(model: nn.Module, timelines: list[CaseTimeline],
              cfg: TrainConfig) -> tuple[dict, bool]:
    """Mirror of liquid_legal.train.train_model with the frozen F2 loss in
    place of the aux-head losses. Returns (history, nan_event). Any
    non-finite loss aborts the run immediately (no restart, no tweaks)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model.to(cfg.device)

    n_val = max(1, int(len(timelines) * cfg.val_fraction))
    idx = rng.permutation(len(timelines))
    val_set = [timelines[i] for i in idx[:n_val]]
    train_set = [timelines[i] for i in idx[n_val:]]

    train_mod.initialize_output_biases(model, train_set, cfg.horizon_days)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(train_set))
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            chunk = [train_set[i] for i in order[start : start + cfg.batch_size]]
            batch = collate_timelines(chunk, horizon_days=cfg.horizon_days).to(cfg.device)
            loss = f2_loss(model, batch, cfg)
            if not torch.isfinite(loss):
                return history, True  # NaN/Inf event: abort run, no restart
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total_loss += float(loss.detach())
            n_batches += 1
        history["train"].append({"loss": total_loss / max(n_batches, 1),
                                 "seconds": time.time() - t0})
        history["val"].append(evaluate(model, val_set, cfg))
    return history, False


def select_lr(train0: list[CaseTimeline], seed: int,
              nan_events: list[str]) -> float:
    """Leakage-free selection: inner split of the TRAIN portion only,
    selected on inner-val settle AUC (mirrors run_stage1.select_lr)."""
    inner_train, inner_val = split(train0, seed)
    best_lr, best_auc = None, -1.0
    for lr in LR_GRID:
        cfg = TrainConfig(epochs=EPOCHS, lr=lr, verbose=False, seed=seed)
        model = make_tpp(seed)
        _, nan = train_tpp(model, inner_train, cfg)
        if nan:
            nan_events.append(f"lr-selection lr={lr} seed={seed}")
            continue
        auc = evaluate(model, inner_val, cfg)["settle_auc"]
        if auc > best_auc:
            best_lr, best_auc = lr, auc
    if best_lr is None:
        raise RuntimeError("all lr candidates produced NaN/Inf — aborting")
    return best_lr


# --------------------------------------------------------------------- #
# Evaluation: identical steps for both models, identical J convention as
# f2_score_calibration.py (masked steps with a next valid event,
# log1p-day quantile scale, full 16-mark distribution)
# --------------------------------------------------------------------- #

@torch.no_grad()
def collect(model: nn.Module, val: list[CaseTimeline], cfg: TrainConfig) -> dict:
    """Masked per-step outputs on the holdout: main-head arrays over all
    valid steps, J arrays over steps with a next observed event, and step
    metadata for stratification."""
    model.eval()
    main, nxt, meta, meta_n = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    is_tpp = isinstance(model, TppTransformerModel)
    for start in range(0, len(val), 256):
        batch = collate_timelines(val[start : start + 256],
                                  horizon_days=cfg.horizon_days)
        out = model(batch.event_ids, batch.event_feats, batch.static,
                    timespans=batch.deltas)
        m = batch.mask.bool()
        nm = batch.mask[:, 1:].bool()
        B, T = batch.event_ids.shape
        days = batch.deltas.cumsum(dim=1)
        lengths = batch.mask.sum(1)
        pos = torch.arange(T).float().unsqueeze(0) / lengths.clamp(min=1).unsqueeze(1)
        case_idx = torch.arange(start, start + B).unsqueeze(1).expand(B, T)

        main["settle_logit"].append(out["settle_logit"][m].numpy())
        main["y_settle"].append(batch.y_settle[m].numpy())
        main["log_recovery"].append(out["log_recovery"][m].numpy())
        main["y_recovery"].append(batch.y_recovery[m].numpy())
        main["log_remaining"].append(out["log_remaining"][m].numpy())
        main["y_remaining"].append(batch.y_remaining[m].numpy())
        if is_tpp:
            main["settle_prob_180d"].append(out["settle_prob_180d"][m].numpy())
            main["lambda_total"].append(out["lambdas"].sum(-1)[m].numpy())
            main["lambda_settled"].append(
                out["lambdas"][..., int(EventType.SETTLED)][m].numpy())
        for key, arr in (("case_idx", case_idx), ("day", days),
                         ("delta", batch.deltas), ("pos_frac", pos),
                         ("length", lengths.unsqueeze(1).expand(B, T)),
                         ("event_id", batch.event_ids.float())):
            meta[key].append(arr[m].numpy())

        # ---- J steps: step k predicts event k+1 (f2_score_calibration masking)
        tgt_type = batch.event_ids[:, 1:]
        tgt_gap = torch.log1p(batch.deltas[:, 1:].clamp(min=0))
        if is_tpp:
            probs = out["next_type_prob"][:, :-1]
            pred_q = out["next_gap_q"][:, :-1]
            nxt["lambdas"].append(out["lambdas"][:, :-1][nm].numpy())
        else:
            probs = torch.softmax(out["next_type_logit"][:, :-1], dim=-1)
            pred_q = out["next_gap_q"][:, :-1]
        ce = -torch.log(probs.gather(-1, tgt_type.unsqueeze(-1)).squeeze(-1))
        pin = torch.zeros_like(tgt_gap)
        for i, q in enumerate(QS):
            err = tgt_gap - pred_q[..., i]
            pin += torch.maximum(q * err, (q - 1.0) * err)
        pin = pin / len(QS)
        nxt["ce"].append(ce[nm].numpy())
        nxt["pin"].append(pin[nm].numpy())
        nxt["next_type_prob"].append(probs[nm].numpy())
        nxt["next_type_true"].append(tgt_type[nm].numpy())
        nxt["gap_target"].append(tgt_gap[nm].numpy())
        nxt["next_gap_q"].append(pred_q[nm].numpy())
        for key, arr in (("case_idx", case_idx[:, :-1]), ("day", days[:, :-1]),
                         ("delta", batch.deltas[:, :-1]),
                         ("pos_frac", pos[:, :-1]),
                         ("length", lengths.unsqueeze(1).expand(B, T)[:, :-1]),
                         ("event_id", batch.event_ids[:, :-1].float())):
            meta_n[key].append(arr[nm].numpy())

    pack = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    return {"main": pack(main), "next": pack(nxt),
            "meta": pack(meta), "meta_next": pack(meta_n)}


def metric_set(coll: dict) -> dict:
    main, nxt = coll["main"], coll["next"]
    probs = 1.0 / (1.0 + np.exp(-main["settle_logit"]))
    j = float(nxt["ce"].mean() + nxt["pin"].mean())
    out = {
        "settle_auc": auc_score(main["y_settle"], main["settle_logit"]),
        "ece": ece(probs, main["y_settle"]),
        "duration_mae_days": mae_days(main["log_remaining"], main["y_remaining"]),
        "next_type_acc": float(
            (nxt["next_type_prob"].argmax(-1) == nxt["next_type_true"]).mean()),
        "next_type_logloss": float(nxt["ce"].mean()),
        "next_gap_pinball": float(nxt["pin"].mean()),
        "J": j,
        "j_steps": int(len(nxt["ce"])),
    }
    if "settle_prob_180d" in main:
        out["settle_auc_tpp_derived"] = auc_score(main["y_settle"],
                                                  main["settle_prob_180d"])
    return out


def archived_aux_mean_seconds() -> float:
    """Mean per-run training wall-clock of tf-native-aux from the archived
    Stage-1 run log (mean of consecutive per-run timestamp deltas)."""
    times = []
    for line in (ARCHIVE_S1 / "stage1_run.log").read_text().splitlines():
        mm = re.match(r"seed=\d+ tf-native-aux/\w+ auc=.*\((\d+)s\)", line)
        if mm:
            times.append(int(mm.group(1)))
    diffs = [b - a for a, b in zip(times, times[1:])]
    return float(np.mean(diffs))


# --------------------------------------------------------------------- #
# Section 6: generator validation (architecture-independent, run first)
# --------------------------------------------------------------------- #

def fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 500):
    """Small deterministic logistic probe (torch, full-batch Adam).
    Returns (weights incl. bias, train mean, train std) so predictions use
    the training standardization."""
    torch.manual_seed(0)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    w = torch.zeros(Xs.shape[1], dtype=torch.float64, requires_grad=True)
    Xt = torch.from_numpy(Xs)
    yt = torch.from_numpy(y.astype(np.float64))
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(iters):
        opt.zero_grad()
        p = torch.sigmoid(Xt @ w)
        loss = nn.functional.binary_cross_entropy(p, yt)
        loss.backward()
        opt.step()
    return w.detach().numpy(), mu, sd


def predict_logistic(model, X: np.ndarray) -> np.ndarray:
    w, mu, sd = model
    Xs = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    return 1.0 / (1.0 + np.exp(-(Xs @ w)))


def generator_validation() -> dict:
    """Prereg section 6: per-mark gap quantiles (3 probe seeds x 1024
    cases, descriptive) and the silence probe (elapsed-context lift,
    paired over probe seeds, CI, no pass/fail)."""
    gaps_by_mark: dict[int, list[float]] = defaultdict(list)
    lifts, probe_detail = [], {}
    for seed in PROBE_SEEDS:
        gen = GeneratorV2(GeneratorV2Config(seed=seed))
        tls, _ = gen.generate_with_latents(CASES)
        for tl in tls:
            for i in range(len(tl.events) - 1):
                gaps_by_mark[int(tl.events[i].event_type)].append(
                    max(0.0, tl.events[i + 1].day - tl.events[i].day))

        # ---- silence probe: above/below-median next gap
        train, val = split(tls, seed)

        def rows(timelines):
            feats, ys = [], []
            for tl in timelines:
                pf = prefix_features(tl)
                for i in range(len(tl.events) - 1):
                    feats.append(pf[i])
                    ys.append(max(0.0, tl.events[i + 1].day - tl.events[i].day))
            return np.asarray(feats, dtype=np.float64), np.asarray(ys)

        Xtr, gap_tr = rows(train)
        Xva, gap_va = rows(val)
        med = float(np.median(gap_tr))
        ytr = (gap_tr > med).astype(float)
        yva = (gap_va > med).astype(float)
        aucs = {}
        for name, cols in (("history_plus_elapsed", None),
                           ("history_only", _TIME_COL_IDX)):
            xtr = Xtr if cols is None else np.delete(Xtr, cols, axis=1)
            xva = Xva if cols is None else np.delete(Xva, cols, axis=1)
            w = fit_logistic(xtr, ytr)
            aucs[name] = auc_score(yva, predict_logistic(w, xva))
        lifts.append(aucs["history_plus_elapsed"] - aucs["history_only"])
        probe_detail[str(seed)] = {**aucs, "lift": lifts[-1],
                                   "median_gap_days": med}
        log(f"gen-val probe seed={seed}: silence-probe AUC "
            f"hist+elapsed={aucs['history_plus_elapsed']:.4f} "
            f"hist-only={aucs['history_only']:.4f} "
            f"lift={lifts[-1]:+.4f}")

    mark_rows = {}
    for m in sorted(gaps_by_mark):
        arr = np.asarray(gaps_by_mark[m])
        mark_rows[EventType(m).name] = {
            "n": int(len(arr)),
            "q10": float(np.quantile(arr, 0.1)),
            "q50": float(np.quantile(arr, 0.5)),
            "q90": float(np.quantile(arr, 0.9)),
        }
    lift_ci = paired_ci(lifts, t_crit=T_CRIT_DF2)
    return {
        "probe_seeds": PROBE_SEEDS, "cases_per_seed": CASES,
        "per_mark_gap_quantiles_days": mark_rows,
        "silence_probe": {
            "definition": "predict above/below-median next gap; engineered "
                          "history features with vs without elapsed-context "
                          "columns (day, last_delta, mean_delta, max_delta); "
                          "logistic probe; paired over probe seeds; no pass/fail",
            "per_seed": probe_detail,
            "lift_auc": lift_ci,
        },
    }


# --------------------------------------------------------------------- #
# Supporting descriptive baselines (never part of the screen)
# --------------------------------------------------------------------- #

def sm_empirical_pinball(train: list[CaseTimeline], coll_next: dict) -> float:
    """sm-empirical: next-gap quantiles from per-(mark, phase-tercile)
    empirical quantiles fit on the training portion; scored as pinball on
    the same holdout J-steps (descriptive only)."""
    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_mark: dict[int, list[float]] = defaultdict(list)
    all_gaps: list[float] = []
    for tl in train:
        n = len(tl.events)
        for i in range(n - 1):
            g = math.log1p(max(0.0, tl.events[i + 1].day - tl.events[i].day))
            terc = min(2, int(3 * i / max(n, 1)))
            buckets[(int(tl.events[i].event_type), terc)].append(g)
            by_mark[int(tl.events[i].event_type)].append(g)
            all_gaps.append(g)
    q_global = np.quantile(all_gaps, QS)
    q_mark = {m: np.quantile(v, QS) for m, v in by_mark.items()}
    q_bucket = {k: np.quantile(v, QS) for k, v in buckets.items() if len(v) >= 30}

    meta = coll_next["meta"]
    n = len(coll_next["next"]["gap_target"])
    pred = np.zeros((n, 3))
    marks = meta["event_id"].astype(int)
    tercs = np.minimum(2, (3 * meta["pos_frac"]).astype(int))
    for i in range(n):
        pred[i] = q_bucket.get((marks[i], tercs[i]),
                               q_mark.get(marks[i], q_global))
    tgt = coll_next["next"]["gap_target"]
    pin = np.zeros(n)
    for j, q in enumerate(QS):
        err = tgt - pred[:, j]
        pin += np.maximum(q * err, (q - 1.0) * err)
    return float((pin / len(QS)).mean())


def cr_discrete_settle_auc(train: list[CaseTimeline],
                           val: list[CaseTimeline]) -> dict:
    """cr-discrete: discrete-time cause-specific hazard baseline (30-day
    monthly grid, logistic hazards on the same engineered features as xgb).
    Descriptive competing-risk comparison only. Returns the 180-day
    settle-within-horizon AUC on holdout steps from the discrete cause-
    specific cumulative incidence, covariates fixed at the landmark step."""
    GRID = 30.0

    def grid_rows(timelines):
        X, ys = [], []
        for tl in timelines:
            pf = prefix_features(tl)
            days = np.array([ev.day for ev in tl.events])
            dur = float(tl.outcome.get("duration_days", days[-1]))
            term = int(tl.events[-1].event_type)
            g = 0
            while GRID * g < dur:
                t_g = GRID * g
                k = int(np.searchsorted(days, t_g, side="right") - 1)
                X.append(pf[max(k, 0)])
                in_month = [0.0, 0.0, 0.0]
                if t_g < dur <= t_g + GRID and term in TERMINAL_IDS:
                    in_month[TERMINAL_IDS.index(term)] = 1.0
                ys.append(in_month)
                g += 1
        return np.asarray(X, dtype=np.float64), np.asarray(ys)

    Xtr, Ytr = grid_rows(train)
    fits = [fit_logistic(Xtr, Ytr[:, c]) for c in range(3)]
    mu, sd = fits[0][1], fits[0][2]  # shared train standardization

    # holdout: per valid step, CIF of SETTLED over the next 180 d with
    # covariates fixed at the step (landmark approach)
    probs, labels = [], []
    for tl in val:
        pf = prefix_features(tl)
        days = [ev.day for ev in tl.events]
        dur = float(tl.outcome.get("duration_days", days[-1]))
        settled = bool(tl.outcome.get("settled", 0.0))
        settle_day = float(tl.outcome.get("settle_day", -1.0))
        for k, day in enumerate(days):
            x = (pf[k] - mu) / sd
            x = np.concatenate([x, [1.0]])
            h = [1.0 / (1.0 + np.exp(-(x @ f[0]))) for f in fits]
            surv, cif = 1.0, 0.0
            for _ in range(6):  # 6 monthly steps = 180 d
                cif += surv * h[0]
                surv *= max(0.0, 1.0 - sum(h))
            probs.append(cif)
            labels.append(1.0 if settled and 0.0 <= (settle_day - day) <= 180.0 else 0.0)
    return {"settle_auc": auc_score(np.asarray(labels), np.asarray(probs)),
            "grid_days": GRID,
            "note": "landmark CIF, covariates fixed at step; descriptive"}


# --------------------------------------------------------------------- #
# Section 8: probe-7 strata applied to the paired J comparison
# --------------------------------------------------------------------- #

def build_strata_next(coll: dict, val: list[CaseTimeline], latents: dict) -> dict:
    """F1 probe-7 stratum definitions, applied to the J-scored steps
    (steps with a next observed event)."""
    meta = coll["meta_next"]
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


def subgroup_analysis(per_seed_colls: dict, latents_by_seed: dict,
                      vals_by_seed: dict) -> dict:
    """Per-stratum paired relative J reduction (exploratory; Holm within
    families). Eligibility: >= 100 J-steps in the stratum."""
    families: dict[str, dict[str, dict]] = {}
    for seed in sorted(per_seed_colls):
        ct = per_seed_colls[seed]["tpp"]
        ca = per_seed_colls[seed]["aux"]
        strata = build_strata_next(ct, vals_by_seed[seed], latents_by_seed[seed])
        for family, masks in strata.items():
            for sname, mask in masks.items():
                entry = families.setdefault(family, {}).setdefault(
                    sname, {"steps": [], "rel_red": [], "J_tpp": [], "J_aux": []})
                entry["steps"].append(int(mask.sum()))
                if mask.sum() < 100:
                    continue
                j_tpp = float(ct["next"]["ce"][mask].mean() + ct["next"]["pin"][mask].mean())
                j_aux = float(ca["next"]["ce"][mask].mean() + ca["next"]["pin"][mask].mean())
                entry["J_tpp"].append(j_tpp)
                entry["J_aux"].append(j_aux)
                entry["rel_red"].append((j_aux - j_tpp) / j_aux)

    out = {}
    for family, strata_d in families.items():
        fam_out = {}
        for sname, e in strata_d.items():
            ci = paired_ci(e["rel_red"])
            fam_out[sname] = {
                "steps_total": int(np.sum(e["steps"])),
                "seeds_used": len(e["rel_red"]),
                "mean_J_tpp": float(np.mean(e["J_tpp"])) if e["J_tpp"] else None,
                "mean_J_aux": float(np.mean(e["J_aux"])) if e["J_aux"] else None,
                "rel_reduction": ci,
                "p_value": ttest_rel_p(e["rel_red"]),
            }
        names = list(fam_out)
        pvals = [fam_out[s]["p_value"] for s in names]
        ok = [p is not None for p in pvals]
        adj = holm_adjust([p for p, k in zip(pvals, ok) if k]) if any(ok) else []
        it = iter(adj)
        for sname, k in zip(names, ok):
            fam_out[sname]["holm_adj_p_descriptive"] = next(it) if k else None
        out[family] = fam_out
    return out


# --------------------------------------------------------------------- #
# Section 7: mechanism checks (ONLY if the screen is survived)
# --------------------------------------------------------------------- #

def mechanism_checks(per_seed_colls: dict, latents_by_seed: dict,
                     vals_by_seed: dict) -> dict:
    """M1: mean predicted total intensity Lambda lower inside logged
    judge-backlog episodes than outside (directional, CI).
    M2: predicted lambda_SETTLED lower after logged adverse-regime flips
    (directional, CI). Labeled MECHANISM CHECK; explains, does not rescue."""
    m1_diffs, m2_diffs = [], []
    m1_in, m1_out = [], []
    for seed in sorted(per_seed_colls):
        coll = per_seed_colls[seed]["tpp"]
        meta, lam = coll["meta"], coll["main"]["lambda_total"]
        lam_s = coll["main"]["lambda_settled"]
        inside = np.zeros(len(lam), dtype=bool)
        after_flip, before_flip = [], []
        for i, (ci, day) in enumerate(zip(meta["case_idx"].astype(int), meta["day"])):
            tl = vals_by_seed[seed][ci]
            lat = latents_by_seed[seed][tl.case_id]
            inside[i] = backlog_state_at(lat["judge_backlog"], float(day))
            flip = lat["regime_flip_day"]
            if flip is not None:
                (after_flip if day >= flip - 1e-9 else before_flip).append(float(lam_s[i]))
        m1_in.append(float(lam[inside].mean()))
        m1_out.append(float(lam[~inside].mean()))
        m1_diffs.append(m1_in[-1] - m1_out[-1])
        if after_flip and before_flip:
            m2_diffs.append(float(np.mean(after_flip) - np.mean(before_flip)))
    return {
        "M1_backlog_intensity": {
            "definition": "mean predicted Lambda on holdout steps inside vs "
                          "outside logged judge-backlog episodes; directional "
                          "(expected lower inside)",
            "mean_lambda_inside": float(np.mean(m1_in)),
            "mean_lambda_outside": float(np.mean(m1_out)),
            "diff_inside_minus_outside": paired_ci(m1_diffs),
            "direction_ok": bool(np.mean(m1_diffs) < 0),
        },
        "M2_adverse_lambda_settled": {
            "definition": "mean predicted lambda_SETTLED after vs before "
                          "logged adverse-regime flips (flipped cases only); "
                          "directional (expected lower after)",
            "diff_after_minus_before": paired_ci(m2_diffs),
            "seeds_with_flips": len(m2_diffs),
            "direction_ok": bool(np.mean(m2_diffs) < 0) if m2_diffs else None,
        },
    }


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "weights").mkdir(exist_ok=True)
    (OUT_DIR / "predictions").mkdir(exist_ok=True)
    cfg = TrainConfig(verbose=False)
    nan_events: list[str] = []

    log("=== F2 v1: tf-tpp under F2_PREREGISTRATION.md (frozen) ===")
    n_params = sum(p.numel() for p in make_tpp(0).parameters())
    log(f"candidate params: {n_params} (tf-native-aux 23,129; budget +/-20%)")

    # ---- section 6: generator validation (before the candidate) ---- #
    log("\n--- generator validation (section 6, architecture-independent) ---")
    gen_val = generator_validation()
    gm = gen_val["per_mark_gap_quantiles_days"]
    spreads = [gm[m]["q90"] - gm[m]["q10"] for m in gm]
    log(f"per-mark gap quantiles: {len(gm)} marks, n range "
        f"{min(r['n'] for r in gm.values())}..{max(r['n'] for r in gm.values())}, "
        f"min (q90-q10) spread {min(spreads):.1f} d (non-degenerate)")
    sl = gen_val["silence_probe"]["lift_auc"]
    log(f"silence probe lift (hist+elapsed - hist-only): "
        f"{sl['mean']:+.4f} +/- {sl['ci95']:.4f} (df=2; descriptive, no pass/fail)")

    # ---- lr selection (inner split of seed-0 train portion, per regime) ---- #
    log("\n--- lr selection (seed-0 inner split) ---")
    gen0 = GeneratorV2(GeneratorV2Config(seed=0))
    tls0, _ = gen0.generate_with_latents(CASES)
    selected: dict[str, float] = {}
    for regime in REGIMES:
        data0 = strip_statics(tls0) if regime == "hidden" else tls0
        train0, _ = split(data0, 0)
        selected[regime] = select_lr(train0, 0, nan_events)
        log(f"selected lr={selected[regime]} for tf-tpp/{regime} "
            f"({time.time() - t0:.0f}s)")

    # ---- training: 10 seeds x 2 regimes (20 runs) ---- #
    log("\n--- training tf-tpp (frozen protocol) ---")
    wall: dict[str, list[float]] = {r: [] for r in REGIMES}
    final_loss: dict[str, list[float]] = {r: [] for r in REGIMES}
    datasets: dict[int, dict] = {}
    for seed in SEEDS:
        gen = GeneratorV2(GeneratorV2Config(seed=seed))
        tls, latents = gen.generate_with_latents(CASES)
        datasets[seed] = {"tls": tls, "latents": latents}
        for regime in REGIMES:
            data = strip_statics(tls) if regime == "hidden" else tls
            scfg = TrainConfig(epochs=EPOCHS, lr=selected[regime],
                               verbose=False, seed=seed)
            model = make_tpp(seed)
            t_start = time.time()
            history, nan = train_tpp(model, data, scfg)
            secs = time.time() - t_start
            wall[regime].append(secs)
            if nan:
                nan_events.append(f"tf-tpp/{regime}/seed{seed}")
                log(f"seed={seed} tf-tpp/{regime}: NaN/Inf EVENT — run aborted "
                    f"(counts against the stability guard; no restart)")
                continue
            final_loss[regime].append(history["train"][-1]["loss"])
            torch.save(model.state_dict(),
                       OUT_DIR / "weights" / f"tf-tpp_{regime}_seed{seed}.pt")
            log(f"seed={seed} trained tf-tpp/{regime:8s} "
                f"auc={history['val'][-1]['settle_auc']:.3f} "
                f"({secs:.1f}s, total {time.time() - t0:.0f}s)")

    # ---- evaluation: identical steps for both models ---- #
    log("\n--- evaluation (J + guards; hidden primary, visible secondary) ---")
    record = json.loads((ARCHIVE_S1 / "stage1_baselines.json").read_text())
    metrics: dict[str, dict[str, dict]] = {r: {"tpp": {}, "aux": {}} for r in REGIMES}
    colls_hidden: dict[int, dict] = {}
    auc_crosscheck = []
    for seed in SEEDS:
        tls, latents = datasets[seed]["tls"], datasets[seed]["latents"]
        hidden_colls = {}
        for regime in REGIMES:
            data = strip_statics(tls) if regime == "hidden" else tls
            _, val = split(data, seed)
            scfg = TrainConfig(verbose=False, seed=seed)

            tpp = make_tpp(seed)
            wp = OUT_DIR / "weights" / f"tf-tpp_{regime}_seed{seed}.pt"
            if wp.exists():
                tpp.load_state_dict(torch.load(wp, weights_only=True))
                coll_tpp = collect(tpp, val, scfg)
                metrics[regime]["tpp"][str(seed)] = metric_set(coll_tpp)
                if regime == "hidden":
                    hidden_colls["tpp"] = coll_tpp
                np.savez_compressed(
                    OUT_DIR / "predictions" / f"tf-tpp_{regime}_seed{seed}.npz",
                    **{f"main/{k}": v for k, v in coll_tpp["main"].items()},
                    **{f"next/{k}": v for k, v in coll_tpp["next"].items()},
                    **{f"meta_next/{k}": v for k, v in coll_tpp["meta_next"].items()})

            aux = make_aux(seed)
            aux.load_state_dict(torch.load(
                ARCHIVE_S1 / "weights" / f"tf-native-aux_{regime}_seed{seed}.pt",
                weights_only=True))
            coll_aux = collect(aux, val, scfg)
            metrics[regime]["aux"][str(seed)] = metric_set(coll_aux)
            if regime == "hidden":
                hidden_colls["aux"] = coll_aux
                np.savez_compressed(
                    OUT_DIR / "predictions" / f"tf-native-aux_hidden_seed{seed}.npz",
                    **{f"main/{k}": v for k, v in coll_aux["main"].items()},
                    **{f"next/{k}": v for k, v in coll_aux["next"].items()},
                    **{f"meta_next/{k}": v for k, v in coll_aux["meta_next"].items()})
                rec_auc = record["runs"]["tf-native-aux/hidden"][seed]["settle_auc"]
                auc_crosscheck.append(
                    abs(metrics["hidden"]["aux"][str(seed)]["settle_auc"] - rec_auc))
        if "tpp" in hidden_colls:
            colls_hidden[seed] = hidden_colls
        jt = metrics["hidden"]["tpp"].get(str(seed), {}).get("J", float("nan"))
        ja = metrics["hidden"]["aux"][str(seed)]["J"]
        log(f"seed={seed} hidden: J tpp={jt:.4f} aux={ja:.4f} "
            f"({time.time() - t0:.0f}s)")

    log(f"cross-check tf-native-aux hidden settle AUC vs stage1_baselines.json "
        f"of record: max |diff| = {max(auc_crosscheck):.2e} "
        f"({'MATCH' if max(auc_crosscheck) < 1e-4 else 'MISMATCH'})")

    # ---- supporting descriptive rows (never part of the screen) ---- #
    log("\n--- supporting descriptive baselines (hidden, never screened) ---")
    sm_pins, cr_aucs = [], []
    for seed in SEEDS:
        if seed not in colls_hidden or "tpp" not in colls_hidden[seed]:
            continue
        tls = datasets[seed]["tls"]
        data = strip_statics(tls)
        train, val = split(data, seed)
        j_steps = {"next": colls_hidden[seed]["tpp"]["next"],
                   "meta": colls_hidden[seed]["tpp"]["meta_next"]}
        sm_pins.append(sm_empirical_pinball(train, j_steps))
        cr_aucs.append(cr_discrete_settle_auc(train, val)["settle_auc"])
    supporting = {
        "idn_reference": json.loads(
            (Path(__file__).parent / "results" / "f2_score_calibration.json")
            .read_text())["paired_delta"]["J"],
        "xgb_reference_hidden": {
            k: float(np.mean([r[k] for r in record["runs"]["xgb/hidden"]]))
            for k in ("settle_auc", "duration_mae_days", "ece")},
        "sm_empirical_next_gap_pinball": paired_ci(sm_pins),
        "cr_discrete_settle_auc_180d": paired_ci(cr_aucs),
        "note": "descriptive only, never part of the screen; idn/xgb rows "
                "quoted from archives of record",
    }
    log(f"sm-empirical next-gap pinball: {supporting['sm_empirical_next_gap_pinball']['mean']:.4f}; "
        f"cr-discrete settle AUC(180d): {supporting['cr_discrete_settle_auc_180d']['mean']:.4f}")

    # ---- primary screen (section 4) ---- #
    rel_red, guards = [], {}
    per_seed_screen = {}
    for seed in SEEDS:
        jt = metrics["hidden"]["tpp"].get(str(seed))
        ja = metrics["hidden"]["aux"][str(seed)]
        if jt is None:
            continue
        rel_red.append((ja["J"] - jt["J"]) / ja["J"])
        per_seed_screen[str(seed)] = {
            "J_tpp": jt["J"], "J_aux": ja["J"],
            "rel_reduction": rel_red[-1],
        }
    rel_ci = paired_ci(rel_red)
    n_seeds_complete = sum(1 for r in REGIMES
                           for s in SEEDS if str(s) in metrics[r]["tpp"])

    def paired_mean(key, regime="hidden"):
        t = [metrics[regime]["tpp"][str(s)][key] for s in SEEDS
             if str(s) in metrics[regime]["tpp"]]
        a = [metrics[regime]["aux"][str(s)][key] for s in SEEDS
             if str(s) in metrics[regime]["tpp"]]
        return float(np.mean(t)), float(np.mean(a))

    auc_t, auc_a = paired_mean("settle_auc")
    ece_t, ece_a = paired_mean("ece")
    dur_t, dur_a = paired_mean("duration_mae_days")
    nta_t, nta_a = paired_mean("next_type_acc")
    wall_tpp = float(np.mean([s for r in REGIMES for s in wall[r]])) if wall[REGIMES[0]] else float("nan")
    wall_arch = archived_aux_mean_seconds()

    guards = {
        "settle_auc": {"tpp": auc_t, "aux": auc_a, "delta": auc_t - auc_a,
                       "threshold": ">= -0.005", "pass": bool(auc_t - auc_a >= -0.005)},
        "ece": {"tpp": ece_t, "aux": ece_a, "delta": ece_t - ece_a,
                "threshold": "<= +0.01", "pass": bool(ece_t - ece_a <= 0.01)},
        "duration_mae": {"tpp": dur_t, "aux": dur_a,
                         "ratio": dur_t / dur_a if dur_a else float("nan"),
                         "threshold": "<= +5%", "pass": bool(dur_t <= 1.05 * dur_a)},
        "next_type_acc": {"tpp": nta_t, "aux": nta_a, "delta": nta_t - nta_a,
                          "threshold": ">= -0.01", "pass": bool(nta_t - nta_a >= -0.01)},
        "wall_clock": {"tpp_mean_s": wall_tpp, "aux_archived_mean_s": wall_arch,
                       "threshold": "<= 2x archived mean",
                       "pass": bool(wall_tpp <= 2.0 * wall_arch)},
        "stability": {"nan_inf_events": nan_events, "seeds_complete": n_seeds_complete,
                      "threshold": "zero NaN/Inf events and all 10 seeds x 2 regimes",
                      "pass": bool(not nan_events and n_seeds_complete == 20)},
    }
    crit_margin = bool(rel_ci["mean"] is not None and rel_ci["mean"] >= J_MARGIN)
    crit_ci = bool(rel_ci["ci95"] is not None
                   and rel_ci["mean"] - rel_ci["ci95"] > 0.0)
    crit_guards = all(g["pass"] for g in guards.values())
    survived = crit_margin and crit_ci and crit_guards
    verdict = "SURVIVE" if survived else "KILLED"

    log("\n=== PRIMARY SCREEN (section 4, hidden statics, paired by seed) ===")
    log(f"mean paired relative J reduction: {rel_ci['mean']:+.4f} "
        f"(required >= {J_MARGIN}) -> {'PASS' if crit_margin else 'FAIL'}")
    log(f"95% paired t-interval (df=9): [{rel_ci['mean'] - rel_ci['ci95']:+.4f}, "
        f"{rel_ci['mean'] + rel_ci['ci95']:+.4f}] "
        f"-> {'excludes zero: PASS' if crit_ci else 'includes zero: FAIL'}")
    for name, g in guards.items():
        detail = (f"delta={g['delta']:+.4f}" if "delta" in g else
                  f"ratio={g['ratio']:.4f}" if "ratio" in g else
                  f"tpp={g.get('tpp_mean_s', 0):.1f}s vs 2x{g.get('aux_archived_mean_s', 0):.1f}s"
                  if name == "wall_clock" else
                  f"{len(g['nan_inf_events'])} events, {g['seeds_complete']}/20 runs")
        log(f"guard {name:15s} {detail:45s} ({g['threshold']}) -> "
            f"{'PASS' if g['pass'] else 'FAIL'}")
    log(f"\nVERDICT: {verdict}"
        + ("" if survived else " — killed candidate (interval-supervision "
                              "branch on Generator v2 terminated per section 4)"))

    # ---- section 8: subgroups (exploratory) ---- #
    log("\n--- subgroup analyses (section 8, EXPLORATORY) ---")
    subgroups = subgroup_analysis(
        colls_hidden,
        {s: datasets[s]["latents"] for s in colls_hidden},
        {s: split(strip_statics(datasets[s]["tls"]), s)[1] for s in colls_hidden})
    for family, strata_d in subgroups.items():
        for sname, e in strata_d.items():
            r = e["rel_reduction"]
            if r["mean"] is None:
                continue
            log(f"  {family}/{sname:12s} rel J red {r['mean']:+.4f} +/- "
                f"{r['ci95']:.4f} (seeds={e['seeds_used']}, "
                f"holm_p={e['holm_adj_p_descriptive']})")

    # ---- section 7: mechanism checks (gated) ---- #
    if survived:
        log("\n--- MECHANISM CHECKS M1/M2 (section 7; explain, do not rescue) ---")
        mech = mechanism_checks(
            colls_hidden,
            {s: datasets[s]["latents"] for s in colls_hidden},
            {s: split(strip_statics(datasets[s]["tls"]), s)[1] for s in colls_hidden})
        log(f"M1 Lambda inside-vs-outside backlog: "
            f"{mech['M1_backlog_intensity']['diff_inside_minus_outside']['mean']:+.5f} "
            f"(direction ok: {mech['M1_backlog_intensity']['direction_ok']})")
        log(f"M2 lambda_SETTLED after-vs-before adverse flip: "
            f"{mech['M2_adverse_lambda_settled']['diff_after_minus_before']['mean']:+.5f} "
            f"(direction ok: {mech['M2_adverse_lambda_settled']['direction_ok']})")
    else:
        mech = {"status": "NOT RUN — gated on screen survival (section 7); "
                          "the screen failed, so mechanism checks were not run"}
        log("\n--- mechanism checks M1/M2: NOT RUN (gated; screen failed) ---")

    # ---- visible-regime summary (secondary) ---- #
    vis = {}
    for name in ("J", "settle_auc", "ece", "duration_mae_days", "next_type_acc"):
        t, a = paired_mean(name, regime="visible")
        vis[name] = {"tpp": t, "aux": a}

    # ---- outputs ---- #
    results = {
        "status": "CONFIRMATORY — F2_PREREGISTRATION.md (frozen before any "
                  "candidate parameter was trained; see F2_FREEZE.md)",
        "verdict": verdict,
        "config": {
            "seeds": SEEDS, "cases": CASES, "epochs": EPOCHS,
            "lr_grid": LR_GRID, "selected_lr": selected,
            "regimes": REGIMES, "primary_regime": "hidden",
            "loss": {"settle_bce": 1.0, "recovery_log1p_mse": 0.3,
                     "duration_log1p_mse": 0.3, "duration_pinball": 0.2,
                     "tpp_nll": TPP_NLL_WEIGHT},
            "quantile_levels": list(QS),
            "params": n_params, "j_margin": J_MARGIN,
            "batch_size": 32, "grad_clip": 1.0, "optimizer": "Adam(0.9, 0.999)",
            "device": "cpu",
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
        "generator_validation": gen_val,
        "crosscheck_aux_hidden_settle_auc_vs_record": {
            "max_abs_diff": float(max(auc_crosscheck)),
            "per_seed": metrics["hidden"]["aux"],
        },
        "screen": {
            "per_seed": per_seed_screen,
            "mean_rel_reduction": rel_ci,
            "criterion_margin": crit_margin,
            "criterion_ci_excludes_zero": crit_ci,
            "guards": guards,
            "criterion_guards": crit_guards,
            "verdict": verdict,
        },
        "metrics": metrics,
        "wall_clock": {"tpp_per_run_s": wall, "aux_archived_mean_s": wall_arch,
                       "total_s": time.time() - t0},
        "train_final_loss": final_loss,
        "visible_secondary": vis,
        "supporting_descriptive": supporting,
        "subgroups_exploratory": subgroups,
        "mechanism_checks": mech,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {OUT_JSON}")

    # archive: log, config, environment, code snapshot, hashes
    (OUT_DIR / "run_f2_tpp.log").write_text("\n".join(LOG) + "\n")
    (OUT_DIR / "config.json").write_text(json.dumps(results["config"], indent=2))
    (OUT_DIR / "environment.json").write_text(
        json.dumps(results["environment"], indent=2))
    (OUT_DIR / "f2_results.json").write_text(json.dumps(results, indent=2))
    code_dir = OUT_DIR / "code"
    code_dir.mkdir(exist_ok=True)
    for f in ("tpp_model.py", "run_f2_tpp.py", "f2_prefreeze_checks.py",
              "F2_PREREGISTRATION.md"):
        shutil.copy2(Path(__file__).parent / f, code_dir / f)
    (OUT_DIR / "generator_validation.json").write_text(
        json.dumps(gen_val, indent=2))

    manifest = {}
    for f in sorted(OUT_DIR.rglob("*")):
        if f.is_file() and f.name != "hashes.json":
            manifest[str(f.relative_to(OUT_DIR))] = hashlib.sha256(
                f.read_bytes()).hexdigest()
    (OUT_DIR / "hashes.json").write_text(json.dumps(manifest, indent=2))
    # verify the manifest against the directory contents
    ok = all(
        hashlib.sha256((OUT_DIR / rel).read_bytes()).hexdigest() == h
        for rel, h in manifest.items())
    log(f"wrote {OUT_DIR} (hashes.json: {len(manifest)} files, "
        f"verification {'OK' if ok else 'FAILED'})")
    if not ok:
        raise RuntimeError("hashes.json verification failed")

    log(f"\n=== F2 v1 complete: verdict {verdict}; total wall-clock "
        f"{time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    main()
