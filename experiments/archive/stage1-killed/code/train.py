"""Training and evaluation for trajectory models.

A single masked multi-task loss trains all three heads at every real
(non-padded) timestep: the model learns to predict the final outcome from any
prefix of the docket — which is exactly how a litigation-finance analyst uses
it mid-case.

Models whose forward accepts ``lengths`` (e.g. IDN) receive per-row valid
step counts derived from the batch mask so padding cannot contaminate state.
"""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .events import CaseTimeline
from .featurize import Batch, collate_timelines
from .metrics import auc_score, mae_days, mae_log_dollars


def _model_forward(model: nn.Module, batch: Batch) -> dict[str, torch.Tensor]:
    kwargs = {}
    if "lengths" in inspect.signature(model.forward).parameters:
        kwargs["lengths"] = batch.mask.sum(dim=1).long()
    return model(batch.event_ids, batch.event_feats, batch.static,
                 timespans=batch.deltas, **kwargs)


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 32
    lr: float = 3e-3
    weight_settle: float = 1.0
    weight_recovery: float = 0.3
    weight_duration: float = 0.3
    weight_next_type: float = 0.3   # auxiliary (models with marked-event heads)
    weight_next_gap: float = 0.2
    weight_duration_q: float = 0.2
    val_fraction: float = 0.2
    horizon_days: float = 180.0
    grad_clip: float = 1.0
    device: str = "cpu"
    seed: int = 0
    verbose: bool = True


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


_QUANTILE_LEVELS = (0.1, 0.5, 0.9)


def _pinball(pred_q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss over quantile levels; pred_q (..., 3), target (...)."""
    e = target.unsqueeze(-1) - pred_q
    per_q = torch.stack([
        torch.maximum(q * e[..., i], (q - 1.0) * e[..., i])
        for i, q in enumerate(_QUANTILE_LEVELS)
    ], dim=-1)
    return per_q.mean(dim=-1)


def _aux_losses(out: dict[str, torch.Tensor], batch: Batch, cfg: TrainConfig) -> torch.Tensor:
    """Marked-event auxiliary losses, active only for models that emit the
    auxiliary heads (IDN, auxiliary-matched Transformer)."""
    total = torch.zeros((), device=batch.event_ids.device)
    if "next_type_logit" not in out:
        return total
    # next-event targets: shifted by one step, masked at each row's last step
    next_mask = batch.mask[:, 1:]
    ce = nn.functional.cross_entropy(
        out["next_type_logit"][:, :-1].reshape(-1, out["next_type_logit"].shape[-1]),
        batch.event_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(batch.event_ids[:, 1:].shape)
    total = total + cfg.weight_next_type * _masked_mean(ce, next_mask)
    next_gap = torch.log1p(batch.deltas[:, 1:].clamp(min=0))
    pb_gap = _pinball(out["next_gap_q"][:, :-1], next_gap)
    total = total + cfg.weight_next_gap * _masked_mean(pb_gap, next_mask)
    pb_dur = _pinball(out["duration_q"], batch.y_remaining)
    total = total + cfg.weight_duration_q * _masked_mean(pb_dur, batch.mask)
    return total


def _loss(model: nn.Module, batch: Batch, cfg: TrainConfig) -> torch.Tensor:
    out = _model_forward(model, batch)
    bce = nn.functional.binary_cross_entropy_with_logits(
        out["settle_logit"], batch.y_settle, reduction="none"
    )
    mse_rec = (out["log_recovery"] - batch.y_recovery) ** 2
    mse_rem = (out["log_remaining"] - batch.y_remaining) ** 2
    return (
        cfg.weight_settle * _masked_mean(bce, batch.mask)
        + cfg.weight_recovery * _masked_mean(mse_rec, batch.mask)
        + cfg.weight_duration * _masked_mean(mse_rem, batch.mask)
        + _aux_losses(out, batch, cfg)
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    timelines: list[CaseTimeline],
    cfg: TrainConfig,
    batch_size: int = 256,
) -> dict[str, float]:
    """Masked validation metrics: BCE, settle AUC, duration MAE, recovery MAE."""
    model.eval()
    device = cfg.device
    total_bce, total_n = 0.0, 0.0
    logits, settles = [], []
    pred_rem, true_rem = [], []
    pred_rec, true_rec = [], []
    for start in range(0, len(timelines), batch_size):
        batch = collate_timelines(
            timelines[start : start + batch_size], horizon_days=cfg.horizon_days
        ).to(device)
        out = _model_forward(model, batch)
        bce = nn.functional.binary_cross_entropy_with_logits(
            out["settle_logit"], batch.y_settle, reduction="none"
        )
        m = batch.mask.bool()
        n = int(m.sum())
        total_bce += float((bce * batch.mask).sum())
        total_n += n
        logits.append(out["settle_logit"][m].cpu().numpy())
        settles.append(batch.y_settle[m].cpu().numpy())
        pred_rem.append(out["log_remaining"][m].cpu().numpy())
        true_rem.append(batch.y_remaining[m].cpu().numpy())
        pred_rec.append(out["log_recovery"][m].cpu().numpy())
        true_rec.append(batch.y_recovery[m].cpu().numpy())
    logits = np.concatenate(logits)
    settles = np.concatenate(settles)
    return {
        "bce": total_bce / max(total_n, 1),
        "settle_auc": auc_score(settles, logits),
        "duration_mae_days": mae_days(np.concatenate(pred_rem), np.concatenate(true_rem)),
        "recovery_mae_log": mae_log_dollars(np.concatenate(pred_rec), np.concatenate(true_rec)),
    }


@torch.no_grad()
def initialize_output_biases(
    model: nn.Module, timelines: list[CaseTimeline], horizon_days: float = 180.0
) -> None:
    """Set the three head biases to the dataset's base statistics.

    log1p-dollar targets sit near 13.8 and log1p-day targets near 5.5, so a
    zero-initialized head spends most of training just climbing to the mean.
    Starting at the label mean (and at the base-rate log-odds for the
    settlement head) lets the liquid dynamics learn the signal immediately.
    """
    ys, yrec, yrem = [], [], []
    for start in range(0, len(timelines), 256):
        batch = collate_timelines(
            timelines[start : start + 256], horizon_days=horizon_days
        )
        m = batch.mask.bool()
        ys.append(batch.y_settle[m])
        yrec.append(batch.y_recovery[m])
        yrem.append(batch.y_remaining[m])
    ys = torch.cat(ys)
    rate = float(ys.mean().clamp(1e-4, 1 - 1e-4))
    model.head_settle.bias.fill_(float(np.log(rate / (1.0 - rate))))
    model.head_recovery.bias.fill_(float(torch.cat(yrec).mean()))
    model.head_remaining.bias.fill_(float(torch.cat(yrem).mean()))


def train_model(
    model: nn.Module,
    timelines: list[CaseTimeline],
    cfg: TrainConfig | None = None,
) -> dict[str, list[dict[str, float]]]:
    """Train ``model`` on ``timelines``; returns per-epoch train/val history."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = cfg.device
    model.to(device)

    n_val = max(1, int(len(timelines) * cfg.val_fraction))
    idx = rng.permutation(len(timelines))
    val_set = [timelines[i] for i in idx[:n_val]]
    train_set = [timelines[i] for i in idx[n_val:]]

    initialize_output_biases(model, train_set, cfg.horizon_days)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(train_set))
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            chunk = [train_set[i] for i in order[start : start + cfg.batch_size]]
            batch = collate_timelines(chunk, horizon_days=cfg.horizon_days).to(device)
            loss = _loss(model, batch, cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total_loss += float(loss.detach())
            n_batches += 1
        train_metrics = {"loss": total_loss / max(n_batches, 1), "seconds": time.time() - t0}
        val_metrics = evaluate(model, val_set, cfg)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        if cfg.verbose:
            auc = val_metrics["settle_auc"]
            auc_s = f"{auc:.3f}" if not math.isnan(auc) else " n/a "
            print(
                f"epoch {epoch + 1:>3}/{cfg.epochs}  loss={train_metrics['loss']:.4f}  "
                f"val_bce={val_metrics['bce']:.4f}  settle_auc={auc_s}  "
                f"dur_mae={val_metrics['duration_mae_days']:.1f}d  "
                f"rec_mae_log={val_metrics['recovery_mae_log']:.3f}"
            )
    return history
