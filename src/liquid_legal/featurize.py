"""Featurization and batching of docket event streams.

Converts :class:`CaseTimeline` objects into padded tensor batches with
per-timestep supervision targets:

* ``y_settle``: 1 if the case settles within ``horizon_days`` of time t.
* ``y_recovery``: log1p of the case's final recovery (regression target).
* ``y_remaining``: log1p of remaining days until resolution from time t.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from .events import STATIC_DIM, STATIC_FIELDS, CaseTimeline

#: Per-event numeric features: [log1p(delta_days), normalized log amount, flag].
EVENT_FEATURE_DIM = 3


class Batch(NamedTuple):
    """A padded batch of featurized timelines."""

    event_ids: torch.Tensor      # (B, T) long
    event_feats: torch.Tensor    # (B, T, EVENT_FEATURE_DIM) float
    deltas: torch.Tensor         # (B, T) float — days since previous event
    static: torch.Tensor         # (B, STATIC_DIM) float
    mask: torch.Tensor           # (B, T) float, 1.0 for real steps
    y_settle: torch.Tensor       # (B, T) float
    y_recovery: torch.Tensor     # (B, T) float, log1p dollars
    y_remaining: torch.Tensor    # (B, T) float, log1p days

    def to(self, device: str | torch.device) -> "Batch":
        return Batch(*(t.to(device) for t in self))


def featurize_timeline(timeline: CaseTimeline) -> dict[str, np.ndarray]:
    """Convert one timeline into numpy arrays (no padding, no labels)."""
    events = timeline.events
    n = len(events)
    event_ids = np.zeros(n, dtype=np.int64)
    feats = np.zeros((n, EVENT_FEATURE_DIM), dtype=np.float32)
    deltas = np.zeros(n, dtype=np.float32)
    prev_day = 0.0
    for i, ev in enumerate(events):
        delta = max(0.0, ev.day - prev_day)
        prev_day = ev.day
        event_ids[i] = int(ev.event_type)
        deltas[i] = delta
        feats[i] = (
            np.log1p(delta),
            np.log1p(max(ev.amount, 0.0)) / 15.0,
            ev.flag,
        )
    static = np.array(
        [timeline.static.get(k, 0.0) for k in STATIC_FIELDS], dtype=np.float32
    )
    return {
        "event_ids": event_ids,
        "event_feats": feats,
        "deltas": deltas,
        "static": static,
    }


def _labels(timeline: CaseTimeline, horizon_days: float) -> tuple[np.ndarray, ...]:
    events = timeline.events
    n = len(events)
    outcome = timeline.outcome
    settled = bool(outcome.get("settled", 0.0))
    settle_day = float(outcome.get("settle_day", -1.0))
    duration = float(outcome.get("duration_days", events[-1].day))
    log_recovery = np.log1p(max(float(outcome.get("recovery", 0.0)), 0.0))

    y_settle = np.zeros(n, dtype=np.float32)
    y_remaining = np.zeros(n, dtype=np.float32)
    for i, ev in enumerate(events):
        if settled and 0.0 <= (settle_day - ev.day) <= horizon_days:
            y_settle[i] = 1.0
        y_remaining[i] = np.log1p(max(duration - ev.day, 0.0))
    y_recovery = np.full(n, log_recovery, dtype=np.float32)
    return y_settle, y_recovery, y_remaining


def collate_timelines(
    timelines: list[CaseTimeline], horizon_days: float = 180.0
) -> Batch:
    """Pad and stack a list of timelines into a :class:`Batch` with labels."""
    B = len(timelines)
    feats = [featurize_timeline(t) for t in timelines]
    labels = [_labels(t, horizon_days) for t in timelines]
    T = max(f["event_ids"].shape[0] for f in feats)

    event_ids = np.zeros((B, T), dtype=np.int64)
    event_feats = np.zeros((B, T, EVENT_FEATURE_DIM), dtype=np.float32)
    deltas = np.zeros((B, T), dtype=np.float32)
    static = np.zeros((B, STATIC_DIM), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    y_settle = np.zeros((B, T), dtype=np.float32)
    y_recovery = np.zeros((B, T), dtype=np.float32)
    y_remaining = np.zeros((B, T), dtype=np.float32)

    for i, (f, (ys, yrec, yrem)) in enumerate(zip(feats, labels)):
        n = f["event_ids"].shape[0]
        event_ids[i, :n] = f["event_ids"]
        event_feats[i, :n] = f["event_feats"]
        deltas[i, :n] = f["deltas"]
        static[i] = f["static"]
        mask[i, :n] = 1.0
        y_settle[i, :n] = ys
        y_recovery[i, :n] = yrec
        y_remaining[i, :n] = yrem

    return Batch(
        event_ids=torch.from_numpy(event_ids),
        event_feats=torch.from_numpy(event_feats),
        deltas=torch.from_numpy(deltas),
        static=torch.from_numpy(static),
        mask=torch.from_numpy(mask),
        y_settle=torch.from_numpy(y_settle),
        y_recovery=torch.from_numpy(y_recovery),
        y_remaining=torch.from_numpy(y_remaining),
    )
