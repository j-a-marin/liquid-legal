"""Agent-facing inference hooks.

The engine's hidden state is a compact, continuously evolving memory of the
case. :func:`snapshot` packages the current predictions plus that state into
a plain dataclass that higher-level LLM agents (LangGraph, CrewAI, custom
orchestrators) can read, log, or condition on — the "world model" query
interface for a multi-agent legal workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .events import CaseTimeline
from .featurize import featurize_timeline


@dataclass
class CaseSnapshot:
    """Point-in-time read of a case trajectory, safe to hand to an agent."""

    case_id: str
    n_events: int
    last_event: str
    day: float
    p_settle_within_horizon: float
    expected_recovery: float
    expected_remaining_days: float
    velocity: str  # "accelerating" | "stalled" | "steady"
    hidden_state: list[float] = field(default_factory=list)


def _velocity(days: np.ndarray) -> str:
    """Classify recent case velocity from inter-arrival times."""
    deltas = np.diff(days)
    deltas = deltas[deltas > 0]
    if deltas.size < 4:
        return "steady"
    recent = float(np.median(deltas[-3:]))
    overall = float(np.median(deltas))
    if overall <= 0:
        return "steady"
    ratio = recent / overall
    if ratio < 0.6:
        return "accelerating"
    if ratio > 1.6:
        return "stalled"
    return "steady"


@torch.no_grad()
def snapshot(
    engine: torch.nn.Module,
    timeline: CaseTimeline,
    horizon_days: float = 180.0,
    device: str = "cpu",
) -> CaseSnapshot:
    """Run the engine over a (possibly unresolved) timeline prefix.

    Pass any prefix of a case — e.g. everything on the docket so far — and
    get back the current settlement probability, expected recovery, expected
    remaining duration, a velocity classification, and the liquid hidden
    state.
    """
    engine.eval()
    f = featurize_timeline(timeline)
    event_ids = torch.from_numpy(f["event_ids"]).unsqueeze(0).to(device)
    event_feats = torch.from_numpy(f["event_feats"]).unsqueeze(0).to(device)
    deltas = torch.from_numpy(f["deltas"]).unsqueeze(0).to(device)
    static = torch.from_numpy(f["static"]).unsqueeze(0).to(device)

    out = engine(event_ids, event_feats, static, timespans=deltas)
    p_settle = torch.sigmoid(out["settle_logit"][0, -1]).item()
    recovery = float(np.expm1(out["log_recovery"][0, -1].cpu().numpy()))
    remaining = float(np.expm1(out["log_remaining"][0, -1].cpu().numpy()))

    days = np.array([ev.day for ev in timeline.events], dtype=np.float64)
    last = timeline.events[-1]
    return CaseSnapshot(
        case_id=timeline.case_id,
        n_events=len(timeline.events),
        last_event=last.event_type.name,
        day=float(last.day),
        p_settle_within_horizon=float(p_settle),
        expected_recovery=max(recovery, 0.0),
        expected_remaining_days=max(remaining, 0.0),
        velocity=_velocity(days),
        hidden_state=[float(v) for v in out["hx"][0].cpu().numpy()],
    )
