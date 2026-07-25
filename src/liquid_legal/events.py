"""Core data structures for docket event streams.

A legal case is modeled as an irregularly sampled sequence of docket events
(:class:`CaseEvent`) plus static covariates (judge, district, plaintiff
characteristics) that drive the case's latent dynamics.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class EventType(enum.IntEnum):
    """Canonical docket event taxonomy used across the library."""

    FILED = 0
    ANSWER = 1
    MOTION_TO_DISMISS = 2
    DISCOVERY_OPEN = 3
    MOTION_TO_COMPEL = 4
    DEPOSITION = 5
    EXPERT_DISCLOSURE = 6
    DISCOVERY_CLOSE = 7
    MOTION_SUMMARY_JUDGMENT = 8
    SETTLEMENT_OFFER = 9
    MEDIATION = 10
    TRIAL_DATE_SET = 11
    TRIAL_START = 12
    SETTLED = 13
    DISMISSED = 14
    VERDICT = 15


N_EVENT_TYPES = len(EventType)

TERMINAL_EVENTS = frozenset(
    {EventType.SETTLED, EventType.DISMISSED, EventType.VERDICT}
)

#: Static covariates carried by every case, in canonical featurization order.
#: These capture the sources of non-linear volatility called out in the
#: project rationale: who the judge is, where the case is venued, and whether
#: the plaintiff is equipped for case-management tasks.
STATIC_FIELDS = [
    "underwriting_score",     # case-merit score, as from a funder's underwriter
    "judge_speed",            # multiplicative effect on all gaps (<1 fast, >1 slow)
    "judge_volatility",       # erraticness of the judge's timing (variance scale)
    "judge_defense_tilt",     # pro-defense lean affecting dispositive motions
    "district_congestion",    # multiplicative calendar congestion of the district
    "plaintiff_capability",   # plaintiff's case-management capability in [0, 1]
    "log_damages",            # log of claimed damages
]

STATIC_DIM = len(STATIC_FIELDS)


@dataclass
class CaseEvent:
    """A single docket event.

    Attributes:
        day: Days since the case was filed (continuous, irregularly spaced).
        event_type: The kind of event.
        amount: Dollar amount when meaningful (settlement offers, verdicts,
            settlements, claimed damages at filing); 0 otherwise.
        flag: Outcome flag when meaningful: 1.0 = motion granted / plaintiff
            verdict win, 0.0 = denied / defense win.
    """

    day: float
    event_type: EventType
    amount: float = 0.0
    flag: float = 0.0


@dataclass
class CaseTimeline:
    """A full (or prefix of a) case history.

    Attributes:
        case_id: Unique identifier.
        events: Ordered docket events, strictly increasing in ``day``.
        static: Static covariates, see :data:`STATIC_FIELDS`.
        judge_id: Index of the assigned judge.
        district_id: Index of the district.
        outcome: Resolution summary with keys ``settled`` (0/1), ``recovery``
            (dollars), ``duration_days``, ``settle_day`` (-1 if not settled),
            and ``n_stalls`` (discovery blockers encountered).
    """

    case_id: str
    events: list[CaseEvent]
    static: dict[str, float] = field(default_factory=dict)
    judge_id: int = 0
    district_id: int = 0
    outcome: dict[str, float] = field(default_factory=dict)

    @property
    def duration_days(self) -> float:
        return float(self.outcome.get("duration_days", self.events[-1].day))

    @property
    def is_resolved(self) -> bool:
        return self.events[-1].event_type in TERMINAL_EVENTS

    @property
    def n_events(self) -> int:
        return len(self.events)
