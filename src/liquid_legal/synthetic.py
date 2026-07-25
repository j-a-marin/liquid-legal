"""Synthetic litigation timeline generator.

A hazard-based generative process over docket events that encodes the domain
assumptions behind liquid-legal:

* Volatility is driven by the judge (speed, erraticness, defense lean), the
  district (calendar congestion), and the plaintiff (case-management
  capability). Timing noise is multiplicative and heavy-tailed — not linear.
* Discovery is always a potential blocker. Under-equipped plaintiffs face a
  quadratically higher stall hazard, stalls stretch the calendar, and each
  stall erodes settlement leverage — with weak cases suffering more.
* Regime shifts: cases drift for months, then accelerate sharply (a trial
  date is set, summary judgment is denied) or stall without warning.

The generator exists because real PACER-style event data is hard to obtain;
it produces labeled sequences (settlement, recovery, duration) suitable for
training and benchmarking trajectory models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .events import CaseEvent, CaseTimeline, EventType


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class GeneratorConfig:
    """Configuration for :class:`SyntheticLitigationGenerator`.

    Attributes:
        n_judges: Size of the synthetic judge pool.
        n_districts: Size of the synthetic district pool.
        horizon_days: Default prediction horizon used for labels downstream.
        seed: Master seed; generation is deterministic per seed.
    """

    n_judges: int = 12
    n_districts: int = 8
    horizon_days: float = 180.0
    seed: int = 0


class SyntheticLitigationGenerator:
    """Generates synthetic :class:`CaseTimeline` objects.

    Judge and district pools are fixed at construction so that the same judge
    is consistently fast/erratic across cases — models can (and should) learn
    to associate static covariates with dynamics.
    """

    def __init__(self, config: GeneratorConfig | None = None):
        self.config = config or GeneratorConfig()
        rng = np.random.default_rng(self.config.seed)
        self._rng = rng
        # Latent judge traits: multiplicative speed, timing erraticness, and
        # a pro-defense lean on dispositive rulings.
        self.judges_: list[dict[str, float]] = [
            {
                "speed": float(rng.lognormal(0.0, 0.45)),
                "volatility": float(rng.uniform(0.15, 1.0)),
                "defense_tilt": float(rng.normal(0.0, 0.6)),
            }
            for _ in range(self.config.n_judges)
        ]
        # Latent district traits: calendar congestion multiplier.
        self.districts_: list[float] = [
            float(rng.lognormal(0.0, 0.35)) for _ in range(self.config.n_districts)
        ]
        self._counter = 0

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _gap(self, base_days: float, mult: float, judge: dict[str, float]) -> float:
        """Sample a multiplicative, heavy-tailed inter-arrival time in days."""
        rng = self._rng
        jitter = math.exp(judge["volatility"] * float(rng.standard_normal()))
        regime = 1.0
        if rng.random() < 0.06:  # sudden acceleration or stall episode
            regime = float(rng.uniform(0.15, 3.0))
        return max(1.0, base_days * mult * judge["speed"] * jitter * regime)

    @staticmethod
    def _expected_settlement(damages: float, score: float, leverage: float) -> float:
        fraction = (0.03 + 0.42 * _sigmoid(0.9 * score)) * leverage
        return damages * fraction

    def _settle(self, day: float, damages: float, score: float, leverage: float,
                events: list[CaseEvent]) -> tuple[float, float]:
        recovery = self._expected_settlement(damages, score, leverage) * float(
            self._rng.lognormal(0.0, 0.2)
        )
        events.append(CaseEvent(day, EventType.SETTLED, amount=recovery))
        return recovery, day

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def sample(self) -> CaseTimeline:
        """Sample one synthetic case timeline."""
        rng = self._rng
        j = int(rng.integers(0, self.config.n_judges))
        d = int(rng.integers(0, self.config.n_districts))
        judge = self.judges_[j]
        cong = self.districts_[d]

        capability = float(rng.beta(2.2, 2.0))
        score = float(rng.standard_normal())
        damages = float(rng.lognormal(mean=math.log(2.0e6), sigma=1.1))

        def gap(base: float, mult: float = 1.0) -> float:
            return self._gap(base, mult * cong, judge)

        day = 0.0
        events: list[CaseEvent] = [
            CaseEvent(0.0, EventType.FILED, amount=damages)
        ]
        pressure = 0.0   # accumulated settlement pressure
        leverage = 1.0   # plaintiff's negotiating leverage, eroded by stalls
        n_stalls = 0
        settled = False
        recovery = 0.0
        settle_day = -1.0

        def maybe_settle() -> None:
            nonlocal settled, recovery, settle_day, day
            if not settled and rng.random() < _sigmoid(-2.0 + pressure):
                recovery, settle_day = self._settle(day, damages, score, leverage, events)
                settled = True

        # ---------------- pleadings ---------------- #
        day += gap(25.0)
        events.append(CaseEvent(day, EventType.ANSWER))

        if rng.random() < _sigmoid(0.5 + judge["defense_tilt"] - 0.3 * score):
            day += gap(45.0)
            granted = rng.random() < _sigmoid(-0.9 + judge["defense_tilt"] - 0.9 * score)
            events.append(CaseEvent(day, EventType.MOTION_TO_DISMISS, flag=float(granted)))
            if granted:
                return self._finish(events, j, d, capability, score, damages, n_stalls)

        # ---------------- discovery ---------------- #
        day += gap(30.0)
        events.append(CaseEvent(day, EventType.DISCOVERY_OPEN))

        discovery_budget = 240.0 * cong * judge["speed"] * float(rng.lognormal(0.0, 0.25))
        n_steps = int(rng.integers(4, 10))
        # Non-linear blocker hazard: under-equipped plaintiffs are
        # disproportionately likely to bog down in discovery.
        stall_hazard = 0.04 + 0.5 * (1.0 - capability) ** 2 * min(cong, 2.5) / 2.5

        for _ in range(n_steps):
            if settled:
                break
            step_days = discovery_budget / n_steps
            if rng.random() < stall_hazard:
                # Discovery blocker: motion to compel, then an extended stall.
                n_stalls += 1
                granted = rng.random() < 0.6
                day += step_days * float(rng.uniform(0.3, 0.7))
                events.append(CaseEvent(day, EventType.MOTION_TO_COMPEL, flag=float(granted)))
                day += gap(35.0, mult=1.5 + 2.5 * (1.0 - capability))
                # Weak cases suffer more from each stall (non-linear decay).
                fragility = 1.0 - _sigmoid(score)
                leverage *= 0.95 - 0.18 * fragility
                pressure -= 0.15
            else:
                day += step_days * float(rng.uniform(0.7, 1.3))
                # NB: index into a list rather than rng.choice over the enum —
                # rng.choice would return np.int64, breaking identity checks.
                options = [EventType.DEPOSITION, EventType.EXPERT_DISCLOSURE, EventType.SETTLEMENT_OFFER]
                ev = options[rng.choice(3, p=[0.5, 0.3, 0.2])]
                if ev is EventType.SETTLEMENT_OFFER:
                    pressure += 0.35
                    amount = self._expected_settlement(damages, score, leverage) * float(
                        rng.lognormal(0.0, 0.25)
                    )
                    events.append(CaseEvent(day, ev, amount=amount))
                    maybe_settle()
                else:
                    events.append(CaseEvent(day, ev))

        if not settled:
            day += gap(20.0)
            events.append(CaseEvent(day, EventType.DISCOVERY_CLOSE))

            # ---------------- dispositive motions ---------------- #
            if rng.random() < _sigmoid(0.4 - 0.2 * score + 0.3 * judge["defense_tilt"]):
                day += gap(75.0)
                granted = rng.random() < _sigmoid(-1.3 + judge["defense_tilt"] - 1.1 * score)
                events.append(
                    CaseEvent(day, EventType.MOTION_SUMMARY_JUDGMENT, flag=float(granted))
                )
                if granted:
                    return self._finish(events, j, d, capability, score, damages, n_stalls)
                pressure += 1.2  # surviving MSJ is the biggest pressure spike

            # ---------------- negotiation / pre-trial ---------------- #
            trial_date_set = False
            for _ in range(6):
                if settled:
                    break
                options = [EventType.SETTLEMENT_OFFER, EventType.MEDIATION, EventType.TRIAL_DATE_SET]
                ev = options[rng.choice(3, p=[0.45, 0.25, 0.30])]
                if ev is EventType.TRIAL_DATE_SET and trial_date_set:
                    ev = EventType.SETTLEMENT_OFFER
                day += gap(40.0, mult=0.5 if trial_date_set else 1.0)
                if ev is EventType.TRIAL_DATE_SET:
                    trial_date_set = True
                    pressure += 0.9  # deadline effect: the case accelerates
                    events.append(CaseEvent(day, ev))
                elif ev is EventType.MEDIATION:
                    pressure += 0.55
                    events.append(CaseEvent(day, ev))
                    maybe_settle()
                else:
                    pressure += 0.4
                    amount = self._expected_settlement(damages, score, leverage) * float(
                        rng.lognormal(0.0, 0.25)
                    )
                    events.append(CaseEvent(day, ev, amount=amount))
                    maybe_settle()

            # ---------------- trial ---------------- #
            if not settled:
                day += gap(30.0, mult=0.5 if trial_date_set else 1.0)
                events.append(CaseEvent(day, EventType.TRIAL_START))
                day += float(rng.uniform(3.0, 15.0))
                win = rng.random() < _sigmoid(0.9 * score - 0.7 * judge["defense_tilt"])
                recovery = damages * float(rng.uniform(0.15, 0.9)) if win else 0.0
                events.append(CaseEvent(day, EventType.VERDICT, amount=recovery, flag=float(win)))

        return self._finish(events, j, d, capability, score, damages, n_stalls,
                            settled=settled, recovery=recovery, settle_day=settle_day)

    def generate(self, n: int) -> list[CaseTimeline]:
        """Sample ``n`` synthetic case timelines."""
        return [self.sample() for _ in range(n)]

    def _finish(
        self,
        events: list[CaseEvent],
        judge_id: int,
        district_id: int,
        capability: float,
        score: float,
        damages: float,
        n_stalls: int,
        settled: bool = False,
        recovery: float = 0.0,
        settle_day: float = -1.0,
    ) -> CaseTimeline:
        last = events[-1]
        if last.event_type is EventType.DISMISSED:
            pass
        elif last.event_type is EventType.SETTLED:
            settled, settle_day = True, last.day
            recovery = last.amount
        elif last.event_type is EventType.VERDICT:
            recovery = last.amount
        elif last.event_type in (
            EventType.MOTION_TO_DISMISS,
            EventType.MOTION_SUMMARY_JUDGMENT,
        ):
            # Granted dispositive motion -> dismissal at the same day.
            events.append(CaseEvent(last.day, EventType.DISMISSED))
        else:  # pragma: no cover - defensive
            raise ValueError(f"timeline ended on non-terminal event {last.event_type}")

        self._counter += 1
        return CaseTimeline(
            case_id=f"syn-{self.config.seed}-{self._counter}",
            events=events,
            static={
                "underwriting_score": score,
                "judge_speed": self.judges_[judge_id]["speed"],
                "judge_volatility": self.judges_[judge_id]["volatility"],
                "judge_defense_tilt": self.judges_[judge_id]["defense_tilt"],
                "district_congestion": self.districts_[district_id],
                "plaintiff_capability": capability,
                "log_damages": math.log(damages),
            },
            judge_id=judge_id,
            district_id=district_id,
            outcome={
                "settled": float(settled),
                "recovery": float(recovery),
                "duration_days": float(events[-1].day),
                "settle_day": float(settle_day),
                "n_stalls": float(n_stalls),
            },
        )
