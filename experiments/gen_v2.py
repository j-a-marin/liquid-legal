"""Generator v2: time-varying institutional latents, hidden case regimes,
selective observation, and full latent logging.

Extends the v1 world (which remains frozen for the E1–E6 record) with the
mechanisms the preregistration (experiments/PREREGISTRATION.md) requires:

* **Judge backlog episodes** — each case experiences its assigned judge's
  capacity as a two-state continuous-time chain (normal/backlogged); backlog
  stretches gaps multiplicatively. The *current* state is hidden from the
  static covariates, so recent gap patterns carry information statics don't.
  Backlog days also erode plaintiff leverage (litigation fatigue): delay is
  not merely a timing symptom, it is outcome-relevant.
* **Hidden case regime** — a case can flip Normal -> Adverse mid-course
  (unobserved): higher discovery-stall hazard, halved settlement-pressure
  gains, and continuous leverage decay while adverse. Regime changes alter
  transition dynamics *and* outcomes.
* **Selective observation** — the docket is an *observation* of the true
  event process, not the process itself: deposition-class events are dropped
  with congestion-dependent rates. The true log and observation mask are
  retained in the latent log for ground-truth evaluation.

`generate_with_latents(n)` returns (timelines, latents_by_case_id).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from liquid_legal.events import CaseEvent, CaseTimeline, EventType
from liquid_legal.synthetic import GeneratorConfig, SyntheticLitigationGenerator, _sigmoid

#: Event types subject to selective (non-total) observation, with base rates.
_OBSERVATION_RATES = {
    EventType.DEPOSITION: 0.95,
    EventType.EXPERT_DISCLOSURE: 0.95,
    EventType.MOTION_TO_COMPEL: 0.90,
    EventType.SETTLEMENT_OFFER: 0.85,
}
_CONGESTION_OBS_PENALTY = 0.35  # rate reduction at max congestion


@dataclass
class GeneratorV2Config(GeneratorConfig):
    """v2 mechanisms on top of the v1 config."""

    mean_normal_days: float = 400.0     # mean judge normal-episode length
    mean_backlog_days: float = 350.0    # mean judge backlog-episode length
    backlog_gap_mult: float = 2.5       # gap multiplier while backlogged
    backlog_fatigue_rate: float = 0.0015  # leverage erosion per backlog day
    backlog_settle_penalty: float = 2.0   # logit penalty on settlement while
                                          # backlogged (no hearings, no settlements)
    regime_flip_hazard: float = 0.04    # per-event Normal->Adverse probability
    adverse_stall_bonus: float = 0.15   # added discovery stall hazard when adverse
    adverse_accept_penalty: float = 0.8   # logit penalty on settlement acceptance
    adverse_decay_rate: float = 0.002   # leverage erosion per day while adverse
    adverse_pressure_mult: float = 0.5  # pressure gains are halved when adverse


class GeneratorV2(SyntheticLitigationGenerator):
    """v2 generative process. Returns v1-compatible timelines whose events
    are the *observed* docket, plus a latent log per case."""

    config: GeneratorV2Config

    def __init__(self, config: GeneratorV2Config | None = None):
        super().__init__(config or GeneratorV2Config())

    # ------------------------------------------------------------------ #

    def sample_with_latents(self) -> tuple[CaseTimeline, dict]:
        rng = self._rng
        cfg = self.config
        j = int(rng.integers(0, cfg.n_judges))
        d = int(rng.integers(0, cfg.n_districts))
        judge = self.judges_[j]
        cong = self.districts_[d]

        capability = float(rng.beta(2.2, 2.0))
        score = float(rng.standard_normal())
        damages = float(rng.lognormal(mean=math.log(2.0e6), sigma=1.1))

        # --- hidden time-varying state --- #
        backlogged = bool(rng.random() < 0.5)
        backlog_log: list[tuple[float, str]] = [(0.0, "backlog" if backlogged else "normal")]
        adverse = False
        regime_flip_day: float | None = None

        day = 0.0
        leverage = 1.0

        def gap(base: float, mult: float = 1.0) -> float:
            m = mult * cong * (cfg.backlog_gap_mult if backlogged else 1.0)
            return self._gap(base, m, judge)

        def _tick(g: float) -> None:
            nonlocal day, leverage
            day += g
            if backlogged:
                leverage *= math.exp(-cfg.backlog_fatigue_rate * g)
            if adverse:
                leverage *= math.exp(-cfg.adverse_decay_rate * g)

        def advance(g: float) -> None:
            """Advance the calendar across an interval of length ``g`` (already
            generated from the judge state at interval start), evolving the
            judge capacity chain WITHIN the interval: flips are stamped at
            their true sub-interval days, and fatigue is attributed to the
            state actually in force in each sub-interval — so the logged
            episode, the gap-generating state, and the fatigue attribution
            always agree.
            """
            nonlocal day, leverage, backlogged
            lam = cfg.mean_backlog_days if backlogged else cfg.mean_normal_days
            n_flips = int(rng.poisson(g / lam))
            us = sorted(rng.uniform(0.0, 1.0, size=n_flips)) if n_flips else []
            t0 = 0.0
            for u in us:
                _tick((u - t0) * g)
                backlogged = not backlogged
                backlog_log.append((day, "backlog" if backlogged else "normal"))
                t0 = u
            _tick((1.0 - t0) * g)

        def maybe_flip_regime() -> None:
            nonlocal adverse, regime_flip_day, pressure
            if not adverse and rng.random() < cfg.regime_flip_hazard:
                adverse = True
                regime_flip_day = day
                pressure = 0.0  # regime shock resets negotiations

        events: list[CaseEvent] = [CaseEvent(0.0, EventType.FILED, amount=damages)]
        pressure = 0.0
        n_stalls = 0
        settled = False
        recovery = 0.0
        settle_day = -1.0

        def add_pressure(x: float) -> None:
            nonlocal pressure
            pressure += x * (cfg.adverse_pressure_mult if adverse else 1.0)

        def maybe_settle() -> None:
            nonlocal settled, recovery, settle_day
            logit = (
                -2.0
                + pressure
                - (cfg.adverse_accept_penalty if adverse else 0.0)
                - (cfg.backlog_settle_penalty if backlogged else 0.0)
            )
            if not settled and rng.random() < _sigmoid(logit):
                recovery, settle_day = self._settle(day, damages, score, leverage, events)
                settled = True

        # ---------------- pleadings ---------------- #
        advance(gap(25.0))
        events.append(CaseEvent(day, EventType.ANSWER))
        maybe_flip_regime()

        if rng.random() < _sigmoid(0.5 + judge["defense_tilt"] - 0.3 * score):
            advance(gap(45.0))
            granted = rng.random() < _sigmoid(-0.9 + judge["defense_tilt"] - 0.9 * score)
            events.append(CaseEvent(day, EventType.MOTION_TO_DISMISS, flag=float(granted)))
            maybe_flip_regime()
            if granted:
                return self._finish_v2(events, j, d, capability, score, damages,
                                       n_stalls, backlog_log, regime_flip_day)

        # ---------------- discovery ---------------- #
        advance(gap(30.0))
        events.append(CaseEvent(day, EventType.DISCOVERY_OPEN))
        maybe_flip_regime()

        discovery_budget = 240.0 * cong * judge["speed"] * float(rng.lognormal(0.0, 0.25))
        n_steps = int(rng.integers(4, 10))

        for _ in range(n_steps):
            if settled:
                break
            stall_hazard = (
                0.04
                + 0.5 * (1.0 - capability) ** 2 * min(cong, 2.5) / 2.5
                + (cfg.adverse_stall_bonus if adverse else 0.0)
            )
            step_days = discovery_budget / n_steps
            if rng.random() < stall_hazard:
                n_stalls += 1
                granted = rng.random() < 0.6
                advance(step_days * float(rng.uniform(0.3, 0.7))
                        * (cfg.backlog_gap_mult if backlogged else 1.0))
                events.append(CaseEvent(day, EventType.MOTION_TO_COMPEL, flag=float(granted)))
                advance(gap(35.0, mult=1.5 + 2.5 * (1.0 - capability)))
                fragility = 1.0 - _sigmoid(score)
                leverage *= 0.95 - 0.18 * fragility
                pressure -= 0.15
            else:
                advance(step_days * float(rng.uniform(0.7, 1.3))
                        * (cfg.backlog_gap_mult if backlogged else 1.0))
                options = [EventType.DEPOSITION, EventType.EXPERT_DISCLOSURE, EventType.SETTLEMENT_OFFER]
                ev = options[rng.choice(3, p=[0.5, 0.3, 0.2])]
                if ev is EventType.SETTLEMENT_OFFER:
                    add_pressure(0.35)
                    amount = self._expected_settlement(damages, score, leverage) * float(
                        rng.lognormal(0.0, 0.25)
                    )
                    events.append(CaseEvent(day, ev, amount=amount))
                    maybe_settle()
                else:
                    events.append(CaseEvent(day, ev))
            maybe_flip_regime()

        if not settled:
            advance(gap(20.0))
            events.append(CaseEvent(day, EventType.DISCOVERY_CLOSE))
            maybe_flip_regime()

            # ---------------- dispositive motions ---------------- #
            if rng.random() < _sigmoid(0.4 - 0.2 * score + 0.3 * judge["defense_tilt"]):
                advance(gap(75.0))
                granted = rng.random() < _sigmoid(-1.3 + judge["defense_tilt"] - 1.1 * score)
                events.append(CaseEvent(day, EventType.MOTION_SUMMARY_JUDGMENT, flag=float(granted)))
                maybe_flip_regime()
                if granted:
                    return self._finish_v2(events, j, d, capability, score, damages,
                                           n_stalls, backlog_log, regime_flip_day)
                add_pressure(1.2)

            # ---------------- negotiation / pre-trial ---------------- #
            trial_date_set = False
            for _ in range(6):
                if settled:
                    break
                options = [EventType.SETTLEMENT_OFFER, EventType.MEDIATION, EventType.TRIAL_DATE_SET]
                ev = options[rng.choice(3, p=[0.45, 0.25, 0.30])]
                if ev is EventType.TRIAL_DATE_SET and trial_date_set:
                    ev = EventType.SETTLEMENT_OFFER
                advance(gap(40.0, mult=0.5 if trial_date_set else 1.0))
                if ev is EventType.TRIAL_DATE_SET:
                    trial_date_set = True
                    add_pressure(0.9)
                    events.append(CaseEvent(day, ev))
                elif ev is EventType.MEDIATION:
                    add_pressure(0.55)
                    events.append(CaseEvent(day, ev))
                    maybe_settle()
                else:
                    add_pressure(0.4)
                    amount = self._expected_settlement(damages, score, leverage) * float(
                        rng.lognormal(0.0, 0.25)
                    )
                    events.append(CaseEvent(day, ev, amount=amount))
                    maybe_settle()
                maybe_flip_regime()

            # ---------------- trial ---------------- #
            if not settled:
                advance(gap(30.0, mult=0.5 if trial_date_set else 1.0))
                events.append(CaseEvent(day, EventType.TRIAL_START))
                advance(float(rng.uniform(3.0, 15.0)))
                win = rng.random() < _sigmoid(0.9 * score - 0.7 * judge["defense_tilt"])
                recovery = damages * float(rng.uniform(0.15, 0.9)) if win else 0.0
                events.append(CaseEvent(day, EventType.VERDICT, amount=recovery, flag=float(win)))

        return self._finish_v2(events, j, d, capability, score, damages,
                               n_stalls, backlog_log, regime_flip_day,
                               settled=settled, recovery=recovery, settle_day=settle_day)

    def generate_with_latents(self, n: int) -> tuple[list[CaseTimeline], dict[str, dict]]:
        timelines, latents = [], {}
        for _ in range(n):
            tl, lat = self.sample_with_latents()
            timelines.append(tl)
            latents[tl.case_id] = lat
        return timelines, latents

    # ------------------------------------------------------------------ #
    # paired interventional check (preregistration A6.2, amendment v1.2)
    # ------------------------------------------------------------------ #

    def intervene_landmark(self, n_landmarks: int = 200, n_reps: int = 8,
                           treatment: str = "backlog") -> dict:
        """Force a mechanism at sampled mid-negotiation landmarks and continue
        with common random numbers (each arm sees identical innovations up to
        treatment-driven divergence), so differences are attributable to the
        treatment alone.

        ``treatment="backlog"``: forced backlog vs forced normal judge state.
        ``treatment="adverse"``: regime shock (pressure reset, halved
        pressure gains, acceptance penalty) vs continued normal regime, judge
        state held normal in both arms.

        Returns the mean paired next-gap ratio (backlog treatment) and the
        settlement-incidence difference within 180 days of the landmark.
        """
        if treatment not in ("backlog", "adverse"):
            raise ValueError(f"unknown treatment: {treatment!r}")
        cfg = self.config
        landmark_rng = np.random.default_rng(cfg.seed * 1000 + 17)
        gap_ratios, settle_normal, settle_treated = [], [], []
        for lm in range(n_landmarks):
            judge = self.judges_[int(landmark_rng.integers(0, cfg.n_judges))]
            cong = self.districts_[int(landmark_rng.integers(0, cfg.n_districts))]
            score = float(landmark_rng.standard_normal())
            damages = float(landmark_rng.lognormal(mean=math.log(2.0e6), sigma=1.1))
            pressure0 = float(landmark_rng.uniform(1.0, 2.5))
            trial0 = bool(landmark_rng.random() < 0.3)
            for rep in range(n_reps):
                arms = {}
                for treated in (False, True):
                    # common random numbers per (landmark, rep) across arms
                    rng = np.random.default_rng(cfg.seed * 1_000_003 + lm * 97 + rep)
                    arms[treated] = self._continuation(
                        rng, judge=judge, cong=cong, score=score, damages=damages,
                        pressure=pressure0, trial_date_set=trial0,
                        forced_backlog=(treated and treatment == "backlog"),
                        adverse_arm=(treated and treatment == "adverse"))
                gap_ratios.append(arms[True][0] / arms[False][0])
                settle_normal.append(float(arms[False][1]))
                settle_treated.append(float(arms[True][1]))
        return {
            "gap_ratio": float(np.mean(gap_ratios)),
            "settle_normal": float(np.mean(settle_normal)),
            "settle_treated": float(np.mean(settle_treated)),
            "settle_diff": float(np.mean(settle_normal) - np.mean(settle_treated)),
        }

    def _continuation(self, rng, *, judge, cong, score, damages, pressure,
                      trial_date_set, forced_backlog, adverse_arm=False) -> tuple[float, bool]:
        """Negotiation continuation from a landmark, mirroring the main
        flow's negotiation mechanics. Returns (first_gap_days,
        settled_within_180d)."""
        cfg = self.config
        if adverse_arm:
            pressure = 0.0  # regime shock resets negotiations

        def gap(base: float, mult: float = 1.0) -> float:
            m = mult * cong * (cfg.backlog_gap_mult if forced_backlog else 1.0)
            return self._gap(base, m, judge)

        def maybe_settle(p: float) -> bool:
            logit = (
                -2.0 + p
                - (cfg.backlog_settle_penalty if forced_backlog else 0.0)
                - (cfg.adverse_accept_penalty if adverse_arm else 0.0)
            )
            return bool(rng.random() < _sigmoid(logit))

        pressure_mult = cfg.adverse_pressure_mult if adverse_arm else 1.0
        day = 0.0
        first_gap = None
        settled = False
        for _ in range(6):
            options = [EventType.SETTLEMENT_OFFER, EventType.MEDIATION, EventType.TRIAL_DATE_SET]
            ev = options[rng.choice(3, p=[0.45, 0.25, 0.30])]
            if ev is EventType.TRIAL_DATE_SET and trial_date_set:
                ev = EventType.SETTLEMENT_OFFER
            g = gap(40.0, mult=0.5 if trial_date_set else 1.0)
            if first_gap is None:
                first_gap = g
            day += g
            if day > 180.0:
                break
            if ev is EventType.TRIAL_DATE_SET:
                trial_date_set = True
                pressure += 0.9 * pressure_mult
            elif ev is EventType.MEDIATION:
                pressure += 0.55 * pressure_mult
                settled = maybe_settle(pressure)
            else:
                pressure += 0.4 * pressure_mult
                settled = maybe_settle(pressure)
            if settled:
                break
        return first_gap, settled

    # ------------------------------------------------------------------ #

    def _finish_v2(self, true_events, judge_id, district_id, capability, score,
                   damages, n_stalls, backlog_log, regime_flip_day,
                   settled=False, recovery=0.0, settle_day=-1.0):
        """Build the outcome from the TRUE process, then apply selective
        observation to produce the observed docket the model sees."""
        tl_true = self._finish(true_events, judge_id, district_id, capability,
                               score, damages, n_stalls,
                               settled=settled, recovery=recovery, settle_day=settle_day)

        rng = self._rng
        cong_norm = min(self.districts_[district_id], 2.5) / 2.5
        observed: list[CaseEvent] = []
        mask: list[bool] = []
        for i, ev in enumerate(true_events):
            if i == 0 or i == len(true_events) - 1:
                keep = True  # FILED and the terminal event are always docketed
            else:
                base = _OBSERVATION_RATES.get(ev.event_type, 1.0)
                rate = base - _CONGESTION_OBS_PENALTY * cong_norm * (base < 1.0)
                keep = bool(rng.random() < rate)
            mask.append(keep)
            if keep:
                observed.append(ev)

        observed_tl = CaseTimeline(
            case_id=tl_true.case_id,
            events=observed,
            static=tl_true.static,
            judge_id=judge_id,
            district_id=district_id,
            outcome=tl_true.outcome,
        )
        latents = {
            "judge_backlog": backlog_log,
            "backlog_time_fraction": _episode_fraction(backlog_log, observed_tl.duration_days),
            "regime_flip_day": regime_flip_day,
            "adverse_final": regime_flip_day is not None,
            "true_events": [(ev.day, ev.event_type.name) for ev in true_events],
            "observed_mask": mask,
            "n_true_events": len(true_events),
            "n_observed_events": len(observed),
        }
        return observed_tl, latents


def _episode_fraction(log: list[tuple[float, str]], duration: float) -> float:
    """Fraction of the case's calendar spent in backlog episodes."""
    if duration <= 0 or not log:
        return 0.0
    total = 0.0
    for i, (day, state) in enumerate(log):
        end = log[i + 1][0] if i + 1 < len(log) else duration
        if state == "backlog":
            total += max(0.0, min(end, duration) - day)
    return total / duration
