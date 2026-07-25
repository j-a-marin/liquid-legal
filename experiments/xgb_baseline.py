"""Non-neural sanity baseline: XGBoost on engineered temporal features.

Answers the reviewer's obligatory question — "could conventional ML on
hand-crafted features solve this?" — with per-timestep predictions on the
same labels, splits, and metrics as the neural models.

Time-aware and time-ablated variants differ only in whether the temporal
columns (day, gaps) are included; counts, amounts, statics, and event-type
one-hots are always present.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from liquid_legal.events import N_EVENT_TYPES, STATIC_FIELDS, CaseTimeline, EventType
from liquid_legal.metrics import auc_score, mae_days, mae_log_dollars

#: Engineered per-prefix features beyond the static covariates.
ENGINEERED = [
    "n_events", "day", "last_delta", "mean_delta", "max_delta",
    "n_offers", "n_compels", "n_mtd", "n_msj", "n_mediations",
    "n_trial_dates", "log_total_offer_amount",
]

#: Columns dropped in the time-ablated variant.
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


def _time_col_indices() -> list[int]:
    return [len(STATIC_FIELDS) + ENGINEERED.index(f) for f in TIME_FEATURES]


def prefix_features(tl: CaseTimeline) -> np.ndarray:
    """One feature row per timestep, built from the prefix up to that step."""
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
        eng = [
            i + 1, ev.day, delta, ev.day / i if i else 0.0, max_delta,
            counts["n_offers"], counts["n_compels"], counts["n_mtd"],
            counts["n_msj"], counts["n_mediations"], counts["n_trial_dates"],
            np.log1p(offer_sum),
        ]
        row = np.concatenate([
            static,
            eng,
            [1.0 if j == int(ev.event_type) else 0.0 for j in range(N_EVENT_TYPES)],
        ])
        rows[i] = row
    return rows


def _labels(tl: CaseTimeline, horizon_days: float):
    n = len(tl.events)
    out = tl.outcome
    settled = bool(out.get("settled", 0.0))
    settle_day = float(out.get("settle_day", -1.0))
    duration = float(out.get("duration_days", tl.events[-1].day))
    ys = np.zeros(n, dtype=np.float32)
    yrem = np.zeros(n, dtype=np.float32)
    for i, ev in enumerate(tl.events):
        if settled and 0.0 <= (settle_day - ev.day) <= horizon_days:
            ys[i] = 1.0
        yrem[i] = np.log1p(max(duration - ev.day, 0.0))
    yrec = np.full(n, np.log1p(max(float(out.get("recovery", 0.0)), 0.0)), dtype=np.float32)
    return ys, yrec, yrem


def _dataset(timelines: list[CaseTimeline], horizon_days: float, time_aware: bool):
    X = np.concatenate([prefix_features(t) for t in timelines])
    ys, yrec, yrem = zip(*(_labels(t, horizon_days) for t in timelines))
    if not time_aware:
        X[:, _time_col_indices()] = 0.0
    return X, np.concatenate(ys), np.concatenate(yrec), np.concatenate(yrem)


def train_xgb(timelines: list[CaseTimeline], horizon_days: float = 180.0,
              time_aware: bool = True, seed: int = 0) -> dict:
    X, ys, yrec, yrem = _dataset(timelines, horizon_days, time_aware)
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.9, colsample_bytree=0.8, random_state=seed)
    clf = XGBClassifier(**params, eval_metric="logloss").fit(X, ys)
    reg_rec = XGBRegressor(**params).fit(X, yrec)
    reg_rem = XGBRegressor(**params).fit(X, yrem)
    return {"clf": clf, "reg_rec": reg_rec, "reg_rem": reg_rem,
            "time_aware": time_aware}


def eval_xgb(models: dict, timelines: list[CaseTimeline],
             horizon_days: float = 180.0) -> dict[str, float]:
    X, ys, yrec, yrem = _dataset(timelines, horizon_days, models["time_aware"])
    p_settle = models["clf"].predict_proba(X)[:, 1]
    return {
        "settle_auc": auc_score(ys, p_settle),
        "duration_mae_days": mae_days(models["reg_rem"].predict(X), yrem),
        "recovery_mae_log": mae_log_dollars(models["reg_rec"].predict(X), yrec),
    }


def train_and_eval_isolated(train_tls: list[CaseTimeline], eval_tls: list[CaseTimeline],
                            seed: int = 0) -> dict[str, dict[str, float]]:
    """Train both variants and evaluate; intended to be called in a spawned
    subprocess (macOS: xgboost's OpenMP runtime can segfault if torch is
    already loaded in the caller)."""
    aware = train_xgb(train_tls, time_aware=True, seed=seed)
    no_time = train_xgb(train_tls, time_aware=False, seed=seed)
    return {
        "aware": eval_xgb(aware, eval_tls),
        "no_time": eval_xgb(no_time, eval_tls),
    }
