"""Stage-1 XGBoost worker (subprocess entry point).

Implements the frozen XGBoost procedure from STAGE1_SPEC.md: 8-config grid,
selection on an inner split of seed-0's training portion, refit on the full
training portion, evaluation on the holdout. Runs isolated (macOS OpenMP).

Request pkl: {"train": [...], "val": [...], "seed": int, "horizon": float,
              "select": bool, "config": dict | None}
Response pkl: {"selected": {...} | None, "metrics": {...}}
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Import order matters: xgb_baseline's header imports xgboost BEFORE
# liquid_legal (hence torch), so XGBoost's OpenMP runtime initializes first.
# The pickled request contains liquid_legal objects (torch); xgboost must
# already be resident before that unpickling happens.
from xgb_baseline import _dataset  # noqa: F401

GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in (200, 400)
    for d in (3, 5)
    for lr in (0.05, 0.1)
]
FIXED = {"subsample": 0.9, "colsample_bytree": 0.8}


def inner_split(train, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    n = max(1, int(0.2 * len(train)))
    inner_val = [train[int(i)] for i in idx[:n]]
    inner_train = [train[int(i)] for i in idx[n:]]
    return inner_train, inner_val


def ece(probs, labels, n_bins=15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bucket = np.digitize(probs, edges[1:-1])
    val = 0.0
    for b in range(n_bins):
        m = bucket == b
        if m.any():
            val += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(val)


def fit_eval(cfg, train, ev, horizon, seed):
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier, XGBRegressor

    Xtr, ytr, ytr_rec, ytr_rem = _dataset(train, horizon, True)
    Xev, yev, yev_rec, yev_rem = _dataset(ev, horizon, True)
    params = dict(**FIXED, random_state=seed)
    t0 = time.time()
    clf = XGBClassifier(**cfg, **params, eval_metric="logloss").fit(Xtr, ytr)
    r_rec = XGBRegressor(**cfg, **params).fit(Xtr, ytr_rec)
    r_rem = XGBRegressor(**cfg, **params).fit(Xtr, ytr_rem)
    fit_s = time.time() - t0
    probs = clf.predict_proba(Xev)[:, 1]
    return {
        "settle_auc": float(roc_auc_score(yev, probs)),
        "ece": ece(probs, yev),
        "duration_mae_days": float(np.mean(np.abs(np.expm1(r_rem.predict(Xev)) - np.expm1(yev_rem)))),
        "recovery_mae_log": float(np.mean(np.abs(r_rec.predict(Xev) - yev_rec))),
        "fit_seconds": fit_s,
    }


def main(request_path: str, response_path: str) -> None:
    with Path(request_path).open("rb") as fh:
        job = pickle.load(fh)
    train, val = job["train"], job["val"]
    horizon, seed = job["horizon"], job["seed"]

    out = {"selected": None, "metrics": None}
    cfg = job.get("config")
    if job.get("select"):
        inner_train, inner_val = inner_split(train, seed)
        scored = []
        for cand in GRID:
            m = fit_eval(cand, inner_train, inner_val, horizon, seed)
            scored.append((m["settle_auc"], cand))
        scored.sort(key=lambda x: -x[0])
        cfg = scored[0][1]
        out["selected"] = {**cfg, "inner_val_auc": scored[0][0]}
    out["metrics"] = fit_eval(cfg, train, val, horizon, seed)
    with Path(response_path).open("wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
