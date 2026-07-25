"""sklearn-only probe worker for generator validation (subprocess entry point).

Kept torch-free and xgboost-free so it can run safely alongside the neural
validation suite on macOS (OpenMP runtime conflicts). Receives feature
matrices by pickle; trains HistGradientBoosting probes; returns metrics.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


def main(request_path: str, response_path: str) -> None:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.metrics import mean_absolute_error, roc_auc_score

    with Path(request_path).open("rb") as fh:
        jobs = pickle.load(fh)

    out = {}
    for job in jobs:
        params = dict(max_iter=200, max_depth=4, learning_rate=0.08,
                      random_state=job.get("seed", 0))
        if job["kind"] == "clf":
            model = HistGradientBoostingClassifier(**params)
            model.fit(job["X_train"], job["y_train"])
            score = model.predict_proba(job["X_eval"])[:, 1]
            out[job["name"]] = float(roc_auc_score(job["y_eval"], score))
        else:
            model = HistGradientBoostingRegressor(**params)
            model.fit(job["X_train"], job["y_train"])
            pred = model.predict(job["X_eval"])
            out[job["name"]] = float(mean_absolute_error(job["y_eval"], pred))

    with Path(response_path).open("wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
