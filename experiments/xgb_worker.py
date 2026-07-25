"""XGBoost-only process entry point used by :mod:`run_all`.

This module intentionally imports ``xgb_baseline`` before loading the pickled
experiment objects.  That guarantees XGBoost owns the only initialized OpenMP
runtime in this interpreter on macOS.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

from xgb_baseline import eval_xgb, train_xgb


def main(request_path: str, response_path: str) -> None:
    with Path(request_path).open("rb") as fh:
        train, val, corrupted_sets, horizon, seed = pickle.load(fh)

    out = {}
    for name in ("xgb", "xgb-no-time"):
        models = train_xgb(train, horizon, time_aware=(name == "xgb"), seed=seed)
        out[name] = {
            "e2": eval_xgb(models, val, horizon),
            "e3": {
                key: eval_xgb(models, timelines, horizon)["settle_auc"]
                for key, timelines in corrupted_sets.items()
            },
        }

    with Path(response_path).open("wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
