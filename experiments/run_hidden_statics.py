"""E6 — hidden-latent ablation: who needs the covariates?

The synthetic world's static features (judge speed/volatility/tilt, district
congestion, plaintiff capability, damages) are unusually informative — in
effect the model is told the latents that *cause* the irregular timing, which
makes the timing channel redundant (E2's finding). Real dockets are not like
that: no underwriter observes true judge volatility.

This experiment zeroes the static covariates so that event semantics and
*the timestamps themselves* must carry the signal — the condition
continuous-time models are built for. Hypothesis: models with native or
explicit time handling degrade less than feature-only models.

Same splits, budgets, heads, losses, and metrics as run_all.py.
Runtime: ~8 min on CPU. Run with: python experiments/run_hidden_statics.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from liquid_legal import (
    CaseTimeline,
    GeneratorConfig,
    SyntheticLitigationGenerator,
    TrainConfig,
    train_model,
)
from run_all import NEURAL_MODELS, XGB_MODELS, make_model, n_params, run_xgb_isolated, split

SEEDS = [0, 1, 2]
CASES = 1024
EPOCHS = 25
OUT = Path(__file__).parent / "results" / "results_hidden_statics.json"

#: Representative subset spanning each family and time mechanism.
MODELS = [
    ("cfc-native", "cfc", "native"),
    ("cfc-timespans_only", "cfc", "timespans_only"),
    ("cfc-feature", "cfc", "feature"),
    ("lstm-feature", "lstm", "feature"),
    ("tf-native", "transformer", "native"),
    ("tf-feature", "transformer", "feature"),
]


def strip_statics(timelines: list[CaseTimeline]) -> list[CaseTimeline]:
    """Same dockets, but the model is no longer told the latent traits."""
    return [
        CaseTimeline(t.case_id, t.events, {}, t.judge_id, t.district_id, dict(t.outcome))
        for t in timelines
    ]


def main() -> None:
    t0 = time.time()
    results: dict = {"config": {"seeds": SEEDS, "cases": CASES, "epochs": EPOCHS,
                                "statics": "zeroed"}, "params": {}, "e6": {}}

    for seed in SEEDS:
        gen = SyntheticLitigationGenerator(GeneratorConfig(seed=seed))
        timelines = strip_statics(gen.generate(CASES))
        train, val = split(timelines, seed)
        cfg = TrainConfig(epochs=EPOCHS, verbose=False, seed=seed)

        for name, family, time_mode in MODELS:
            model = make_model(family, time_mode, seed)
            results["params"][name] = n_params(model)
            history = train_model(model, timelines, cfg)
            results["e6"].setdefault(name, []).append(history["val"][-1])
            print(f"seed={seed} trained {name:20s} "
                  f"auc={history['val'][-1]['settle_auc']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

        xgb_out = run_xgb_isolated(train, val, {}, cfg.horizon_days, seed)
        for name in XGB_MODELS:
            results["e6"].setdefault(name, []).append(xgb_out[name]["e2"])
            print(f"seed={seed} trained {name:20s} "
                  f"auc={xgb_out[name]['e2']['settle_auc']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT} ({time.time() - t0:.0f}s)")

    print("\n=== E6: hidden statics (mean ± std over seeds) ===")
    for name in [n for n, _, _ in MODELS] + XGB_MODELS:
        runs = results["e6"][name]
        aucs = [r["settle_auc"] for r in runs]
        durs = [r["duration_mae_days"] for r in runs]
        print(f"{name:20s} auc={np.mean(aucs):.3f}±{np.std(aucs):.3f} "
              f"dur_mae={np.mean(durs):.0f}±{np.std(durs):.0f}d")


if __name__ == "__main__":
    main()
