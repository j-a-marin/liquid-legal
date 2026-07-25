"""Ablation and stress-test suite supporting the liquid-legal write-up.

Trains one consistent set of models per seed — matched splits, budgets,
heads, losses, and metrics — and runs all analyses on them:

* E2  — the full model matrix: {CfC, LSTM, Transformer} x time variants,
        plus XGBoost on engineered features (non-neural sanity baseline)
* E3  — irregularity stress: event-dropout and timestamp-jitter degradation
        curves on clean-trained models (incomplete/messy dockets)
* E4  — counterfactual judge probe: same docket prefix, swapped judge traits
* E5  — ground-truthed saliency on the native liquid model

The core question: for irregular legal-event trajectories, does predictive
performance come primarily from recurrence, continuous-time dynamics,
attention, or conventional feature engineering?

Writes experiments/results/results.json and prints summary tables.
Runtime: ~25 min on CPU.

Run with: python experiments/run_all.py
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from liquid_legal import (
    CaseTimeline,
    CaseTrajectoryEngine,
    GeneratorConfig,
    SyntheticLitigationGenerator,
    TrainConfig,
    collate_timelines,
    evaluate,
    event_saliency,
    snapshot,
    train_model,
)
from liquid_legal.baselines import LSTMTrajectoryModel, TemporalTransformerModel
from liquid_legal.events import CaseEvent, EventType

SEEDS = [0, 1, 2]
CASES = 1024
EPOCHS = 25
OUT = Path(__file__).parent / "results" / "results.json"

#: Neural model matrix: (name, family, time_mode).
NEURAL_MODELS = [
    ("cfc-native", "cfc", "native"),
    ("cfc-timespans_only", "cfc", "timespans_only"),
    ("cfc-feature", "cfc", "feature"),
    ("cfc-none", "cfc", "none"),
    ("lstm-feature", "lstm", "feature"),
    ("lstm-none", "lstm", "none"),
    ("tf-native", "transformer", "native"),
    ("tf-timespans_only", "transformer", "timespans_only"),
    ("tf-feature", "transformer", "feature"),
    ("tf-none", "transformer", "none"),
]
XGB_MODELS = ["xgb", "xgb-no-time"]
STRESS_NEURAL = ["cfc-native", "cfc-timespans_only", "cfc-feature",
                 "lstm-feature", "tf-native", "tf-feature"]
DROPOUT_PS = [0.0, 0.1, 0.2, 0.3, 0.5]
JITTER_PS = [0.0, 0.1, 0.25]


def make_model(family: str, time_mode: str, seed: int):
    torch.manual_seed(seed)
    if family == "cfc":
        return CaseTrajectoryEngine(units=64, wiring="ncp", ncp_output_size=16,
                                    time_mode=time_mode)
    if family == "lstm":
        return LSTMTrajectoryModel(units=64, time_mode=time_mode)
    if family == "transformer":
        # Sized to sit between the CfC engine (~16k params) and the LSTM
        # (~24k): d_model=32, 2 layers, positional table capped at 128.
        return TemporalTransformerModel(d_model=32, nhead=4, num_layers=2,
                                        dim_feedforward=64, max_len=128,
                                        time_mode=time_mode)
    raise ValueError(family)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def split(timelines: list[CaseTimeline], seed: int, val_fraction: float = 0.2):
    """Mirror train_model's internal split so val sets align across models."""
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(timelines) * val_fraction))
    idx = rng.permutation(len(timelines))
    val_idx = set(int(i) for i in idx[:n_val])
    train = [t for i, t in enumerate(timelines) if i not in val_idx]
    val = [timelines[int(i)] for i in idx[:n_val]]
    return train, val


# --------------------------------------------------------------------- #
# E3: corruption
# --------------------------------------------------------------------- #

def corrupt(tl: CaseTimeline, rng: np.random.Generator,
            dropout_p: float = 0.0, jitter_p: float = 0.0) -> CaseTimeline:
    """Drop non-anchor events and/or jitter timestamps, keeping the outcome.

    FILED and the terminal event are always kept; absolute calendar days are
    preserved so dropping an event merges its gaps (an incomplete docket).
    """
    kept = [tl.events[0]]
    kept.extend(ev for ev in tl.events[1:-1] if rng.random() >= dropout_p)
    kept.append(tl.events[-1])
    days, last = [], 0.0
    for ev in kept:
        d = ev.day * (1.0 + float(rng.uniform(-jitter_p, jitter_p))) if jitter_p else ev.day
        d = max(d, last)
        days.append(d)
        last = d
    days[0] = 0.0
    events = [CaseEvent(d, ev.event_type, ev.amount, ev.flag) for d, ev in zip(days, kept)]
    outcome = dict(tl.outcome)
    outcome["duration_days"] = events[-1].day
    if outcome.get("settled"):
        outcome["settle_day"] = events[-1].day
    return CaseTimeline(tl.case_id, events, dict(tl.static), tl.judge_id, tl.district_id, outcome)


# --------------------------------------------------------------------- #
# E4: counterfactual judge probe
# --------------------------------------------------------------------- #

def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(np.asarray(a)), _ranks(np.asarray(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom else 0.0


def judge_probe(model, judges: list[dict], val: list[CaseTimeline],
                n_prefixes: int = 100) -> float:
    """Mean Spearman(judge_speed, predicted remaining days) over prefixes."""
    rs = []
    for t in val[:n_prefixes]:
        k = max(2, int(t.n_events * 0.5))
        prefix = CaseTimeline(t.case_id, t.events[:k], dict(t.static),
                              t.judge_id, t.district_id, dict(t.outcome))
        preds = []
        for j in judges:
            prefix.static = {
                **t.static,
                "judge_speed": j["speed"],
                "judge_volatility": j["volatility"],
                "judge_defense_tilt": j["defense_tilt"],
            }
            preds.append(snapshot(model, prefix).expected_remaining_days)
        rs.append(spearman([j["speed"] for j in judges], preds))
    return float(np.mean(rs))


# --------------------------------------------------------------------- #
# E5: ground-truthed saliency
# --------------------------------------------------------------------- #

def saliency_by_event(model, val: list[CaseTimeline]) -> dict:
    batch = collate_timelines(val)
    sal = event_saliency(model, batch)  # (B, T)
    ids = batch.event_ids.numpy()
    mask = batch.mask.numpy().astype(bool)
    per_type: dict[str, list[float]] = {}
    for et in EventType:
        sel = mask & (ids == int(et))
        if sel.any():
            per_type[et.name] = [float(sal[sel].mean()), float(sel.sum())]
    caps = np.array([t.static["plaintiff_capability"] for t in val])
    mtc = int(EventType.MOTION_TO_COMPEL)
    out = {"per_type_mean_saliency": per_type}
    for label, rows in [("low_cap", np.where(caps < 0.35)[0]),
                        ("high_cap", np.where(caps > 0.65)[0])]:
        sel = (ids[rows] == mtc) & mask[rows]
        out[f"mtc_saliency_{label}"] = float(sal[rows][sel].mean()) if sel.any() else None
    return out


# --------------------------------------------------------------------- #
# XGBoost runs in a fresh interpreter: on macOS, loading xgboost after torch
# puts two OpenMP runtimes in one process and segfaults. The worker
# trains both variants and evaluates E2 + all E3 corrupted sets, returning
# plain metric dicts.
# --------------------------------------------------------------------- #

def run_xgb_isolated(train, val, corrupted_sets, horizon, seed):
    worker = Path(__file__).with_name("xgb_worker.py")
    with tempfile.TemporaryDirectory(prefix="liquid-neural-xgb-") as tmp:
        request = Path(tmp) / "request.pkl"
        response = Path(tmp) / "response.pkl"
        with request.open("wb") as fh:
            pickle.dump((train, val, corrupted_sets, horizon, seed), fh)
        subprocess.run(
            [sys.executable, str(worker), str(request), str(response)],
            check=True,
        )
        with response.open("rb") as fh:
            return pickle.load(fh)


def make_corrupted_sets(val, seed):
    """The same corrupted holdout sets are used for every model (paired)."""
    sets = {}
    for kind, ps in (("dropout", DROPOUT_PS), ("jitter", JITTER_PS)):
        for p in ps:
            crng = np.random.default_rng(10_000 + seed)
            sets[(kind, str(p))] = [
                corrupt(t, crng,
                        dropout_p=p if kind == "dropout" else 0.0,
                        jitter_p=p if kind == "jitter" else 0.0)
                for t in val
            ]
    return sets


# --------------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()
    results: dict = {"config": {"seeds": SEEDS, "cases": CASES, "epochs": EPOCHS},
                     "params": {}, "e2": {}, "e3": {"dropout": {}, "jitter": {}},
                     "e4": {}, "e5": {}}

    for seed in SEEDS:
        gen = SyntheticLitigationGenerator(GeneratorConfig(seed=seed))
        timelines = gen.generate(CASES)
        train, val = split(timelines, seed)
        cfg = TrainConfig(epochs=EPOCHS, verbose=False, seed=seed)

        models = {}
        for name, family, time_mode in NEURAL_MODELS:
            models[name] = make_model(family, time_mode, seed)
            results["params"][name] = n_params(models[name])
            history = train_model(models[name], timelines, cfg)
            results["e2"].setdefault(name, []).append(history["val"][-1])
            print(f"seed={seed} trained {name:20s} "
                  f"auc={history['val'][-1]['settle_auc']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

        # XGBoost sanity baseline (time-aware and ablated), subprocess-isolated
        corrupted_sets = make_corrupted_sets(val, seed)
        xgb_out = run_xgb_isolated(train, val, corrupted_sets, cfg.horizon_days, seed)
        for name in XGB_MODELS:
            results["e2"].setdefault(name, []).append(xgb_out[name]["e2"])
            print(f"seed={seed} trained {name:20s} auc={xgb_out[name]['e2']['settle_auc']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

        # E3: stress curves on clean-trained models (same corrupted sets per model)
        for (kind, p), corrupted in corrupted_sets.items():
            for name in STRESS_NEURAL:
                m = evaluate(models[name], corrupted, cfg)
                results["e3"][kind].setdefault(name, {}).setdefault(p, []).append(
                    m["settle_auc"])
            for name in XGB_MODELS:
                results["e3"][kind].setdefault(name, {}).setdefault(p, []).append(
                    xgb_out[name]["e3"][(kind, p)])
        print(f"seed={seed} e3 done ({time.time() - t0:.0f}s)", flush=True)

        # E4: counterfactual judge probe
        for name in ["cfc-native", "lstm-feature", "tf-native"]:
            r = judge_probe(models[name], gen.judges_, val)
            results["e4"].setdefault(name, []).append(r)
        print(f"seed={seed} e4 done ({time.time() - t0:.0f}s)", flush=True)

        # E5: ground-truthed saliency (native liquid model only)
        results["e5"][str(seed)] = saliency_by_event(models["cfc-native"], val)
        print(f"seed={seed} e5 done ({time.time() - t0:.0f}s)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT} ({time.time() - t0:.0f}s)")

    print("\n=== E2: full model matrix (mean ± std over seeds) ===")
    for name in [n for n, _, _ in NEURAL_MODELS] + XGB_MODELS:
        runs = results["e2"][name]
        aucs = [r["settle_auc"] for r in runs]
        durs = [r["duration_mae_days"] for r in runs]
        p = f"{results['params'][name]:,}" if name in results["params"] else "n/a"
        print(f"{name:20s} params={p:>8s} auc={np.mean(aucs):.3f}±{np.std(aucs):.3f} "
              f"dur_mae={np.mean(durs):.0f}±{np.std(durs):.0f}d")

    for kind in ("dropout", "jitter"):
        print(f"\n=== E3: {kind} stress (settle AUC, mean over seeds) ===")
        ps = [str(p) for p in (DROPOUT_PS if kind == "dropout" else JITTER_PS)]
        print("model               " + "".join(f"{float(p):>8.0%}" for p in ps))
        for name in STRESS_NEURAL + XGB_MODELS:
            row = results["e3"][kind][name]
            print(f"{name:20s}" + "".join(f"{np.mean(row[p]):>8.3f}" for p in ps))

    print("\n=== E4: judge probe (mean Spearman judge_speed vs predicted remaining days) ===")
    for name, rs in results["e4"].items():
        print(f"{name:20s} r={np.mean(rs):.3f}±{np.std(rs):.3f}")

    print("\n=== E5: saliency by event type (seed 0, mean per occurrence) ===")
    per_type = results["e5"]["0"]["per_type_mean_saliency"]
    for et, (m, n) in sorted(per_type.items(), key=lambda kv: -kv[1][0]):
        print(f"{et:24s} {m:.4f}  (n={int(n)})")
    def format_saliency(value):
        return "n/a" if value is None else f"{value:.4f}"

    low = format_saliency(results["e5"]["0"]["mtc_saliency_low_cap"])
    high = format_saliency(results["e5"]["0"]["mtc_saliency_high_cap"])
    print(f"MOTION_TO_COMPEL saliency low-cap={low} high-cap={high}")


if __name__ == "__main__":
    main()
