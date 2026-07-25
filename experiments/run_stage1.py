"""Stage-1 frozen baseline suite — STAGE1_SPEC.md execution order, step 1.

Runs tf-native-aux, tf-native, lstm-feature, cfc-native, and xgb across the
10 frozen seeds and both regimes (visible / hidden statics), with leakage-
free config selection (inner split of seed-0's training portion) and full
metric capture (ECE, next-event-type accuracy, latency).

Outputs: experiments/results/stage1_baselines.json
Runtime: ~30 min on CPU. Run with: python experiments/run_stage1.py
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

from liquid_legal import CaseTrajectoryEngine, TrainConfig, evaluate, train_model
from liquid_legal.baselines import LSTMTrajectoryModel, TemporalTransformerModel
from liquid_legal.featurize import collate_timelines
from gen_v2 import GeneratorV2, GeneratorV2Config
from run_all import split
from run_hidden_statics import strip_statics

SEEDS = list(range(10))
CASES = 1024
EPOCHS = 25
LR_GRID = [1e-3, 3e-3]
REGIMES = ("visible", "hidden")
NEURAL = ("tf-native-aux", "tf-native", "lstm-feature", "cfc-native")
OUT = Path(__file__).parent / "results" / "stage1_baselines.json"


def make_neural(name: str, seed: int):
    torch.manual_seed(seed)
    if name == "tf-native-aux":
        return TemporalTransformerModel(d_model=32, nhead=4, num_layers=2,
                                        dim_feedforward=64, max_len=128,
                                        time_mode="native", auxiliary=True)
    if name == "tf-native":
        return TemporalTransformerModel(d_model=32, nhead=4, num_layers=2,
                                        dim_feedforward=64, max_len=128,
                                        time_mode="native")
    if name == "lstm-feature":
        return LSTMTrajectoryModel(units=64, time_mode="feature")
    if name == "cfc-native":
        return CaseTrajectoryEngine(units=64, wiring="ncp", ncp_output_size=16,
                                    time_mode="native")
    raise ValueError(name)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bucket = np.digitize(probs, edges[1:-1])
    val = 0.0
    for b in range(n_bins):
        m = bucket == b
        if m.any():
            val += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(val)


@torch.no_grad()
def evaluate_full(model, val, cfg) -> dict:
    """evaluate() metrics + ECE + next-event-type accuracy + batch-1 latency."""
    metrics = evaluate(model, val, cfg)
    model.eval()
    probs, labels, hits, n_next = [], [], 0, 0
    for start in range(0, len(val), 256):
        batch = collate_timelines(val[start : start + 256], horizon_days=cfg.horizon_days)
        out = model(batch.event_ids, batch.event_feats, batch.static,
                    timespans=batch.deltas,
                    **({"lengths": batch.mask.sum(1).long()}
                       if "lengths" in model.forward.__code__.co_varnames else {}))
        m = batch.mask.bool()
        probs.append(torch.sigmoid(out["settle_logit"])[m].numpy())
        labels.append(batch.y_settle[m].numpy())
        if "next_type_logit" in out:
            pred = out["next_type_logit"][:, :-1].argmax(-1)
            nm = batch.mask[:, 1:].bool()
            hits += int((pred[nm] == batch.event_ids[:, 1:][nm]).sum())
            n_next += int(nm.sum())
    metrics["ece"] = ece(np.concatenate(probs), np.concatenate(labels))
    metrics["next_type_acc"] = hits / max(n_next, 1)
    # latency: batch-1 forward, mean over 100 cases after 20-case warmup
    singles = val[:120]
    for t in singles[:20]:
        b = collate_timelines([t], horizon_days=cfg.horizon_days)
        model(b.event_ids, b.event_feats, b.static, timespans=b.deltas)
    t0 = time.time()
    for t in singles[20:]:
        b = collate_timelines([t], horizon_days=cfg.horizon_days)
        model(b.event_ids, b.event_feats, b.static, timespans=b.deltas)
    metrics["latency_ms"] = (time.time() - t0) / 100 * 1000
    return metrics


def select_lr(name: str, train, seed: int) -> float:
    """Leakage-free selection: inner split of the TRAIN portion only."""
    inner_train, inner_val = split(train, seed)
    best_lr, best_auc = None, -1.0
    for lr in LR_GRID:
        cfg = TrainConfig(epochs=EPOCHS, lr=lr, verbose=False, seed=seed)
        model = make_neural(name, seed)
        train_model(model, inner_train, cfg)
        auc = evaluate(model, inner_val, cfg)["settle_auc"]
        if auc > best_auc:
            best_lr, best_auc = lr, auc
    return best_lr


def run_xgb(train, val, seed: int, select: bool, config: dict | None) -> dict:
    worker = Path(__file__).with_name("stage1_xgb_worker.py")
    with tempfile.TemporaryDirectory(prefix="liquid-neural-stage1xgb-") as tmp:
        req, resp = Path(tmp) / "req.pkl", Path(tmp) / "resp.pkl"
        with req.open("wb") as fh:
            pickle.dump({"train": train, "val": val, "seed": seed, "horizon": 180.0,
                         "select": select, "config": config}, fh)
        subprocess.run([sys.executable, str(worker), str(req), str(resp)], check=True)
        with resp.open("rb") as fh:
            return pickle.load(fh)


def main() -> None:
    t0 = time.time()
    results = {"freeze": "STAGE1_SPEC.md (third freeze)", "selected": {}, "runs": {}}

    # ---- selection on seed 0 (inner split of train portion) ---- #
    gen0 = GeneratorV2(GeneratorV2Config(seed=0))
    tls0, _ = gen0.generate_with_latents(CASES)
    data0 = {"visible": tls0, "hidden": strip_statics(tls0)}
    xgb_configs = {}
    for regime in REGIMES:
        train0, val0 = split(data0[regime], 0)
        for name in NEURAL:
            lr = select_lr(name, train0, 0)
            results["selected"][f"{name}/{regime}"] = {"lr": lr}
            print(f"selected lr={lr} for {name}/{regime} ({time.time() - t0:.0f}s)", flush=True)
        out = run_xgb(train0, val0, 0, select=True, config=None)
        xgb_configs[regime] = {k: v for k, v in out["selected"].items() if k != "inner_val_auc"}
        results["selected"][f"xgb/{regime}"] = out["selected"]
        print(f"selected xgb config {out['selected']} for {regime} ({time.time() - t0:.0f}s)",
              flush=True)

    # ---- frozen runs ---- #
    for seed in SEEDS:
        gen = GeneratorV2(GeneratorV2Config(seed=seed))
        tls, _ = gen.generate_with_latents(CASES)
        data = {"visible": tls, "hidden": strip_statics(tls)}
        for regime in REGIMES:
            train, val = split(data[regime], seed)
            for name in NEURAL:
                cfg = TrainConfig(epochs=EPOCHS, lr=results["selected"][f"{name}/{regime}"]["lr"],
                                  verbose=False, seed=seed)
                model = make_neural(name, seed)
                train_model(model, data[regime], cfg)
                m = evaluate_full(model, val, cfg)
                m["params"] = sum(p.numel() for p in model.parameters())
                results["runs"].setdefault(f"{name}/{regime}", []).append(m)
                print(f"seed={seed} {name}/{regime} auc={m['settle_auc']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            m = run_xgb(train, val, seed, select=False, config=xgb_configs[regime])["metrics"]
            results["runs"].setdefault(f"xgb/{regime}", []).append(m)
            print(f"seed={seed} xgb/{regime} auc={m['settle_auc']:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT} ({time.time() - t0:.0f}s)")

    print("\n=== Stage-1 baselines (mean ± std over 10 seeds) ===")
    for key, runs in results["runs"].items():
        aucs = [r["settle_auc"] for r in runs]
        durs = [r["duration_mae_days"] for r in runs]
        print(f"{key:22s} auc={np.mean(aucs):.3f}±{np.std(aucs):.3f} "
              f"dur_mae={np.mean(durs):.0f}±{np.std(durs):.0f}d")


if __name__ == "__main__":
    main()
