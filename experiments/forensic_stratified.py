"""H8 forensic: is the aggregate screen hiding a narrow timing-sensitive effect?

Stratifies the paired IDN vs tf-native-aux comparison (hidden statics,
archived weights) by preceding-gap size, case age, and procedural phase.
This does not reopen the Stage-1 verdict; it prices where a successor
should aim. Output: experiments/results/forensic_stratified.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from liquid_legal import TrainConfig
from liquid_legal.baselines import TemporalTransformerModel
from liquid_legal.featurize import collate_timelines
from liquid_legal.metrics import auc_score
from gen_v2 import GeneratorV2, GeneratorV2Config
from idn_model import IDNModel
from run_all import split
from run_hidden_statics import strip_statics

ARCHIVE = Path(__file__).parent / "archive" / "stage1-killed"
OUT = Path(__file__).parent / "results" / "forensic_stratified.json"
SEEDS = list(range(10))


def make_model(name):
    if name == "idn":
        return IDNModel()
    return TemporalTransformerModel(d_model=32, nhead=4, num_layers=2,
                                    dim_feedforward=64, max_len=128,
                                    time_mode="native", auxiliary=True)


@torch.no_grad()
def per_step(model, val, cfg):
    """Flattened masked per-step (delta_t, case_day, position_frac, logit, label)."""
    model.eval()
    rows = []
    for start in range(0, len(val), 256):
        batch = collate_timelines(val[start : start + 256], horizon_days=cfg.horizon_days)
        kwargs = {}
        if "lengths" in model.forward.__code__.co_varnames:
            kwargs["lengths"] = batch.mask.sum(1).long()
        out = model(batch.event_ids, batch.event_feats, batch.static,
                    timespans=batch.deltas, **kwargs)
        B, T = batch.event_ids.shape
        days = batch.deltas.cumsum(dim=1)
        pos = torch.arange(T).float().unsqueeze(0) / batch.mask.sum(1, keepdim=True).clamp(min=1)
        for b in range(B):
            L = int(batch.mask[b].sum())
            for t in range(L):
                rows.append((float(batch.deltas[b, t]), float(days[b, t]),
                             float(pos[b, t]), float(out["settle_logit"][b, t]),
                             float(batch.y_settle[b, t])))
    return np.array(rows)


def main() -> None:
    cfg = TrainConfig(verbose=False)
    strata = {"delta_quartile": {}, "long_gap": {}, "age_quartile": {}, "phase": {}}
    for seed in SEEDS:
        gen = GeneratorV2(GeneratorV2Config(seed=seed))
        tls, _ = gen.generate_with_latents(1024)
        _, val = split(strip_statics(tls), seed)
        per_model = {}
        for name in ("idn", "tf-native-aux"):
            model = make_model(name)
            model.load_state_dict(torch.load(ARCHIVE / "weights" / f"{name}_hidden_seed{seed}.pt",
                                             weights_only=True))
            per_model[name] = per_step(model, val, cfg)
        # shared strata definitions from IDN rows (same val set, same mask)
        d = per_model["idn"][:, 0]
        q = np.quantile(d, [0.25, 0.5, 0.75])
        age = per_model["idn"][:, 1]
        qa = np.quantile(age, [0.25, 0.5, 0.75])
        bins = {
            "delta_quartile": [d <= q[0], (d > q[0]) & (d <= q[1]), (d > q[1]) & (d <= q[2]), d > q[2]],
            "long_gap": [d <= 90.0, d > 90.0],
            "age_quartile": [age <= qa[0], (age > qa[0]) & (age <= qa[1]),
                             (age > qa[1]) & (age <= qa[2]), age > qa[2]],
            "phase": [per_model["idn"][:, 2] <= 0.33,
                      (per_model["idn"][:, 2] > 0.33) & (per_model["idn"][:, 2] <= 0.66),
                      per_model["idn"][:, 2] > 0.66],
        }
        for sname, masks in bins.items():
            for si, m in enumerate(masks):
                key = f"{sname}[{si}]"
                row = strata[sname].setdefault(key, {"delta": [], "n": []})
                y = per_model["idn"][m, 4]
                if len(np.unique(y)) < 2 or m.sum() < 100:
                    continue
                a_idn = auc_score(y, per_model["idn"][m, 3])
                a_tf = auc_score(y, per_model["tf-native-aux"][m, 3])
                row["delta"].append(a_idn - a_tf)
                row["n"].append(int(m.sum()))
        print(f"seed={seed} done", flush=True)

    summary = {}
    print("\n=== H8: paired ΔAUC (IDN − tf-aux) by stratum (mean over seeds) ===")
    for sname, rows in strata.items():
        summary[sname] = {}
        for key, r in rows.items():
            d = np.array(r["delta"])
            if len(d) == 0:
                continue
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            summary[sname][key] = {
                "mean_delta": float(d.mean()), "ci95": 2.262 * float(se),
                "seeds": len(d), "mean_steps": float(np.mean(r["n"])),
            }
            print(f"{key:22s} Δ={d.mean():+.4f} ± {2.262 * se:.4f} "
                  f"(seeds={len(d)}, steps/seed≈{np.mean(r['n']):.0f})")
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
