"""Stage-1 IDN run + paired primary screen — STAGE1_SPEC.md steps 2–3.

Trains IDN with the identical frozen protocol (same data, seeds, splits,
selection rule, epochs) as the baseline suite, then computes the paired
primary comparison against tf-native-aux:

  provisional survival := mean paired ΔAUC >= 0.01 AND paired 95% t-CI
  excluding zero AND no material regression (duration MAE within +5%,
  ECE within +0.01).

Outputs: experiments/results/stage1_idn.json + printed screen.
Run with: python experiments/run_stage1_idn.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from liquid_legal import TrainConfig, evaluate, train_model
from gen_v2 import GeneratorV2, GeneratorV2Config
from idn_model import IDNModel
from run_all import split
from run_hidden_statics import strip_statics
from run_stage1 import LR_GRID, REGIMES, evaluate_full

SEEDS = list(range(10))
CASES = 1024
EPOCHS = 25
OUT = Path(__file__).parent / "results" / "stage1_idn.json"
BASELINES = Path(__file__).parent / "results" / "stage1_baselines.json"


def make_idn(seed: int) -> IDNModel:
    torch.manual_seed(seed)
    return IDNModel()


def select_lr(train, seed: int) -> float:
    inner_train, inner_val = split(train, seed)
    best_lr, best_auc = None, -1.0
    for lr in LR_GRID:
        cfg = TrainConfig(epochs=EPOCHS, lr=lr, verbose=False, seed=seed)
        model = make_idn(seed)
        train_model(model, inner_train, cfg)
        auc = evaluate(model, inner_val, cfg)["settle_auc"]
        print(f"  lr={lr} inner_val_auc={auc:.3f}", flush=True)
        if auc > best_auc:
            best_lr, best_auc = lr, auc
    return best_lr


def paired_t_ci(deltas: list[float]) -> tuple[float, float, float]:
    """mean, half-width of 95% paired t-interval (df = n-1)."""
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    mean = float(d.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    se = float(d.std(ddof=1) / np.sqrt(n))
    # t_{0.975, 9} = 2.262
    return mean, 2.262 * se, se


def main() -> None:
    t0 = time.time()
    results = {"selected": {}, "runs": {}}

    gen0 = GeneratorV2(GeneratorV2Config(seed=0))
    tls0, _ = gen0.generate_with_latents(CASES)
    for regime, data0 in (("visible", tls0), ("hidden", strip_statics(tls0))):
        train0, _ = split(data0, 0)
        lr = select_lr(train0, 0)
        results["selected"][f"idn/{regime}"] = {"lr": lr}
        print(f"selected lr={lr} for idn/{regime} ({time.time() - t0:.0f}s)", flush=True)

    for seed in SEEDS:
        gen = GeneratorV2(GeneratorV2Config(seed=seed))
        tls, _ = gen.generate_with_latents(CASES)
        for regime, data in (("visible", tls), ("hidden", strip_statics(tls))):
            _, val = split(data, seed)
            cfg = TrainConfig(epochs=EPOCHS, lr=results["selected"][f"idn/{regime}"]["lr"],
                              verbose=False, seed=seed)
            model = make_idn(seed)
            train_model(model, data, cfg)
            m = evaluate_full(model, val, cfg)
            m["params"] = sum(p.numel() for p in model.parameters())
            results["runs"].setdefault(f"idn/{regime}", []).append(m)
            print(f"seed={seed} idn/{regime} auc={m['settle_auc']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT} ({time.time() - t0:.0f}s)")

    # ---------------- paired primary screen ---------------- #
    base = json.loads(BASELINES.read_text())
    idn_h = [r["settle_auc"] for r in results["runs"]["idn/hidden"]]
    tf_h = [r["settle_auc"] for r in base["runs"]["tf-native-aux/hidden"]]
    deltas = [a - b for a, b in zip(idn_h, tf_h)]
    mean, half, se = paired_t_ci(deltas)
    idn_dur = np.mean([r["duration_mae_days"] for r in results["runs"]["idn/hidden"]])
    tf_dur = np.mean([r["duration_mae_days"] for r in base["runs"]["tf-native-aux/hidden"]])
    idn_ece = np.mean([r["ece"] for r in results["runs"]["idn/hidden"]])
    tf_ece = np.mean([r["ece"] for r in base["runs"]["tf-native-aux/hidden"]])

    pass_auc = mean >= 0.01 and (mean - half) > 0.0
    pass_dur = idn_dur <= tf_dur * 1.05
    pass_ece = idn_ece <= tf_ece + 0.01
    verdict = pass_auc and pass_dur and pass_ece

    print("\n=== paired primary screen (hidden statics, 10 seeds) ===")
    print(f"per-seed ΔAUC (IDN − tf-native-aux): {['%+.3f' % d for d in deltas]}")
    print(f"mean ΔAUC = {mean:+.4f}, 95% paired CI = [{mean - half:+.4f}, {mean + half:+.4f}]")
    print(f"duration MAE: IDN {idn_dur:.0f}d vs tf-aux {tf_dur:.0f}d "
          f"({'ok' if pass_dur else 'REGRESSION'})")
    print(f"ECE: IDN {idn_ece:.3f} vs tf-aux {tf_ece:.3f} "
          f"({'ok' if pass_ece else 'REGRESSION'})")
    print(f"\nPROVISIONAL SURVIVAL: {'PASS' if verdict else 'FAIL — stop the hybrid track'}")


if __name__ == "__main__":
    main()
