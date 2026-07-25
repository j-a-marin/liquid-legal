"""Reproducible benchmark: CfC (NCP wiring) vs LSTM over multiple seeds.

Single-run figures on CPU are not bit-stable across processes (torch
entropy-seeds its default generator at startup, and recurrent training
amplifies init differences), so this reports mean +/- std over seeds
0, 1, 2 — each seed drawing a fresh synthetic dataset, model init, and
train/val split. Runtime: ~10 min on CPU.

Run with: python examples/benchmark.py
"""

import numpy as np
import torch

from liquid_legal import (
    CaseTrajectoryEngine,
    GeneratorConfig,
    SyntheticLitigationGenerator,
    TrainConfig,
    train_model,
)
from liquid_legal.baselines import LSTMTrajectoryModel

SEEDS = [0, 1, 2]
CASES = 1024
EPOCHS = 25

MODELS = {
    "cfc-ncp": lambda: CaseTrajectoryEngine(units=64, wiring="ncp", ncp_output_size=16),
    "lstm": lambda: LSTMTrajectoryModel(units=64),
}


def run(seed: int, make_model) -> dict[str, float]:
    torch.manual_seed(seed)
    timelines = SyntheticLitigationGenerator(GeneratorConfig(seed=seed)).generate(CASES)
    history = train_model(
        make_model(), timelines, TrainConfig(epochs=EPOCHS, verbose=False, seed=seed)
    )
    return history["val"][-1]


def main() -> None:
    keys = ["bce", "settle_auc", "duration_mae_days", "recovery_mae_log"]
    print(f"seeds={SEEDS} cases={CASES} epochs={EPOCHS}\n")
    for name, make in MODELS.items():
        runs = [run(seed, make) for seed in SEEDS]
        per_seed = "  ".join(
            f"s{seed}: auc={m['settle_auc']:.3f} dur={m['duration_mae_days']:.0f}"
            for seed, m in zip(SEEDS, runs)
        )
        print(f"[{name}] {per_seed}")
        stats = {k: (float(np.mean([m[k] for m in runs])), float(np.std([m[k] for m in runs]))) for k in keys}
        print(
            f"[{name}] mean±std  bce={stats['bce'][0]:.3f}±{stats['bce'][1]:.3f}  "
            f"settle_auc={stats['settle_auc'][0]:.3f}±{stats['settle_auc'][1]:.3f}  "
            f"dur_mae_d={stats['duration_mae_days'][0]:.0f}±{stats['duration_mae_days'][1]:.0f}  "
            f"rec_mae_log={stats['recovery_mae_log'][0]:.2f}±{stats['recovery_mae_log'][1]:.2f}\n"
        )


if __name__ == "__main__":
    main()
