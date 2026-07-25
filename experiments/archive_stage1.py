"""Stage-1 archive builder (protocol preservation requirement).

Re-trains IDN and tf-native-aux under the identical frozen protocol
(deterministic: same frozen code, seeds, splits) to preserve what the
in-process runs discarded: weights and raw holdout predictions. Verifies
each reproduced holdout AUC against the recorded Stage-1 metrics and logs
any drift. Writes a self-contained archive under experiments/archive/.

Run with: python experiments/archive_stage1.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from liquid_legal import TrainConfig, train_model
from liquid_legal.baselines import TemporalTransformerModel
from liquid_legal.featurize import collate_timelines
from gen_v2 import GeneratorV2, GeneratorV2Config
from idn_model import IDNModel
from run_all import split
from run_hidden_statics import strip_statics

ARCHIVE = Path(__file__).parent / "archive" / "stage1-killed"
SEEDS = list(range(10))
CASES = 1024
EPOCHS = 25

MODELS = {
    "idn": lambda seed: IDNModel(),
    "tf-native-aux": lambda seed: TemporalTransformerModel(
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64, max_len=128,
        time_mode="native", auxiliary=True),
}
LRS = {"idn": None, "tf-native-aux": None}  # filled from results JSON


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def holdout_predictions(model, val, cfg) -> dict[str, np.ndarray]:
    model.eval()
    logits, labels, dur_p, dur_t, rec_p, rec_t, masks = [], [], [], [], [], [], []
    for start in range(0, len(val), 256):
        batch = collate_timelines(val[start : start + 256],
                                  horizon_days=cfg.horizon_days)
        kwargs = {}
        if "lengths" in model.forward.__code__.co_varnames:
            kwargs["lengths"] = batch.mask.sum(1).long()
        out = model(batch.event_ids, batch.event_feats, batch.static,
                    timespans=batch.deltas, **kwargs)
        m = batch.mask.bool()
        logits.append(out["settle_logit"][m].numpy())
        labels.append(batch.y_settle[m].numpy())
        dur_p.append(out["log_remaining"][m].numpy())
        dur_t.append(batch.y_remaining[m].numpy())
        rec_p.append(out["log_recovery"][m].numpy())
        rec_t.append(batch.y_recovery[m].numpy())
        masks.append(batch.mask[m].numpy())
    return {
        "settle_logit": np.concatenate(logits),
        "settle_label": np.concatenate(labels),
        "log_remaining_pred": np.concatenate(dur_p),
        "log_remaining_true": np.concatenate(dur_t),
        "log_recovery_pred": np.concatenate(rec_p),
        "log_recovery_true": np.concatenate(rec_t),
    }


def main() -> None:
    t0 = time.time()
    (ARCHIVE / "weights").mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "predictions").mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "code").mkdir(parents=True, exist_ok=True)

    recorded = {
        "idn": json.loads((Path(__file__).parent / "results" / "stage1_idn.json").read_text())["runs"],
        "tf-native-aux": json.loads((Path(__file__).parent / "results" / "stage1_baselines.json").read_text())["runs"],
    }
    lrs = {
        "idn": json.loads((Path(__file__).parent / "results" / "stage1_idn.json").read_text())["selected"],
        "tf-native-aux": json.loads((Path(__file__).parent / "results" / "stage1_baselines.json").read_text())["selected"],
    }

    # code snapshots for self-containment
    for f in ["idn_model.py", "run_stage1.py", "run_stage1_idn.py",
              "../src/liquid_legal/baselines.py", "../src/liquid_legal/train.py"]:
        src = Path(__file__).parent / f
        (ARCHIVE / "code" / src.name).write_bytes(src.read_bytes())

    repro_log = []
    for name, make in MODELS.items():
        for regime in ("visible", "hidden"):
            for seed in SEEDS:
                torch.manual_seed(seed)
                model = make(seed)
                gen = GeneratorV2(GeneratorV2Config(seed=seed))
                tls, _ = gen.generate_with_latents(CASES)
                data = tls if regime == "visible" else strip_statics(tls)
                _, val = split(data, seed)
                lr = lrs[name][f"{name}/{regime}"]["lr"]
                cfg = TrainConfig(epochs=EPOCHS, lr=lr, verbose=False, seed=seed)
                train_model(model, data, cfg)

                tag = f"{name}_{regime}_seed{seed}"
                torch.save(model.state_dict(), ARCHIVE / "weights" / f"{tag}.pt")
                preds = holdout_predictions(model, val, cfg)
                np.savez_compressed(ARCHIVE / "predictions" / f"{tag}.npz", **preds)

                from liquid_legal.metrics import auc_score
                auc = auc_score(preds["settle_label"], preds["settle_logit"])
                rec_auc = recorded[name][f"{name}/{regime}"][seed]["settle_auc"]
                drift = auc - rec_auc
                repro_log.append({"tag": tag, "reproduced_auc": auc,
                                  "recorded_auc": rec_auc, "drift": drift})
                print(f"{tag}: auc={auc:.4f} (recorded {rec_auc:.4f}, drift {drift:+.5f}) "
                      f"({time.time() - t0:.0f}s)", flush=True)

    (ARCHIVE / "reproduction_check.json").write_text(json.dumps(repro_log, indent=2))
    for f in ["stage1_baselines.json", "stage1_idn.json"]:
        shutil.copy(Path(__file__).parent / "results" / f, ARCHIVE / f)
    for f in ["stage1_run.log", "stage1_idn_run.log"]:
        shutil.copy(Path(__file__).parent / "results" / f, ARCHIVE / f)

    hashes = {}
    for p in sorted(ARCHIVE.rglob("*")):
        if p.is_file():
            hashes[str(p.relative_to(ARCHIVE))] = sha256(p)
    (ARCHIVE / "hashes.json").write_text(json.dumps(hashes, indent=2))
    print(f"\narchive complete at {ARCHIVE} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
