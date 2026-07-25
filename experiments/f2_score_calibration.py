"""F2 pre-registration calibration (exploratory, pre-candidate).

Computes the candidate F2 primary score J — masked next-event-type log-loss
plus mean next-gap pinball (0.1/0.5/0.9) over steps with a next event — for
the two frozen Stage-1 models (idn, tf-native-aux) from archived weights.
Purpose: measure the paired noise floor of J so the F2 practical margin can
be frozen above it. No F2 candidate exists yet; nothing here can tune it.

Output: experiments/results/f2_score_calibration.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from liquid_legal import TrainConfig
from liquid_legal.baselines import TemporalTransformerModel
from liquid_legal.featurize import collate_timelines
from gen_v2 import GeneratorV2, GeneratorV2Config
from idn_model import IDNModel
from run_all import split
from run_hidden_statics import strip_statics

ARCHIVE = Path(__file__).parent / "archive" / "stage1-killed"
OUT = Path(__file__).parent / "results" / "f2_score_calibration.json"
SEEDS = list(range(10))
QS = (0.1, 0.5, 0.9)


def make_model(name):
    if name == "idn":
        return IDNModel()
    return TemporalTransformerModel(d_model=32, nhead=4, num_layers=2,
                                    dim_feedforward=64, max_len=128,
                                    time_mode="native", auxiliary=True)


@torch.no_grad()
def joint_score(model, val):
    """Masked next-type log-loss and next-gap pinball on steps with a next event."""
    ce_sum, pin_sum, n = 0.0, 0.0, 0
    for start in range(0, len(val), 256):
        batch = collate_timelines(val[start : start + 256], horizon_days=180)
        kwargs = {}
        if "lengths" in model.forward.__code__.co_varnames:
            kwargs["lengths"] = batch.mask.sum(1).long()
        out = model(batch.event_ids, batch.event_feats, batch.static,
                    timespans=batch.deltas, **kwargs)
        B, T = batch.event_ids.shape
        # step t predicts event t+1; valid where t+1 < length
        next_mask = batch.mask[:, 1:]  # (B, T-1)
        tgt_type = batch.event_ids[:, 1:]
        logits = out["next_type_logit"][:, :-1]
        logp = torch.log_softmax(logits, dim=-1)
        ce = -logp.gather(-1, tgt_type.unsqueeze(-1)).squeeze(-1)
        # next-gap target: log1p(dt_{t+1}) in the same units the heads were
        # trained on (train.py uses log1p of deltas in days)
        tgt_gap = torch.log1p(batch.deltas[:, 1:].clamp(min=0))
        pred_q = out["next_gap_q"][:, :-1]  # (B, T-1, 3)
        pin = torch.zeros_like(tgt_gap)
        for i, q in enumerate(QS):
            err = tgt_gap - pred_q[..., i]
            pin += torch.maximum(q * err, (q - 1.0) * err)
        pin = pin / len(QS)
        m = next_mask.bool()
        ce_sum += float(ce[m].sum())
        pin_sum += float(pin[m].sum())
        n += int(m.sum())
    return ce_sum / n, pin_sum / n, n


def main() -> None:
    cfg = TrainConfig(verbose=False)
    rows = {}
    for seed in SEEDS:
        gen = GeneratorV2(GeneratorV2Config(seed=seed))
        tls, _ = gen.generate_with_latents(1024)
        _, val = split(strip_statics(tls), seed)
        per = {}
        for name in ("idn", "tf-native-aux"):
            model = make_model(name)
            model.load_state_dict(torch.load(
                ARCHIVE / "weights" / f"{name}_hidden_seed{seed}.pt",
                weights_only=True))
            model.eval()
            ce, pin, n = joint_score(model, val)
            per[name] = {"next_type_logloss": ce, "next_gap_pinball": pin,
                         "J": ce + pin, "steps": n}
        rows[str(seed)] = per
        print(f"seed={seed} idn J={per['idn']['J']:.4f} "
              f"tf-aux J={per['tf-native-aux']['J']:.4f}", flush=True)

    d_ce, d_pin, d_J = [], [], []
    for s in SEEDS:
        a, b = rows[str(s)]["idn"], rows[str(s)]["tf-native-aux"]
        d_ce.append(a["next_type_logloss"] - b["next_type_logloss"])
        d_pin.append(a["next_gap_pinball"] - b["next_gap_pinball"])
        d_J.append(a["J"] - b["J"])

    def stats(x):
        x = np.asarray(x)
        return {"mean": float(x.mean()),
                "ci95": float(2.262 * x.std(ddof=1) / np.sqrt(len(x))),
                "per_seed": [float(v) for v in x]}

    j_ref = float(np.mean([rows[str(s)]["tf-native-aux"]["J"] for s in SEEDS]))
    ce_ref = float(np.mean([rows[str(s)]["tf-native-aux"]["next_type_logloss"] for s in SEEDS]))
    pin_ref = float(np.mean([rows[str(s)]["tf-native-aux"]["next_gap_pinball"] for s in SEEDS]))
    out = {
        "note": "paired deltas are idn - tf-native-aux (noise floor between two "
                "equally supervised strong models); J = next-type log-loss + "
                "mean next-gap pinball(0.1/0.5/0.9), hidden statics, holdout",
        "tf_native_aux_reference": {"J": j_ref, "next_type_logloss": ce_ref,
                                    "next_gap_pinball": pin_ref},
        "paired_delta": {"J": stats(d_J), "next_type_logloss": stats(d_ce),
                         "next_gap_pinball": stats(d_pin)},
        "per_seed": rows,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\ntf-aux reference: J={j_ref:.4f} (CE {ce_ref:.4f} + pin {pin_ref:.4f})")
    print(f"paired dJ (idn-tf): {out['paired_delta']['J']['mean']:+.5f} "
          f"+/- {out['paired_delta']['J']['ci95']:.5f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
