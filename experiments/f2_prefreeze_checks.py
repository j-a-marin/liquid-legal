"""F2 pre-freeze checks for the tf-tpp candidate — FORWARD PASS ONLY.

No optimizer step is run here or anywhere before the freeze recorded in
experiments/F2_FREEZE.md. Checks (printed, all must PASS):

1. parameter count within the declared budget (23,129 +/- 20%);
2. output shapes on a real padded batch (lambdas (B,T,16), main heads (B,T),
   derived next-type probs (B,T,16), next-gap quantiles (B,T,3),
   TPP-derived settle-within-180d probability (B,T));
3. lambda > 0 everywhere (floor 1e-6), including padded positions;
4. no NaN/Inf at any position, including masked/padded ones;
5. closed-form gap quantiles monotone in q (0.1 <= 0.5 <= 0.9);
6. next-type probabilities form a distribution (rows sum to 1);
7. the masked TPP NLL is finite on a real batch.

Run with: cd experiments && ../.venv/bin/python f2_prefreeze_checks.py
"""

from __future__ import annotations

import numpy as np
import torch

from liquid_legal.featurize import collate_timelines
from gen_v2 import GeneratorV2, GeneratorV2Config
from run_all import split
from run_hidden_statics import strip_statics
from tpp_model import LAMBDA_FLOOR, TppTransformerModel

BUDGET_CENTER = 23_129  # tf-native-aux parameter count (of record)
BUDGET_TOL = 0.20


def main() -> None:
    torch.manual_seed(0)
    results: list[tuple[str, bool, str]] = []

    model = TppTransformerModel()
    n_params = sum(p.numel() for p in model.parameters())
    lo, hi = BUDGET_CENTER * (1 - BUDGET_TOL), BUDGET_CENTER * (1 + BUDGET_TOL)
    results.append((
        "param count within 23,129 +/- 20%",
        lo <= n_params <= hi,
        f"{n_params:,} params (budget [{lo:,.0f}, {hi:,.0f}]; expected 23,030)",
    ))

    gen = GeneratorV2(GeneratorV2Config(seed=0))
    tls, _ = gen.generate_with_latents(128)
    _, val = split(strip_statics(tls), 0)
    batch = collate_timelines(val[:32], horizon_days=180)

    with torch.no_grad():
        out = model(batch.event_ids, batch.event_feats, batch.static,
                    timespans=batch.deltas)

    B, T = batch.event_ids.shape
    lam = out["lambdas"]
    shapes_ok = (
        lam.shape == (B, T, 16)
        and out["settle_logit"].shape == (B, T)
        and out["log_recovery"].shape == (B, T)
        and out["log_remaining"].shape == (B, T)
        and out["duration_q"].shape == (B, T, 3)
        and out["next_type_prob"].shape == (B, T, 16)
        and out["next_gap_q"].shape == (B, T, 3)
        and out["settle_prob_180d"].shape == (B, T)
    )
    results.append(("output shapes", shapes_ok,
                    f"lambdas {tuple(lam.shape)}, settle_logit "
                    f"{tuple(out['settle_logit'].shape)}, B={B} T={T}"))

    lam_min = float(lam.min())
    results.append(("lambda > 0 everywhere", bool((lam > 0).all()),
                    f"min lambda = {lam_min:.3e} (floor {LAMBDA_FLOOR:.0e})"))

    finite = all(
        bool(torch.isfinite(out[k]).all())
        for k in ("settle_logit", "log_recovery", "log_remaining",
                  "duration_q", "lambdas", "next_type_prob", "next_gap_q",
                  "settle_prob_180d")
    )
    n_padded = int((batch.mask == 0).sum())
    results.append(("no NaN/Inf incl. padded positions", finite,
                    f"{n_padded} padded positions in batch checked"))

    gq = out["next_gap_q"]
    mono = bool((gq[..., 0] <= gq[..., 1]).all() and (gq[..., 1] <= gq[..., 2]).all())
    results.append(("gap quantiles monotone in q", mono,
                    "q0.1 <= q0.5 <= q0.9 at every position"))

    prob_sums = out["next_type_prob"].sum(-1)
    dist_ok = bool(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5))
    results.append(("next-type probs sum to 1", dist_ok,
                    f"max |sum-1| = {float((prob_sums - 1).abs().max()):.2e}"))

    # Masked TPP NLL on the real batch (value only; no backward, no step).
    with torch.no_grad():
        next_mask = batch.mask[:, 1:]
        tgt = batch.event_ids[:, 1:]
        lam_prev = lam[:, :-1]
        log_lam_tgt = torch.log(lam_prev).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        dt = batch.deltas[:, 1:].clamp(min=0)
        ell = log_lam_tgt - dt * lam_prev.sum(-1)
        nll = -(ell * next_mask).sum() / next_mask.sum().clamp(min=1.0)
    results.append(("masked TPP NLL finite on real batch",
                    bool(torch.isfinite(nll)), f"NLL = {float(nll):.4f}"))

    print("=== F2 pre-freeze checks (FORWARD ONLY — no optimizer step) ===")
    ok_all = True
    for name, ok, detail in results:
        ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\noverall: {'ALL CHECKS PASS' if ok_all else 'FAILURES PRESENT — DO NOT FREEZE'}")
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
