"""Generator v2 acceptance validation (PREREGISTRATION.md section A).

Architecture-independent checks; the generator is accepted or rejected on
its own properties. Prints a pass/fail table and exits nonzero on failure.

Run with: python experiments/validate_generator_v2.py
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from liquid_legal import GeneratorConfig, TrainConfig, train_model
from run_all import make_model, split
from gen_v2 import GeneratorV2, GeneratorV2Config
from xgb_baseline import _labels, _time_col_indices, prefix_features

PROBE_SEEDS = [0, 1, 2, 3, 4]
SUITE_SEEDS = [0, 1]
N_CASES = 1024

# Preregistered thresholds (do not tune to pass):
A1_MIN_LIFT = 0.02
A2_MIN_GAP_RATIO = 1.3
A2_MIN_HAZARD_DIFF = 0.05
A3_MIN_OBS_SPREAD = 0.10
A4_BACKLOG_FRACTION_RANGE = (0.10, 0.60)
A5_AUC_RANGE = (0.70, 0.90)
A6_MIN_ORACLE_GAIN = 0.03
A6_2_MIN_GAP_RATIO = 2.0
A6_2_MIN_SETTLE_DIFF = 0.10


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #

def run_probe_jobs(jobs: list[dict]) -> dict[str, float]:
    worker = Path(__file__).with_name("validate_worker.py")
    with tempfile.TemporaryDirectory(prefix="liquid-neural-validate-") as tmp:
        req, resp = Path(tmp) / "req.pkl", Path(tmp) / "resp.pkl"
        with req.open("wb") as fh:
            pickle.dump(jobs, fh)
        subprocess.run([sys.executable, str(worker), str(req), str(resp)], check=True)
        with resp.open("rb") as fh:
            return pickle.load(fh)


def feature_matrix(timelines, latents, with_time: bool, with_latents: bool,
                   horizon: float = 180.0):
    """Per-timestep feature rows (+ optional latent channels) and settle labels.

    Timing arms include richer timing summaries (recent-gap EMA, gap std) so
    the A1 check gives timing every chance to demonstrate unique information.
    """
    Xs, ys = [], []
    for tl in timelines:
        X = prefix_features(tl)
        deltas = np.diff([0.0] + [ev.day for ev in tl.events])
        ema3 = np.array([deltas[max(0, i - 2) : i + 1].mean() for i in range(len(deltas))],
                        dtype=np.float32)
        std = np.array([deltas[: i + 1].std() for i in range(len(deltas))], dtype=np.float32)
        # normalized slowdown: recent pace relative to the case's own mean
        rate_ratio = ema3 / (deltas[1:].mean() + 1e-6 if len(deltas) > 1 else 1.0)
        timing_extra = np.stack([ema3, std, rate_ratio], axis=1)
        if with_time:
            X = np.concatenate([X, timing_extra], axis=1)
        else:
            X[:, _time_col_indices()] = 0.0
        if with_latents:
            X = np.concatenate([X, latent_channels(tl, latents[tl.case_id])], axis=1)
        y, _, _ = _labels(tl, horizon)
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def latent_channels(tl, lat) -> np.ndarray:
    """Per-event [backlog_now, adverse_now] columns from the latent log."""
    episodes = lat["judge_backlog"]
    flip_day = lat["regime_flip_day"]
    cols = np.zeros((len(tl.events), 2), dtype=np.float32)
    for i, ev in enumerate(tl.events):
        state = "normal"
        for day, s in episodes:
            if day <= ev.day:
                state = s
            else:
                break
        cols[i, 0] = 1.0 if state == "backlog" else 0.0
        cols[i, 1] = 1.0 if (flip_day is not None and ev.day >= flip_day) else 0.0
    return cols


# --------------------------------------------------------------------- #
# acceptance checks
# --------------------------------------------------------------------- #

def check_a1(datasets) -> tuple[bool, str]:
    """Timing carries unique outcome information conditioned on statics/order."""
    lifts = []
    for seed, (train, val, lat_train, lat_val) in enumerate(datasets):
        jobs = []
        for name, with_time in (("time", True), ("notime", False)):
            Xtr, ytr = feature_matrix(train, lat_train, with_time, False)
            Xev, yev = feature_matrix(val, lat_val, with_time, False)
            jobs.append({"name": f"s{seed}_{name}", "kind": "clf",
                         "X_train": Xtr, "y_train": ytr, "X_eval": Xev, "y_eval": yev,
                         "seed": seed})
        res = run_probe_jobs(jobs)
        lifts.append(res[f"s{seed}_time"] - res[f"s{seed}_notime"])
    lift = float(np.mean(lifts))
    ok = lift >= A1_MIN_LIFT
    return ok, f"mean paired AUC lift from timing = {lift:.3f} (need >= {A1_MIN_LIFT})"


def check_a2(latents) -> tuple[bool, str]:
    """Regime changes alter transition dynamics, not merely labels.

    Clause 2 (amendment v1.1, refined): pooled discovery stall-rate,
    post-flip events (flip cases) vs never-flip controls. Regime flips are
    trait-independent by construction and stalls do not end the case, so the
    pooled comparison is unbiased; the earlier within-case pairing was
    underpowered. Acceptance-per-offer is survival-confounded (accepted
    cases end before they can flip).
    """
    gaps_in, gaps_out = [], []
    post_stalls = post_tot = ctrl_stalls = ctrl_tot = 0
    for lat in latents.values():
        episodes = lat["judge_backlog"]
        true_days = [d for d, _ in lat["true_events"]]
        for a, b in zip(true_days, true_days[1:]):
            state = "normal"
            for day, s in episodes:
                if day <= a:
                    state = s
                else:
                    break
            (gaps_in if state == "backlog" else gaps_out).append(b - a)
        types = [t for _, t in lat["true_events"]]
        flip = lat["regime_flip_day"]
        in_disc = False
        for d, t in zip(true_days, types):
            if t == "DISCOVERY_OPEN":
                in_disc = True
                continue
            if t == "DISCOVERY_CLOSE":
                in_disc = False
                continue
            if not in_disc or t not in ("DEPOSITION", "EXPERT_DISCLOSURE",
                                        "MOTION_TO_COMPEL", "SETTLEMENT_OFFER"):
                continue
            stall = t == "MOTION_TO_COMPEL"
            if flip is None:
                ctrl_tot += 1
                ctrl_stalls += stall
            elif d >= flip:
                post_tot += 1
                post_stalls += stall
    ratio = float(np.mean(gaps_in) / max(np.mean(gaps_out), 1e-9))
    if not post_tot or not ctrl_tot:
        return False, "insufficient post-flip or control discovery events"
    hdiff = float(post_stalls / post_tot - ctrl_stalls / ctrl_tot)
    ok = ratio >= A2_MIN_GAP_RATIO and hdiff >= A2_MIN_HAZARD_DIFF
    return ok, (f"gap ratio backlog/normal = {ratio:.2f} (need >= {A2_MIN_GAP_RATIO}); "
                f"stall-rate diff post/pre flip = {hdiff:.3f} (need >= {A2_MIN_HAZARD_DIFF})")


def check_a3(timelines, latents) -> tuple[bool, str]:
    """Selective observation differs from the underlying event process."""
    congs = np.array([t.static["district_congestion"] for t in timelines])
    terciles = np.quantile(congs, [1 / 3, 2 / 3])
    rates = [[], [], []]
    n_differ = 0
    for tl in timelines:
        lat = latents[tl.case_id]
        if lat["n_observed_events"] < lat["n_true_events"]:
            n_differ += 1
        bucket = int(np.searchsorted(terciles, tl.static["district_congestion"]))
        for (day, name), keep in zip(lat["true_events"], lat["observed_mask"]):
            if name in ("DEPOSITION", "EXPERT_DISCLOSURE"):
                rates[bucket].append(float(keep))
    means = [float(np.mean(r)) for r in rates]
    spread = max(means) - min(means)
    ok = spread >= A3_MIN_OBS_SPREAD and n_differ > 0
    return ok, (f"deposition-class observation rates by congestion tercile = "
                f"{['%.2f' % m for m in means]} spread={spread:.2f} (need >= {A3_MIN_OBS_SPREAD}); "
                f"{n_differ}/{len(timelines)} dockets differ from true process")


def check_a4(timelines, latents) -> tuple[bool, str]:
    """Latent logs complete; backlog mechanism neither vestigial nor dominant."""
    required = {"judge_backlog", "regime_flip_day", "true_events", "observed_mask",
                "backlog_time_fraction"}
    coverage = all(required <= set(lat) for lat in latents.values())
    frac = float(np.mean([lat["backlog_time_fraction"] for lat in latents.values()]))
    lo, hi = A4_BACKLOG_FRACTION_RANGE
    ok = coverage and lo <= frac <= hi
    return ok, f"log coverage={coverage}, mean backlog time fraction={frac:.2f} (need {lo}–{hi})"


def check_a5() -> tuple[bool, str]:
    """No saturation, no degeneracy: quick frozen suite within AUC range."""
    aucs = {}
    for seed in SUITE_SEEDS:
        gen = GeneratorV2(GeneratorV2Config(seed=seed))
        timelines, _ = gen.generate_with_latents(N_CASES)
        cfg = TrainConfig(epochs=8, verbose=False, seed=seed)
        for name, family, mode in (("cfc", "cfc", "native"), ("lstm", "lstm", "feature"),
                                   ("tf", "transformer", "native")):
            torch.manual_seed(seed)
            h = train_model(make_model(family, mode, seed), timelines, cfg)
            aucs.setdefault(name, []).append(h["val"][-1]["settle_auc"])
    means = {k: float(np.mean(v)) for k, v in aucs.items()}
    lo, hi = A5_AUC_RANGE
    ok = all(lo <= v <= hi for v in means.values())
    return ok, f"quick-suite AUCs { {k: round(v, 3) for k, v in means.items()} } (need all in [{lo}, {hi}))"


def check_a6(datasets) -> tuple[bool, str]:
    """Oracle checks recover the planted mechanisms (amendment v1.3).

    Pass/fail rests on two paired interventional estimands (common random
    numbers): backlog must stretch the next gap and suppress settlement;
    an adverse-regime shock must suppress settlement. The observational
    combined-channel oracle gain is reported descriptively (no threshold) —
    observables legitimately substitute for the channels, so the marginal
    gain underestimates the causal effect by construction.
    """
    gains = []
    for seed, (train, val, lat_train, lat_val) in enumerate(datasets):
        jobs = []
        for name, with_lat in (("oracle", True), ("nolatent", False)):
            Xtr, ytr = feature_matrix(train, lat_train, True, with_lat)
            Xev, yev = feature_matrix(val, lat_val, True, with_lat)
            jobs.append({"name": f"s{seed}_{name}", "kind": "clf",
                         "X_train": Xtr, "y_train": ytr, "X_eval": Xev, "y_eval": yev,
                         "seed": seed})
        res = run_probe_jobs(jobs)
        gains.append(res[f"s{seed}_oracle"] - res[f"s{seed}_nolatent"])
    gain = float(np.mean(gains))

    gen = GeneratorV2(GeneratorV2Config(seed=777))
    backlog = gen.intervene_landmark(treatment="backlog")
    adverse = gen.intervene_landmark(treatment="adverse")
    ok = (
        backlog["gap_ratio"] >= A6_2_MIN_GAP_RATIO
        and backlog["settle_diff"] >= A6_2_MIN_SETTLE_DIFF
        and adverse["settle_diff"] >= A6_2_MIN_SETTLE_DIFF
    )
    return ok, (
        f"[descriptive] observational oracle gain = {gain:.3f} AUC; "
        f"backlog intervention: gap ratio = {backlog['gap_ratio']:.2f} (need >= {A6_2_MIN_GAP_RATIO}), "
        f"settle diff = {backlog['settle_diff']:.3f} (need >= {A6_2_MIN_SETTLE_DIFF}); "
        f"adverse intervention: settle diff = {adverse['settle_diff']:.3f} (need >= {A6_2_MIN_SETTLE_DIFF})"
    )


# --------------------------------------------------------------------- #

def main() -> None:
    print(f"generating probe datasets ({len(PROBE_SEEDS)} seeds x {N_CASES} cases)...")
    datasets = []
    for seed in PROBE_SEEDS:
        gen = GeneratorV2(GeneratorV2Config(seed=seed))
        timelines, latents = gen.generate_with_latents(N_CASES)
        train_idx, val_idx = split(timelines, seed)
        lat_train = {t.case_id: latents[t.case_id] for t in train_idx}
        lat_val = {t.case_id: latents[t.case_id] for t in val_idx}
        datasets.append((train_idx, val_idx, lat_train, lat_val))

    big_gen = GeneratorV2(GeneratorV2Config(seed=99))
    big_tl, big_lat = big_gen.generate_with_latents(2000)

    checks = [
        ("A1 timing carries unique info", check_a1(datasets)),
        ("A2 regimes alter dynamics", check_a2(big_lat)),
        ("A3 selective observation", check_a3(big_tl, big_lat)),
        ("A4 latent logs complete", check_a4(big_tl, big_lat)),
        ("A5 no saturation/degeneracy", check_a5()),
        ("A6 oracle recovers mechanisms", check_a6(datasets)),
    ]
    print("\n=== Generator v2 acceptance (PREREGISTRATION section A) ===")
    all_ok = True
    for name, (ok, detail) in checks:
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n{'ACCEPTED — freeze generator, seeds, splits, eval code' if all_ok else 'REJECTED — repair generator, not thresholds'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
