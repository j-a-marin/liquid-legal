"""Command-line interface: ``liquid-legal demo`` and ``liquid-legal generate``."""

from __future__ import annotations

import argparse
import json
import math

from .agents import snapshot
from .baselines import LSTMTrajectoryModel
from .events import CaseTimeline
from .interpret import wiring_report
from .models import CaseTrajectoryEngine
from .synthetic import GeneratorConfig, SyntheticLitigationGenerator
from .train import TrainConfig, train_model


def _fmt_auc(x: float) -> str:
    return f"{x:.3f}" if not math.isnan(x) else "n/a"


def _timeline_to_dict(t: CaseTimeline) -> dict:
    return {
        "case_id": t.case_id,
        "judge_id": t.judge_id,
        "district_id": t.district_id,
        "static": t.static,
        "outcome": t.outcome,
        "events": [
            {"day": e.day, "type": e.event_type.name, "amount": e.amount, "flag": e.flag}
            for e in t.events
        ],
    }


def _cmd_generate(args: argparse.Namespace) -> None:
    gen = SyntheticLitigationGenerator(GeneratorConfig(seed=args.seed))
    timelines = gen.generate(args.n)
    with open(args.out, "w") as fh:
        for t in timelines:
            fh.write(json.dumps(_timeline_to_dict(t)) + "\n")
    print(f"wrote {len(timelines)} timelines to {args.out}")


def _run_one(name: str, model, timelines, cfg: TrainConfig) -> dict[str, float]:
    print(f"\n=== {name} ===")
    history = train_model(model, timelines, cfg)
    return history["val"][-1]


def _cmd_demo(args: argparse.Namespace) -> None:
    import torch

    gen = SyntheticLitigationGenerator(GeneratorConfig(seed=args.seed))
    timelines = gen.generate(args.cases)
    settled = sum(t.outcome["settled"] for t in timelines) / len(timelines)
    stalls = sum(t.outcome["n_stalls"] for t in timelines) / len(timelines)
    duration = sum(t.duration_days for t in timelines) / len(timelines)
    print(
        f"generated {len(timelines)} cases | settled {settled:.0%} | "
        f"avg stalls {stalls:.2f} | avg duration {duration:.0f}d"
    )
    # Seed before construction: torch entropy-seeds its default generator at
    # process start, so unseeded construction differs run to run.
    torch.manual_seed(args.seed)

    cfg = TrainConfig(
        epochs=args.epochs, device=args.device, seed=args.seed, verbose=True
    )
    engine = CaseTrajectoryEngine(
        units=args.units, wiring=args.wiring, cell=args.cell
    )
    print(f"engine wiring: {wiring_report(engine.wiring)}")
    results = {"liquid": _run_one(f"{args.cell.upper()} engine", engine, timelines, cfg)}

    if args.baseline:
        lstm = LSTMTrajectoryModel(units=args.units)
        results["lstm"] = _run_one("LSTM baseline", lstm, timelines, cfg)

    print("\n=== final validation metrics ===")
    header = f"{'model':<12} {'bce':>7} {'settle_auc':>10} {'dur_mae_d':>10} {'rec_mae_log':>11}"
    print(header)
    for name, m in results.items():
        print(
            f"{name:<12} {m['bce']:>7.4f} {_fmt_auc(m['settle_auc']):>10} "
            f"{m['duration_mae_days']:>10.1f} {m['recovery_mae_log']:>11.3f}"
        )

    snap = snapshot(engine, timelines[0], horizon_days=cfg.horizon_days, device=args.device)
    print("\nexample snapshot (case 0):")
    print(
        f"  case={snap.case_id} events={snap.n_events} last={snap.last_event} "
        f"day={snap.day:.0f}\n"
        f"  p_settle_180d={snap.p_settle_within_horizon:.2f} "
        f"expected_recovery=${snap.expected_recovery:,.0f} "
        f"eta={snap.expected_remaining_days:.0f}d velocity={snap.velocity}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="liquid-legal",
        description="Liquid neural networks for legal case trajectory modeling.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="synthetic data -> train -> evaluate")
    p_demo.add_argument("--cases", type=int, default=1024)
    p_demo.add_argument("--epochs", type=int, default=25)
    p_demo.add_argument("--units", type=int, default=64)
    p_demo.add_argument("--wiring", choices=["ncp", "fully_connected"], default="ncp")
    p_demo.add_argument("--cell", choices=["cfc", "ltc"], default="cfc")
    p_demo.add_argument("--baseline", action="store_true", help="also train LSTM baseline")
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument("--device", default="cpu")
    p_demo.set_defaults(func=_cmd_demo)

    p_gen = sub.add_parser("generate", help="dump synthetic timelines as JSONL")
    p_gen.add_argument("--n", type=int, default=100)
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.set_defaults(func=_cmd_generate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
