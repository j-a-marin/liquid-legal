"""Regenerate all paper figures with one command:

    .venv/bin/python paper/figures/make_figures.py

Outputs PDF + PNG (200 dpi) into paper/figures/. Grayscale-friendly,
sans-serif. Every number comes from the JSON files in experiments/results/
(read-only; experiments/ is frozen).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
OUT = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "0.2",
    "xtick.color": "0.2",
    "ytick.color": "0.2",
    "text.color": "0.1",
    "axes.labelcolor": "0.1",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

# grayscale-friendly fills
C_LIGHT = "0.93"
C_MID = "0.82"
C_DARK = "0.65"
C_WHITE = "1.0"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def box(ax, x, y, w, h, text, fc=C_LIGHT, fs=8.5, bold_first=False, ec="0.15",
        lw=1.0, tc="0.1", style="round,pad=0.004,rounding_size=0.006", zorder=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder))
    if bold_first and "\n" in text:
        head, rest = text.split("\n", 1)
        ax.text(x + w / 2, y + h - 0.024, head, ha="center", va="top",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)
        ax.text(x + w / 2, y + h - 0.068, rest, ha="center", va="top",
                fontsize=fs - 1.2, color=tc, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3)


def arrow(ax, p1, p2, label=None, fs=7.5, lw=1.2, style="-|>", ls="-",
          color="0.15", rad=0.0, label_offset=(0.0, 0.012), zorder=4):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 lw=lw, ls=ls, color=color, zorder=zorder,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_offset[0]
        my = (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs,
                color="0.25", zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.6))


def new_canvas(w_in, h_in, xlim=(0, 1), ylim=(0, 1)):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------- #
# Fig 1 — Stage-1 IDN architecture (killed candidate)
# --------------------------------------------------------------------- #

def fig_stage1_architecture():
    fig, ax = new_canvas(11.5, 7.4)

    # inputs (left column)
    box(ax, 0.02, 0.80, 0.18, 0.11, "event history\n$e_{\\leq k-1},\\ t_{\\leq k-1}$",
        fc=C_WHITE, fs=9)
    box(ax, 0.02, 0.61, 0.18, 0.09, "interval $\\Delta t_k$", fc=C_WHITE, fs=9)
    box(ax, 0.38, 0.47, 0.16, 0.09, "current event $e_k$", fc=C_WHITE, fs=9)

    # history encoder + context
    box(ax, 0.28, 0.79, 0.30, 0.14,
        "History encoder\ncausal-masked temporal Transformer\n+ position & time encodings",
        fc=C_MID, bold_first=True)
    box(ax, 0.66, 0.80, 0.31, 0.12,
        "context $c_{k-1}$\npre-event only — never sees $e_k$",
        fc=C_DARK, bold_first=True)

    # state partitions
    box(ax, 0.06, 0.18, 0.28, 0.22,
        "Clock partition $z_{clock}$\ncontinuous-time flow (pre-event)\n"
        "$r=\\mathrm{softplus}(R(c))$;  $\\alpha=1-e^{-r\\Delta t_k}$\n"
        "$g=\\sigma(G([c,\\log(1{+}\\Delta t_k)]))$;  $\\beta=g\\cdot\\alpha$\n"
        "$z^-_{clock}=(1-\\beta)\\,z+\\beta\\tanh(T(c))$",
        fc=C_MID, bold_first=True, fs=8)
    box(ax, 0.38, 0.18, 0.24, 0.22,
        "Event partition $z_{event}$\nGRU jump (sees $e_k$)\n\n"
        "$z_{event}=\\mathrm{GRU}(\\,[\\mathrm{emb}(e_k),\\ c_{k-1}]\\,)$",
        fc=C_MID, bold_first=True, fs=8)
    box(ax, 0.66, 0.18, 0.28, 0.22,
        "Context partition $z_{context}$\nblend toward post-event context\n\n"
        "$b=\\sigma(B(c_k))$\n$z_{context}=\\mathrm{LN}((1{-}b)\\,z+b\\,c_k)$",
        fc=C_MID, bold_first=True, fs=8)

    # concat + heads
    box(ax, 0.08, 0.055, 0.46, 0.07,
        "state $z_k=[z_{event},\\ z_{clock},\\ z_{context}]$   (+ static covariates)",
        fc=C_DARK, fs=8.5)
    box(ax, 0.64, 0.095, 0.33, 0.075,
        "Main heads\nsettle · recovery · duration", fc=C_LIGHT, fs=8)
    box(ax, 0.64, 0.005, 0.33, 0.075,
        "Auxiliary heads\nnext-event type · next-gap quantiles · duration quantiles",
        fc=C_LIGHT, fs=7.5)

    # arrows
    arrow(ax, (0.20, 0.855), (0.28, 0.855))                     # history -> encoder
    arrow(ax, (0.58, 0.855), (0.66, 0.855))                     # encoder -> context
    arrow(ax, (0.11, 0.61), (0.17, 0.40))                       # dt -> clock
    arrow(ax, (0.46, 0.47), (0.48, 0.40), rad=-0.1)             # e_k -> event part
    arrow(ax, (0.68, 0.80), (0.13, 0.40), label="$c_{k-1}$", rad=0.0,
          label_offset=(-0.06, 0.03))
    arrow(ax, (0.79, 0.80), (0.50, 0.40), label="$c_{k-1}$", rad=-0.05)
    arrow(ax, (0.88, 0.80), (0.80, 0.40), label="$c_k$ (post-event, legal)",
          label_offset=(0.0, 0.03))
    arrow(ax, (0.20, 0.18), (0.24, 0.125))                      # clock -> state
    arrow(ax, (0.50, 0.18), (0.38, 0.125))                      # event -> state
    arrow(ax, (0.80, 0.18), (0.50, 0.125))                      # context -> state
    arrow(ax, (0.54, 0.115), (0.64, 0.13))                      # state -> main heads
    arrow(ax, (0.50, 0.055), (0.66, 0.04), rad=-0.15)           # state -> aux heads

    ax.text(0.02, 0.005,
            "Chronology contract: the pre-event flow uses only $z_{k-1}$, $c_{k-1}$, "
            "$\\Delta t_k$;  $e_k$ enters exclusively at the jump.\n"
            "State frozen at padded steps; $\\Delta t_k=0$ is the identity flow.",
            fontsize=7.5, style="italic", color="0.3", va="bottom")
    ax.set_title("Stage-1 IDN: hybrid flow–jump architecture (killed candidate)",
                 fontsize=11, pad=8)
    save(fig, "fig_stage1_architecture")


# --------------------------------------------------------------------- #
# Fig 2 — Generator v1 causal mechanisms
# --------------------------------------------------------------------- #

def fig_gen_v1_mechanisms():
    fig, ax = new_canvas(11.5, 7.0)

    # row A: exogenous traits
    box(ax, 0.01, 0.87, 0.21, 0.11,
        "Judge traits\nspeed · erraticness · defense-tilt", fc=C_MID, bold_first=True)
    box(ax, 0.26, 0.87, 0.17, 0.11, "District\ncongestion", fc=C_MID, bold_first=True)
    box(ax, 0.47, 0.87, 0.17, 0.11, "Plaintiff\ncapability", fc=C_MID, bold_first=True)
    box(ax, 0.68, 0.87, 0.14, 0.11, "Case merit\nscore", fc=C_MID, bold_first=True)
    box(ax, 0.86, 0.87, 0.13, 0.11, "Claimed\ndamages", fc=C_MID, bold_first=True)

    # row B: mechanisms
    box(ax, 0.01, 0.58, 0.19, 0.17,
        "Inter-event gaps\nmultiplicative, heavy-tailed;\nrare acceleration / stall\nepisodes",
        fc=C_LIGHT, bold_first=True, fs=8)
    box(ax, 0.24, 0.58, 0.24, 0.17,
        "Discovery stall hazard\n$0.04 + 0.5\\,(1-\\mathrm{cap})^2"
        "\\cdot\\min(c,2.5)/2.5$\nquadratic in incapability",
        fc=C_LIGHT, bold_first=True, fs=8)
    box(ax, 0.52, 0.58, 0.18, 0.17,
        "Dispositive rulings\nMTD / MSJ granted?\n(judge defense-tilt,\ncase score)",
        fc=C_LIGHT, bold_first=True, fs=8)
    box(ax, 0.74, 0.58, 0.19, 0.17,
        "Procedural events\nMSJ denied +1.2 · trial date\n+0.9 · mediation +0.55\n· offer +0.35",
        fc=C_LIGHT, bold_first=True, fs=8)

    # row C: intermediates
    box(ax, 0.14, 0.32, 0.16, 0.15,
        "Stalls\nmotion to compel +\nextended delay;\npressure $-0.15$",
        fc=C_LIGHT, bold_first=True, fs=8)
    box(ax, 0.34, 0.32, 0.16, 0.15,
        "Leverage erosion\n$\\times\\,(0.95-0.18\\cdot"
        "\\mathrm{fragility})$\nper stall; weak cases\nsuffer more",
        fc=C_LIGHT, bold_first=True, fs=8)
    box(ax, 0.70, 0.32, 0.22, 0.15,
        "Settlement pressure\naccumulates over\nprocedural events",
        fc=C_LIGHT, bold_first=True, fs=8)

    # row D: outcomes
    box(ax, 0.16, 0.04, 0.34, 0.14,
        "Settlement & duration\n$P(\\mathrm{settle})=\\sigma(-2+\\mathrm{pressure})$;\n"
        "duration = calendar of gaps & stalls",
        fc=C_DARK, bold_first=True, fs=8)
    box(ax, 0.60, 0.04, 0.35, 0.14,
        "Recovery\ndamages $\\times\\ (0.03+0.42\\,\\sigma(0.9\\,"
        "\\mathrm{score}))\\times$ leverage",
        fc=C_DARK, bold_first=True, fs=8)

    # edges
    arrow(ax, (0.08, 0.87), (0.08, 0.75))                       # judge -> gaps
    arrow(ax, (0.30, 0.87), (0.13, 0.75), rad=0.1)              # cong -> gaps
    arrow(ax, (0.36, 0.87), (0.36, 0.75))                       # cong -> stall hazard
    arrow(ax, (0.55, 0.87), (0.42, 0.75), rad=-0.08,
          label="quadratic in\nincapability", label_offset=(-0.065, 0.005))
    arrow(ax, (0.73, 0.87), (0.63, 0.75), rad=-0.05)            # score -> rulings
    arrow(ax, (0.32, 0.58), (0.24, 0.47))                       # hazard -> stalls
    arrow(ax, (0.30, 0.395), (0.34, 0.395))                     # stalls -> leverage
    arrow(ax, (0.82, 0.58), (0.80, 0.47))                       # events -> pressure
    arrow(ax, (0.05, 0.58), (0.20, 0.18), rad=0.15,
          label="calendar", label_offset=(-0.045, 0.0))         # gaps -> duration
    arrow(ax, (0.60, 0.58), (0.48, 0.18), rad=0.1,
          label="dismissal /\ndenial", label_offset=(0.065, 0.01))  # rulings -> outcomes
    arrow(ax, (0.24, 0.32), (0.30, 0.18))                       # stalls -> duration
    arrow(ax, (0.48, 0.36), (0.62, 0.18), rad=-0.08)            # leverage -> recovery
    arrow(ax, (0.76, 0.32), (0.44, 0.18), rad=-0.1)             # pressure -> settlement
    arrow(ax, (0.965, 0.87), (0.945, 0.18), rad=0.15)           # damages -> recovery

    ax.set_title("Generator v1: planted causal mechanisms (synthetic litigation world)",
                 fontsize=11, pad=8)
    save(fig, "fig_gen_v1_mechanisms")


# --------------------------------------------------------------------- #
# Fig 3 — Generator v2 latents and selective observation
# --------------------------------------------------------------------- #

def fig_gen_v2_latents():
    fig, ax = new_canvas(11.5, 7.2)

    # latent layer
    ax.add_patch(FancyBboxPatch((0.015, 0.60), 0.97, 0.37,
                                boxstyle="round,pad=0.004", fc="0.97", ec="0.6",
                                ls="--", lw=1.0, zorder=1))
    ax.text(0.03, 0.945, "LATENT LAYER — hidden from the model, recorded in the latent logs",
            fontsize=8.5, fontweight="bold", color="0.35")

    box(ax, 0.04, 0.63, 0.44, 0.27,
        "Judge backlog episodes\ntwo-state chain (normal $\\rightleftharpoons$ backlogged);\n"
        "flips within intervals at true sub-interval days\n"
        "gaps $\\times 2.5$ while backlogged; settle logit $-2$\n"
        "fatigue: leverage $\\times e^{-0.0015\\,d}$ per backlog day",
        fc=C_MID, bold_first=True, fs=8)
    box(ax, 0.52, 0.63, 0.44, 0.27,
        "Hidden case regime\nNormal $\\to$ Adverse flip (0.04 / event, unobserved)\n"
        "settlement pressure reset to 0; gains halved\n"
        "stall hazard $+0.15$; accept logit $-0.8$\n"
        "leverage decays $e^{-0.002\\,d}$ per day while adverse",
        fc=C_MID, bold_first=True, fs=8)

    # process layer
    box(ax, 0.06, 0.40, 0.54, 0.13,
        "True event process\nfull event log — every filing, offer, deposition, ruling",
        fc=C_LIGHT, bold_first=True)
    box(ax, 0.06, 0.22, 0.54, 0.12,
        "Observation mask\ndeposition-class events dropped (base rates 0.85–0.95);\n"
        "drop rate rises with district congestion; FILED & terminal kept",
        fc=C_LIGHT, bold_first=True, fs=8)
    box(ax, 0.06, 0.045, 0.54, 0.11,
        "Observed docket\nwhat the model sees (and is trained/evaluated on)",
        fc=C_DARK, bold_first=True)

    # side annotation: what the logs record
    box(ax, 0.68, 0.22, 0.29, 0.31,
        "Latent logs record\n(case-level ground truth)\n\n"
        "· backlog episode log & time fraction\n"
        "· regime flip day\n"
        "· true event log\n"
        "· observation mask\n\n"
        "used for acceptance tests and\nstratified forensics only",
        fc=C_WHITE, bold_first=True, fs=8)

    # arrows: latents drive the true process
    arrow(ax, (0.26, 0.63), (0.28, 0.53))
    arrow(ax, (0.60, 0.63), (0.40, 0.53), rad=0.1)
    # latents are recorded (dashed)
    arrow(ax, (0.48, 0.70), (0.72, 0.53), ls="--", color="0.55", rad=-0.1)
    arrow(ax, (0.85, 0.63), (0.83, 0.53), ls="--", color="0.55")
    # observation pipeline
    arrow(ax, (0.33, 0.40), (0.33, 0.34))
    ax.text(0.345, 0.368, "selectively observed", fontsize=7.5, color="0.3", ha="left")
    arrow(ax, (0.33, 0.22), (0.33, 0.155))
    # mask reconciles logs and docket
    arrow(ax, (0.60, 0.26), (0.68, 0.30), ls="--", color="0.55", style="<|-|>",
          label="mask reconciles\ndocket ↔ true log", label_offset=(0.0, -0.055), fs=6.5)

    ax.text(0.02, 0.005,
            "Labels (settlement, recovery, duration) derive from the TRUE process; "
            "the model never sees the mask, the backlog state, or the regime.",
            fontsize=7.5, style="italic", color="0.3", va="bottom")
    ax.set_title("Generator v2: latent processes and selective observation",
                 fontsize=11, pad=8)
    save(fig, "fig_gen_v2_latents")


# --------------------------------------------------------------------- #
# Fig 4 — protocol flowchart
# --------------------------------------------------------------------- #

def fig_protocol_flowchart():
    fig, ax = new_canvas(9.5, 8.5)

    box(ax, 0.10, 0.88, 0.50, 0.09,
        "Generator v2 acceptance\nA1–A6, architecture-independent  →  PASS",
        fc=C_LIGHT, bold_first=True)
    box(ax, 0.10, 0.73, 0.50, 0.09,
        "Freeze\nsha256 hashes · seeds 0–9 · STAGE1_SPEC (3rd freeze)",
        fc=C_LIGHT, bold_first=True)
    box(ax, 0.10, 0.56, 0.50, 0.11,
        "Train frozen core baselines + IDN\n10 paired seeds · equal supervision\n"
        "(tf-native-aux carries IDN's auxiliary heads)",
        fc=C_LIGHT, bold_first=True, fs=8)

    # decision diamond
    dx, dy, dw, dh = 0.35, 0.40, 0.16, 0.085
    ax.add_patch(plt.Polygon([(dx - dw, dy), (dx, dy + dh), (dx + dw, dy), (dx, dy - dh)],
                             closed=True, fc=C_MID, ec="0.15", lw=1.0, zorder=2))
    ax.text(dx, dy + 0.002, "Primary screen\nmean paired $\\Delta$AUC $\\geq +0.01$\n"
            "AND 95% CI excludes 0\nAND no regression",
            ha="center", va="center", fontsize=7, zorder=3)

    box(ax, 0.10, 0.135, 0.50, 0.11,
        "FAIL\nmean $\\Delta$AUC $= -0.0039$;  95% CI $[-0.0082, +0.0004]$\n"
        "duration MAE and ECE within bounds (regression checks pass)",
        fc=C_DARK, bold_first=True, fs=8)
    box(ax, 0.10, 0.01, 0.50, 0.09,
        "Kill hybrid track + archive\nweights · predictions · code · hashes",
        fc="0.35", bold_first=True, tc="white")

    # gated path (NOT RUN)
    box(ax, 0.70, 0.50, 0.28, 0.10,
        "Mechanism ablations\n(event / clock / context / static)",
        fc=C_WHITE, bold_first=True, fs=8)
    box(ax, 0.70, 0.35, 0.28, 0.09, "Hostile-world battery",
        fc=C_WHITE, bold_first=True, fs=8)
    ax.text(0.84, 0.305, "NOT RUN — gated on\nprovisional survival",
            ha="center", fontsize=8, fontweight="bold", color="0.45")
    arrow(ax, (dx + dw, dy + 0.01), (0.70, 0.545), ls="--", color="0.5",
          label="PASS", label_offset=(0.0, 0.03))
    arrow(ax, (0.84, 0.50), (0.84, 0.44), ls="--", color="0.5")

    # main flow arrows
    arrow(ax, (0.35, 0.88), (0.35, 0.82))
    arrow(ax, (0.35, 0.73), (0.35, 0.67))
    arrow(ax, (0.35, 0.56), (0.35, 0.485))
    arrow(ax, (0.35, 0.315), (0.35, 0.245), label="FAIL", label_offset=(0.04, 0.0))
    arrow(ax, (0.35, 0.135), (0.35, 0.10))

    ax.set_title("Stage-1 evaluation protocol: preregistered screen and kill path",
                 fontsize=11, pad=8)
    save(fig, "fig_protocol_flowchart")


# --------------------------------------------------------------------- #
# Fig 5 — paired per-seed dAUC
# --------------------------------------------------------------------- #

def fig_paired_dauc():
    base = json.loads((RESULTS / "stage1_baselines.json").read_text())
    idn = json.loads((RESULTS / "stage1_idn.json").read_text())
    auc_b = [r["settle_auc"] for r in base["runs"]["tf-native-aux/hidden"]]
    auc_i = [r["settle_auc"] for r in idn["runs"]["idn/hidden"]]
    assert len(auc_b) == len(auc_i) == 10
    deltas = np.array(auc_i) - np.array(auc_b)
    seeds = np.arange(10)
    mean = float(deltas.mean())
    se = float(deltas.std(ddof=1) / np.sqrt(len(deltas)))
    ci = 2.262 * se  # t, df=9
    lo, hi = mean - ci, mean + ci

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(0.0, color="0.4", lw=0.9, ls="--")
    ax.axhspan(lo, hi, color="0.85", zorder=0,
               label=f"95% CI [{lo:+.4f}, {hi:+.4f}]")
    ax.axhline(mean, color="0.1", lw=1.4, label=f"mean = {mean:+.4f}")
    ax.axhline(0.01, color="0.1", lw=1.1, ls=":",
               label="preregistered requirement ≥ +0.01")
    ax.plot(seeds, deltas, "-o", color="0.25", mfc="white", mec="0.1",
            ms=6, lw=1.0, zorder=5, label="per-seed ΔAUC (IDN − tf-native-aux)")
    for s, d in zip(seeds, deltas):
        ax.annotate(f"{d:+.3f}", (s, d), textcoords="offset points",
                    xytext=(0, 8 if d >= mean else -13), ha="center",
                    fontsize=7, color="0.35")
    ax.set_xticks(seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel("Δ settle AUC (hidden statics)")
    ax.set_title("Primary screen: paired per-seed ΔAUC, IDN − tf-native-aux", fontsize=10.5)
    ax.set_ylim(-0.022, 0.020)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True, edgecolor="0.7")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_paired_dauc")


# --------------------------------------------------------------------- #
# Fig 6 — stratified exploratory dAUC
# --------------------------------------------------------------------- #

def fig_stratified_dauc():
    data = json.loads((RESULTS / "forensic_stratified.json").read_text())

    groups = [
        ("preceding-gap quartile (short → long)",
         [("Q1", "delta_quartile", "delta_quartile[0]"),
          ("Q2", "delta_quartile", "delta_quartile[1]"),
          ("Q3", "delta_quartile", "delta_quartile[2]"),
          ("Q4", "delta_quartile", "delta_quartile[3]")]),
        ("long gap",
         [("≤ 90 d", "long_gap", "long_gap[0]"),
          ("> 90 d", "long_gap", "long_gap[1]")]),
        ("case-age quartile (young → old)",
         [("Q1", "age_quartile", "age_quartile[0]"),
          ("Q2", "age_quartile", "age_quartile[1]"),
          ("Q3", "age_quartile", "age_quartile[2]"),
          ("Q4", "age_quartile", "age_quartile[3]")]),
        ("procedural phase",
         [("early", "phase", "phase[0]"),
          ("mid", "phase", "phase[1]"),
          ("late", "phase", "phase[2]")]),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.axhline(0.0, color="0.4", lw=0.9, ls="--")
    tick_labels, tick_pos = [], []
    x = 0.0
    for gi, (gname, strata) in enumerate(groups):
        start = x
        for label, sname, key in strata:
            row = data[sname][key]
            m, c = row["mean_delta"], row["ci95"]
            ax.errorbar(x, m, yerr=c, fmt="o", color="0.15", mfc="white",
                        mec="0.1", ms=6, capsize=3, lw=1.1, zorder=5)
            ax.annotate(f"{m:+.3f}", (x, m + c), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=6.5, color="0.4")
            tick_labels.append(label)
            tick_pos.append(x)
            x += 1.0
        ax.text((start + x - 1) / 2, -0.046, gname, ha="center", fontsize=8,
                fontweight="bold", color="0.25")
        if gi < len(groups) - 1:
            ax.axvline(x - 0.5, color="0.8", lw=0.7)
            x += 0.8
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("paired ΔAUC (IDN − tf-native-aux)\nmean ± 95% CI, 10 seeds")
    ax.set_ylim(-0.052, 0.030)
    ax.set_title("EXPLORATORY — stratified paired ΔAUC on archived predictions "
                 "(hidden statics)", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig_stratified_dauc")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_stage1_architecture()
    fig_gen_v1_mechanisms()
    fig_gen_v2_latents()
    fig_protocol_flowchart()
    fig_paired_dauc()
    fig_stratified_dauc()


if __name__ == "__main__":
    main()
