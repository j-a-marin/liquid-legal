"""DIAGRAM 1 — Institutional state as a continuous field.

A matter trajectory moving through a field shaped by judge, court, procedural
regime, party resources, and doctrine. Filings and rulings appear as impulses;
settlement, dismissal, and judgment as boundaries; a period of silence still
contains drift.

Matplotlib 3D surface + seaborn theming. Sequential single-hue surface
(magnitude), one warm accent for the trajectory (identity), ink for labels.
Outputs: PNG (Substack) + PDF (article).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

sns.set_theme(style="white", font="DejaVu Sans")

rng = np.random.default_rng(7)

# ── The institutional field ─────────────────────────────────────────────────
# x: calendar time (days). y: settlement pressure (latent). z: field intensity
# (how strongly institutional conditions push the matter's evolution).
T, P = np.meshgrid(np.linspace(0, 900, 220), np.linspace(0, 1, 160))

# Baseline: pressure raises intensity, gently nonlinear.
Z = 0.55 * P**1.6

# Judge/court congestion: backlog episodes as ridges across time bands.
for center, width, height in [(180, 55, 0.30), (430, 70, 0.38), (700, 45, 0.22)]:
    Z += height * np.exp(-((T - center) ** 2) / (2 * width**2)) * (0.35 + 0.65 * P)

# Doctrine/procedural regime: a slow swell late in the calendar.
Z += 0.16 * (T / 900) ** 2

# Resource asymmetry: a shallow well at low pressure early (cheap to do nothing).
Z -= 0.10 * np.exp(-(((T - 90) / 130) ** 2 + ((P - 0.12) / 0.18) ** 2))

# Settlement basin: intensity funnels down as pressure saturates near the end.
Z -= 0.35 * np.exp(-(((T - 860) / 90) ** 2 + ((P - 0.92) / 0.16) ** 2))

# ── The matter trajectory ───────────────────────────────────────────────────
# Piecewise: drift during silence, jumps at events.
events = [
    (60, 0.06, "complaint"),
    (150, 0.06, "answer"),
    (260, 0.09, "discovery order"),
    (410, 0.13, "motion denied"),
    (620, 0.10, "trial date set"),
    (760, 0.08, "mediation"),
]
t_traj = np.linspace(0, 880, 600)
p_traj = np.zeros_like(t_traj) + 0.06
for et, jump, _ in events:
    p_traj += jump / (1 + np.exp(-(t_traj - et) / 6))          # event impulse
p_traj += 0.06 * (t_traj / 880) ** 2                            # slow drift
p_traj += 0.012 * np.sin(t_traj / 47) * (t_traj / 880)          # micro-drift
p_traj = np.clip(p_traj, 0, 0.97)

def field(t, p):
    """Sample Z at (t, p) by nearest grid lookup."""
    ti = np.clip(np.searchsorted(T[0], t), 0, T.shape[1] - 1)
    pi = np.clip(np.searchsorted(P[:, 0], p), 0, P.shape[0] - 1)
    return Z[pi, ti]

z_traj = np.array([field(t, p) for t, p in zip(t_traj, p_traj)]) + 0.03

# ── Figure ──────────────────────────────────────────────────────────────────
INK = "#1F2937"
MUTED = "#6B7280"
ACCENT = "#C2410C"          # trajectory (identity)
BOUND = "#111827"           # terminal boundary (ink, not a series color)

# Sequential single hue: light slate-blue → deep indigo (magnitude job).
cmap = LinearSegmentedColormap.from_list(
    "field", ["#EEF2F9", "#C7D5EC", "#8FA9D6", "#5577B5", "#2E4D86", "#1B3159"]
)

fig = plt.figure(figsize=(11, 7.5), dpi=200)
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=26, azim=-58)
ax.invert_yaxis()  # low-pressure lane (the trajectory) faces the camera
ax.set_box_aspect((2.1, 1.0, 0.55))

ax.plot_surface(
    T, P, Z, cmap=cmap, rcount=110, ccount=110,
    linewidth=0, antialiased=True, alpha=0.92,
)
# Trajectory draped on the surface.
ax.plot(t_traj, p_traj, z_traj, color=ACCENT, lw=3.2, zorder=10)
# Event impulses: thin stems from floor to path.
for et, _, name in events:
    i = np.searchsorted(t_traj, et)
    zbase = field(et, 0.0)
    ax.plot([et, et], [p_traj[i], p_traj[i]], [zbase - 0.06, z_traj[i]],
            color=MUTED, lw=1.2, alpha=0.9, zorder=9)
    ax.scatter([et], [p_traj[i]], [z_traj[i]], color=ACCENT, s=34,
               edgecolor="white", linewidth=0.9, zorder=11, depthshade=False)

# Terminal boundary: settlement edge as a bold rim at the far end.
tb = np.full(40, 880.0)
pb = np.linspace(0.55, 0.97, 40)
zb = np.array([field(880, p) for p in pb]) + 0.02
ax.plot(tb, pb, zb, color=BOUND, lw=3.2, zorder=8)

# ── Annotations (screen space: always on top, never occluded) ──────────────
ax.text2D(0.14, 0.80, "institutional field\n(judge · court · regime · resources · doctrine)",
          transform=ax.transAxes, color="#2E4D86", fontsize=10, ha="left")
ax.text2D(0.30, 0.68, "rulings and filings arrive as impulses",
          transform=ax.transAxes, color=INK, fontsize=10, ha="center")
ax.text2D(0.63, 0.38, "silence — the state still drifts",
          transform=ax.transAxes, color=MUTED, fontsize=10, style="italic", ha="center")
ax.text2D(0.86, 0.62, "terminal boundary\n(settlement · dismissal · judgment)",
          transform=ax.transAxes, color=BOUND, fontsize=10, ha="center")

ax.set_xlabel("calendar time (days)", color=INK, fontsize=10, labelpad=10)
ax.set_ylabel("settlement pressure", color=INK, fontsize=10, labelpad=8)
ax.set_zlabel("field intensity", color=INK, fontsize=10, labelpad=6)
ax.tick_params(colors=MUTED, labelsize=8)

# Recessive panes.
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.set_pane_color((1, 1, 1, 0))
    pane._axinfo["grid"].update(color="#E5E7EB", linewidth=0.5)
ax.set_zlim(Z.min() - 0.06, Z.max() + 0.05)

fig.tight_layout()
fig.savefig("fig_institutional_field_3d.png", dpi=300, bbox_inches="tight")
fig.savefig("fig_institutional_field_3d.pdf", bbox_inches="tight")
print("saved png+pdf")
