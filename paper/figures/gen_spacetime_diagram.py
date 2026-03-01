#!/usr/bin/env python3
"""
gen_spacetime_diagram.py
=========================
Figure: Price-time Minkowski diagram.

Shows:
  - The light cone at origin (|β| = 1 boundaries)
  - TIMELIKE / SPACELIKE / LIGHTLIKE regions coloured
  - Example bar trajectories in each region
  - Hyperbola of constant proper time
  - Lorentz boost visualisation (boosted axes)

Output: paper/figures/spacetime_diagram.pdf
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BG      = "#0D1117"
SURFACE = "#161B22"
BORDER  = "#30363D"
TEXT    = "#C9D1D9"
MUTED   = "#8B949E"
BLUE    = "#58A6FF"
RED     = "#F85149"
GREEN   = "#3FB950"
YELLOW  = "#D29922"
PURPLE  = "#BC8CFF"

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "font.size":         9,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
})

LIM = 3.0


def make_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    # ── Background regions ────────────────────────────────────────────────────
    x = np.linspace(-LIM, LIM, 400)

    # TIMELIKE: |x| < ct  (upper and lower)
    ax.fill_between(x, x,  LIM, where=(np.abs(x) <= x), alpha=0.0)

    # Upper TIMELIKE cone (future)
    ax.fill_between(x,  np.abs(x), LIM, alpha=0.18, color=BLUE)
    # Lower TIMELIKE cone (past)
    ax.fill_between(x, -LIM, -np.abs(x), alpha=0.18, color=BLUE)
    # SPACELIKE: left and right
    x_sp = np.linspace(0, LIM, 200)
    ax.fill_between( x_sp,  x_sp, -x_sp, alpha=0.18, color=RED)
    ax.fill_between(-x_sp,  x_sp, -x_sp, alpha=0.18, color=RED)

    # ── Light cone ────────────────────────────────────────────────────────────
    for slope in [1, -1]:
        ax.plot(x, slope * x, color=YELLOW, linewidth=1.8,
                label="Light cone ($|\\beta|=1$)" if slope == 1 else "")

    # ── Axes (ct vertical, x horizontal) ─────────────────────────────────────
    ax.axhline(0, color=BORDER, linewidth=0.8)
    ax.axvline(0, color=BORDER, linewidth=0.8)
    ax.set_xlabel("Log-price displacement  $\\Delta x$  (normalised by $c_{\\mathrm{mkt}}\\Delta t$)")
    ax.set_ylabel("Time  $c_{\\mathrm{mkt}} \\Delta t$  (normalised)")
    ax.set_title("Price-Time Minkowski Diagram")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")

    # ── Hyperbola of constant proper time τ₀ = 1 ─────────────────────────────
    t_hyp = np.linspace(1.001, LIM, 300)
    x_hyp = np.sqrt(t_hyp**2 - 1)
    ax.plot( x_hyp, t_hyp, "--", color=GREEN, linewidth=1.2,
             label="Constant proper time  $\\tau = 1$")
    ax.plot(-x_hyp, t_hyp, "--", color=GREEN, linewidth=1.2)

    # ── Example bar events ───────────────────────────────────────────────────
    events = [
        {"pos": (0.5, 2.0),  "label": "TIMELIKE\n($|\\beta|<1$)",  "col": BLUE,   "regime": "TL"},
        {"pos": (2.2, 0.8),  "label": "SPACELIKE\n($|\\beta|>1$)", "col": RED,    "regime": "SL"},
        {"pos": (1.5, 1.5),  "label": "LIGHTLIKE\n($|\\beta|=1$)", "col": YELLOW, "regime": "LL"},
    ]
    for ev in events:
        x0, y0 = ev["pos"]
        ax.annotate(
            ev["label"],
            xy=(x0, y0), xytext=(x0 + 0.25, y0 + 0.25),
            arrowprops=dict(arrowstyle="->", color=ev["col"], lw=1.2),
            color=ev["col"], fontsize=8, ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE,
                      edgecolor=ev["col"], alpha=0.85),
        )
        ax.scatter([x0], [y0], s=50, color=ev["col"], zorder=5)

    # ── Boosted frame axes (β = 0.5) ─────────────────────────────────────────
    beta_boost = 0.5
    gamma_boost = 1 / np.sqrt(1 - beta_boost**2)
    # Boosted time axis: x' = 0 → x = β·ct
    t_range = np.linspace(0, LIM * 0.85, 100)
    ax.plot(beta_boost * t_range, t_range, ":", color=PURPLE, linewidth=1.2,
            label=f"Boosted $t'$-axis ($\\beta={beta_boost}$)")
    # Boosted space axis: t' = 0 → ct = β·x
    x_range = np.linspace(0, LIM * 0.85, 100)
    ax.plot(x_range, beta_boost * x_range, ":", color=PURPLE, linewidth=1.2,
            label=f"Boosted $x'$-axis ($\\beta={beta_boost}$)")

    # ── Region labels ─────────────────────────────────────────────────────────
    ax.text(0,  2.4, "FUTURE\n(TIMELIKE)", ha="center", va="center",
            color=BLUE, fontsize=8, alpha=0.85)
    ax.text(0, -2.4, "PAST\n(TIMELIKE)",   ha="center", va="center",
            color=BLUE, fontsize=8, alpha=0.85)
    ax.text( 2.2, 0, "SPACELIKE",          ha="center", va="center",
            color=RED,  fontsize=8, alpha=0.85)
    ax.text(-2.2, 0, "SPACELIKE",          ha="center", va="center",
            color=RED,  fontsize=8, alpha=0.85)

    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, alpha=0.15)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[srfm] Saved: {out_path}")


if __name__ == "__main__":
    out = Path(__file__).parent / "spacetime_diagram.pdf"
    make_figure(out)
