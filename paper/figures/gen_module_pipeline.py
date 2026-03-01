#!/usr/bin/env python3
"""
gen_module_pipeline.py
=======================
Figure: SRFM pipeline data-flow diagram (for the implementation section).

Shows each module as a box with input/output labels and latency budget.
Pure matplotlib implementation (alternative to TikZ for quick preview).

Output: paper/figures/module_pipeline.pdf
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

BG      = "#0D1117"
SURFACE = "#161B22"
BORDER  = "#30363D"
TEXT    = "#C9D1D9"
MUTED   = "#8B949E"
BLUE    = "#58A6FF"
GREEN   = "#3FB950"
RED     = "#F85149"
YELLOW  = "#D29922"
PURPLE  = "#BC8CFF"
ORANGE  = "#E3B341"

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "font.size":         8.5,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
})

MODULES = [
    {
        "name":    "OhlcvBar\nReader",
        "inputs":  ["raw OHLCV"],
        "outputs": ["OhlcvBar"],
        "latency": "<1 µs",
        "color":   BORDER,
    },
    {
        "name":    "Beta\nCalculator",
        "inputs":  ["OhlcvBar", "c_market"],
        "outputs": ["BetaValue?"],
        "latency": "48 ns",
        "color":   BLUE,
    },
    {
        "name":    "Lorentz\nTransform",
        "inputs":  ["BetaValue"],
        "outputs": ["GammaValue?"],
        "latency": "11 ns",
        "color":   BLUE,
    },
    {
        "name":    "Interval\nClassifier",
        "inputs":  ["BetaValue"],
        "outputs": ["Regime"],
        "latency": "6 ns",
        "color":   YELLOW,
    },
    {
        "name":    "Cov\nManifold",
        "inputs":  ["features"],
        "outputs": ["Σ ∈ P_n"],
        "latency": "2.1 µs",
        "color":   GREEN,
    },
    {
        "name":    "Christoffel\nSolver",
        "inputs":  ["Σ", "g_μν"],
        "outputs": ["Γ^λ_μν"],
        "latency": "180 µs",
        "color":   GREEN,
    },
    {
        "name":    "Geodesic\nSolver",
        "inputs":  ["Γ^λ_μν", "x₀", "v₀"],
        "outputs": ["x(τ), v(τ)"],
        "latency": "8.4 µs",
        "color":   PURPLE,
    },
    {
        "name":    "Deviation\nSignal",
        "inputs":  ["J(τ)", "Regime"],
        "outputs": ["signal ∈ {-1,0,+1}"],
        "latency": "250 µs total",
        "color":   ORANGE,
    },
]


def draw_module(ax, cx, cy, mod, w=1.5, h=0.9):
    """Draw a single module box."""
    col = mod["color"]
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=SURFACE, edgecolor=col, linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(cx, cy + 0.12, mod["name"], ha="center", va="center",
            fontsize=8, color=col, fontweight="bold")
    ax.text(cx, cy - 0.22, mod["latency"], ha="center", va="center",
            fontsize=6.5, color=MUTED, style="italic")


def make_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.set_xlim(-0.5, len(MODULES) * 2.0 - 0.5)
    ax.set_ylim(-1.2, 1.5)
    ax.axis("off")
    ax.set_aspect("equal")

    positions = [(i * 2.0, 0.0) for i in range(len(MODULES))]

    for i, (mod, (cx, cy)) in enumerate(zip(MODULES, positions)):
        draw_module(ax, cx, cy, mod)

        # I/O labels
        for j, inp in enumerate(mod["inputs"]):
            ax.text(cx, cy + 0.55 + j * 0.18, inp,
                    ha="center", va="bottom", fontsize=6, color=MUTED)

        for j, out in enumerate(mod["outputs"]):
            ax.text(cx, cy - 0.55 - j * 0.18, out,
                    ha="center", va="top", fontsize=6, color=TEXT)

        # Arrow to next module
        if i < len(MODULES) - 1:
            ax.annotate(
                "", xy=(positions[i+1][0] - 0.75, 0),
                xytext=(cx + 0.75, 0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=MUTED,
                    lw=1.2,
                    mutation_scale=12,
                ),
            )

    ax.set_title(
        "SRFM C++20 Pipeline  ·  P99 end-to-end latency < 3 ms",
        color=TEXT, fontsize=10, pad=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[srfm] Saved: {out_path}")


if __name__ == "__main__":
    out = Path(__file__).parent / "module_pipeline.pdf"
    make_figure(out)
