#!/usr/bin/env python3
"""
gen_q1_variance_ratio_heatmap.py
=================================
Figure: Variance ratio VR = Var(SPACELIKE) / Var(TIMELIKE) as a function of
c_market calibration percentile (x-axis) and rolling window n (y-axis).

Shows sensitivity of regime separation to hyperparameter choices.
The optimal (99.5th pctile, n=20) combination is marked with a white star.

Output: paper/figures/q1_variance_ratio_heatmap.pdf
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Dark theme ────────────────────────────────────────────────────────────────
BG      = "#0D1117"
SURFACE = "#161B22"
BORDER  = "#30363D"
TEXT    = "#C9D1D9"
MUTED   = "#8B949E"

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "font.size":         10,
    "axes.titlesize":    11,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
})

PERCENTILES = [90, 93, 95, 97, 98, 99, 99.5, 99.9]
WINDOWS     = [5, 10, 15, 20, 30, 40, 60]


def synthetic_variance_ratio(pctile: float, window: int, seed: int = 7) -> float:
    """
    Synthetic VR surface with realistic shape:
    - Increases with percentile (tighter speed limit → more separation)
    - Has a moderate peak around window=20
    - Replace with empirical sweep from AGT-07

    VR ~ 1 + A * f(pctile) * g(window) + noise
    """
    rng = np.random.default_rng(seed + int(pctile * 10) + window)

    # Monotone in percentile: higher pctile → tighter c_market → larger VR
    f = (pctile - 90) / 10          # 0 → 0.95
    # Unimodal in window with peak ~20
    g = np.exp(-0.5 * ((window - 20) / 15) ** 2)

    vr = 1.8 + 3.2 * f * g + rng.normal(0, 0.08)
    return max(1.01, vr)


def make_heatmap(out_path: Path, data_file: str | None = None) -> None:

    if data_file:
        raw = np.load(data_file)
        vr_matrix = raw["vr_matrix"]     # shape: (len(WINDOWS), len(PERCENTILES))
        percentiles = raw["percentiles"]
        windows     = raw["windows"]
    else:
        percentiles = np.array(PERCENTILES)
        windows     = np.array(WINDOWS)
        vr_matrix   = np.array([
            [synthetic_variance_ratio(p, w) for p in percentiles]
            for w in windows
        ])

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(
        vr_matrix,
        aspect="auto",
        cmap="magma",
        origin="lower",
        extent=[0, len(percentiles), 0, len(windows)],
        vmin=1.0,
        vmax=vr_matrix.max() * 1.05,
    )

    # ── Tick labels ───────────────────────────────────────────────────────────
    ax.set_xticks(np.arange(len(percentiles)) + 0.5)
    ax.set_xticklabels([f"{p:.1f}" for p in percentiles], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(windows)) + 0.5)
    ax.set_yticklabels([str(w) for w in windows])

    ax.set_xlabel("$c_{\\mathrm{mkt}}$ calibration percentile")
    ax.set_ylabel("Rolling window $n$ (bars)")
    ax.set_title("Variance ratio  $\\mathrm{VR} = \\sigma^2_{\\mathrm{SL}} / "
                 "\\sigma^2_{\\mathrm{TL}}$  by hyperparameter")

    # ── Annotate each cell ────────────────────────────────────────────────────
    for i, w in enumerate(windows):
        for j, p in enumerate(percentiles):
            vr = vr_matrix[i, j]
            ax.text(j + 0.5, i + 0.5, f"{vr:.2f}",
                    ha="center", va="center",
                    fontsize=7, color="white" if vr > 2.5 else MUTED,
                    fontfamily="monospace")

    # ── Mark optimal hyperparameter (★) ──────────────────────────────────────
    opt_p = list(percentiles).index(99.5) if 99.5 in percentiles else -1
    opt_w = list(windows).index(20) if 20 in windows else -1
    if opt_p >= 0 and opt_w >= 0:
        ax.plot(opt_p + 0.5, opt_w + 0.5, "w*", markersize=14,
                label="Primary (99.5%, n=20)", zorder=10)
        ax.legend(loc="upper left", fontsize=8,
                  facecolor=SURFACE, edgecolor=BORDER)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("VR", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
    cbar.outline.set_edgecolor(BORDER)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[srfm] Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SRFM Figure: variance ratio hyperparameter heatmap"
    )
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--out",
                        default=str(Path(__file__).parent / "q1_variance_ratio_heatmap.pdf"))
    args = parser.parse_args()
    make_heatmap(Path(args.out), args.data_file)


if __name__ == "__main__":
    main()
