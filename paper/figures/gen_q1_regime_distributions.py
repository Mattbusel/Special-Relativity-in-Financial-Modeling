#!/usr/bin/env python3
"""
gen_q1_regime_distributions.py
===============================
Figure: Return distributions conditioned on spacetime interval regime.

Generates a publication-quality plot showing:
  - Blue: TIMELIKE bar return distribution (|beta| < 1)
  - Red:  SPACELIKE bar return distribution (|beta| > 1)
  - Dashed: Gaussian fits for each

Uses placeholder synthetic data with realistic parameters.
AGT-07 will replace with empirical data by calling this script with
--data-file path/to/empirical_returns.npz

Output: paper/figures/q1_regime_distributions.pdf
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path

# ── Dark terminal aesthetic ───────────────────────────────────────────────────
BG      = "#0D1117"
SURFACE = "#161B22"
BORDER  = "#30363D"
TEXT    = "#C9D1D9"
MUTED   = "#8B949E"
BLUE    = "#58A6FF"   # TIMELIKE
RED     = "#F85149"   # SPACELIKE
GREEN   = "#3FB950"
YELLOW  = "#D29922"

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  BORDER,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
})


def generate_synthetic_data(seed: int = 42) -> dict:
    """
    Generate synthetic return samples with realistic regime properties.

    TIMELIKE: lower variance, near-Gaussian (calm trending regime)
    SPACELIKE: higher variance, heavier tails (information shock regime)

    Replace with empirical data from AGT-07 results file.
    """
    rng = np.random.default_rng(seed)

    n_timelike  = 180_000   # ~75% of bars are TIMELIKE
    n_spacelike =  60_000   # ~25% are SPACELIKE

    # TIMELIKE: mean≈0, vol≈0.8 bps, slightly leptokurtic
    r_timelike = rng.standard_normal(n_timelike) * 0.0008

    # SPACELIKE: mean≈0, vol≈2.1 bps, heavy-tailed (Student-t df=3)
    r_spacelike = rng.standard_t(df=3, size=n_spacelike) * 0.0009

    return {
        "timelike":  r_timelike,
        "spacelike": r_spacelike,
    }


def load_empirical_data(path: str) -> dict:
    """Load empirical return arrays from AGT-07 results file."""
    data = np.load(path)
    return {
        "timelike":  data["timelike_returns"],
        "spacelike": data["spacelike_returns"],
    }


def make_figure(data: dict, out_path: Path) -> None:
    fig = plt.figure(figsize=(9, 5.5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.08)

    ax_hist = fig.add_subplot(gs[0])
    ax_qq   = fig.add_subplot(gs[1])

    r_tl = data["timelike"]
    r_sl = data["spacelike"]

    # ── Clip for display (±5σ of TIMELIKE) ──────────────────────────────────
    clip = 5 * np.std(r_tl)
    bins = np.linspace(-clip, clip, 120)

    # ── Left panel: overlaid histograms ──────────────────────────────────────
    ax_hist.hist(r_tl, bins=bins, density=True, color=BLUE,
                 alpha=0.55, label="TIMELIKE  ($\\Delta s^2 > 0$)", zorder=2)
    ax_hist.hist(r_sl, bins=bins, density=True, color=RED,
                 alpha=0.55, label="SPACELIKE ($\\Delta s^2 < 0$)", zorder=2)

    # Gaussian fits
    xg = np.linspace(-clip, clip, 400)
    for r, col in [(r_tl, BLUE), (r_sl, RED)]:
        mu, sig = np.mean(r), np.std(r)
        y = stats.norm.pdf(xg, mu, sig)
        ax_hist.plot(xg, y, "--", color=col, linewidth=1.4,
                     alpha=0.9, zorder=3)

    ax_hist.set_xlim(-clip, clip)
    ax_hist.set_xlabel("Log-return $r_t$ (per 1-min bar)")
    ax_hist.set_ylabel("Probability density")
    ax_hist.set_title("Return distributions by regime")
    ax_hist.legend(loc="upper right")
    ax_hist.grid(True, axis="y", alpha=0.4)
    ax_hist.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x*1e4:.1f} bps"))

    # Annotate variance ratio
    vr = np.var(r_sl) / np.var(r_tl)
    ax_hist.text(0.03, 0.97, f"VR = {vr:.2f}×",
                 transform=ax_hist.transAxes, va="top",
                 color=YELLOW, fontsize=9, fontfamily="monospace")

    # ── Right panel: Q-Q plot against Gaussian ───────────────────────────────
    for r, col, label in [
        (r_tl, BLUE, "TIMELIKE"),
        (r_sl, RED,  "SPACELIKE"),
    ]:
        r_std = (r - np.mean(r)) / np.std(r)
        n = min(len(r_std), 5000)
        q_emp = np.quantile(r_std, np.linspace(0.005, 0.995, n))
        q_th  = stats.norm.ppf(np.linspace(0.005, 0.995, n))
        ax_qq.scatter(q_th, q_emp, s=1.5, color=col, alpha=0.35,
                      label=label, rasterized=True)

    # 45° reference line
    lim = 5
    ax_qq.plot([-lim, lim], [-lim, lim], "--",
               color=MUTED, linewidth=0.8, zorder=0)
    ax_qq.set_xlim(-lim, lim)
    ax_qq.set_ylim(-lim, lim)
    ax_qq.set_xlabel("Theoretical Gaussian quantiles")
    ax_qq.set_ylabel("Empirical quantiles")
    ax_qq.set_title("Q-Q plot vs. Gaussian")
    ax_qq.legend(loc="upper left", markerscale=4)
    ax_qq.set_aspect("equal")
    ax_qq.grid(True, alpha=0.3)

    # ── Figure-level annotation ───────────────────────────────────────────────
    fig.suptitle(
        "Q1 2025  ·  S&P 500 universe  ·  1-min OHLCV bars",
        color=MUTED, fontsize=9, y=0.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[srfm] Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SRFM Figure: regime return distributions"
    )
    parser.add_argument("--data-file", default=None,
                        help="Path to AGT-07 .npz results file "
                             "(omit for synthetic placeholder)")
    parser.add_argument("--out",
                        default=str(Path(__file__).parent / "q1_regime_distributions.pdf"),
                        help="Output PDF path")
    args = parser.parse_args()

    data = (load_empirical_data(args.data_file)
            if args.data_file else generate_synthetic_data())

    make_figure(data, Path(args.out))


if __name__ == "__main__":
    main()
