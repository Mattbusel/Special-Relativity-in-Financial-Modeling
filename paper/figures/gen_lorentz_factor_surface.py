#!/usr/bin/env python3
"""
gen_lorentz_factor_surface.py
==============================
Figure: γ(β) surface and key properties.

Panel A: γ as a function of β with Taylor expansion overlay.
Panel B: Proper-time fraction dτ/dt = 1/γ vs β.
Panel C: Rapidity φ = arctanh(β) vs β (showing linearity advantage).
Panel D: Velocity composition β12(β1, β2) heatmap vs Galilean addition.

Output: paper/figures/lorentz_factor_surface.pdf
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
ORANGE  = "#E3B341"

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
    "font.size":         9,
    "axes.titlesize":    9,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  BORDER,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
})


def gamma(b: np.ndarray) -> np.ndarray:
    return 1.0 / np.sqrt(1 - b**2)


def gamma_taylor(b: np.ndarray) -> np.ndarray:
    return 1 + 0.5 * b**2 + (3/8) * b**4


def rapidity(b: np.ndarray) -> np.ndarray:
    return np.arctanh(b)


def velocity_composition(b1: float, b2: float) -> float:
    return (b1 + b2) / (1 + b1 * b2)


def make_figure(out_path: Path) -> None:
    beta    = np.linspace(-0.999, 0.999, 1200)
    beta_lo = np.linspace(-0.8, 0.8, 400)

    fig = plt.figure(figsize=(11, 8))
    gs  = GridSpec(2, 2, figure=fig, wspace=0.32, hspace=0.42)

    # ── A: γ(β) ──────────────────────────────────────────────────────────────
    ax_g = fig.add_subplot(gs[0, 0])
    ax_g.plot(beta, gamma(beta),          color=BLUE,   linewidth=1.8,
              label="$\\gamma(\\beta) = (1-\\beta^2)^{-1/2}$")
    ax_g.plot(beta_lo, gamma_taylor(beta_lo), color=YELLOW, linewidth=1.2,
              linestyle="--",
              label="Taylor: $1 + \\frac{1}{2}\\beta^2 + \\frac{3}{8}\\beta^4$")
    ax_g.axhline(1, color=BORDER, linewidth=0.6)
    ax_g.set_ylim(0.9, 8)
    ax_g.set_xlabel("Price velocity $\\beta$")
    ax_g.set_ylabel("Lorentz factor $\\gamma$")
    ax_g.set_title("(A) Lorentz factor")
    ax_g.legend(fontsize=8)
    ax_g.grid(True, alpha=0.3)

    # ── B: Proper-time fraction ───────────────────────────────────────────────
    ax_pt = fig.add_subplot(gs[0, 1])
    ax_pt.plot(beta, 1 / gamma(beta), color=GREEN, linewidth=1.8,
               label="$d\\tau/dt = \\sqrt{1-\\beta^2}$")
    ax_pt.axhline(1, color=BORDER, linewidth=0.6, linestyle="--",
                  label="Newtonian limit ($\\gamma=1$)")
    ax_pt.set_ylim(0, 1.05)
    ax_pt.set_xlabel("Price velocity $\\beta$")
    ax_pt.set_ylabel("Proper-time fraction $d\\tau/dt$")
    ax_pt.set_title("(B) Time dilation")
    ax_pt.legend(fontsize=8)
    ax_pt.grid(True, alpha=0.3)
    ax_pt.fill_between(beta, 1 / gamma(beta), 1.0, alpha=0.15, color=RED,
                       label="Time lost to dilation")

    # ── C: Rapidity φ(β) ─────────────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[1, 0])
    ax_r.plot(beta, rapidity(beta), color=BLUE,  linewidth=1.8,
              label="$\\phi = \\mathrm{arctanh}(\\beta)$  (additive)")
    ax_r.plot(beta, beta,            color=YELLOW, linewidth=1.2, linestyle="--",
              label="$\\phi \\approx \\beta$  (small-$\\beta$ limit)")
    ax_r.axhline(0,  color=BORDER, linewidth=0.6)
    ax_r.axvline(0,  color=BORDER, linewidth=0.6)
    ax_r.set_ylim(-4, 4)
    ax_r.set_xlabel("Price velocity $\\beta$")
    ax_r.set_ylabel("Rapidity $\\phi$")
    ax_r.set_title("(C) Rapidity — additive under composition")
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.3)

    # ── D: Velocity composition heatmap ──────────────────────────────────────
    ax_vc = fig.add_subplot(gs[1, 1])
    bv    = np.linspace(-0.9, 0.9, 200)
    B1, B2 = np.meshgrid(bv, bv)
    B12_rel = (B1 + B2) / (1 + B1 * B2)
    B12_gal = np.clip(B1 + B2, -0.999, 0.999)      # Galilean (capped)
    diff     = B12_rel - B12_gal                     # relativistic correction

    im = ax_vc.imshow(
        diff, extent=[-0.9, 0.9, -0.9, 0.9],
        origin="lower", cmap="RdBu_r", aspect="equal",
        vmin=-0.4, vmax=0.4,
    )
    ax_vc.contour(bv, bv, diff, levels=[0], colors=[MUTED],
                  linewidths=0.6)
    cbar = plt.colorbar(im, ax=ax_vc, fraction=0.05, pad=0.02)
    cbar.set_label("$\\beta_{12}^{\\rm rel} - \\beta_{12}^{\\rm Gal}$",
                   color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
    cbar.outline.set_edgecolor(BORDER)

    ax_vc.set_xlabel("$\\beta_1$")
    ax_vc.set_ylabel("$\\beta_2$")
    ax_vc.set_title("(D) Relativistic correction to velocity addition")
    ax_vc.grid(False)

    fig.suptitle("Special-Relativistic Kinematics: Core Functions",
                 color=TEXT, fontsize=11, y=0.98)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[srfm] Saved: {out_path}")


if __name__ == "__main__":
    out = Path(__file__).parent / "lorentz_factor_surface.pdf"
    make_figure(out)
