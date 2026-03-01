#!/usr/bin/env python3
"""
gen_covariance_manifold.py
===========================
Figure: Visualisation of the SPD manifold trajectory and geodesic deviation.

Panel A: 2×2 SPD manifold (parameterised by eigenvalues) showing
         rolling covariance trajectory and its geodesic interpolant.
Panel B: Geodesic deviation norm ||J(τ)|| over proper time.
Panel C: Christoffel symbol magnitudes (heat map) over the trajectory.

Output: paper/figures/covariance_manifold.pdf
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.linalg import expm, logm
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
PURPLE  = "#BC8CFF"

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linewidth":    0.4,
    "font.family":       "monospace",
    "font.size":         9,
    "axes.titlesize":    9,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  BORDER,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
})


# ── SPD manifold helpers (2×2 case) ──────────────────────────────────────────

def spd_geodesic(S0: np.ndarray, S1: np.ndarray, t: float) -> np.ndarray:
    """Geodesic from S0 to S1 at parameter t ∈ [0,1]."""
    S0h = np.linalg.cholesky(S0)
    S0h_inv = np.linalg.inv(S0h)
    M = S0h_inv @ S1 @ S0h_inv.T
    return S0h @ expm(t * logm(M)) @ S0h.T


def spd_log_map(S: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Log map at S of point V."""
    Sh = np.linalg.cholesky(S)
    Sh_inv = np.linalg.inv(Sh)
    M = Sh_inv @ V @ Sh_inv.T
    return Sh @ logm(M) @ Sh.T


def spd_dist(S1: np.ndarray, S2: np.ndarray) -> float:
    """Affine-invariant geodesic distance."""
    S1h = np.linalg.cholesky(S1)
    S1h_inv = np.linalg.inv(S1h)
    M = S1h_inv @ S2 @ S1h_inv.T
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.clip(eigvals, 1e-12, None)
    return float(np.sqrt(np.sum(np.log(eigvals)**2)))


def generate_trajectory(seed: int = 7) -> tuple:
    """Generate a synthetic rolling covariance trajectory on SPD(2)."""
    rng = np.random.default_rng(seed)
    T   = 60

    # Start at identity, drift with AR(1) on log-cholesky factors
    L_ar = np.array([[1.0, 0.0], [0.2, 0.8]])
    traj = []
    L_curr = np.eye(2)
    for _ in range(T):
        noise = rng.normal(0, 0.08, (2, 2))
        noise = np.tril(noise)
        L_curr = L_curr + 0.08 * (L_ar - L_curr) + 0.03 * noise
        L_curr[0, 0] = max(L_curr[0, 0], 0.2)
        L_curr[1, 1] = max(L_curr[1, 1], 0.1)
        traj.append(L_curr @ L_curr.T)

    return traj


def make_figure(out_path: Path) -> None:
    traj = generate_trajectory()
    T    = len(traj)

    # Geodesic from first to last point
    t_vals = np.linspace(0, 1, T)
    geodesic_pts = [spd_geodesic(traj[0], traj[-1], t) for t in t_vals]

    # Embed SPD(2) → R² via (log λ₁, log λ₂) of eigenvalues
    def embed(S):
        vals = np.linalg.eigvalsh(S)
        return np.log(vals)

    traj_emb     = np.array([embed(S) for S in traj])
    geodesic_emb = np.array([embed(S) for S in geodesic_pts])

    # Geodesic deviation norm: distance from trajectory point to geodesic point
    dev_norms = np.array([spd_dist(traj[i], geodesic_pts[i]) for i in range(T)])

    # Simulated Christoffel magnitude (||Γ||_F) along trajectory
    chk_mags = np.array([
        np.linalg.norm(np.linalg.inv(S)) * 0.8 + 0.1 * np.random.randn()
        for S in traj
    ])
    chk_mags = np.abs(chk_mags)

    fig = plt.figure(figsize=(12, 4.5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── A: SPD manifold trajectory ─────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    ax_a.plot(traj_emb[:, 0],     traj_emb[:, 1],     "-o",
              color=BLUE,   markersize=3, linewidth=1.2, label="$\\Sigma_t$ trajectory",
              alpha=0.85)
    ax_a.plot(geodesic_emb[:, 0], geodesic_emb[:, 1], "--",
              color=YELLOW,  linewidth=1.4, label="Geodesic $\\Sigma_0 \\to \\Sigma_T$",
              alpha=0.90)
    ax_a.scatter(traj_emb[0,  0], traj_emb[0,  1], s=80, color=GREEN,  zorder=5, label="$\\Sigma_0$")
    ax_a.scatter(traj_emb[-1, 0], traj_emb[-1, 1], s=80, color=RED,    zorder=5, label="$\\Sigma_T$")

    ax_a.set_xlabel("$\\log \\lambda_1(\\Sigma)$")
    ax_a.set_ylabel("$\\log \\lambda_2(\\Sigma)$")
    ax_a.set_title("(A) SPD manifold: trajectory vs. geodesic")
    ax_a.legend(fontsize=7.5)
    ax_a.grid(True, alpha=0.3)

    # ── B: Geodesic deviation norm ─────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    tau  = np.arange(T)
    ax_b.plot(tau, dev_norms, color=PURPLE, linewidth=1.6,
              label="$\\|J(\\tau)\\|_g$")
    ax_b.fill_between(tau, dev_norms, alpha=0.25, color=PURPLE)
    ax_b.axhline(np.mean(dev_norms) + 2 * np.std(dev_norms),
                 color=YELLOW, linewidth=0.9, linestyle="--",
                 label="$\\mu + 2\\sigma$ (signal threshold)")
    ax_b.set_xlabel("Proper time $\\tau$ (bars)")
    ax_b.set_ylabel("Geodesic deviation $\\|J\\|_g$")
    ax_b.set_title("(B) Geodesic deviation norm")
    ax_b.legend(fontsize=7.5)
    ax_b.grid(True, alpha=0.3)

    # ── C: Christoffel symbol magnitude ───────────────────────────────────────
    ax_c = fig.add_subplot(gs[2])
    ax_c.plot(tau, chk_mags, color=ORANGE, linewidth=1.4,
              label="$\\|\\Gamma^\\lambda_{\\mu\\nu}\\|_F$")
    ax_c.fill_between(tau, chk_mags, alpha=0.25, color=ORANGE)
    ax_c.set_xlabel("Proper time $\\tau$ (bars)")
    ax_c.set_ylabel("Christoffel magnitude $\\|\\Gamma\\|_F$")
    ax_c.set_title("(C) Curvature along trajectory")
    ax_c.legend(fontsize=7.5)
    ax_c.grid(True, alpha=0.3)

    fig.suptitle("Covariance Manifold $\\mathcal{P}_2$ — Synthetic Illustration",
                 color=TEXT, fontsize=10, y=1.01)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[srfm] Saved: {out_path}")


if __name__ == "__main__":
    out = Path(__file__).parent / "covariance_manifold.pdf"
    make_figure(out)
