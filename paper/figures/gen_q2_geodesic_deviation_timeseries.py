#!/usr/bin/env python3
"""
gen_q2_geodesic_deviation_timeseries.py
=========================================
Figure: Rolling geodesic deviation z-score and regime classification
for a representative asset, with earnings announcement markers.

Upper panel: 5-day rolling median of |J_t| z-score.
Lower panel: Fraction of TIMELIKE bars per day.
Vertical dashed lines: earnings announcement dates.

Output: paper/figures/q2_geodesic_deviation_timeseries.pdf
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from pathlib import Path
import datetime

BG      = "#0D1117"
SURFACE = "#161B22"
BORDER  = "#30363D"
TEXT    = "#C9D1D9"
MUTED   = "#8B949E"
BLUE    = "#58A6FF"
GREEN   = "#3FB950"
RED     = "#F85149"
PURPLE  = "#BC8CFF"
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
    "grid.linewidth":    0.4,
    "font.family":       "monospace",
    "font.size":         9,
    "axes.titlesize":    10,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  BORDER,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
})

Q1_START = datetime.date(2025, 1, 2)
Q1_END   = datetime.date(2025, 3, 31)

EARNINGS_DATES = [
    datetime.date(2025, 1, 23),
    datetime.date(2025, 2,  6),
    datetime.date(2025, 2, 20),
    datetime.date(2025, 3,  6),
]


def trading_days(start, end):
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def rolling_median(x: np.ndarray, window: int = 5) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for i in range(window - 1, len(x)):
        out[i] = np.median(x[i - window + 1 : i + 1])
    return out


def generate_synthetic_series(seed: int = 13) -> dict:
    rng  = np.random.default_rng(seed)
    days = trading_days(Q1_START, Q1_END)
    T    = len(days)

    # Base geodesic deviation z-score: mean-reverting around 0
    j_zscore = np.zeros(T)
    ar_coef   = 0.92
    for t in range(1, T):
        j_zscore[t] = ar_coef * j_zscore[t-1] + rng.normal(0, 0.40)

    # Spike before each earnings announcement (geodesic deviation precursor)
    earnings_idx = []
    for ed in EARNINGS_DATES:
        for i, d in enumerate(days):
            if d == ed or (d > ed and (d - ed).days <= 2):
                earnings_idx.append(i)
                # Spike 2-4 days before earnings
                for k in range(max(0, i-4), i):
                    j_zscore[k] += rng.uniform(1.5, 3.0)
                break

    # TIMELIKE fraction: inversely correlated with |J|
    tl_frac = np.clip(
        0.75 - 0.12 * j_zscore / (np.std(j_zscore) + 1e-9)
        + rng.normal(0, 0.04, T),
        0.30, 0.95,
    )

    return {
        "dates":         days,
        "j_zscore":      j_zscore,
        "tl_fraction":   tl_frac,
        "earnings_dates": EARNINGS_DATES,
    }


def make_figure(data: dict, out_path: Path) -> None:
    dates        = [datetime.datetime(d.year, d.month, d.day) for d in data["dates"]]
    j_z          = data["j_zscore"]
    j_smooth     = rolling_median(j_z, window=5)
    tl_frac      = data["tl_fraction"]
    earn_dates   = [datetime.datetime(d.year, d.month, d.day)
                    for d in data["earnings_dates"]]
    theta        = 2.0   # entry threshold

    fig = plt.figure(figsize=(11, 6))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.10)

    ax_j  = fig.add_subplot(gs[0])
    ax_tl = fig.add_subplot(gs[1], sharex=ax_j)

    # ── Upper: geodesic deviation z-score ────────────────────────────────────
    ax_j.plot(dates, j_smooth, color=PURPLE, linewidth=1.5,
              label="5-day rolling median  $\\hat{J}_t$")
    ax_j.fill_between(dates, j_smooth, alpha=0.25, color=PURPLE)
    ax_j.axhline(theta,  color=YELLOW, linewidth=0.9, linestyle="--",
                 label=f"Entry threshold $\\theta = {theta:.1f}$")
    ax_j.axhline(-theta, color=YELLOW, linewidth=0.9, linestyle="--")
    ax_j.axhline(0,       color=BORDER, linewidth=0.6)

    # Shade exceedances
    exceed = j_smooth > theta
    ax_j.fill_between(dates, j_smooth, theta,
                      where=exceed, alpha=0.45, color=GREEN,
                      label="Signal exceedance")

    ax_j.set_ylabel("Geodesic deviation $\\hat{J}_t$ (z-score)")
    ax_j.set_title("Q2: Geodesic Deviation Signal — Representative Asset")
    ax_j.legend(loc="upper left", fontsize=8)
    ax_j.grid(True, alpha=0.3)

    # Earnings annotations
    for ed in earn_dates:
        ax_j.axvline(ed, color=RED, linewidth=0.9, linestyle=":", alpha=0.9)
    ax_j.plot([], [], color=RED, linestyle=":", linewidth=0.9,
              label="Earnings announcement")
    ax_j.legend(loc="upper left", fontsize=8)

    # ── Lower: TIMELIKE fraction ──────────────────────────────────────────────
    ax_tl.fill_between(dates, tl_frac, alpha=0.7, color=BLUE,
                       label="Fraction TIMELIKE bars/day")
    ax_tl.fill_between(dates, tl_frac, 1.0, alpha=0.4, color=RED)
    ax_tl.axhline(0.5, color=YELLOW, linewidth=0.7, linestyle="--",
                  label="50% threshold")
    ax_tl.set_ylim(0.2, 1.0)
    ax_tl.set_ylabel("TIMELIKE fraction")
    ax_tl.legend(loc="lower left", fontsize=8)
    ax_tl.grid(True, alpha=0.2)

    for ed in earn_dates:
        ax_tl.axvline(ed, color=RED, linewidth=0.9, linestyle=":", alpha=0.9)

    # ── Date formatting ───────────────────────────────────────────────────────
    locator   = mdates.WeekdayLocator(byweekday=mdates.MO, interval=2)
    formatter = mdates.DateFormatter("%b %d")
    ax_tl.xaxis.set_major_locator(locator)
    ax_tl.xaxis.set_major_formatter(formatter)
    plt.setp(ax_tl.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax_j.get_xticklabels(), visible=False)

    fig.text(0.5, 0.01,
             "Synthetic placeholder — replace with AGT-07 empirical data",
             ha="center", va="bottom", color=MUTED, fontsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[srfm] Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--out",
                        default=str(Path(__file__).parent /
                                    "q2_geodesic_deviation_timeseries.pdf"))
    args = parser.parse_args()

    data = (np.load(args.data_file, allow_pickle=True)
            if args.data_file else generate_synthetic_series())

    make_figure(data, Path(args.out))


if __name__ == "__main__":
    main()
