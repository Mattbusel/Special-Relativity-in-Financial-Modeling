"""
generate_figures.py — SRFM Q1/Q2 Publication-Quality Figures
==============================================================
Generates 4 figures for Q1 (regime variance analysis) and 1 figure for Q2
(strategy comparison) from the validation outputs.

Figures produced:
  validation/figures/q1_violin_plot.png      — Next-bar return distributions by regime
  validation/figures/q1_rolling_timelike.png — Rolling TIMELIKE% over time
  validation/figures/q1_pvalue_heatmap.png   — p-value heatmap across tickers
  validation/figures/q1_effect_size.png      — Cohen's d per ticker
  validation/figures/q2_strategy_comparison.png — RAW/RELATIVISTIC/GEODESIC comparison

Usage
-----
    python validation/generate_figures.py [--results-dir PATH]
                                          [--backtest-dir PATH]
                                          [--output-dir PATH]
                                          [--analysis-csv PATH]

Dependencies
------------
    pip install matplotlib seaborn pandas numpy scipy
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False
    warnings.warn("seaborn not installed; violin plots will use matplotlib fallback")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Style ────────────────────────────────────────────────────────────────────

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
}
TIMELIKE_COLOR = "#2c7bb6"
SPACELIKE_COLOR = "#d7191c"
GEODESIC_COLOR = "#1a9641"
PALETTE = {
    "Timelike": TIMELIKE_COLOR,
    "Spacelike": SPACELIKE_COLOR,
    "Lightlike": "#fdae61",
}
TICKERS_ORDER = ["AAPL", "SPY", "QQQ", "TSLA", "GS", "JPM", "NVDA", "META", "BTC-USD", "GLD"]

# ─── Data Helpers ─────────────────────────────────────────────────────────────


def _load_regime_data(results_dir: str) -> pd.DataFrame:
    path = Path(results_dir)
    frames = [pd.read_csv(f) for f in sorted(path.glob("*.csv")) if f.is_file()]
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {path}")
    df = pd.concat(frames, ignore_index=True)
    df["interval_type"] = df["interval_type"].str.strip().str.title()
    df = df[np.isfinite(df["next_bar_abs_return"])].copy()
    return df


def _load_analysis_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _load_backtest_data(backtest_dir: str) -> Optional[pd.DataFrame]:
    path = Path(backtest_dir)
    frames = [pd.read_csv(f) for f in sorted(path.glob("*.csv")) if f.is_file()]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ─── Figure 1: Violin Plot ────────────────────────────────────────────────────


def plot_violin(
    df: pd.DataFrame,
    output_dir: Path,
    max_return: float = 0.05,
) -> None:
    """
    Violin plot of next_bar_abs_return by regime, one panel per ticker.
    2-row × 5-col grid, log-scale y-axis.
    """
    log.info("Generating violin plot...")
    tickers = [t for t in TICKERS_ORDER if t in df["ticker"].unique()]
    n_tickers = len(tickers)
    n_cols = 5
    n_rows = math.ceil(n_tickers / n_cols)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), squeeze=False)
        fig.suptitle(
            "SRFM Q1 — Next-Bar Absolute Return Distribution by Regime",
            fontsize=15, fontweight="bold", y=1.01,
        )

        for idx, ticker in enumerate(tickers):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]

            sub = df[df["ticker"] == ticker].copy()
            sub = sub[sub["next_bar_abs_return"] <= max_return]
            sub = sub[sub["interval_type"].isin(["Timelike", "Spacelike"])]

            if _HAS_SEABORN and len(sub) > 0:
                sns.violinplot(
                    data=sub,
                    x="interval_type",
                    y="next_bar_abs_return",
                    palette={"Timelike": TIMELIKE_COLOR, "Spacelike": SPACELIKE_COLOR},
                    inner="quartile",
                    order=["Timelike", "Spacelike"],
                    ax=ax,
                    cut=0,
                    scale="width",
                )
                # Overlay median markers
                for i, label in enumerate(["Timelike", "Spacelike"]):
                    vals = sub.loc[sub["interval_type"] == label, "next_bar_abs_return"]
                    if len(vals) > 0:
                        ax.scatter(i, vals.median(), color="white", zorder=5,
                                   s=40, edgecolors="black", linewidths=1.0)
            else:
                # Fallback: boxplot
                groups = [
                    sub.loc[sub["interval_type"] == "Timelike", "next_bar_abs_return"].values,
                    sub.loc[sub["interval_type"] == "Spacelike", "next_bar_abs_return"].values,
                ]
                ax.boxplot(groups, labels=["Timelike", "Spacelike"],
                           patch_artist=True,
                           boxprops=dict(facecolor="lightblue"),
                           medianprops=dict(color="black"))

            ax.set_title(ticker, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("|Δ return|" if col == 0 else "")
            ax.tick_params(axis="x", rotation=15)
            ax.grid(axis="y")

        # Hide unused panels
        for idx in range(n_tickers, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        plt.tight_layout()
        out = output_dir / "q1_violin_plot.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
    log.info("  Saved %s", out)


# ─── Figure 2: Rolling TIMELIKE% ─────────────────────────────────────────────


def plot_rolling_timelike(
    df: pd.DataFrame,
    output_dir: Path,
    window: int = 500,
) -> None:
    """Rolling TIMELIKE% over bar index for each ticker."""
    log.info("Generating rolling TIMELIKE%% plot...")
    tickers = [t for t in TICKERS_ORDER if t in df["ticker"].unique()]

    colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))  # type: ignore[attr-defined]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(16, 7))

        for i, (ticker, color) in enumerate(zip(tickers, colors)):
            sub = df[df["ticker"] == ticker].copy()
            sub = sub.sort_values("bar_index").reset_index(drop=True)

            is_tl = (sub["interval_type"] == "Timelike").astype(float)
            rolling_pct = is_tl.rolling(window, min_periods=max(1, window // 10)).mean() * 100

            ax.plot(
                sub["bar_index"].values,
                rolling_pct.values,
                label=ticker,
                color=color,
                linewidth=1.0,
                alpha=0.8,
            )

        ax.axhline(50, color="black", linestyle="--", linewidth=0.8,
                   label="50% reference")
        ax.set_xlabel(f"Bar Index (rolling {window}-bar window)")
        ax.set_ylabel("TIMELIKE Bars (%)")
        ax.set_title("SRFM Q1 — Rolling TIMELIKE% by Ticker",
                     fontsize=14, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=9, ncol=2)
        ax.grid(axis="y")

        plt.tight_layout()
        out = output_dir / "q1_rolling_timelike.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
    log.info("  Saved %s", out)


# ─── Figure 3: p-Value Heatmap ────────────────────────────────────────────────


def plot_pvalue_heatmap(
    analysis_csv: Optional[str],
    results: Optional[dict],
    output_dir: Path,
    alpha: float = 0.05,
) -> None:
    """
    Heatmap: tickers × tests, colored by -log10(p_value).
    """
    log.info("Generating p-value heatmap...")

    # Build a DataFrame from whichever source is available
    if analysis_csv and Path(analysis_csv).exists():
        full_df = pd.read_csv(analysis_csv)
        rows = []
        for _, row in full_df.iterrows():
            if row.get("group") == "POOLED":
                continue
            rows.append({
                "ticker": row["group"],
                "Levene": row.get("levene_p", float("nan")),
                "Bartlett": row.get("bartlett_p", float("nan")),
            })
    elif results:
        rows = []
        for t, r in results.items():
            if t == "POOLED":
                continue
            rows.append({
                "ticker": t,
                "Levene": r.get("levene_p", float("nan")),
                "Bartlett": r.get("bartlett_p", float("nan")),
            })
    else:
        log.warning("No analysis data available for heatmap — skipping")
        return

    df_heat = pd.DataFrame(rows)
    if df_heat.empty:
        return

    tickers = [t for t in TICKERS_ORDER if t in df_heat["ticker"].values]
    df_heat = df_heat.set_index("ticker").reindex(tickers)

    # Convert to -log10(p)
    neg_log_p = -np.log10(df_heat.values.astype(float).clip(min=1e-300))
    threshold = -np.log10(alpha)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))

        im = ax.imshow(neg_log_p.T, cmap="RdYlGn", aspect="auto",
                       vmin=0, vmax=max(float(neg_log_p.max()), threshold * 2 + 1))

        # Axis labels
        ax.set_xticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=30, ha="right")
        ax.set_yticks(range(len(df_heat.columns)))
        ax.set_yticklabels(df_heat.columns)
        ax.set_title("SRFM Q1 — Variance Test p-Values  (−log₁₀ scale; green = more significant)",
                     fontweight="bold")

        # Annotate cells with actual p-values
        for ci, ticker in enumerate(tickers):
            for ri, test in enumerate(df_heat.columns):
                raw_p = df_heat.loc[ticker, test]
                if np.isfinite(raw_p):
                    txt = f"{raw_p:.3f}" if raw_p >= 0.001 else f"{raw_p:.1e}"
                    color = "white" if neg_log_p[ci, ri] > threshold * 1.5 else "black"
                    ax.text(ci, ri, txt, ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label("−log₁₀(p)", fontsize=10)
        ref_line = -np.log10(alpha)
        cbar.ax.axhline(ref_line, color="red", linewidth=1.5, linestyle="--")
        cbar.ax.text(1.05, ref_line / neg_log_p.max().max(),
                     f"  α={alpha}", transform=cbar.ax.transAxes,
                     color="red", fontsize=8, va="center")

        plt.tight_layout()
        out = output_dir / "q1_pvalue_heatmap.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
    log.info("  Saved %s", out)


# ─── Figure 4: Effect Size Bar Chart ─────────────────────────────────────────


def plot_effect_sizes(
    analysis_csv: Optional[str],
    results: Optional[dict],
    output_dir: Path,
    alpha: float = 0.05,
) -> None:
    """Horizontal bar chart of Cohen's d per ticker."""
    log.info("Generating effect size plot...")

    if analysis_csv and Path(analysis_csv).exists():
        full_df = pd.read_csv(analysis_csv)
        rows = []
        for _, row in full_df.iterrows():
            if row.get("group") == "POOLED":
                continue
            rows.append({
                "ticker": row["group"],
                "cohens_d": row.get("cohens_d", float("nan")),
                "significant": bool(row.get("levene_significant", False)),
            })
    elif results:
        rows = []
        for t, r in results.items():
            if t == "POOLED":
                continue
            rows.append({
                "ticker": t,
                "cohens_d": r.get("cohens_d", float("nan")),
                "significant": bool(r.get("levene_significant", False)),
            })
    else:
        log.warning("No analysis data for effect size plot — skipping")
        return

    df_es = pd.DataFrame(rows).dropna(subset=["cohens_d"])
    df_es = df_es.sort_values("cohens_d", ascending=True).reset_index(drop=True)

    colors = [SPACELIKE_COLOR if sig else "#aaaaaa" for sig in df_es["significant"]]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(df_es["ticker"], df_es["cohens_d"], color=colors,
                       edgecolor="black", linewidth=0.5)

        # Reference lines
        ax.axvline(0.2, color="#2196F3", linestyle="--", linewidth=1.2,
                   label="Small effect (d=0.2)")
        ax.axvline(0.5, color="#FF9800", linestyle="--", linewidth=1.2,
                   label="Medium effect (d=0.5)")
        ax.axvline(0.8, color="#F44336", linestyle="--", linewidth=1.2,
                   label="Large effect (d=0.8)")
        ax.axvline(0, color="black", linewidth=0.8)

        # Value labels
        for bar, val in zip(bars, df_es["cohens_d"]):
            ax.text(max(val + 0.01, 0.01), bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

        # Legend for significance
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color=SPACELIKE_COLOR, label="Bonferroni significant"),
            Patch(color="#aaaaaa", label="Not significant"),
        ]
        ax.legend(handles=legend_handles + [
            plt.Line2D([0], [0], color="#2196F3", linestyle="--", label="Small (0.2)"),
            plt.Line2D([0], [0], color="#FF9800", linestyle="--", label="Medium (0.5)"),
            plt.Line2D([0], [0], color="#F44336", linestyle="--", label="Large (0.8)"),
        ], loc="lower right", fontsize=9)

        ax.set_xlabel("Cohen's d  (positive = SPACELIKE has higher mean abs return)")
        ax.set_title("SRFM Q1 — Effect Size by Ticker\n"
                     "Cohen's d: SPACELIKE vs TIMELIKE Next-Bar Absolute Return",
                     fontweight="bold")
        ax.grid(axis="x")

        plt.tight_layout()
        out = output_dir / "q1_effect_size.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
    log.info("  Saved %s", out)


# ─── Figure 5: Strategy Comparison (Q2) ──────────────────────────────────────


def plot_strategy_comparison(
    backtest_dir: str,
    output_dir: Path,
) -> None:
    """Grouped bar chart: RAW / RELATIVISTIC / GEODESIC — Sharpe, Sortino, MDD."""
    log.info("Generating Q2 strategy comparison plot...")
    df = _load_backtest_data(backtest_dir)
    if df is None or df.empty:
        log.warning("No backtest data found in %s — skipping Q2 figure", backtest_dir)
        return

    tickers = [t for t in TICKERS_ORDER if t in df["ticker"].unique()]
    strategies = ["RAW", "RELATIVISTIC", "GEODESIC_DEVIATION"]
    strategy_colors = {
        "RAW": "#607D8B",
        "RELATIVISTIC": TIMELIKE_COLOR,
        "GEODESIC_DEVIATION": GEODESIC_COLOR,
    }
    metrics = ["sharpe", "sortino", "max_drawdown"]
    metric_labels = ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown (lower = better)"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle("SRFM Q2 — Strategy Comparison: RAW vs RELATIVISTIC vs GEODESIC",
                     fontsize=14, fontweight="bold")

        x = np.arange(len(tickers))
        bar_width = 0.25

        for col_idx, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[col_idx]

            for s_idx, strategy in enumerate(strategies):
                vals = []
                for ticker in tickers:
                    row = df[(df["ticker"] == ticker) & (df["strategy"] == strategy)]
                    vals.append(float(row[metric].iloc[0]) if len(row) > 0 else 0.0)

                offset = (s_idx - 1) * bar_width
                bars = ax.bar(x + offset, vals, bar_width,
                              label=strategy, color=strategy_colors[strategy],
                              edgecolor="black", linewidth=0.4, alpha=0.9)

            ax.set_xticks(x)
            ax.set_xticklabels(tickers, rotation=30, ha="right")
            ax.set_title(ylabel, fontweight="bold")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.legend(fontsize=9)
            ax.grid(axis="y")

        plt.tight_layout()
        out = output_dir / "q2_strategy_comparison.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
    log.info("  Saved %s", out)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate SRFM Q1/Q2 validation figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", default="validation/results")
    p.add_argument("--backtest-dir", default="validation/backtest_results")
    p.add_argument("--output-dir", default="validation/figures")
    p.add_argument("--analysis-csv", default="validation/Q1_RESULTS_RAW.csv")
    p.add_argument("--rolling-window", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load raw regime data
    try:
        df = _load_regime_data(args.results_dir)
    except FileNotFoundError as exc:
        log.error("Could not load regime data: %s", exc)
        df = pd.DataFrame(columns=[
            "ticker", "bar_index", "interval_type",
            "next_bar_abs_return", "beta", "geodesic_deviation",
        ])

    analysis_csv = args.analysis_csv if Path(args.analysis_csv).exists() else None

    if not df.empty:
        plot_violin(df, out)
        plot_rolling_timelike(df, out, window=args.rolling_window)
    else:
        log.warning("No regime data — skipping violin and rolling plots")

    plot_pvalue_heatmap(analysis_csv, None, out)
    plot_effect_sizes(analysis_csv, None, out)
    plot_strategy_comparison(args.backtest_dir, out)

    log.info("All figures saved to %s", out)


if __name__ == "__main__":
    main()
