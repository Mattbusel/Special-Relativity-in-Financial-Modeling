"""
backtest_comparison.py — SRFM Q2: Strategy Comparison Report
=============================================================
Loads backtest output CSVs produced by the C++ backtest_runner binary and
generates a side-by-side performance table for RAW, RELATIVISTIC, and
GEODESIC_DEVIATION strategies across all tickers.

Input CSVs format (one per ticker):
    ticker, strategy, sharpe, sortino, max_drawdown

Output:
    - Console formatted table (and LaTeX table)
    - validation/backtest_results/Q2_COMPARISON.csv
    - validation/figures/q2_strategy_comparison.png (via generate_figures.py)

Usage
-----
    python validation/backtest_comparison.py [--backtest-dir PATH]
                                             [--output-dir PATH]
                                             [--latex]

Dependencies
------------
    pip install pandas numpy tabulate
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_BACKTEST_DIR = "validation/backtest_results"
DEFAULT_OUTPUT_DIR = "validation"

TICKERS_ORDER = ["AAPL", "SPY", "QQQ", "TSLA", "GS", "JPM", "NVDA", "META", "BTC-USD", "GLD"]
STRATEGIES = ["RAW", "RELATIVISTIC", "GEODESIC_DEVIATION"]
METRICS = ["sharpe", "sortino", "max_drawdown"]
METRIC_LABELS = {
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "MDD",
}

# ─── Data Loading ─────────────────────────────────────────────────────────────


def load_backtest_results(results_dir: str = DEFAULT_BACKTEST_DIR) -> pd.DataFrame:
    """
    Load all backtest result CSVs from the given directory.

    Expected columns: ticker, strategy, sharpe, sortino, max_drawdown
    """
    path = Path(results_dir)
    if not path.exists():
        raise FileNotFoundError(f"Backtest results directory not found: {path}")

    frames: list[pd.DataFrame] = []
    for f in sorted(path.glob("*.csv")):
        try:
            df = pd.read_csv(f)
            if "ticker" in df.columns and "strategy" in df.columns:
                frames.append(df)
                log.info("  Loaded %s — %d rows", f.name, len(df))
        except Exception as exc:  # noqa: BLE001
            log.warning("  Could not read %s: %s", f.name, exc)

    if not frames:
        raise FileNotFoundError(f"No valid backtest CSVs found in {path}")

    df = pd.concat(frames, ignore_index=True)
    df["strategy"] = df["strategy"].str.upper().str.strip()
    return df


# ─── Analysis ─────────────────────────────────────────────────────────────────


def compute_relativistic_lift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker lift: RELATIVISTIC − RAW for Sharpe, Sortino; RAW − REL for MDD.
    """
    rows = []
    for ticker in df["ticker"].unique():
        sub = df[df["ticker"] == ticker].set_index("strategy")
        raw = sub.loc["RAW"] if "RAW" in sub.index else None
        rel = sub.loc["RELATIVISTIC"] if "RELATIVISTIC" in sub.index else None
        if raw is None or rel is None:
            continue
        rows.append({
            "ticker": ticker,
            "sharpe_lift_rel": float(rel["sharpe"]) - float(raw["sharpe"]),
            "sortino_lift_rel": float(rel["sortino"]) - float(raw["sortino"]),
            "mdd_delta_rel": float(raw["max_drawdown"]) - float(rel["max_drawdown"]),
        })
    return pd.DataFrame(rows)


def compute_geodesic_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker alpha: GEODESIC_DEVIATION − RAW for Sharpe, Sortino.
    """
    rows = []
    for ticker in df["ticker"].unique():
        sub = df[df["ticker"] == ticker].set_index("strategy")
        raw = sub.loc["RAW"] if "RAW" in sub.index else None
        geo = sub.loc["GEODESIC_DEVIATION"] if "GEODESIC_DEVIATION" in sub.index else None
        if raw is None or geo is None:
            continue
        rows.append({
            "ticker": ticker,
            "sharpe_alpha_geo": float(geo["sharpe"]) - float(raw["sharpe"]),
            "sortino_alpha_geo": float(geo["sortino"]) - float(raw["sortino"]),
            "mdd_delta_geo": float(raw["max_drawdown"]) - float(geo["max_drawdown"]),
        })
    return pd.DataFrame(rows)


def generate_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a wide-format comparison table.

    Index: ticker
    Columns: MultiIndex (metric, strategy)
    """
    tickers = [t for t in TICKERS_ORDER if t in df["ticker"].unique()]
    strategies = [s for s in STRATEGIES if s in df["strategy"].unique()]

    col_tuples = [(m, s) for m in METRICS for s in strategies]
    multi_idx = pd.MultiIndex.from_tuples(col_tuples, names=["metric", "strategy"])
    result = pd.DataFrame(index=tickers, columns=multi_idx, dtype=float)

    for ticker in tickers:
        sub = df[df["ticker"] == ticker].set_index("strategy")
        for metric in METRICS:
            for strategy in strategies:
                if strategy in sub.index and metric in sub.columns:
                    result.loc[ticker, (metric, strategy)] = float(sub.loc[strategy, metric])

    return result


def _fmt(val: float, metric: str) -> str:
    if not np.isfinite(val):
        return "  N/A  "
    if metric == "max_drawdown":
        return f"{val:.3f}"
    return f"{val:+.3f}"


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print a formatted three-strategy comparison to stdout."""
    tickers = [t for t in TICKERS_ORDER if t in df["ticker"].unique()]
    strategies = [s for s in STRATEGIES if s in df["strategy"].unique()]

    print("\n" + "═" * 110)
    print("  SRFM Q2 — Strategy Performance Comparison")
    print("  Strategies: RAW vs RELATIVISTIC vs GEODESIC_DEVIATION")
    print("═" * 110)

    # Header
    header = f"  {'Ticker':<12}"
    for metric in METRICS:
        for strategy in strategies:
            short = {"RAW": "Raw", "RELATIVISTIC": "Rel", "GEODESIC_DEVIATION": "Geo"}
            header += f"  {METRIC_LABELS[metric]}/{short.get(strategy, strategy):<7}"
    print(header)
    print("─" * 110)

    for ticker in tickers:
        sub = df[df["ticker"] == ticker].set_index("strategy")
        line = f"  {ticker:<12}"
        for metric in METRICS:
            for strategy in strategies:
                if strategy in sub.index:
                    val = float(sub.loc[strategy, metric])
                    line += f"  {_fmt(val, metric):<12}"
                else:
                    line += f"  {'N/A':<12}"
        print(line)

    print("═" * 110)

    # Summary: mean lift
    rel_lift = compute_relativistic_lift(df)
    geo_alpha = compute_geodesic_alpha(df)

    if not rel_lift.empty:
        print(f"\n  Mean Relativistic Sharpe lift:   {rel_lift['sharpe_lift_rel'].mean():+.4f}")
        print(f"  Mean Relativistic MDD delta:     {rel_lift['mdd_delta_rel'].mean():+.4f}")
    if not geo_alpha.empty:
        print(f"  Mean Geodesic Sharpe alpha:      {geo_alpha['sharpe_alpha_geo'].mean():+.4f}")
        print(f"  Mean Geodesic MDD delta:         {geo_alpha['mdd_delta_geo'].mean():+.4f}")
    print()


def print_latex_table(df: pd.DataFrame) -> None:
    """Print a LaTeX tabular for inclusion in a paper."""
    tickers = [t for t in TICKERS_ORDER if t in df["ticker"].unique()]
    strategies = [s for s in STRATEGIES if s in df["strategy"].unique()]

    short = {"RAW": "Raw", "RELATIVISTIC": "Rel.", "GEODESIC_DEVIATION": "Geo."}
    col_spec = "l" + "r" * (len(METRICS) * len(strategies))

    print("\n% ─── LaTeX Table ────────────────────────────────────────────────")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\small")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")

    # Header row 1: metric groups
    metric_header = "Ticker"
    for m in METRICS:
        metric_header += f" & \\multicolumn{{{len(strategies)}}}{{c}}{{{METRIC_LABELS[m]}}}"
    print(metric_header + " \\\\")

    # Header row 2: strategy names
    strat_header = ""
    for m in METRICS:
        for s in strategies:
            strat_header += f" & {short.get(s, s)}"
    print(strat_header + " \\\\")
    print("\\midrule")

    for ticker in tickers:
        sub = df[df["ticker"] == ticker].set_index("strategy")
        line = ticker
        for metric in METRICS:
            for strategy in strategies:
                if strategy in sub.index:
                    val = float(sub.loc[strategy, metric])
                    line += f" & {_fmt(val, metric)}"
                else:
                    line += " & ---"
        print(line + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{SRFM strategy comparison: RAW vs RELATIVISTIC vs GEODESIC\\_DEVIATION.}")
    print("\\label{tab:srfm-comparison}")
    print("\\end{table}")
    print("% ─────────────────────────────────────────────────────────────────\n")


def save_comparison_csv(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    """Write a comprehensive comparison CSV."""
    out = Path(output_dir) / "backtest_results"
    out.mkdir(parents=True, exist_ok=True)

    rel_lift = compute_relativistic_lift(df)
    geo_alpha = compute_geodesic_alpha(df)

    summary = df.copy()
    if not rel_lift.empty:
        summary = summary.merge(rel_lift, on="ticker", how="left")
    if not geo_alpha.empty:
        summary = summary.merge(geo_alpha, on="ticker", how="left")

    path = out / "Q2_COMPARISON.csv"
    summary.to_csv(path, index=False)
    log.info("Q2 comparison CSV written to %s", path)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SRFM Q2: backtest strategy comparison report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backtest-dir", default=DEFAULT_BACKTEST_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--latex", action="store_true", help="Also print LaTeX table")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_backtest_results(args.backtest_dir)
    print_comparison_table(df)
    if args.latex:
        print_latex_table(df)
    save_comparison_csv(df, args.output_dir)
    log.info("Q2 comparison complete.")


if __name__ == "__main__":
    main()
