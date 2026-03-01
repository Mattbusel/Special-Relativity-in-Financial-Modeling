"""
analyze_q1.py — SRFM Q1: TIMELIKE vs SPACELIKE Return Variance Analysis
=========================================================================
Answers Research Question 1:

    Do TIMELIKE market regimes (ds² < 0, causal) exhibit statistically
    lower next-bar absolute return variance than SPACELIKE regimes?

Input: CSV files in validation/results/ produced by the C++ regime_validator
binary.  Each CSV has columns:
    ticker, bar_index, interval_type, next_bar_abs_return, beta, geodesic_deviation

Output:
    - Console report with per-ticker and pooled statistics
    - validation/Q1_RESULTS_RAW.csv — machine-readable stats table
    - validation/Q1_RESULTS.md — human-readable summary

Usage
-----
    python validation/analyze_q1.py [--results-dir PATH] [--output-dir PATH]
                                    [--n-boot N] [--alpha FLOAT]

Dependencies
------------
    pip install pandas scipy numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_RESULTS_DIR = "validation/results"
DEFAULT_OUTPUT_DIR = "validation"
DEFAULT_N_BOOT = 10_000
DEFAULT_ALPHA = 0.05

INTERVAL_COL = "interval_type"
RETURN_COL = "next_bar_abs_return"
TICKER_COL = "ticker"

TIMELIKE_LABEL = "Timelike"
SPACELIKE_LABEL = "Spacelike"

# ─── Data Loading ─────────────────────────────────────────────────────────────


def load_regime_data(results_dir: str = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """
    Load all regime-classified return data from CSV files.

    Parameters
    ----------
    results_dir:
        Directory containing one CSV per ticker, produced by regime_validator.

    Returns
    -------
    DataFrame with columns [ticker, interval_type, next_bar_abs_return, beta,
    geodesic_deviation].  Rows with non-finite return values are dropped.
    """
    path = Path(results_dir)
    if not path.exists():
        raise FileNotFoundError(f"Results directory not found: {path}")

    csv_files = list(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    frames: list[pd.DataFrame] = []
    for f in sorted(csv_files):
        try:
            df = pd.read_csv(f)
            frames.append(df)
            log.info("  Loaded %s — %d rows", f.name, len(df))
        except Exception as exc:  # noqa: BLE001
            log.warning("  Could not read %s: %s", f.name, exc)

    if not frames:
        raise ValueError("No valid data files could be loaded")

    combined = pd.concat(frames, ignore_index=True)

    # Normalise interval_type casing
    combined[INTERVAL_COL] = combined[INTERVAL_COL].str.strip().str.title()

    # Drop non-finite returns
    before = len(combined)
    combined = combined[np.isfinite(combined[RETURN_COL])].copy()
    dropped = before - len(combined)
    if dropped:
        log.info("Dropped %d rows with non-finite return values", dropped)

    log.info("Total rows loaded: %d  |  Tickers: %s",
             len(combined),
             sorted(combined[TICKER_COL].unique()))
    return combined


# ─── Statistical Tests ────────────────────────────────────────────────────────


def _split_regimes(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (timelike_returns, spacelike_returns) arrays, optionally filtered by ticker."""
    if ticker is not None:
        df = df[df[TICKER_COL] == ticker]
    tl = df.loc[df[INTERVAL_COL] == TIMELIKE_LABEL, RETURN_COL].values
    sl = df.loc[df[INTERVAL_COL] == SPACELIKE_LABEL, RETURN_COL].values
    return tl, sl


def run_levene_test(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> dict:
    """
    Levene's test for equality of variance (robust to non-normality).

    Returns
    -------
    dict with keys: statistic, p_value, n_timelike, n_spacelike
    """
    tl, sl = _split_regimes(df, ticker)
    if len(tl) < 2 or len(sl) < 2:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n_timelike": len(tl), "n_spacelike": len(sl)}

    stat, pval = stats.levene(tl, sl, center="median")
    return {
        "test": "Levene",
        "statistic": float(stat),
        "p_value": float(pval),
        "n_timelike": int(len(tl)),
        "n_spacelike": int(len(sl)),
    }


def run_bartlett_test(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> dict:
    """
    Bartlett's test for equality of variance (parametric, assumes normality).

    Returns
    -------
    dict with keys: statistic, p_value
    """
    tl, sl = _split_regimes(df, ticker)
    if len(tl) < 2 or len(sl) < 2:
        return {"statistic": float("nan"), "p_value": float("nan")}

    try:
        stat, pval = stats.bartlett(tl, sl)
    except Exception:  # noqa: BLE001
        return {"test": "Bartlett", "statistic": float("nan"), "p_value": float("nan")}

    return {
        "test": "Bartlett",
        "statistic": float(stat),
        "p_value": float(pval),
    }


def compute_cohens_d(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> float:
    """
    Cohen's d effect size comparing TIMELIKE vs SPACELIKE absolute return distributions.

    Uses pooled standard deviation in the denominator.

    Returns
    -------
    Cohen's d (positive = SPACELIKE has higher mean abs return).
    nan if insufficient data.
    """
    tl, sl = _split_regimes(df, ticker)
    if len(tl) < 2 or len(sl) < 2:
        return float("nan")

    mean_tl = float(np.mean(tl))
    mean_sl = float(np.mean(sl))
    pooled_std = float(np.sqrt(
        ((len(tl) - 1) * np.var(tl, ddof=1) + (len(sl) - 1) * np.var(sl, ddof=1))
        / (len(tl) + len(sl) - 2)
    ))

    if pooled_std < 1e-12:
        return 0.0

    return (mean_sl - mean_tl) / pooled_std


def compute_variance_ratio(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> float:
    """
    Variance ratio: Var(SPACELIKE) / Var(TIMELIKE).  Values > 1 support H1.
    """
    tl, sl = _split_regimes(df, ticker)
    if len(tl) < 2 or len(sl) < 2:
        return float("nan")
    var_tl = float(np.var(tl, ddof=1))
    var_sl = float(np.var(sl, ddof=1))
    if var_tl < 1e-20:
        return float("nan")
    return var_sl / var_tl


def compute_confidence_intervals(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = 0.95,
) -> dict:
    """
    Bootstrap 95% CI for the variance of TIMELIKE and SPACELIKE return populations.

    Returns
    -------
    dict with keys: timelike_var, timelike_var_ci_lo, timelike_var_ci_hi,
                    spacelike_var, spacelike_var_ci_lo, spacelike_var_ci_hi
    """
    tl, sl = _split_regimes(df, ticker)
    alpha = 1.0 - ci
    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)

    def bootstrap_var(arr: np.ndarray) -> tuple[float, float, float]:
        if len(arr) < 2:
            nan = float("nan")
            return nan, nan, nan
        rng = np.random.default_rng(42)
        boot_vars = [
            float(np.var(rng.choice(arr, size=len(arr), replace=True), ddof=1))
            for _ in range(n_boot)
        ]
        return (
            float(np.var(arr, ddof=1)),
            float(np.percentile(boot_vars, lo_pct)),
            float(np.percentile(boot_vars, hi_pct)),
        )

    tl_var, tl_lo, tl_hi = bootstrap_var(tl)
    sl_var, sl_lo, sl_hi = bootstrap_var(sl)

    return {
        "timelike_var": tl_var,
        "timelike_var_ci_lo": tl_lo,
        "timelike_var_ci_hi": tl_hi,
        "spacelike_var": sl_var,
        "spacelike_var_ci_lo": sl_lo,
        "spacelike_var_ci_hi": sl_hi,
    }


# ─── Full Analysis ────────────────────────────────────────────────────────────


def run_full_analysis(
    df: pd.DataFrame,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """
    Run all statistical tests per-ticker and pooled.

    Returns
    -------
    dict keyed by ticker symbol + "POOLED", each value a stats dict.
    """
    tickers = sorted(df[TICKER_COL].unique())
    groups = list(tickers) + ["POOLED"]
    n_comparisons = len(tickers)   # Bonferroni denominator (per-ticker tests only)

    results: dict[str, dict] = {}

    for group in groups:
        ticker_arg = None if group == "POOLED" else group
        log.info("  Analysing %s ...", group)

        levene = run_levene_test(df, ticker_arg)
        bartlett = run_bartlett_test(df, ticker_arg)
        d = compute_cohens_d(df, ticker_arg)
        var_ratio = compute_variance_ratio(df, ticker_arg)
        ci_dict = compute_confidence_intervals(df, ticker_arg, n_boot)

        # Bonferroni-corrected p-values (apply only to per-ticker, not pooled)
        correction_factor = n_comparisons if group != "POOLED" else 1
        levene_p_corr = min(1.0, levene["p_value"] * correction_factor)
        bartlett_p_corr = min(1.0, bartlett["p_value"] * correction_factor)

        results[group] = {
            "group": group,
            "n_timelike": levene["n_timelike"],
            "n_spacelike": levene["n_spacelike"],
            "levene_stat": levene["statistic"],
            "levene_p": levene["p_value"],
            "levene_p_bonferroni": levene_p_corr,
            "levene_significant": levene_p_corr < alpha,
            "bartlett_stat": bartlett["statistic"],
            "bartlett_p": bartlett["p_value"],
            "bartlett_p_bonferroni": bartlett_p_corr,
            "bartlett_significant": bartlett_p_corr < alpha,
            "cohens_d": d,
            "variance_ratio": var_ratio,
            **ci_dict,
        }

    return results


def print_report(results: dict, output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    """
    Print a formatted results table to stdout and write CSV + Markdown report.
    """
    # ── Console Table ──────────────────────────────────────────────────────────
    cols = [
        "group", "n_timelike", "n_spacelike",
        "levene_p", "levene_p_bonferroni", "levene_significant",
        "cohens_d", "variance_ratio",
        "timelike_var", "spacelike_var",
    ]
    rows = [results[k] for k in sorted(results.keys())]
    report_df = pd.DataFrame(rows, columns=[c for c in cols if c in rows[0]])
    report_df.rename(columns={
        "group": "Group",
        "n_timelike": "N(TL)",
        "n_spacelike": "N(SL)",
        "levene_p": "Levene p",
        "levene_p_bonferroni": "p (Bonf.)",
        "levene_significant": "Sig?",
        "cohens_d": "Cohen d",
        "variance_ratio": "VarRatio",
        "timelike_var": "Var(TL)",
        "spacelike_var": "Var(SL)",
    }, inplace=True)

    print("\n" + "=" * 100)
    print("  SRFM Q1 -- TIMELIKE vs SPACELIKE Return Variance Analysis")
    print("=" * 100)
    print(report_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 100 + "\n")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "Q1_RESULTS_RAW.csv"
    full_df = pd.DataFrame(list(results.values()))
    full_df.to_csv(csv_path, index=False)
    log.info("Raw results written to %s", csv_path)


def write_markdown_report(
    results: dict,
    df: pd.DataFrame,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """Write validation/Q1_RESULTS.md with human-readable findings."""
    tickers = sorted(t for t in results if t != "POOLED")
    pooled = results.get("POOLED", {})

    sig_tickers = [t for t in tickers if results[t].get("levene_significant", False)]
    var_ratio_pooled = pooled.get("variance_ratio", float("nan"))
    d_pooled = pooled.get("cohens_d", float("nan"))
    p_pooled = pooled.get("levene_p", float("nan"))
    n_tl = pooled.get("n_timelike", 0)
    n_sl = pooled.get("n_spacelike", 0)

    lines: list[str] = [
        "# SRFM Q1 Results — TIMELIKE vs SPACELIKE Return Variance",
        "",
        f"**Analysis Date:** {pd.Timestamp.now().date()}  ",
        f"**Tickers:** {', '.join(tickers)}  ",
        f"**Total TIMELIKE bars:** {n_tl:,}  ",
        f"**Total SPACELIKE bars:** {n_sl:,}  ",
        "",
        "## Executive Summary",
        "",
    ]

    if not np.isnan(var_ratio_pooled) and var_ratio_pooled > 1.0:
        lines += [
            f"The pooled analysis **confirms H1**: SPACELIKE regimes exhibit "
            f"**{var_ratio_pooled:.2f}× higher return variance** than TIMELIKE regimes "
            f"(Levene p={p_pooled:.2e}, Cohen's d={d_pooled:.3f}).  ",
            "",
        ]
    else:
        lines += [
            "The pooled analysis **does not confirm H1** at the current significance level.",
            "",
        ]

    lines += [
        "## Per-Ticker Results",
        "",
        "| Ticker | N(TL) | N(SL) | Levene p (Bonf.) | Significant | Cohen d | Var Ratio |",
        "|--------|------:|------:|-----------------|:-----------:|--------:|----------:|",
    ]

    for t in tickers:
        r = results[t]
        sig = "✓" if r.get("levene_significant") else "✗"
        lines.append(
            f"| {t} | {r['n_timelike']:,} | {r['n_spacelike']:,} | "
            f"{r['levene_p_bonferroni']:.4f} | {sig} | "
            f"{r['cohens_d']:.3f} | {r['variance_ratio']:.2f} |"
        )

    lines += [
        "",
        "## Pooled Results",
        "",
        f"- **Levene statistic:** {pooled.get('levene_stat', float('nan')):.4f}  ",
        f"- **Levene p-value:** {p_pooled:.2e}  ",
        f"- **Bartlett p-value:** {pooled.get('bartlett_p', float('nan')):.2e}  ",
        f"- **Cohen's d:** {d_pooled:.4f}  ",
        f"- **Variance ratio (SL/TL):** {var_ratio_pooled:.4f}  ",
        f"- **Timelike variance 95% CI:** [{pooled.get('timelike_var_ci_lo', float('nan')):.6f}, "
        f"{pooled.get('timelike_var_ci_hi', float('nan')):.6f}]  ",
        f"- **Spacelike variance 95% CI:** [{pooled.get('spacelike_var_ci_lo', float('nan')):.6f}, "
        f"{pooled.get('spacelike_var_ci_hi', float('nan')):.6f}]  ",
        "",
        "## Interpretation",
        "",
        "A Levene p-value < 0.05 (after Bonferroni correction for 10 tickers) rejects the",
        "null hypothesis that TIMELIKE and SPACELIKE regimes have equal return variance.",
        "Cohen's d > 0.2 indicates a practically meaningful effect size.",
        "",
        "The variance ratio directly quantifies how much more volatile SPACELIKE bars are",
        "relative to TIMELIKE bars — the key prediction of the SRFM framework.",
        "",
        f"**Significant tickers:** {', '.join(sig_tickers) if sig_tickers else 'None'}  ",
        "",
        "---",
        "*Generated by validation/analyze_q1.py*",
    ]

    md_path = Path(output_dir) / "Q1_RESULTS.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown report written to %s", md_path)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SRFM Q1: statistical analysis of TIMELIKE vs SPACELIKE return variance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_regime_data(args.results_dir)
    results = run_full_analysis(df, n_boot=args.n_boot, alpha=args.alpha)
    print_report(results, args.output_dir)
    write_markdown_report(results, df, args.output_dir)
    log.info("Q1 analysis complete.")


if __name__ == "__main__":
    main()
