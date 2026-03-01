"""
run_validation.py — SRFM Empirical Validation Master Orchestrator
=================================================================
Orchestrates the full SRFM empirical validation pipeline:

  1. (Optional) Download OHLCV data via fetch_data.py
  2. Run regime_validator C++ binary for each ticker
  3. Run backtest_runner C++ binary for each ticker
  4. Run Q1 statistical analysis (analyze_q1.py)
  5. Generate all publication figures (generate_figures.py)
  6. Generate Q2 comparison report (backtest_comparison.py)

Usage
-----
    python validation/run_validation.py \\
        --binary-regime   build/regime_validator \\
        --binary-backtest build/backtest_runner \\
        [--skip-fetch] [--workers N] [--data-dir PATH] [--output-dir PATH]

Each C++ binary must already be compiled via CMake.

Dependencies
------------
    pip install pandas scipy numpy matplotlib seaborn yfinance
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TICKERS = ["AAPL", "SPY", "QQQ", "TSLA", "GS", "JPM", "NVDA", "META", "BTC-USD", "GLD"]
DEFAULT_DATA_DIR = "validation/data"
DEFAULT_OUTPUT_DIR = "validation"
DEFAULT_RESULTS_DIR = "validation/results"
DEFAULT_BACKTEST_DIR = "validation/backtest_results"
BINARY_TIMEOUT_SECS = 600

# ─── Step Runners ─────────────────────────────────────────────────────────────


def run_regime_validator(
    binary_path: str,
    ticker: str,
    csv_path: str,
    output_dir: str,
    timeout: int = BINARY_TIMEOUT_SECS,
) -> tuple[str, bool, str]:
    """
    Invoke the regime_validator binary for a single ticker.

    Parameters
    ----------
    binary_path : str
        Path to the compiled regime_validator binary.
    ticker : str
        Ticker symbol (used for --ticker argument and output filename).
    csv_path : str
        Path to the input OHLCV CSV.
    output_dir : str
        Directory where the output regime CSV is written.
    timeout : int
        Maximum seconds before forcibly terminating the process.

    Returns
    -------
    (ticker, success, message)
    """
    out_path = Path(output_dir) / f"{ticker.replace('-', '_')}_regime.csv"
    cmd = [
        binary_path,
        "--input", csv_path,
        "--output", str(out_path),
        "--ticker", ticker,
    ]
    log.info("[%s] Running regime_validator...", ticker)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            log.info("[%s] regime_validator OK → %s", ticker, out_path.name)
            return ticker, True, result.stdout.strip()
        else:
            msg = f"exit {result.returncode}: {result.stderr.strip()[:200]}"
            log.error("[%s] regime_validator FAILED — %s", ticker, msg)
            return ticker, False, msg
    except subprocess.TimeoutExpired:
        msg = f"Timed out after {timeout}s"
        log.error("[%s] regime_validator TIMEOUT", ticker)
        return ticker, False, msg
    except FileNotFoundError:
        msg = f"Binary not found: {binary_path}"
        log.error("[%s] %s", ticker, msg)
        return ticker, False, msg


def run_backtest_runner(
    binary_path: str,
    ticker: str,
    regime_csv: str,
    output_dir: str,
    timeout: int = BINARY_TIMEOUT_SECS,
) -> tuple[str, bool, str]:
    """
    Invoke the backtest_runner binary for a single ticker.

    Returns
    -------
    (ticker, success, message)
    """
    cmd = [
        binary_path,
        "--input", regime_csv,
        "--output-dir", output_dir,
        "--ticker", ticker,
    ]
    log.info("[%s] Running backtest_runner...", ticker)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            log.info("[%s] backtest_runner OK", ticker)
            return ticker, True, result.stdout.strip()
        else:
            msg = f"exit {result.returncode}: {result.stderr.strip()[:200]}"
            log.error("[%s] backtest_runner FAILED — %s", ticker, msg)
            return ticker, False, msg
    except subprocess.TimeoutExpired:
        return ticker, False, f"Timed out after {timeout}s"
    except FileNotFoundError:
        return ticker, False, f"Binary not found: {binary_path}"


def run_all_regime_validators(
    binary_path: str,
    tickers: list[str],
    data_dir: str,
    results_dir: str,
    n_workers: int = 4,
) -> dict[str, bool]:
    """
    Run regime_validator for all tickers, optionally in parallel.

    Returns
    -------
    {ticker → success}
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    outcomes: dict[str, bool] = {}

    tasks = []
    for ticker in tickers:
        safe = ticker.replace("-", "_")
        csv_path = Path(data_dir) / f"{safe}_1m.csv"
        if not csv_path.exists():
            log.warning("[%s] Data file not found: %s — skipping", ticker, csv_path)
            outcomes[ticker] = False
            continue
        tasks.append((ticker, str(csv_path)))

    if not tasks:
        log.warning("No data files found in %s — no validators to run", data_dir)
        return outcomes

    if n_workers <= 1:
        for ticker, csv_path in tasks:
            _, ok, _ = run_regime_validator(binary_path, ticker, csv_path, results_dir)
            outcomes[ticker] = ok
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(run_regime_validator, binary_path, t, c, results_dir): t
                for t, c in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                ticker_key = futures[future]
                try:
                    _, ok, _ = future.result()
                    outcomes[ticker_key] = ok
                except Exception as exc:  # noqa: BLE001
                    log.error("[%s] Unexpected error: %s", ticker_key, exc)
                    outcomes[ticker_key] = False

    return outcomes


def run_all_backtest_runners(
    binary_path: str,
    tickers: list[str],
    results_dir: str,
    backtest_dir: str,
    n_workers: int = 4,
) -> dict[str, bool]:
    """
    Run backtest_runner for all tickers that have regime CSVs.

    Returns
    -------
    {ticker → success}
    """
    Path(backtest_dir).mkdir(parents=True, exist_ok=True)
    outcomes: dict[str, bool] = {}

    tasks = []
    for ticker in tickers:
        safe = ticker.replace("-", "_")
        regime_csv = Path(results_dir) / f"{safe}_regime.csv"
        if not regime_csv.exists():
            log.warning("[%s] Regime CSV not found: %s — skipping", ticker, regime_csv)
            outcomes[ticker] = False
            continue
        tasks.append((ticker, str(regime_csv)))

    if n_workers <= 1:
        for ticker, regime_csv in tasks:
            _, ok, _ = run_backtest_runner(binary_path, ticker, regime_csv, backtest_dir)
            outcomes[ticker] = ok
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(run_backtest_runner, binary_path, t, c, backtest_dir): t
                for t, c in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                ticker_key = futures[future]
                try:
                    _, ok, _ = future.result()
                    outcomes[ticker_key] = ok
                except Exception as exc:  # noqa: BLE001
                    log.error("[%s] Unexpected error: %s", ticker_key, exc)
                    outcomes[ticker_key] = False

    return outcomes


def run_python_analysis(output_dir: str, results_dir: str, backtest_dir: str) -> None:
    """
    Run the Python analysis scripts in-process (import and call main()).
    """
    import importlib.util

    def _call_main(module_path: str, argv_override: list[str]) -> None:
        spec = importlib.util.spec_from_file_location("_mod", module_path)
        if spec is None or spec.loader is None:
            log.warning("Could not load %s", module_path)
            return
        mod = importlib.util.module_from_spec(spec)
        old_argv = sys.argv
        sys.argv = [module_path] + argv_override
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            if hasattr(mod, "main"):
                mod.main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

    base = Path(__file__).parent

    log.info("Running Q1 statistical analysis...")
    _call_main(str(base / "analyze_q1.py"), [
        "--results-dir", results_dir,
        "--output-dir", output_dir,
    ])

    log.info("Generating figures...")
    _call_main(str(base / "generate_figures.py"), [
        "--results-dir", results_dir,
        "--backtest-dir", backtest_dir,
        "--output-dir", str(Path(output_dir) / "figures"),
        "--analysis-csv", str(Path(output_dir) / "Q1_RESULTS_RAW.csv"),
    ])

    log.info("Running Q2 comparison report...")
    _call_main(str(base / "backtest_comparison.py"), [
        "--backtest-dir", backtest_dir,
        "--output-dir", output_dir,
        "--latex",
    ])


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SRFM full empirical validation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--binary-regime",
                   default="build/regime_validator",
                   help="Path to compiled regime_validator binary")
    p.add_argument("--binary-backtest",
                   default="build/backtest_runner",
                   help="Path to compiled backtest_runner binary")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help="Directory containing *_1m.csv files")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Root output directory")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                   help="Directory for regime classifier output CSVs")
    p.add_argument("--backtest-dir", default=DEFAULT_BACKTEST_DIR,
                   help="Directory for backtest output CSVs")
    p.add_argument("--tickers", nargs="+", default=TICKERS)
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel workers for C++ binary execution")
    p.add_argument("--skip-fetch", action="store_true",
                   help="Skip the yfinance data download step")
    p.add_argument("--skip-cpp", action="store_true",
                   help="Skip C++ binary execution (use existing CSVs)")
    p.add_argument("--skip-analysis", action="store_true",
                   help="Skip Python analysis (only run C++ binaries)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.monotonic()

    # ── Step 1: Data download ──────────────────────────────────────────────────
    if not args.skip_fetch:
        log.info("Step 1: Fetching OHLCV data via yfinance...")
        from fetch_data import fetch_all
        fetch_all(
            tickers=args.tickers,
            output_dir=args.data_dir,
        )
    else:
        log.info("Step 1: Skipping data fetch (--skip-fetch)")

    # ── Step 2: Regime validation ──────────────────────────────────────────────
    if not args.skip_cpp:
        log.info("Step 2: Running regime_validator for %d tickers...", len(args.tickers))
        regime_outcomes = run_all_regime_validators(
            binary_path=args.binary_regime,
            tickers=args.tickers,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            n_workers=args.workers,
        )
        n_ok = sum(1 for v in regime_outcomes.values() if v)
        log.info("Regime validation: %d/%d tickers succeeded", n_ok, len(args.tickers))

        # ── Step 3: Backtest ───────────────────────────────────────────────────
        log.info("Step 3: Running backtest_runner for %d tickers...", len(args.tickers))
        backtest_outcomes = run_all_backtest_runners(
            binary_path=args.binary_backtest,
            tickers=args.tickers,
            results_dir=args.results_dir,
            backtest_dir=args.backtest_dir,
            n_workers=args.workers,
        )
        n_ok = sum(1 for v in backtest_outcomes.values() if v)
        log.info("Backtest: %d/%d tickers succeeded", n_ok, len(args.tickers))
    else:
        log.info("Steps 2-3: Skipping C++ binaries (--skip-cpp)")

    # ── Step 4–6: Python analysis + figures + report ───────────────────────────
    if not args.skip_analysis:
        log.info("Steps 4-6: Running Python analysis, figures, and comparison report...")
        run_python_analysis(
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            backtest_dir=args.backtest_dir,
        )
    else:
        log.info("Steps 4-6: Skipping Python analysis (--skip-analysis)")

    elapsed = time.monotonic() - t0
    log.info("Pipeline complete in %.1fs.", elapsed)


if __name__ == "__main__":
    main()
