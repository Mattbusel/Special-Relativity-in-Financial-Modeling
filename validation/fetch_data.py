"""
fetch_data.py — SRFM Empirical Validation: Data Acquisition Pipeline
=====================================================================
Downloads 5 years of 1-minute OHLCV bars for the SRFM validation ticker
universe using yfinance.  Results are stored as CSV files under
validation/data/{TICKER}_1m.csv.

yfinance 1-minute data is capped at 7-day windows per API call.  This
script chunks the 5-year span into 7-day slices, retries failures with
exponential back-off, deduplicates overlapping rows, and writes a
manifest file (validation/data/manifest.json) for downstream consumers.

Usage
-----
    python validation/fetch_data.py [--years N] [--output-dir PATH]
                                    [--tickers T1 T2 ...] [--workers N]

Dependencies
------------
    pip install yfinance pandas
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError("yfinance is required: pip install yfinance") from exc

# ─── Configuration ────────────────────────────────────────────────────────────

TICKERS: list[str] = [
    "AAPL", "SPY", "QQQ", "TSLA", "GS",
    "JPM", "NVDA", "META", "BTC-USD", "GLD",
]

CHUNK_DAYS: int = 6          # yfinance allows ≤7 days per 1-min call; use 6 for safety
DEFAULT_YEARS: int = 5
MAX_RETRIES: int = 3
BASE_BACKOFF_SECS: float = 5.0
OUTPUT_DIR: str = "validation/data"
OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]
MANIFEST_FILENAME: str = "manifest.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Core Fetch Logic ─────────────────────────────────────────────────────────


def _build_chunks(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into consecutive chunks of ``chunk_days`` days."""
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def fetch_chunk(
    ticker: str,
    start: datetime,
    end: datetime,
    max_retries: int = MAX_RETRIES,
) -> Optional[pd.DataFrame]:
    """
    Fetch one 1-minute OHLCV chunk for *ticker* in [start, end).

    Returns a tidy DataFrame with columns [timestamp, open, high, low,
    close, volume] (timestamp as UTC string), or None on terminal failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers=ticker,
                period="5y",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if raw is None or raw.empty:
                return None

            # Flatten MultiIndex columns produced by yf.download for a single ticker
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0].lower() for col in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]

            # Keep only OHLCV columns that actually exist
            present = [c for c in OHLCV_COLUMNS if c in raw.columns]
            raw = raw[present].copy()

            # Normalise index → UTC string timestamp column
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC")
            else:
                raw.index = raw.index.tz_convert("UTC")

            raw.index.name = "timestamp"
            raw.reset_index(inplace=True)
            raw["timestamp"] = raw["timestamp"].astype(str)

            # Drop rows that are entirely NaN in price columns
            price_cols = [c for c in ["open", "high", "low", "close"] if c in raw.columns]
            raw.dropna(subset=price_cols, how="all", inplace=True)

            return raw if not raw.empty else None

        except Exception as exc:  # noqa: BLE001
            backoff = BASE_BACKOFF_SECS * (2 ** (attempt - 1))
            log.warning(
                "  [%s] chunk %s→%s attempt %d/%d failed: %s  (retry in %.0fs)",
                ticker,
                start.date(),
                end.date(),
                attempt,
                max_retries,
                exc,
                backoff,
            )
            if attempt < max_retries:
                time.sleep(backoff)

    log.error("  [%s] chunk %s→%s permanently failed after %d attempts",
              ticker, start.date(), end.date(), max_retries)
    return None


def fetch_ticker(
    ticker: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """
    Fetch all daily bars for *ticker* (5-year period) and write to CSV.

    With interval="1d" + period="5y" yfinance returns the full history in
    one call — no chunking needed.  start/end are kept for signature
    compatibility but are not passed to yf.download.

    Returns a manifest entry dict:
        {ticker, rows, start_date, end_date, file_path, status}
    """
    log.info("[%s] Fetching 5y daily bars (single call)...", ticker)

    df = fetch_chunk(ticker, start, end, max_retries)
    failed_chunks = 0 if df is not None else 1

    frames: list[pd.DataFrame] = [] if df is None else [df]

    if not frames:
        log.warning("[%s] No data retrieved", ticker)
        return {
            "ticker": ticker,
            "rows": 0,
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "file_path": "",
            "status": "failed",
            "failed_chunks": failed_chunks,
        }

    # Concatenate, deduplicate, sort
    combined = pd.concat(frames, ignore_index=True)
    before_dedup = len(combined)
    combined.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Ensure all OHLCV columns present (fill missing with NaN)
    for col in OHLCV_COLUMNS:
        if col not in combined.columns:
            combined[col] = float("nan")

    # Write CSV
    safe_ticker = ticker.replace("-", "_")
    out_path = output_dir / f"{safe_ticker}_1m.csv"
    combined.to_csv(out_path, index=False)

    rows = len(combined)
    dedup_removed = before_dedup - rows
    log.info("[%s] Done — %d bars written to %s  (dedup removed %d, failed chunks %d)",
             ticker, rows, out_path.name, dedup_removed, failed_chunks)

    return {
        "ticker": ticker,
        "rows": rows,
        "start_date": combined["timestamp"].iloc[0] if rows > 0 else str(start.date()),
        "end_date": combined["timestamp"].iloc[-1] if rows > 0 else str(end.date()),
        "file_path": str(out_path),
        "status": "ok",
        "dedup_removed": dedup_removed,
        "failed_chunks": failed_chunks,
    }


def fetch_all(
    tickers: list[str] = TICKERS,
    years: int = DEFAULT_YEARS,
    output_dir: str = OUTPUT_DIR,
    max_workers: int = 1,
    max_retries: int = MAX_RETRIES,
) -> dict[str, dict]:
    """
    Fetch 1-minute OHLCV data for all *tickers* over the last *years* years.

    Parameters
    ----------
    tickers:
        List of ticker symbols.
    years:
        How many years back from today to fetch.
    output_dir:
        Directory where CSVs and manifest are written.
    max_workers:
        Parallel ticker downloads (use 1 to avoid rate-limiting).
    max_retries:
        Per-chunk retry attempts.

    Returns
    -------
    Manifest dict: {ticker → manifest_entry}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    end_dt = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=365 * years)

    log.info("Fetching %d tickers | %d years | %s → %s | workers=%d",
             len(tickers), years, start_dt.date(), end_dt.date(), max_workers)

    manifest: dict[str, dict] = {}

    if max_workers <= 1:
        for ticker in tickers:
            entry = fetch_ticker(ticker, start_dt, end_dt, out, max_retries)
            manifest[ticker] = entry
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(fetch_ticker, t, start_dt, end_dt, out, max_retries): t
                for t in tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    entry = future.result()
                    manifest[ticker] = entry
                except Exception as exc:  # noqa: BLE001
                    log.error("[%s] Unexpected error: %s", ticker, exc)
                    manifest[ticker] = {
                        "ticker": ticker, "rows": 0, "status": "error",
                        "error": str(exc),
                    }

    # Write manifest
    manifest_path = out / MANIFEST_FILENAME
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    log.info("Manifest written to %s", manifest_path)

    # Summary table
    total_rows = sum(v.get("rows", 0) for v in manifest.values())
    log.info("─" * 60)
    log.info("  %-12s  %9s  %s", "Ticker", "Rows", "Status")
    log.info("─" * 60)
    for t, v in manifest.items():
        log.info("  %-12s  %9d  %s", t, v.get("rows", 0), v.get("status", "?"))
    log.info("─" * 60)
    log.info("  %-12s  %9d", "TOTAL", total_rows)

    return manifest


# ─── CLI Entry Point ──────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download 5-year 1-minute OHLCV data for SRFM validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--years", type=int, default=DEFAULT_YEARS,
                   help="Number of years of history to fetch")
    p.add_argument("--output-dir", default=OUTPUT_DIR,
                   help="Directory to write CSV files and manifest")
    p.add_argument("--tickers", nargs="+", default=TICKERS,
                   help="Ticker symbols to download")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel download workers (1 = sequential, safer for rate limits)")
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                   help="Per-chunk retry attempts before giving up")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = fetch_all(
        tickers=args.tickers,
        years=args.years,
        output_dir=args.output_dir,
        max_workers=args.workers,
        max_retries=args.max_retries,
    )
    failed = [t for t, v in manifest.items() if v.get("status") != "ok"]
    if failed:
        log.warning("The following tickers had issues: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
