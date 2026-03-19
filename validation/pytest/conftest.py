"""
conftest.py — shared pytest fixtures for the SRFM validation suite.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the repository root by walking up from this file."""
    here = Path(__file__).resolve()
    # Walk up until we find a directory that contains CMakeLists.txt or Cargo.toml.
    for parent in here.parents:
        if (parent / "CMakeLists.txt").exists() or (parent / "Cargo.toml").exists():
            return parent
    # Fallback: three levels up from validation/pytest/
    return here.parent.parent.parent


REPO_ROOT = _repo_root()
FIXTURE_CSV = REPO_ROOT / "tests" / "fixtures" / "sample_ohlcv.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_ohlcv_path() -> Path:
    """Absolute path to the sample OHLCV fixture CSV file."""
    if not FIXTURE_CSV.exists():
        pytest.skip(f"Fixture file not found: {FIXTURE_CSV}")
    return FIXTURE_CSV


@pytest.fixture(scope="session")
def sample_ohlcv_rows(sample_ohlcv_path: Path) -> list[dict]:
    """
    Load sample_ohlcv.csv and return a list of row dicts with typed fields.

    Each dict has keys: Date (str), Open, High, Low, Close (float), Volume (int).
    """
    rows: list[dict] = []
    with sample_ohlcv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "Date": row["Date"],
                    "Open": float(row["Open"]),
                    "High": float(row["High"]),
                    "Low": float(row["Low"]),
                    "Close": float(row["Close"]),
                    "Volume": int(row["Volume"]),
                }
            )
    if not rows:
        pytest.skip("sample_ohlcv.csv contains no data rows")
    return rows


@pytest.fixture(scope="session")
def close_prices(sample_ohlcv_rows: list[dict]) -> list[float]:
    """Convenience fixture: just the Close prices as a plain Python list."""
    return [row["Close"] for row in sample_ohlcv_rows]


@pytest.fixture(scope="session")
def simple_returns(close_prices: list[float]) -> list[float]:
    """
    Close-to-close simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}.

    Length is len(close_prices) - 1.
    """
    prices = close_prices
    return [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
