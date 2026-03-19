"""
test_fetch_mock.py — unit tests for fetch_data.py using mocked yfinance.

These tests verify that fetch_data.fetch_chunk processes a mocked yfinance
response correctly without making any real network calls.  The mocked
DataFrame mirrors the structure produced by yf.download for a single ticker.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Locate the validation package root so we can import fetch_data
# ---------------------------------------------------------------------------

_VALIDATION_DIR = Path(__file__).resolve().parent.parent  # validation/
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))


# ---------------------------------------------------------------------------
# Helpers: build a mock DataFrame that mirrors yf.download output
# ---------------------------------------------------------------------------

_FIXTURE_CSV = (
    Path(__file__).resolve().parent.parent.parent  # repo root
    / "tests" / "fixtures" / "sample_ohlcv.csv"
)


def _make_mock_df() -> pd.DataFrame:
    """
    Build a DataFrame that resembles what yf.download returns for a single
    ticker (flat columns, DatetimeIndex with UTC timezone).
    """
    raw = pd.read_csv(_FIXTURE_CSV, parse_dates=["Date"])
    raw = raw.rename(columns={
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.set_index("timestamp")
    return raw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_yf_download() -> pd.DataFrame:
    """Return the fixture DataFrame that the mock will produce."""
    if not _FIXTURE_CSV.exists():
        pytest.skip(f"Fixture not found: {_FIXTURE_CSV}")
    return _make_mock_df()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fetch_chunk_returns_dataframe_when_mocked(mock_yf_download: pd.DataFrame) -> None:
    """
    fetch_chunk returns a processed DataFrame when yf.download is mocked to
    return the fixture data.
    """
    import fetch_data  # noqa: PLC0415

    with patch("fetch_data.yf.download", return_value=mock_yf_download) as mock_dl:
        from datetime import datetime, timezone

        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 6, 1, tzinfo=timezone.utc)
        result = fetch_data.fetch_chunk("AAPL", start, end, max_retries=1)

        mock_dl.assert_called_once()

    assert result is not None, "fetch_chunk should return a DataFrame, not None"
    assert isinstance(result, pd.DataFrame)


def test_fetch_chunk_result_has_expected_columns(mock_yf_download: pd.DataFrame) -> None:
    """
    The DataFrame returned by fetch_chunk must include a 'timestamp' column
    and at least the OHLCV columns that were present in the mock data.
    """
    import fetch_data  # noqa: PLC0415

    with patch("fetch_data.yf.download", return_value=mock_yf_download):
        from datetime import datetime, timezone

        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 6, 1, tzinfo=timezone.utc)
        result = fetch_data.fetch_chunk("AAPL", start, end, max_retries=1)

    assert result is not None
    assert "timestamp" in result.columns, "'timestamp' column must be present"
    for col in ("open", "high", "low", "close"):
        assert col in result.columns, f"'{col}' column must be present"


def test_fetch_chunk_timestamps_are_strings(mock_yf_download: pd.DataFrame) -> None:
    """
    fetch_chunk normalises the timestamp column to UTC strings, not
    datetime objects, matching the downstream CSV schema.
    """
    import fetch_data  # noqa: PLC0415

    with patch("fetch_data.yf.download", return_value=mock_yf_download):
        from datetime import datetime, timezone

        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 6, 1, tzinfo=timezone.utc)
        result = fetch_data.fetch_chunk("AAPL", start, end, max_retries=1)

    assert result is not None
    assert result["timestamp"].dtype == object, (
        "timestamp column should be dtype=object (string), "
        f"got {result['timestamp'].dtype}"
    )


def test_fetch_chunk_drops_all_nan_price_rows(mock_yf_download: pd.DataFrame) -> None:
    """
    Rows where all price columns (open, high, low, close) are NaN must be
    dropped by fetch_chunk.
    """
    import fetch_data  # noqa: PLC0415

    # Inject a fully-NaN price row into the mock DataFrame
    nan_row = pd.DataFrame(
        [{col: float("nan") for col in ("open", "high", "low", "close", "volume")}],
        index=pd.DatetimeIndex(["2023-06-01"], tz="UTC", name="timestamp"),
    )
    mock_with_nan = pd.concat([mock_yf_download, nan_row])

    with patch("fetch_data.yf.download", return_value=mock_with_nan):
        from datetime import datetime, timezone

        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 7, 1, tzinfo=timezone.utc)
        result = fetch_data.fetch_chunk("AAPL", start, end, max_retries=1)

    assert result is not None
    # The result must not contain a row where all price columns are NaN
    price_cols = [c for c in ("open", "high", "low", "close") if c in result.columns]
    all_nan_mask = result[price_cols].isna().all(axis=1)
    assert not all_nan_mask.any(), "fetch_chunk must drop rows where all prices are NaN"


def test_fetch_chunk_returns_none_on_empty_response() -> None:
    """
    fetch_chunk returns None when yf.download returns an empty DataFrame.
    """
    import fetch_data  # noqa: PLC0415

    empty_df = pd.DataFrame()

    with patch("fetch_data.yf.download", return_value=empty_df):
        from datetime import datetime, timezone

        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 1, 7, tzinfo=timezone.utc)
        result = fetch_data.fetch_chunk("AAPL", start, end, max_retries=1)

    assert result is None, "fetch_chunk must return None when yf.download returns empty DataFrame"


def test_fetch_chunk_no_real_network_call(mock_yf_download: pd.DataFrame) -> None:
    """
    Confirm that the mock intercepts all calls and no real yfinance network
    request is made (call count == 1).
    """
    import fetch_data  # noqa: PLC0415

    with patch("fetch_data.yf.download", return_value=mock_yf_download) as mock_dl:
        from datetime import datetime, timezone

        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 1, 7, tzinfo=timezone.utc)
        fetch_data.fetch_chunk("SPY", start, end, max_retries=1)

    assert mock_dl.call_count == 1, (
        f"Expected exactly 1 call to yf.download, got {mock_dl.call_count}"
    )
