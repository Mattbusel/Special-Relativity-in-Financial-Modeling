"""
test_backtest.py — reproducibility and correctness checks for Sharpe ratio.

These tests use only the Python reference implementations of the performance
metrics (no C++ binary required) to verify:
  1. Sharpe ratio formula correctness against known values.
  2. Deterministic / reproducible results for the same input data.
  3. Sharpe computed on the sample_ohlcv fixture is in a reasonable range.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

import pytest


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def annualised_sharpe(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    annualisation: float = 252.0,
) -> float | None:
    """
    Compute the annualised Sharpe ratio.

    Formula: (mean(R) - r_f) / std(R) * sqrt(ann)

    Returns None when the series is too short (<2 elements) or std == 0.
    """
    if len(returns) < 2:
        return None
    mean_r = statistics.mean(returns)
    std_r = statistics.stdev(returns)
    if std_r == 0.0:
        return None
    return (mean_r - risk_free_rate) / std_r * math.sqrt(annualisation)


def max_drawdown(returns: Sequence[float]) -> float | None:
    """
    Compute maximum peak-to-trough drawdown from a return series.

    Builds the cumulative equity curve then measures the largest fractional
    decline from a running peak.

    Returns None on empty input.
    """
    if not returns:
        return None
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for r in returns:
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak
        if drawdown > mdd:
            mdd = drawdown
    return mdd


# ---------------------------------------------------------------------------
# Known-value correctness tests
# ---------------------------------------------------------------------------


def test_sharpe_constant_positive_returns() -> None:
    """Constant positive excess returns → large positive Sharpe."""
    returns = [0.001] * 100
    sharpe = annualised_sharpe(returns)
    assert sharpe is not None
    assert sharpe > 0.0
    assert math.isfinite(sharpe)


def test_sharpe_constant_negative_returns() -> None:
    """Constant negative returns → negative Sharpe."""
    returns = [-0.001] * 100
    sharpe = annualised_sharpe(returns)
    assert sharpe is not None
    assert sharpe < 0.0


def test_sharpe_zero_mean_returns() -> None:
    """
    Returns that alternate ±0.01 have mean = 0 and positive std.
    Sharpe should be approximately 0 (with small numerical error).
    """
    returns = [0.01 if i % 2 == 0 else -0.01 for i in range(100)]
    sharpe = annualised_sharpe(returns, risk_free_rate=0.0)
    assert sharpe is not None
    assert abs(sharpe) < 0.1, f"Expected Sharpe ≈ 0, got {sharpe:.6f}"


def test_sharpe_formula_manual_verification() -> None:
    """
    Manually compute Sharpe for a small series and compare to the function.

    Series: [0.01, 0.02, 0.03] with r_f=0, ann=252.
    """
    returns = [0.01, 0.02, 0.03]
    mean_r = (0.01 + 0.02 + 0.03) / 3  # 0.02
    std_r = statistics.stdev(returns)   # sample std
    expected = mean_r / std_r * math.sqrt(252.0)

    result = annualised_sharpe(returns, risk_free_rate=0.0, annualisation=252.0)
    assert result is not None
    assert abs(result - expected) < 1e-12, (
        f"Formula mismatch: got {result:.10f}, expected {expected:.10f}"
    )


def test_sharpe_too_few_elements_returns_none() -> None:
    """Series with fewer than 2 elements is undefined → None."""
    assert annualised_sharpe([]) is None
    assert annualised_sharpe([0.01]) is None


def test_sharpe_zero_variance_returns_none() -> None:
    """Constant return series has zero variance → None (division by zero)."""
    returns = [0.005] * 50
    assert annualised_sharpe(returns) is None


def test_sharpe_risk_free_rate_shifts_result() -> None:
    """Increasing the risk-free rate reduces the Sharpe ratio."""
    returns = [0.001] * 100
    sharpe_0 = annualised_sharpe(returns, risk_free_rate=0.0)
    sharpe_rf = annualised_sharpe(returns, risk_free_rate=0.0005)
    assert sharpe_0 is not None
    assert sharpe_rf is not None
    assert sharpe_0 > sharpe_rf


# ---------------------------------------------------------------------------
# Max drawdown tests
# ---------------------------------------------------------------------------


def test_max_drawdown_monotone_up_is_zero() -> None:
    """Monotonically increasing equity → zero drawdown."""
    returns = [0.005] * 50
    mdd = max_drawdown(returns)
    assert mdd is not None
    assert mdd == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_single_drop() -> None:
    """
    Equity rises then falls: peak=1.01, trough=0.99 → MDD ≈ 2%.
    """
    returns = [0.01, -0.02]
    mdd = max_drawdown(returns)
    assert mdd is not None
    # peak after r=0.01 is 1.01; after r=-0.02, equity = 1.01 * 0.98 ≈ 0.9898
    # drawdown = (1.01 - 0.9898) / 1.01
    peak = 1.01
    trough = 1.01 * 0.98
    expected = (peak - trough) / peak
    assert abs(mdd - expected) < 1e-12


def test_max_drawdown_in_unit_interval() -> None:
    """Max drawdown is always in [0, 1] for any return series."""
    test_cases = [
        [0.01, -0.05, 0.02, -0.03],
        [-0.1, -0.1, 0.2, 0.1],
        [0.001] * 30,
        [-0.001] * 30,
    ]
    for returns in test_cases:
        mdd = max_drawdown(returns)
        assert mdd is not None
        assert 0.0 <= mdd <= 1.0, f"MDD={mdd:.6f} out of [0,1] for {returns[:5]}..."


def test_max_drawdown_empty_returns_none() -> None:
    """Empty series → None."""
    assert max_drawdown([]) is None


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


def test_sharpe_deterministic_same_input(simple_returns: list[float]) -> None:
    """
    Two independent calls with the same return series produce identical Sharpe.
    """
    returns = simple_returns
    sharpe_a = annualised_sharpe(returns)
    sharpe_b = annualised_sharpe(returns)

    assert sharpe_a is not None, "Sharpe returned None on first call"
    assert sharpe_b is not None, "Sharpe returned None on second call"
    assert sharpe_a == sharpe_b, (
        f"Non-deterministic result: first={sharpe_a}, second={sharpe_b}"
    )


def test_sharpe_fixture_in_reasonable_range(simple_returns: list[float]) -> None:
    """
    Sharpe on the sample_ohlcv fixture must be finite and in [-5, 5].

    The fixture uses a slowly-drifting synthetic series (roughly +0.5% daily
    drift) so the Sharpe should be positive and well-defined.
    """
    returns = simple_returns
    sharpe = annualised_sharpe(returns)

    assert sharpe is not None, (
        "annualised_sharpe returned None; check return series length and variance"
    )
    assert math.isfinite(sharpe), f"Sharpe is not finite: {sharpe}"
    assert -5.0 <= sharpe <= 5.0, (
        f"Sharpe={sharpe:.4f} outside reasonable range [-5, 5]"
    )


def test_sharpe_fixture_positive(simple_returns: list[float]) -> None:
    """
    The fixture has a consistent upward trend, so Sharpe should be positive.
    """
    returns = simple_returns
    sharpe = annualised_sharpe(returns)
    assert sharpe is not None
    assert sharpe > 0.0, (
        f"Expected positive Sharpe for trending fixture, got {sharpe:.4f}"
    )


def test_max_drawdown_fixture_in_unit_interval(simple_returns: list[float]) -> None:
    """Max drawdown on the fixture must be in [0, 1]."""
    mdd = max_drawdown(simple_returns)
    assert mdd is not None
    assert 0.0 <= mdd <= 1.0, f"MDD={mdd:.6f} out of [0, 1]"


def test_sharpe_fixture_stable_across_windows(simple_returns: list[float]) -> None:
    """
    Sharpe computed on two non-overlapping halves of the fixture should both
    be finite.  This verifies there are no degenerate sub-periods.
    """
    returns = simple_returns
    n = len(returns)
    if n < 10:
        pytest.skip("Too few returns for windowed test")

    half = n // 2
    sharpe_first = annualised_sharpe(returns[:half])
    sharpe_second = annualised_sharpe(returns[half:])

    # Either half may be None (zero variance) but both should be finite if set.
    if sharpe_first is not None:
        assert math.isfinite(sharpe_first), f"First-half Sharpe not finite: {sharpe_first}"
    if sharpe_second is not None:
        assert math.isfinite(sharpe_second), f"Second-half Sharpe not finite: {sharpe_second}"
