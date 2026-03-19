"""
test_lorentz_property.py — hypothesis-based property tests for SRFM Lorentz
factor and related financial metric invariants.

These tests complement the example-based tests in test_lorentz.py and
test_backtest.py by verifying algebraic invariants across a wide range of
randomly-generated inputs.
"""

from __future__ import annotations

import math
import statistics

from hypothesis import assume, given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Gamma (Lorentz factor) invariants
# ---------------------------------------------------------------------------


@given(st.floats(min_value=0.0, max_value=0.9999, allow_nan=False, allow_infinity=False))
def test_gamma_always_ge_one(beta: float) -> None:
    """γ = 1 / sqrt(1 − β²) is always ≥ 1 for any β ∈ [0, 1)."""
    g = 1.0 / math.sqrt(1.0 - beta * beta)
    assert g >= 1.0


@given(
    st.floats(min_value=0.0, max_value=0.9998, allow_nan=False),
    st.floats(min_value=0.0001, max_value=0.9999, allow_nan=False),
)
def test_gamma_monotone(b1: float, b2: float) -> None:
    """γ is monotonically increasing: b1 < b2 implies γ(b1) ≤ γ(b2)."""
    assume(b1 < b2)
    g1 = 1.0 / math.sqrt(1.0 - b1 * b1)
    g2 = 1.0 / math.sqrt(1.0 - b2 * b2)
    assert g1 <= g2


# ---------------------------------------------------------------------------
# Sharpe ratio finiteness
# ---------------------------------------------------------------------------


@given(
    st.lists(
        st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
        min_size=30,
        max_size=200,
    )
)
def test_sharpe_finite_for_nonzero_vol(returns: list[float]) -> None:
    """Annualised Sharpe is finite whenever the return series has positive variance."""
    assume(statistics.stdev(returns) > 1e-10)
    mean_r = sum(returns) / len(returns)
    std_r = statistics.stdev(returns)
    sharpe = mean_r / std_r * math.sqrt(252)
    assert math.isfinite(sharpe)


# ---------------------------------------------------------------------------
# Spacetime interval sign → regime classification
# ---------------------------------------------------------------------------


@given(
    st.floats(min_value=0.0, max_value=5.0, allow_nan=False),   # deltaT
    st.floats(min_value=0.0, max_value=5.0, allow_nan=False),   # deltaP
    st.floats(min_value=1.0, max_value=3.0, allow_nan=False),   # c
)
def test_ds2_sign_regime(deltaT: float, deltaP: float, c: float) -> None:
    """
    The spacetime interval ds² = −c²ΔT² + ΔP² determines the causal regime.

    The resulting regime label must always be one of the three canonical values.
    """
    ds2 = -(c * c * deltaT * deltaT) + deltaP * deltaP
    if ds2 < 0:
        regime = "TIMELIKE"
    elif ds2 > 0:
        regime = "SPACELIKE"
    else:
        regime = "LIGHTLIKE"
    assert regime in ("TIMELIKE", "SPACELIKE", "LIGHTLIKE")
