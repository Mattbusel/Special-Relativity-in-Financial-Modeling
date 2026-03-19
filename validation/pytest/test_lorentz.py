"""
test_lorentz.py — validates gamma/beta Lorentz factor calculations.

The SRFM system maps normalised market velocity β ∈ (-1, 1) to the Lorentz
factor γ = 1 / sqrt(1 - β²).  These tests verify that the Python-side
reference implementation of those formulas is correct to 6 decimal places
and that derived quantities (e.g. relativistic momentum) behave as expected.
"""

from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Reference implementations (pure Python, formula-level)
# ---------------------------------------------------------------------------


def lorentz_gamma(beta: float) -> float:
    """Compute γ = 1 / sqrt(1 − β²).  Raises ValueError for |β| >= 1."""
    if abs(beta) >= 1.0:
        raise ValueError(f"beta={beta} is outside the valid range (-1, 1)")
    return 1.0 / math.sqrt(1.0 - beta * beta)


def relativistic_momentum(beta: float, mass: float = 1.0) -> float:
    """Compute p_rel = γ · m · v  (market-space analogue)."""
    return lorentz_gamma(beta) * mass * beta


# ---------------------------------------------------------------------------
# Known-value table
# Each entry: (beta, expected_gamma, expected_gamma_minus_one_approx)
# gamma values taken from SR textbooks / Wikipedia to 8 significant figures.
# ---------------------------------------------------------------------------

KNOWN_VALUES: list[tuple[float, float]] = [
    (0.0,   1.000000),
    (0.1,   1.005038),
    (0.2,   1.020621),
    (0.3,   1.048285),
    (0.4,   1.091089),
    (0.5,   1.154701),
    (0.6,   1.250000),
    (0.7,   1.400280),
    (0.8,   1.666667),
    (0.9,   2.294157),
    (0.95,  3.202563),
    (0.99,  7.088812),
    (-0.5,  1.154701),
    (-0.8,  1.666667),
]


@pytest.mark.parametrize("beta, expected_gamma", KNOWN_VALUES)
def test_lorentz_gamma_known_values(beta: float, expected_gamma: float) -> None:
    """gamma(beta) matches reference values to 6 decimal places."""
    result = lorentz_gamma(beta)
    assert abs(result - expected_gamma) < 5e-7, (
        f"gamma({beta}): got {result:.8f}, expected {expected_gamma:.8f}, "
        f"delta={abs(result - expected_gamma):.2e}"
    )


def test_lorentz_gamma_zero_beta_is_one() -> None:
    """At zero velocity, γ = 1 exactly (Newtonian limit)."""
    assert lorentz_gamma(0.0) == pytest.approx(1.0, abs=1e-15)


def test_lorentz_gamma_symmetric_in_beta() -> None:
    """γ depends only on |β|; sign of β is irrelevant."""
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert lorentz_gamma(beta) == pytest.approx(lorentz_gamma(-beta), rel=1e-12)


def test_lorentz_gamma_monotone_increasing() -> None:
    """γ must be strictly increasing as |β| increases from 0 toward 1."""
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    gammas = [lorentz_gamma(b) for b in betas]
    for i in range(len(gammas) - 1):
        assert gammas[i] < gammas[i + 1], (
            f"Monotonicity violated at beta={betas[i]}: "
            f"gamma={gammas[i]:.6f} >= gamma={gammas[i+1]:.6f}"
        )


def test_lorentz_gamma_always_at_least_one() -> None:
    """γ ≥ 1 for all valid β."""
    for beta in [-0.99, -0.5, -0.1, 0.0, 0.1, 0.5, 0.99]:
        assert lorentz_gamma(beta) >= 1.0 - 1e-15, (
            f"gamma({beta}) = {lorentz_gamma(beta):.8f} < 1"
        )


def test_lorentz_gamma_invalid_beta_raises() -> None:
    """beta = ±1 (speed of light) is not allowed."""
    with pytest.raises(ValueError):
        lorentz_gamma(1.0)
    with pytest.raises(ValueError):
        lorentz_gamma(-1.0)
    with pytest.raises(ValueError):
        lorentz_gamma(1.5)


def test_lorentz_gamma_identity_formula() -> None:
    """
    Verify the identity: γ² (1 − β²) = 1.

    This is a direct algebraic consequence of the definition.
    """
    for beta in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        gamma = lorentz_gamma(beta)
        lhs = gamma * gamma * (1.0 - beta * beta)
        assert abs(lhs - 1.0) < 1e-12, (
            f"Identity γ²(1−β²)=1 failed at beta={beta}: lhs={lhs:.15f}"
        )


# ---------------------------------------------------------------------------
# Relativistic momentum tests
# ---------------------------------------------------------------------------


def test_relativistic_momentum_zero_beta() -> None:
    """At β = 0, relativistic momentum is 0 (no motion)."""
    assert relativistic_momentum(0.0) == pytest.approx(0.0, abs=1e-15)


def test_relativistic_momentum_positive_beta() -> None:
    """Momentum is positive for positive β."""
    assert relativistic_momentum(0.5) > 0.0
    assert relativistic_momentum(0.9) > 0.0


def test_relativistic_momentum_antisymmetric() -> None:
    """Momentum is an odd function of β: p(-β) = -p(β)."""
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert relativistic_momentum(beta) == pytest.approx(
            -relativistic_momentum(-beta), rel=1e-12
        )


def test_relativistic_momentum_mass_scaling() -> None:
    """Doubling mass doubles relativistic momentum."""
    beta = 0.6
    p1 = relativistic_momentum(beta, mass=1.0)
    p2 = relativistic_momentum(beta, mass=2.0)
    assert p2 == pytest.approx(2.0 * p1, rel=1e-12)


def test_relativistic_momentum_exceeds_classical_for_high_beta() -> None:
    """
    Relativistic momentum (γmv) exceeds classical momentum (mv) for β > 0.
    """
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        p_rel = relativistic_momentum(beta, mass=1.0)
        p_classical = beta  # m=1, so p_classical = m*v = beta
        assert p_rel >= p_classical - 1e-12, (
            f"Relativistic momentum {p_rel:.6f} < classical {p_classical:.6f} "
            f"at beta={beta}"
        )


# ---------------------------------------------------------------------------
# Integration with fixture data
# ---------------------------------------------------------------------------


def test_beta_from_fixture_returns_valid_gamma(simple_returns: list[float]) -> None:
    """
    Compute β from close-to-close returns and verify γ is finite and ≥ 1.

    Beta is normalised by the maximum absolute return in the series, mimicking
    the BetaCalculator in the C++ engine.
    """
    returns = simple_returns
    assert len(returns) >= 10, "Need at least 10 returns"

    max_abs = max(abs(r) for r in returns)
    if max_abs == 0.0:
        pytest.skip("All returns are zero; cannot normalise beta")

    for ret in returns:
        beta = ret / max_abs  # normalised to (-1, 1)
        # Clamp to safe range (mirrors BETA_MAX_SAFE = 0.9999)
        beta = max(-0.9999, min(0.9999, beta))
        gamma = lorentz_gamma(beta)
        assert math.isfinite(gamma), f"gamma is not finite for beta={beta}"
        assert gamma >= 1.0 - 1e-12, f"gamma={gamma:.8f} < 1 for beta={beta}"
