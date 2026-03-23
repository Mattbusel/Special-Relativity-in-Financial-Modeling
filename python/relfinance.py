"""
relfinance.py — High-level Python API for Special Relativity in Financial Modeling
====================================================================================

Provides a simple, pip-installable interface to the core SRFM calculations.
This module wraps the existing ``srfm`` package (pure-Python fallback) and
exposes a simplified dataclass-based API for the research community.

When the compiled ``srfm._srfm`` C++ extension is available it is used
transparently for performance.  The module degrades gracefully to pure Python
when running in documentation generation or CI environments without a build.

Public API
----------
SpacetimeEvent
    Dataclass representing a single market bar as a 4-vector (t, P, V, M).

classify_events(events)
    Classify a list of :class:`SpacetimeEvent` objects as TIMELIKE / LIGHTLIKE /
    SPACELIKE based on the Minkowski interval between consecutive events.

compute_lorentz_factor(beta)
    Compute the Lorentz factor γ = 1 / √(1 − β²).

portfolio_manifold(events)
    Compute the pairwise Minkowski spacetime-interval covariance matrix for a
    multi-asset portfolio, returned as a NumPy array.

relativistic_options_price(spot, strike, expiry, dt, dp, is_call, cfg)
    Price a European option using the relativistic Black-Scholes model.

lightcone_implied_vol(spot, strike, expiry, beta, is_call, cfg)
    Compute light-cone implied volatility split into TIMELIKE and SPACELIKE vols.

Quick Start
-----------
>>> from relfinance import SpacetimeEvent, classify_events, compute_lorentz_factor
>>> events = [SpacetimeEvent(t=i, P=100+i*0.5, V=1e6, M=1e9) for i in range(5)]
>>> labels = classify_events(events)
>>> print(labels)
['TIMELIKE', 'TIMELIKE', 'TIMELIKE', 'TIMELIKE']

>>> compute_lorentz_factor(0.8)
1.6666666666666667
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Try to import the compiled SRFM extension.  Fall back to pure Python.
# ---------------------------------------------------------------------------

try:
    from srfm import SpacetimeInterval as _SI, LorentzTransform as _LT

    def _compute_ds2(dt: float, dp: float, dv: float, dm: float) -> float:
        return _SI.compute_ds2(dt, dp, dv, dm)

    def _classify_ds2(ds2: float) -> str:
        return _SI.classify_ds2(ds2)

    def _gamma(beta: float) -> float:
        return _LT.gamma(beta)

    _NATIVE = True

except ImportError:
    _NATIVE = False

    # Pure-Python fallbacks.
    _C: float = 1.0
    _BETA_MAX: float = 0.9999
    _LIGHTLIKE_EPS: float = 1e-12

    def _compute_ds2(dt: float, dp: float, dv: float, dm: float) -> float:
        return -(_C * _C * dt * dt) + dp * dp + dv * dv + dm * dm

    def _classify_ds2(ds2: float) -> str:
        if abs(ds2) <= _LIGHTLIKE_EPS:
            return "LIGHTLIKE"
        return "TIMELIKE" if ds2 < 0.0 else "SPACELIKE"

    def _gamma(beta: float) -> float:
        b = min(abs(beta), _BETA_MAX)
        return 1.0 / math.sqrt(1.0 - b * b)


# ---------------------------------------------------------------------------
# SpacetimeEvent
# ---------------------------------------------------------------------------


@dataclass
class SpacetimeEvent:
    """
    A single market bar embedded as a 4-vector in SRFM spacetime.

    The four coordinates are:

    - ``t`` — Bar timestamp (any consistent unit, e.g. seconds or bar index).
    - ``P`` — Close price (or log-price; the API accepts either).
    - ``V`` — Volume (will be normalised as ``V^(1/4)`` internally when computing ds²).
    - ``M`` — Market-impact proxy (e.g. notional traded; normalised as ``M^(1/3)``).

    Optional metadata:
    - ``symbol`` — Ticker label.
    - ``meta``   — Arbitrary key-value dict for annotations.

    Examples
    --------
    >>> ev = SpacetimeEvent(t=0.0, P=100.0, V=1_000_000.0, M=5e9)
    >>> ev.symbol = "AAPL"
    """

    t: float
    """Coordinate time (bar timestamp)."""
    P: float
    """Price (or log-price)."""
    V: float
    """Volume."""
    M: float
    """Market-impact proxy."""
    symbol: str = ""
    """Optional ticker symbol."""
    meta: Dict[str, str] = field(default_factory=dict)
    """Arbitrary metadata."""

    def as_4vector(self) -> np.ndarray:
        """Return the event as a normalised 4-vector (c·t, P, V^(1/4), M^(1/3))."""
        v_norm = abs(self.V) ** 0.25 if self.V != 0 else 0.0
        m_norm = abs(self.M) ** (1.0 / 3.0) if self.M != 0 else 0.0
        return np.array([self.t, self.P, v_norm, m_norm], dtype=float)

    def interval_to(self, other: "SpacetimeEvent") -> float:
        """
        Compute ds² between ``self`` and ``other``.

        ds² = −c²·Δt² + ΔP² + ΔV⁴ + ΔM³   (using normalised coordinates)
        """
        a = self.as_4vector()
        b = other.as_4vector()
        diff = b - a
        return _compute_ds2(diff[0], diff[1], diff[2], diff[3])

    def classify_to(self, other: "SpacetimeEvent") -> str:
        """Return ``'TIMELIKE'``, ``'LIGHTLIKE'``, or ``'SPACELIKE'`` between events."""
        return _classify_ds2(self.interval_to(other))


# ---------------------------------------------------------------------------
# classify_events
# ---------------------------------------------------------------------------


def classify_events(events: List[SpacetimeEvent]) -> List[str]:
    """
    Classify the spacetime interval between consecutive events.

    For *n* input events, returns *n − 1* classification strings:
    ``'TIMELIKE'``, ``'LIGHTLIKE'``, or ``'SPACELIKE'``.

    Parameters
    ----------
    events:
        Ordered list of :class:`SpacetimeEvent` objects (must have ≥ 2 entries).

    Returns
    -------
    List[str]
        Length ``len(events) - 1``.  The *i*-th entry classifies the interval
        between ``events[i]`` and ``events[i+1]``.

    Raises
    ------
    ValueError
        If fewer than 2 events are provided.

    Examples
    --------
    >>> evs = [SpacetimeEvent(t=i, P=100+i*0.1, V=1e6, M=1e9) for i in range(4)]
    >>> classify_events(evs)
    ['TIMELIKE', 'TIMELIKE', 'TIMELIKE']
    """
    if len(events) < 2:
        raise ValueError("classify_events requires at least 2 events.")
    return [events[i].classify_to(events[i + 1]) for i in range(len(events) - 1)]


# ---------------------------------------------------------------------------
# compute_lorentz_factor
# ---------------------------------------------------------------------------


def compute_lorentz_factor(beta: float) -> float:
    """
    Compute the Lorentz factor γ = 1 / √(1 − β²).

    The input ``beta`` is clamped to the safe range ``[−0.9999, 0.9999]``
    to avoid numerical divergence at the light cone.

    Parameters
    ----------
    beta:
        Normalised price velocity β = ΔP / (c · Δt).  Must satisfy ``|β| ≤ 1``.

    Returns
    -------
    float
        Lorentz factor γ ≥ 1.

    Examples
    --------
    >>> compute_lorentz_factor(0.0)
    1.0
    >>> compute_lorentz_factor(0.5)
    1.1547005383792515
    >>> compute_lorentz_factor(0.8)
    1.6666666666666667
    """
    return _gamma(beta)


# ---------------------------------------------------------------------------
# portfolio_manifold
# ---------------------------------------------------------------------------


def portfolio_manifold(events: Dict[str, SpacetimeEvent]) -> np.ndarray:
    """
    Compute the pairwise Minkowski spacetime-interval covariance matrix.

    Each entry ``(i, j)`` is the Minkowski inner product between the
    4-vectors of assets *i* and *j*:

    .. math::

        C_{ij} = \\exp(-|ds^2(i, j)|)

    This produces a matrix with ones on the diagonal and off-diagonal values
    in ``(0, 1]``, analogous to a correlation matrix.  Pairs that are
    LIGHTLIKE (``|ds²| ≈ 0``) have the maximum covariance of 1.

    Parameters
    ----------
    events:
        Dictionary mapping asset label → :class:`SpacetimeEvent`.
        Must contain ≥ 2 assets.

    Returns
    -------
    np.ndarray
        Symmetric ``(n, n)`` covariance matrix.  Row/column ordering follows
        ``sorted(events.keys())``.

    Raises
    ------
    ValueError
        If fewer than 2 assets are provided.

    Examples
    --------
    >>> import numpy as np
    >>> from relfinance import SpacetimeEvent, portfolio_manifold
    >>> events = {
    ...     "AAPL": SpacetimeEvent(t=1.0, P=150.0, V=1e8, M=2e12),
    ...     "MSFT": SpacetimeEvent(t=1.0, P=290.0, V=8e7, M=2e12),
    ... }
    >>> C = portfolio_manifold(events)
    >>> C.shape
    (2, 2)
    >>> float(C[0, 0])
    1.0
    """
    if len(events) < 2:
        raise ValueError("portfolio_manifold requires at least 2 assets.")

    symbols = sorted(events.keys())
    n = len(symbols)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                ds2 = events[symbols[i]].interval_to(events[symbols[j]])
                matrix[i, j] = math.exp(-abs(ds2))

    return matrix


# ---------------------------------------------------------------------------
# Relativistic Options Pricing convenience wrappers
# ---------------------------------------------------------------------------


@dataclass
class OptionsConfig:
    """
    Configuration for relativistic options pricing.

    Attributes
    ----------
    c_scale:
        Financial speed of light (max log-return per unit time). Default 0.05.
    sigma_base:
        Baseline annualised volatility. Default 0.20.
    risk_free_rate:
        Annualised risk-free rate (continuously compounded). Default 0.05.
    gamma_cap:
        Maximum Lorentz factor applied to the delta hedge ratio. Default 5.0.
    """

    c_scale: float = 0.05
    sigma_base: float = 0.20
    risk_free_rate: float = 0.05
    gamma_cap: float = 5.0


@dataclass
class OptionResult:
    """Result of a relativistic option pricing computation."""

    price: float
    """Option fair value."""
    sigma_effective: float
    """Effective volatility derived from spacetime metric."""
    interval_class: str
    """TIMELIKE / LIGHTLIKE / SPACELIKE regime of the underlying's last move."""
    gamma_factor: float
    """Lorentz factor γ for this bar."""
    proper_time_to_expiry: float
    """Remaining proper time τ (< calendar time T for TIMELIKE regimes)."""
    is_call: bool
    """True for call, False for put."""


@dataclass
class DeltaResult:
    """Result of a relativistic delta computation."""

    delta_classical: float
    """Standard Black-Scholes N(d₁) or N(d₁)−1."""
    delta_relativistic: float
    """Relativistic delta = γ · Δ_BS."""
    gamma_factor: float
    """Lorentz factor γ applied."""
    beta: float
    """Normalised velocity β of the underlying."""
    interval_class: str
    """Regime classification."""


def _standard_normal_cdf(x: float) -> float:
    """Rational approximation to the standard normal CDF (max error < 7.5e-8)."""
    from math import sqrt, exp, pi

    neg = x < 0.0
    z = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (
        0.319381530
        + t * (
            -0.356563782
            + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))
        )
    )
    phi = exp(-0.5 * z * z) / sqrt(2.0 * pi) * poly
    return phi if neg else 1.0 - phi


def relativistic_options_price(
    spot: float,
    strike: float,
    expiry: float,
    dt: float,
    dp: float,
    is_call: bool = True,
    cfg: Optional[OptionsConfig] = None,
) -> OptionResult:
    """
    Price a European option using the relativistic Black-Scholes model.

    The effective volatility is derived from the Minkowski spacetime interval
    between the current and previous price event.  TIMELIKE trajectories
    (causal, sub-luminal) receive a reduced implied vol; SPACELIKE trajectories
    (acausal, super-luminal) receive an enhanced implied vol.

    Parameters
    ----------
    spot:
        Current underlying price S.
    strike:
        Option strike price K.
    expiry:
        Time to expiry in years T.
    dt:
        Coordinate-time step for the last price move (ticks or seconds).
    dp:
        Price change for the last move ΔP.
    is_call:
        ``True`` for call, ``False`` for put.
    cfg:
        :class:`OptionsConfig` instance.  Defaults to ``OptionsConfig()``.

    Returns
    -------
    OptionResult

    Raises
    ------
    ValueError
        If ``spot ≤ 0``, ``strike ≤ 0``, or ``expiry ≤ 0``.

    Examples
    --------
    >>> result = relativistic_options_price(100.0, 100.0, 1.0, 1.0, 0.5)
    >>> result.price > 0
    True
    >>> result.interval_class in ("TIMELIKE", "LIGHTLIKE", "SPACELIKE")
    True
    """
    if spot <= 0:
        raise ValueError("spot must be > 0")
    if strike <= 0:
        raise ValueError("strike must be > 0")
    if expiry <= 0:
        raise ValueError("expiry must be > 0")

    cfg = cfg or OptionsConfig()

    # Normalised velocity.
    if abs(dt) < 1e-12 or spot <= 0:
        beta_raw = 0.0
    else:
        beta_raw = (dp / spot) / (cfg.c_scale * dt)
    beta = max(-0.9999, min(0.9999, beta_raw))
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)

    # ds² classification.
    ds2 = -(cfg.c_scale * dt) ** 2 + dp ** 2
    cls = _classify_ds2(ds2)

    # Effective volatility.
    if cls == "TIMELIKE":
        sigma_eff = max(1e-4, min(10.0, cfg.sigma_base * math.sqrt(1.0 - beta * beta)))
    elif cls == "LIGHTLIKE":
        sigma_eff = max(1e-4, min(10.0, cfg.sigma_base))
    else:  # SPACELIKE
        beta_excess = max(0.0, abs(beta) - 1.0)
        enhancement = max(1.0, min(10.0 / cfg.sigma_base, 1.0 + math.sqrt(beta_excess)))
        sigma_eff = max(1e-4, min(10.0, cfg.sigma_base * enhancement))

    # Proper-time adjusted expiry.
    proper_t = max(1e-6, expiry * math.sqrt(1.0 - beta * beta))

    # Black-Scholes formula.
    r = cfg.risk_free_rate
    log_ratio = math.log(spot / strike)
    d1 = (log_ratio + (r + 0.5 * sigma_eff * sigma_eff) * proper_t) / (
        sigma_eff * math.sqrt(proper_t)
    )
    d2 = d1 - sigma_eff * math.sqrt(proper_t)

    if is_call:
        price = (
            spot * _standard_normal_cdf(d1)
            - strike * math.exp(-r * proper_t) * _standard_normal_cdf(d2)
        )
    else:
        price = (
            strike * math.exp(-r * proper_t) * _standard_normal_cdf(-d2)
            - spot * _standard_normal_cdf(-d1)
        )

    return OptionResult(
        price=max(0.0, price),
        sigma_effective=sigma_eff,
        interval_class=cls,
        gamma_factor=gamma,
        proper_time_to_expiry=proper_t,
        is_call=is_call,
    )


def lightcone_implied_vol(
    spot: float,
    strike: float,
    expiry: float,
    beta: float,
    is_call: bool = True,
    cfg: Optional[OptionsConfig] = None,
) -> Dict[str, float]:
    """
    Compute TIMELIKE and SPACELIKE implied vols under the light-cone model.

    Returns a dictionary with keys:

    - ``sigma_timelike``  — Vol assigned to TIMELIKE (inside light cone) trajectories.
    - ``sigma_spacelike`` — Vol assigned to SPACELIKE (outside light cone) trajectories.
    - ``sigma_applied``   — Actual vol applied based on current regime (from ``beta``).
    - ``interval_class``  — Regime string: ``'TIMELIKE'``, ``'LIGHTLIKE'``, ``'SPACELIKE'``.
    - ``price``           — Option price under the applied vol.
    - ``vol_reduction_ratio`` — ``sigma_timelike / sigma_base``.

    Parameters
    ----------
    spot:
        Current price.
    strike:
        Strike price.
    expiry:
        Time to expiry (years).
    beta:
        Observed normalised velocity β ∈ (−1, 1) for TIMELIKE; ``|β| ≥ 1`` for SPACELIKE.
    is_call:
        Option type.
    cfg:
        :class:`OptionsConfig`. Defaults to ``OptionsConfig()``.

    Examples
    --------
    >>> vols = lightcone_implied_vol(100.0, 100.0, 1.0, beta=0.1)
    >>> vols["sigma_timelike"] < vols["sigma_spacelike"]
    True
    """
    if spot <= 0:
        raise ValueError("spot must be > 0")
    if strike <= 0:
        raise ValueError("strike must be > 0")
    if expiry <= 0:
        raise ValueError("expiry must be > 0")

    cfg = cfg or OptionsConfig()
    beta_safe = max(-0.9999, min(0.9999, beta))
    beta2 = beta_safe * beta_safe
    sigma_base = cfg.sigma_base

    sigma_tl = max(1e-4, min(10.0, sigma_base * math.sqrt(1.0 - beta2)))
    sigma_sl = max(1e-4, min(10.0, sigma_base / math.sqrt(1.0 - beta2)))

    # TIMELIKE ↔ |β| < 1 (ds2_proxy = beta2 - 1 < 0).
    ds2_proxy = -(1.0 - beta2)  # negative → TIMELIKE
    cls = _classify_ds2(ds2_proxy)

    sigma_applied = {"TIMELIKE": sigma_tl, "LIGHTLIKE": sigma_base, "SPACELIKE": sigma_sl}.get(
        cls, sigma_base
    )

    proper_t = max(1e-6, expiry * math.sqrt(1.0 - beta2))
    r = cfg.risk_free_rate
    log_ratio = math.log(spot / strike)
    d1 = (log_ratio + (r + 0.5 * sigma_applied * sigma_applied) * proper_t) / (
        sigma_applied * math.sqrt(proper_t)
    )
    d2 = d1 - sigma_applied * math.sqrt(proper_t)
    if is_call:
        price = (
            spot * _standard_normal_cdf(d1)
            - strike * math.exp(-r * proper_t) * _standard_normal_cdf(d2)
        )
    else:
        price = (
            strike * math.exp(-r * proper_t) * _standard_normal_cdf(-d2)
            - spot * _standard_normal_cdf(-d1)
        )

    return {
        "sigma_timelike": sigma_tl,
        "sigma_spacelike": sigma_sl,
        "sigma_applied": sigma_applied,
        "interval_class": cls,
        "price": max(0.0, price),
        "vol_reduction_ratio": sigma_tl / sigma_base if sigma_base > 1e-12 else 1.0,
    }


def compute_spacetime_delta(
    spot: float,
    strike: float,
    expiry: float,
    sigma: float,
    dt: float,
    dp: float,
    is_call: bool = True,
    cfg: Optional[OptionsConfig] = None,
) -> DeltaResult:
    """
    Compute the relativistic delta hedge ratio Δ_rel = γ(β) · Δ_BS.

    Parameters
    ----------
    spot:
        Current underlying price.
    strike:
        Strike price.
    expiry:
        Time to expiry (years).
    sigma:
        Volatility estimate (annualised).
    dt:
        Coordinate-time step for the last bar.
    dp:
        Price change for the last bar ΔP.
    is_call:
        Option type.
    cfg:
        :class:`OptionsConfig`.

    Returns
    -------
    DeltaResult

    Examples
    --------
    >>> dr = compute_spacetime_delta(100., 100., 1., 0.20, 1., 0.5)
    >>> dr.delta_relativistic >= dr.delta_classical
    True
    """
    cfg = cfg or OptionsConfig()

    if abs(dt) < 1e-12 or spot <= 0:
        beta_raw = 0.0
    else:
        beta_raw = (dp / spot) / (cfg.c_scale * dt)
    beta = max(-0.9999, min(0.9999, beta_raw))
    gamma = min(cfg.gamma_cap, 1.0 / math.sqrt(1.0 - beta * beta))

    ds2 = -(cfg.c_scale * dt) ** 2 + dp ** 2
    cls = _classify_ds2(ds2)

    sigma_safe = max(1e-4, min(10.0, sigma))
    r = cfg.risk_free_rate
    log_ratio = math.log(max(spot, 1e-12) / max(strike, 1e-12))
    d1 = (log_ratio + (r + 0.5 * sigma_safe * sigma_safe) * expiry) / (
        sigma_safe * math.sqrt(max(expiry, 1e-12))
    )
    delta_bs = _standard_normal_cdf(d1) if is_call else _standard_normal_cdf(d1) - 1.0
    delta_rel = max(-cfg.gamma_cap, min(cfg.gamma_cap, gamma * delta_bs))

    return DeltaResult(
        delta_classical=delta_bs,
        delta_relativistic=delta_rel,
        gamma_factor=gamma,
        beta=beta,
        interval_class=cls,
    )


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------

__version__: str = "2.0.0"
__author__: str = "Matthew Busel"

__all__ = [
    # Core
    "SpacetimeEvent",
    "classify_events",
    "compute_lorentz_factor",
    "portfolio_manifold",
    # Options
    "OptionsConfig",
    "OptionResult",
    "DeltaResult",
    "relativistic_options_price",
    "lightcone_implied_vol",
    "compute_spacetime_delta",
    # Meta
    "__version__",
    "__author__",
]
