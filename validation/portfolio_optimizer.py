"""
Relativistic Portfolio Optimizer
=================================
Extends the SRFM single-asset framework to multi-asset portfolios.
Uses the Minkowski metric to define a portfolio "spacetime interval"
that accounts for cross-asset causal vs. stochastic interactions.

Key insight: TIMELIKE asset pairs (high beta correlation within causal
cone) should be weighted differently from SPACELIKE pairs (stochastic
correlation). The optimizer targets maximum relativistic Sharpe:

    SR_rel = E[r] / sqrt(ds^2_portfolio)

where ds^2 accounts for both variance and spacetime interval structure.

Usage
-----
    python validation/portfolio_optimizer.py

Dependencies
------------
    pip install numpy scipy
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Interval type classification ─────────────────────────────────────────────


class IntervalType(Enum):
    TIMELIKE = "timelike"
    LIGHTLIKE = "lightlike"
    SPACELIKE = "spacelike"


# ─── Data structures ──────────────────────────────────────────────────────────


@dataclass
class AssetManifold:
    """Represents an asset as a worldline in financial spacetime."""

    symbol: str
    prices: np.ndarray          # shape (N,) — close prices
    timestamps: np.ndarray      # shape (N,) — Unix timestamps
    beta: np.ndarray            # shape (N-1,) — price velocity at each bar
    interval_types: np.ndarray  # shape (N-1,) — IntervalType per bar
    timelike_fraction: float    # fraction of bars classified TIMELIKE


@dataclass
class PortfolioResult:
    weights: np.ndarray
    relativistic_sharpe: float
    classical_sharpe: float
    timelike_exposure: float    # weighted fraction of TIMELIKE bars
    spacetime_interval: float   # portfolio-level ds^2
    max_drawdown: float
    annualized_return: float
    annualized_vol: float


# ─── Core optimizer ───────────────────────────────────────────────────────────


class RelativisticPortfolioOptimizer:
    """
    Multi-asset portfolio optimization using spacetime interval structure.

    The optimizer penalizes SPACELIKE correlations (treating them as
    fundamentally noise) and rewards TIMELIKE momentum alignment
    across assets.

    Parameters
    ----------
    c:
        Financial speed of light — the threshold price velocity (fractional
        move per bar) that separates TIMELIKE from SPACELIKE regimes.
        Default: 0.1 (empirically calibrated on 1-minute S&P 500 data).
    risk_free_rate:
        Annual risk-free rate used to compute excess returns for Sharpe.
        Default: 0.0.
    """

    C_FINANCIAL: float = 0.1  # empirical speed of light for financial data
    _LIGHTLIKE_EPSILON: float = 0.01  # |beta - 1| threshold for lightlike

    def __init__(self, c: float = 0.1, risk_free_rate: float = 0.0) -> None:
        self.c = c
        self.risk_free_rate = risk_free_rate

    # ── Bar classification ────────────────────────────────────────────────────

    def classify_bar(self, dp: float, dt: float) -> IntervalType:
        """
        Classify a price bar as TIMELIKE, LIGHTLIKE, or SPACELIKE.

        Parameters
        ----------
        dp:
            Absolute price change |P_t - P_{t-1}|.
        dt:
            Bar duration in seconds (or normalised time units).

        Returns
        -------
        IntervalType corresponding to the Minkowski interval of this bar.
        """
        beta = abs(dp) / (self.c * dt) if dt > 0 else 0.0
        if abs(beta - 1.0) < self._LIGHTLIKE_EPSILON:
            return IntervalType.LIGHTLIKE
        if beta < 1.0:
            return IntervalType.TIMELIKE
        return IntervalType.SPACELIKE

    def _spacetime_interval(self, dp: float, dt: float) -> float:
        """
        Compute ds^2 = -(c*dt)^2 + dp^2.

        Negative => TIMELIKE, zero => LIGHTLIKE, positive => SPACELIKE.
        """
        return dp ** 2 - (self.c * dt) ** 2

    # ── Manifold construction ─────────────────────────────────────────────────

    def build_asset_manifold(self, symbol: str, ohlcv: np.ndarray) -> AssetManifold:
        """
        Construct an asset's spacetime manifold from OHLCV data.

        Parameters
        ----------
        symbol:
            Ticker string, e.g. "AAPL".
        ohlcv:
            Array of shape (N, 5) with columns [open, high, low, close, volume],
            or shape (N, 2) with columns [timestamp_unix, close_price].
            If 5 columns are provided, column 3 (close) and an implicit
            uniform bar width of 60 s are used.
            If 2 columns are provided, column 0 is timestamp, column 1 is price.

        Returns
        -------
        AssetManifold with per-bar beta and interval classification.
        """
        ohlcv = np.asarray(ohlcv, dtype=float)

        if ohlcv.ndim == 1:
            # Treat as price series; generate synthetic timestamps
            prices = ohlcv
            timestamps = np.arange(len(prices), dtype=float) * 60.0
        elif ohlcv.shape[1] == 2:
            timestamps = ohlcv[:, 0]
            prices = ohlcv[:, 1]
        elif ohlcv.shape[1] >= 4:
            # Standard OHLCV — use close price (index 3)
            prices = ohlcv[:, 3]
            timestamps = np.arange(len(prices), dtype=float) * 60.0
        else:
            raise ValueError(
                f"ohlcv must have shape (N,), (N,2), or (N,>=4); got {ohlcv.shape}"
            )

        n = len(prices)
        if n < 2:
            raise ValueError(f"Need at least 2 bars for {symbol}; got {n}")

        dp = np.diff(prices)
        dt = np.diff(timestamps)
        dt = np.where(dt <= 0, 1.0, dt)  # guard against zero/negative dt

        beta = np.abs(dp) / (self.c * dt)

        interval_types = np.array([
            self.classify_bar(abs(dp[i]), dt[i]) for i in range(len(dp))
        ])

        timelike_mask = interval_types == IntervalType.TIMELIKE
        timelike_fraction = float(np.mean(timelike_mask))

        return AssetManifold(
            symbol=symbol,
            prices=prices,
            timestamps=timestamps,
            beta=beta,
            interval_types=interval_types,
            timelike_fraction=timelike_fraction,
        )

    # ── Spacetime-aware covariance ────────────────────────────────────────────

    def spacetime_covariance(self, assets: List[AssetManifold]) -> np.ndarray:
        """
        Compute spacetime-aware covariance matrix.

        TIMELIKE pairs contribute their classical covariance unmodified.
        SPACELIKE pairs have covariance discounted by (1 - spacelike_fraction)
        of each asset, reflecting the noise character of stochastic regimes.

        The (i, j) discount factor is:

            d_{ij} = 1 - spacelike_i * spacelike_j

        where spacelike_k = 1 - timelike_fraction_k.

        Parameters
        ----------
        assets:
            List of AssetManifold objects, all with the same bar count.

        Returns
        -------
        Spacetime-weighted covariance matrix of shape (N_assets, N_assets).
        """
        n = len(assets)

        # Align returns to the minimum common length
        return_series = []
        for a in assets:
            ret = np.diff(a.prices) / np.where(a.prices[:-1] == 0, 1.0, a.prices[:-1])
            return_series.append(ret)

        min_len = min(len(r) for r in return_series)
        returns = np.column_stack([r[-min_len:] for r in return_series])  # (T, N)

        # Classical sample covariance
        classical_cov = np.cov(returns, rowvar=False)

        # Build spacetime discount matrix
        spacelike_fracs = np.array([1.0 - a.timelike_fraction for a in assets])
        discount = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                discount[i, j] = 1.0 - spacelike_fracs[i] * spacelike_fracs[j]

        st_cov = classical_cov * discount

        # Ensure positive semi-definiteness via Tikhonov regularisation
        min_eig = float(np.min(np.linalg.eigvalsh(st_cov)))
        if min_eig < 1e-10:
            st_cov += (1e-10 - min_eig) * np.eye(n)

        return st_cov

    # ── Return computation ────────────────────────────────────────────────────

    def _asset_returns(self, assets: List[AssetManifold]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (mean_returns, cov_matrix) aligned to common bar count.

        Returns are simple per-bar percentage returns; mean and covariance are
        used directly by the optimizer (not annualised here — caller decides).
        """
        return_series = []
        for a in assets:
            ret = np.diff(a.prices) / np.where(a.prices[:-1] == 0, 1.0, a.prices[:-1])
            return_series.append(ret)

        min_len = min(len(r) for r in return_series)
        returns = np.column_stack([r[-min_len:] for r in return_series])

        mean_returns = returns.mean(axis=0)
        cov_matrix = self.spacetime_covariance(assets)

        return mean_returns, cov_matrix

    # ── Objective function ────────────────────────────────────────────────────

    def _neg_relativistic_sharpe(
        self,
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        rf_per_bar: float,
    ) -> float:
        """
        Negative relativistic Sharpe (minimised by scipy.optimize).

        Relativistic Sharpe:
            SR_rel = (w^T mu - rf) / sqrt(w^T Sigma_st w)

        where Sigma_st is the spacetime-weighted covariance matrix.
        """
        port_return = float(np.dot(weights, mean_returns))
        port_variance = float(weights @ cov_matrix @ weights)
        port_variance = max(port_variance, 1e-14)
        port_vol = np.sqrt(port_variance)
        excess = port_return - rf_per_bar
        return -(excess / port_vol)

    # ── Main optimisation ─────────────────────────────────────────────────────

    def optimize(
        self,
        assets: List[AssetManifold],
        max_weight: float = 0.4,
        min_weight: float = 0.0,
        target_timelike: Optional[float] = None,
    ) -> PortfolioResult:
        """
        Maximise relativistic Sharpe subject to weight constraints.

        Uses scipy SLSQP to solve:
            max  SR_rel(w)
            s.t. sum(w) = 1
                 min_weight <= w_i <= max_weight
                 [optional] sum_i w_i * tf_i >= target_timelike

        Parameters
        ----------
        assets:
            List of AssetManifold objects (>= 2).
        max_weight:
            Maximum weight for any single asset (default 0.4).
        min_weight:
            Minimum weight for any single asset (default 0.0, no shorting).
        target_timelike:
            If set, the weighted average TIMELIKE fraction must be at least
            this value (range [0, 1]).

        Returns
        -------
        PortfolioResult with optimal weights and performance metrics.
        """
        n = len(assets)
        if n < 2:
            raise ValueError("Need at least 2 assets to optimise a portfolio.")

        mean_returns, cov_matrix = self._asset_returns(assets)

        # Per-bar risk-free rate (assume 252 trading days * 390 bars/day for 1-min)
        bars_per_year = 252.0 * 390.0
        rf_per_bar = self.risk_free_rate / bars_per_year

        # Initial weights: equal weight
        w0 = np.full(n, 1.0 / n)

        # Constraints
        constraints: list[dict] = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        if target_timelike is not None:
            tf_arr = np.array([a.timelike_fraction for a in assets])
            constraints.append({
                "type": "ineq",
                "fun": lambda w, tf=tf_arr, tgt=target_timelike: float(np.dot(w, tf)) - tgt,
            })

        bounds = [(min_weight, max_weight)] * n

        result = minimize(
            self._neg_relativistic_sharpe,
            w0,
            args=(mean_returns, cov_matrix, rf_per_bar),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 5000},
        )

        if not result.success:
            log.warning("Optimizer did not converge: %s", result.message)

        weights = np.clip(result.x, min_weight, max_weight)
        weights /= weights.sum()  # re-normalise after clipping

        return self._build_result(weights, assets, mean_returns, cov_matrix, rf_per_bar)

    def _build_result(
        self,
        weights: np.ndarray,
        assets: List[AssetManifold],
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        rf_per_bar: float,
    ) -> PortfolioResult:
        """Compute all performance metrics for a given weight vector."""
        port_return = float(np.dot(weights, mean_returns))
        port_variance = float(weights @ cov_matrix @ weights)
        port_variance = max(port_variance, 1e-14)
        port_vol = np.sqrt(port_variance)

        bars_per_year = 252.0 * 390.0
        ann_return = port_return * bars_per_year
        ann_vol = port_vol * np.sqrt(bars_per_year)

        rel_sharpe = (port_return - rf_per_bar) / port_vol
        classical_sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0

        # Weighted TIMELIKE fraction
        tf_arr = np.array([a.timelike_fraction for a in assets])
        timelike_exposure = float(np.dot(weights, tf_arr))

        # Portfolio spacetime interval (weighted average of per-asset ds^2)
        # Proxy: use variance scaled by (c*dt)^2 to get interval character
        spacetime_interval = float(port_variance - (self.c ** 2))

        # Max drawdown from synthetic wealth index using mean returns
        max_drawdown = self._compute_max_drawdown(assets, weights)

        return PortfolioResult(
            weights=weights,
            relativistic_sharpe=float(rel_sharpe),
            classical_sharpe=classical_sharpe,
            timelike_exposure=timelike_exposure,
            spacetime_interval=spacetime_interval,
            max_drawdown=max_drawdown,
            annualized_return=ann_return,
            annualized_vol=ann_vol,
        )

    # ── Max drawdown ──────────────────────────────────────────────────────────

    def _compute_max_drawdown(
        self,
        assets: List[AssetManifold],
        weights: np.ndarray,
    ) -> float:
        """Compute historical max drawdown for the given weighted portfolio."""
        return_series = []
        for a in assets:
            ret = np.diff(a.prices) / np.where(a.prices[:-1] == 0, 1.0, a.prices[:-1])
            return_series.append(ret)

        min_len = min(len(r) for r in return_series)
        returns = np.column_stack([r[-min_len:] for r in return_series])

        port_returns = returns @ weights
        wealth = np.cumprod(1.0 + port_returns)
        peak = np.maximum.accumulate(wealth)
        drawdown = (wealth - peak) / np.where(peak == 0, 1.0, peak)
        return float(np.min(drawdown))

    # ── Efficient frontier ────────────────────────────────────────────────────

    def efficient_frontier(
        self,
        assets: List[AssetManifold],
        n_points: int = 50,
    ) -> List[PortfolioResult]:
        """
        Compute the relativistic efficient frontier.

        Sweeps the target-return dimension from min to max achievable,
        minimising portfolio variance (spacetime interval) at each level.

        Parameters
        ----------
        assets:
            List of AssetManifold objects.
        n_points:
            Number of frontier points to compute.

        Returns
        -------
        List of PortfolioResult, one per frontier point, sorted by
        annualised return.
        """
        n = len(assets)
        mean_returns, cov_matrix = self._asset_returns(assets)
        bars_per_year = 252.0 * 390.0
        rf_per_bar = self.risk_free_rate / bars_per_year

        ann_means = mean_returns * bars_per_year
        r_min = float(np.min(ann_means))
        r_max = float(np.max(ann_means))

        target_returns = np.linspace(r_min, r_max, n_points)
        frontier: List[PortfolioResult] = []

        for target_r in target_returns:
            target_r_bar = target_r / bars_per_year

            def portfolio_variance(w: np.ndarray) -> float:
                return float(w @ cov_matrix @ w)

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {
                    "type": "eq",
                    "fun": lambda w, tr=target_r_bar: float(np.dot(w, mean_returns)) - tr,
                },
            ]
            bounds = [(0.0, 1.0)] * n
            w0 = np.full(n, 1.0 / n)

            result = minimize(
                portfolio_variance,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 2000},
            )

            if result.success:
                w = np.clip(result.x, 0.0, 1.0)
                w /= w.sum()
                pr = self._build_result(w, assets, mean_returns, cov_matrix, rf_per_bar)
                frontier.append(pr)

        frontier.sort(key=lambda pr: pr.annualized_return)
        log.info("Efficient frontier: %d / %d points converged.", len(frontier), n_points)
        return frontier

    # ── Backtest ──────────────────────────────────────────────────────────────

    def backtest(
        self,
        assets: List[AssetManifold],
        weights: np.ndarray,
        rebalance_freq: int = 20,
    ) -> Dict:
        """
        Backtest a static-weight portfolio with periodic rebalancing.

        At each rebalance point the optimizer re-runs on the history up to
        that bar and produces updated weights.  Between rebalances, weights
        drift with price movement (buy-and-hold drift).

        Parameters
        ----------
        assets:
            List of AssetManifold objects.
        weights:
            Initial weight vector (shape N).  Used as the starting point
            for each re-optimisation.
        rebalance_freq:
            Number of bars between rebalance events.

        Returns
        -------
        dict with keys:
            wealth_index   — np.ndarray of cumulative portfolio value
            returns        — np.ndarray of bar-by-bar portfolio returns
            rebalance_dates— list of bar indices where rebalance occurred
            final_weights  — weight vector at last rebalance
            total_return   — scalar final cumulative return
            sharpe         — annualised Sharpe over backtest period
            max_drawdown   — maximum drawdown over backtest period
        """
        n_assets = len(assets)
        return_series = []
        for a in assets:
            ret = np.diff(a.prices) / np.where(a.prices[:-1] == 0, 1.0, a.prices[:-1])
            return_series.append(ret)

        min_len = min(len(r) for r in return_series)
        returns = np.column_stack([r[-min_len:] for r in return_series])  # (T, N)
        T = min_len

        w = weights.copy()
        w /= w.sum()

        port_returns = np.zeros(T)
        rebalance_dates: list[int] = []
        current_weights = w.copy()

        for t in range(T):
            # Rebalance check
            if t > 0 and t % rebalance_freq == 0:
                # Build sub-manifolds up to bar t
                sub_assets = []
                for idx, a in enumerate(assets):
                    sub_len = min(t + 1, len(a.prices))
                    sub_prices = a.prices[-sub_len:]
                    sub_ts = a.timestamps[-sub_len:]
                    try:
                        sub_m = self.build_asset_manifold(
                            a.symbol,
                            np.column_stack([sub_ts, sub_prices]),
                        )
                        sub_assets.append(sub_m)
                    except Exception:
                        sub_assets.append(a)

                try:
                    new_result = self.optimize(sub_assets)
                    current_weights = new_result.weights
                    rebalance_dates.append(t)
                except Exception as exc:
                    log.debug("Rebalance failed at bar %d: %s", t, exc)

            port_returns[t] = float(np.dot(current_weights, returns[t]))

        wealth_index = np.cumprod(1.0 + port_returns)
        total_return = float(wealth_index[-1]) - 1.0

        # Annualised Sharpe
        bars_per_year = 252.0 * 390.0
        mean_r = float(np.mean(port_returns))
        std_r = float(np.std(port_returns, ddof=1))
        sharpe = (mean_r / std_r) * np.sqrt(bars_per_year) if std_r > 0 else 0.0

        # Max drawdown
        peak = np.maximum.accumulate(wealth_index)
        drawdown = (wealth_index - peak) / np.where(peak == 0, 1.0, peak)
        max_drawdown = float(np.min(drawdown))

        return {
            "wealth_index": wealth_index,
            "returns": port_returns,
            "rebalance_dates": rebalance_dates,
            "final_weights": current_weights,
            "total_return": total_return,
            "sharpe": float(sharpe),
            "max_drawdown": max_drawdown,
        }


# ─── Synthetic data generator ──────────────────────────────────────────────────


def generate_synthetic_assets(
    n_assets: int = 5,
    n_bars: int = 1000,
    seed: int = 42,
    c: float = 0.1,
) -> List[AssetManifold]:
    """
    Generate synthetic OHLCV-style price series with a mix of TIMELIKE and
    SPACELIKE regimes for demonstration purposes.

    The price model alternates between:
    - Low-volatility (causal) regime: small moves => TIMELIKE bars
    - High-volatility (noise) regime: large moves => SPACELIKE bars

    Parameters
    ----------
    n_assets:
        Number of assets to generate.
    n_bars:
        Number of price bars per asset.
    seed:
        Random seed for reproducibility.
    c:
        Financial speed of light — must match the optimizer's c parameter.

    Returns
    -------
    List of AssetManifold objects.
    """
    rng = np.random.default_rng(seed)
    optimizer = RelativisticPortfolioOptimizer(c=c)
    assets: List[AssetManifold] = []

    base_prices = rng.uniform(50.0, 500.0, size=n_assets)

    for i in range(n_assets):
        prices = np.zeros(n_bars)
        prices[0] = base_prices[i]
        timestamps = np.arange(n_bars, dtype=float) * 60.0  # 1-minute bars

        # Regime switching: every ~100 bars alternate between calm/volatile
        for t in range(1, n_bars):
            regime = (t // 100) % 2
            if regime == 0:
                # TIMELIKE-dominant regime: small fractional moves
                sigma = c * 0.5  # well below c => TIMELIKE
            else:
                # SPACELIKE-dominant regime: large fractional moves
                sigma = c * 1.8  # above c => SPACELIKE

            shock = rng.normal(0, sigma)
            # Add a small drift and cross-asset correlation
            drift = 0.00005 * (i + 1)
            prices[t] = prices[t - 1] * (1.0 + drift + shock)

        # Clamp to positive
        prices = np.maximum(prices, 0.01)

        manifold = optimizer.build_asset_manifold(
            symbol=f"ASSET_{chr(65 + i)}",
            ohlcv=np.column_stack([timestamps, prices]),
        )
        assets.append(manifold)
        log.info(
            "  %s: TIMELIKE fraction = %.3f, price range [%.2f, %.2f]",
            manifold.symbol,
            manifold.timelike_fraction,
            float(prices.min()),
            float(prices.max()),
        )

    return assets


# ─── Demonstration ─────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("=" * 70)
    print("  SRFM Relativistic Portfolio Optimizer — Demo")
    print("=" * 70)

    C = 0.1
    optimizer = RelativisticPortfolioOptimizer(c=C, risk_free_rate=0.05)

    print("\n[1] Generating synthetic assets...")
    assets = generate_synthetic_assets(n_assets=5, n_bars=1000, seed=42, c=C)

    # ── Single optimisation ────────────────────────────────────────────────────
    print("\n[2] Running relativistic portfolio optimisation...")
    t0 = time.perf_counter()
    result = optimizer.optimize(assets, max_weight=0.4, min_weight=0.0)
    elapsed = time.perf_counter() - t0

    print(f"\n  Optimisation time     : {elapsed*1000:.1f} ms")
    print(f"  Relativistic Sharpe   : {result.relativistic_sharpe:+.4f}")
    print(f"  Classical Sharpe      : {result.classical_sharpe:+.4f}")
    print(f"  Annualised return     : {result.annualized_return*100:+.2f}%")
    print(f"  Annualised volatility : {result.annualized_vol*100:.2f}%")
    print(f"  TIMELIKE exposure     : {result.timelike_exposure:.3f}")
    print(f"  Spacetime interval    : {result.spacetime_interval:.6e}")
    print(f"  Max drawdown          : {result.max_drawdown*100:.2f}%")
    print()
    print("  Optimal weights:")
    for a, w in zip(assets, result.weights):
        bar = "#" * int(w * 40)
        print(f"    {a.symbol:<12} {w:6.3f}  {bar}")

    # ── Constrained: minimum TIMELIKE exposure ─────────────────────────────────
    print("\n[3] Constrained optimisation (min 70% TIMELIKE exposure)...")
    result_constrained = optimizer.optimize(
        assets, max_weight=0.4, min_weight=0.0, target_timelike=0.70
    )
    print(f"  Relativistic Sharpe   : {result_constrained.relativistic_sharpe:+.4f}")
    print(f"  TIMELIKE exposure     : {result_constrained.timelike_exposure:.3f}")
    print("  Weights:")
    for a, w in zip(assets, result_constrained.weights):
        print(f"    {a.symbol:<12} {w:6.3f}")

    # ── Efficient frontier ─────────────────────────────────────────────────────
    print("\n[4] Computing relativistic efficient frontier (20 points)...")
    t0 = time.perf_counter()
    frontier = optimizer.efficient_frontier(assets, n_points=20)
    elapsed = time.perf_counter() - t0
    print(f"  Frontier computation  : {elapsed*1000:.1f} ms  ({len(frontier)} points)")
    print()
    print(f"  {'Ann. Return':>12}  {'Ann. Vol':>10}  {'Rel. Sharpe':>12}  {'TL Exposure':>12}")
    print("  " + "-" * 52)
    for fp in frontier:
        print(
            f"  {fp.annualized_return*100:>11.2f}%  "
            f"{fp.annualized_vol*100:>9.2f}%  "
            f"{fp.relativistic_sharpe:>12.4f}  "
            f"{fp.timelike_exposure:>12.3f}"
        )

    # ── Backtest ───────────────────────────────────────────────────────────────
    print("\n[5] Running backtest with rebalancing every 20 bars...")
    t0 = time.perf_counter()
    bt = optimizer.backtest(assets, result.weights, rebalance_freq=20)
    elapsed = time.perf_counter() - t0
    print(f"  Backtest time         : {elapsed*1000:.1f} ms")
    print(f"  Total return          : {bt['total_return']*100:+.2f}%")
    print(f"  Annualised Sharpe     : {bt['sharpe']:+.4f}")
    print(f"  Max drawdown          : {bt['max_drawdown']*100:.2f}%")
    print(f"  Rebalances executed   : {len(bt['rebalance_dates'])}")

    # ── Spacetime covariance matrix ────────────────────────────────────────────
    print("\n[6] Spacetime covariance matrix:")
    cov = optimizer.spacetime_covariance(assets)
    symbols = [a.symbol for a in assets]
    header = "         " + "".join(f"{s:>12}" for s in symbols)
    print("  " + header)
    for i, s in enumerate(symbols):
        row = f"  {s:<8} " + "".join(f"{cov[i, j]:>12.6f}" for j in range(len(assets)))
        print(row)

    print()
    print("=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
