"""
srfm — Special Relativity in Financial Modeling
================================================

Python interface to the SRFM C++ library.

When the compiled extension (``srfm._srfm``) is available it is imported
directly.  When running without a build (e.g. during documentation generation
or pure-Python testing) a pure-Python fallback implementation is used that
mirrors the full API but operates in Python/NumPy.

Public API
----------
SpacetimeInterval
    Compute and classify ds² = −c²Δt² + ΔP² + ΔV² + ΔM².

LorentzTransform
    Compute γ(β), β, and relativistic momentum.

Backtester
    Run a full relativistic backtest over a price/volume series.

BacktestResult / BacktestComparison
    Side-by-side raw vs relativistic performance metrics.

MultiAssetEvent
    Multi-asset spacetime event (N prices + volumes + timestamp).

CorrelationMetric
    Rolling correlation-based Lorentzian metric tensor.

MultiAssetInterval
    N-dimensional spacetime interval computation.

MultiAssetLorentz
    Simultaneous Lorentz boosts across correlated assets.

PortfolioGeodesic
    Geodesic path computation in multi-asset spacetime.

Quick Start
-----------
>>> from srfm import SpacetimeInterval, LorentzTransform, Backtester

Interval classification:
>>> SpacetimeInterval.classify(dt=1.0, dp=0.5, dv=0.1, dm=0.05)
'TIMELIKE'

Lorentz factor:
>>> LorentzTransform.gamma(beta=0.8)
1.6666666666666667

Full backtest:
>>> bt = Backtester()
>>> result = bt.run(prices=[100, 101, 99, 102, 103, 100])
>>> print(result.sharpe)
>>> print(result.relativistic_lift)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Try to import the compiled C++ extension.  Fall back to pure Python.
# ---------------------------------------------------------------------------

try:
    from srfm._srfm import (  # type: ignore[import]
        SpacetimeInterval,
        LorentzTransform,
        Backtester,
        BacktestConfig,
        BacktestResult,
        PerformanceMetrics,
        MultiAssetEvent,
        CorrelationMetric,
        CorrelationMetricConfig,
        MultiAssetInterval,
        MultiAssetLorentz,
        MultiAssetTransformResult,
        PortfolioGeodesic,
        GeodesicStep,
        IntervalType,
        SPEED_OF_LIGHT,
        BETA_MAX_SAFE,
        LIGHTLIKE_EPS,
    )
    _NATIVE = True

except ImportError:
    _NATIVE = False

    # -----------------------------------------------------------------------
    # Pure-Python fallback implementation
    # -----------------------------------------------------------------------

    import statistics

    #: Market speed of light (normalised to 1.0 in natural units).
    SPEED_OF_LIGHT: float = 1.0
    #: Safe upper bound on |β| to avoid γ divergence.
    BETA_MAX_SAFE: float = 0.9999
    #: |ds²| threshold for LIGHTLIKE classification.
    LIGHTLIKE_EPS: float = 1e-12

    from enum import Enum

    class IntervalType(Enum):
        """Causal character of a spacetime interval."""
        TIMELIKE  = "TIMELIKE"
        LIGHTLIKE = "LIGHTLIKE"
        SPACELIKE = "SPACELIKE"

    # ── SpacetimeInterval ────────────────────────────────────────────────────

    class SpacetimeInterval:
        """Pure-Python implementation of spacetime interval math."""

        @staticmethod
        def compute_ds2(dt: float, dp: float, dv: float, dm: float,
                        c: float = SPEED_OF_LIGHT) -> float:
            """Compute ds² = −c²·Δt² + ΔP² + ΔV² + ΔM²."""
            return -(c * c * dt * dt) + dp * dp + dv * dv + dm * dm

        @staticmethod
        def classify(dt: float, dp: float, dv: float, dm: float,
                     c: float = SPEED_OF_LIGHT) -> str:
            """Return 'TIMELIKE', 'LIGHTLIKE', or 'SPACELIKE'."""
            ds2 = SpacetimeInterval.compute_ds2(dt, dp, dv, dm, c)
            return SpacetimeInterval.classify_ds2(ds2)

        @staticmethod
        def classify_ds2(ds2: float) -> str:
            """Classify a pre-computed ds² value."""
            if abs(ds2) <= LIGHTLIKE_EPS:
                return "LIGHTLIKE"
            return "TIMELIKE" if ds2 < 0 else "SPACELIKE"

    # ── LorentzTransform ─────────────────────────────────────────────────────

    class LorentzTransform:
        """Pure-Python Lorentz factor utilities."""

        @staticmethod
        def gamma(beta: float) -> float:
            """γ(β) = 1 / √(1 − β²). β is clamped to BETA_MAX_SAFE."""
            b = min(abs(beta), BETA_MAX_SAFE)
            return 1.0 / math.sqrt(1.0 - b * b)

        @staticmethod
        def beta(dp: float, dt: float, c: float = SPEED_OF_LIGHT) -> float:
            """β = ΔP / (c·Δt), clamped to BETA_MAX_SAFE."""
            if abs(dt) < 1e-15 or abs(c) < 1e-15:
                return 0.0
            raw = dp / (c * dt)
            return max(-BETA_MAX_SAFE, min(BETA_MAX_SAFE, raw))

        @staticmethod
        def relativistic_momentum(gamma: float, m_eff: float,
                                   p_raw: float) -> float:
            """p_rel = γ · m_eff · p_raw."""
            return gamma * m_eff * p_raw

    # ── PerformanceMetrics ───────────────────────────────────────────────────

    @dataclass
    class PerformanceMetrics:
        """Performance metrics for a strategy evaluation."""
        sharpe_ratio:      float = 0.0
        sortino_ratio:     float = 0.0
        max_drawdown:      float = 0.0
        gamma_weighted_ir: float = 0.0

        def to_string(self) -> str:
            return (f"Sharpe={self.sharpe_ratio:.4f}  "
                    f"Sortino={self.sortino_ratio:.4f}  "
                    f"MDD={self.max_drawdown:.4f}  "
                    f"IR_γ={self.gamma_weighted_ir:.4f}")

    # ── BacktestResult ───────────────────────────────────────────────────────

    @dataclass
    class BacktestResult:
        """Side-by-side raw vs relativistic backtest results."""
        raw:               PerformanceMetrics = field(default_factory=PerformanceMetrics)
        relativistic:      PerformanceMetrics = field(default_factory=PerformanceMetrics)
        mean_gamma:        float = 1.0
        max_gamma_applied: float = 1.0
        relativistic_lift: float = 0.0

        @property
        def sharpe(self) -> float:
            return self.relativistic.sharpe_ratio

        @property
        def sortino(self) -> float:
            return self.relativistic.sortino_ratio

        @property
        def max_drawdown(self) -> float:
            return self.relativistic.max_drawdown

        @property
        def relativistic_sharpe(self) -> float:
            return self.relativistic.sharpe_ratio

        def sharpe_lift(self) -> float:
            return self.relativistic.sharpe_ratio - self.raw.sharpe_ratio

        def sortino_lift(self) -> float:
            return self.relativistic.sortino_ratio - self.raw.sortino_ratio

        def drawdown_delta(self) -> float:
            return self.raw.max_drawdown - self.relativistic.max_drawdown

        def ir_lift(self) -> float:
            return self.relativistic.gamma_weighted_ir - self.raw.gamma_weighted_ir

        def to_string(self) -> str:
            lines = [
                f"{'Metric':<22} {'Raw':>12} {'Relativistic':>14}",
                "-" * 50,
                f"{'Sharpe ratio':<22} {self.raw.sharpe_ratio:>12.4f} {self.relativistic.sharpe_ratio:>14.4f}",
                f"{'Sortino ratio':<22} {self.raw.sortino_ratio:>12.4f} {self.relativistic.sortino_ratio:>14.4f}",
                f"{'Max drawdown':<22} {self.raw.max_drawdown:>12.4f} {self.relativistic.max_drawdown:>14.4f}",
                f"{'IR_γ':<22} {self.raw.gamma_weighted_ir:>12.4f} {self.relativistic.gamma_weighted_ir:>14.4f}",
                "-" * 50,
                f"{'Mean γ':<22} {self.mean_gamma:>12.4f}",
                f"{'Relativistic lift':<22} {self.relativistic_lift:>12.4f}×",
            ]
            return "\n".join(lines)

        def __repr__(self) -> str:
            return self.to_string()

    #: Alias
    BacktestComparison = BacktestResult

    # ── BacktestConfig ───────────────────────────────────────────────────────

    @dataclass
    class BacktestConfig:
        """Configuration for a Backtester run."""
        risk_free_rate: float = 0.0
        annualisation:  float = 252.0
        effective_mass: float = 1.0
        max_gamma:      float = 3.0
        verbose:        bool  = False

    # ── Backtester ───────────────────────────────────────────────────────────

    class Backtester:
        """Pure-Python relativistic backtester."""

        def __init__(self, config: Optional[BacktestConfig] = None) -> None:
            self.config = config or BacktestConfig()

        def run(self, prices: List[float],
                volumes: Optional[List[float]] = None) -> BacktestResult:
            """Run a backtest over a price series."""
            if len(prices) < 3:
                raise ValueError("Need at least 3 price bars.")
            if volumes and len(volumes) != len(prices):
                raise ValueError("volumes must match prices length.")

            c   = SPEED_OF_LIGHT
            cfg = self.config
            ann = cfg.annualisation

            # Compute log-returns and per-bar β/γ.
            returns: List[float] = []
            gammas:  List[float] = []
            for i in range(1, len(prices)):
                if prices[i - 1] <= 0:
                    returns.append(0.0)
                    gammas.append(1.0)
                    continue
                ret  = math.log(prices[i] / prices[i - 1])
                beta = max(-BETA_MAX_SAFE,
                           min(BETA_MAX_SAFE, (prices[i] - prices[i-1]) / (c * 1.0)))
                gamma = LorentzTransform.gamma(beta)
                gamma = min(gamma, cfg.max_gamma)
                returns.append(ret)
                gammas.append(gamma)

            n = len(returns)
            if n < 2:
                return BacktestResult()

            rf = cfg.risk_free_rate / ann
            mean_g = sum(gammas) / len(gammas)
            max_g  = max(gammas)

            def _metrics(rets: List[float]) -> PerformanceMetrics:
                n_r = len(rets)
                if n_r < 2:
                    return PerformanceMetrics()
                mean_r = sum(rets) / n_r
                variance = sum((r - mean_r) ** 2 for r in rets) / (n_r - 1)
                std_r = math.sqrt(variance) if variance > 0 else 1e-10
                sharpe = (mean_r - rf) / std_r * math.sqrt(ann)

                # Sortino: downside deviation.
                neg_rets = [r for r in rets if r < 0]
                if neg_rets:
                    down_var = sum(r * r for r in neg_rets) / n_r
                    down_std = math.sqrt(down_var) if down_var > 0 else 1e-10
                else:
                    down_std = 1e-10
                sortino = (mean_r - rf) / down_std * math.sqrt(ann)

                # Max drawdown.
                peak = 0.0
                cum  = 0.0
                mdd  = 0.0
                for r in rets:
                    cum += r
                    if cum > peak:
                        peak = cum
                    dd = peak - cum
                    if dd > mdd:
                        mdd = dd

                # γ-weighted IR (simplified: mean/std of returns × mean_gamma).
                ir_gamma = (mean_r / std_r) * mean_g if std_r > 0 else 0.0

                return PerformanceMetrics(sharpe, sortino, mdd, ir_gamma)

            raw_ret = returns
            rel_ret = [g * r for g, r in zip(gammas, returns)]

            raw_metrics = _metrics(raw_ret)
            rel_metrics = _metrics(rel_ret)

            lift = 0.0
            if abs(raw_metrics.gamma_weighted_ir) > 1e-10:
                lift = rel_metrics.gamma_weighted_ir / raw_metrics.gamma_weighted_ir

            return BacktestResult(
                raw=raw_metrics,
                relativistic=rel_metrics,
                mean_gamma=mean_g,
                max_gamma_applied=max_g,
                relativistic_lift=lift,
            )

    # ── MultiAssetEvent ──────────────────────────────────────────────────────

    @dataclass
    class MultiAssetEvent:
        """A snapshot of N financial assets at a point in time."""
        symbols:   List[str]   = field(default_factory=list)
        prices:    List[float] = field(default_factory=list)
        volumes:   List[float] = field(default_factory=list)
        timestamp: int         = 0

        def n_assets(self) -> int:
            return len(self.prices)

        def is_valid(self) -> bool:
            if not self.prices:
                return False
            if self.volumes and len(self.volumes) != len(self.prices):
                return False
            if self.symbols and len(self.symbols) != len(self.prices):
                return False
            return True

    # ── CorrelationMetricConfig / CorrelationMetric ───────────────────────────

    @dataclass
    class CorrelationMetricConfig:
        """Configuration for CorrelationMetric."""
        window_size:        int   = 60
        c_market:           float = 1.0
        regularisation_eps: float = 1e-6

    class CorrelationMetric:
        """Pure-Python rolling correlation metric (NumPy-backed)."""

        def __init__(self, n_assets: int,
                     config: Optional[CorrelationMetricConfig] = None) -> None:
            self.n_assets = n_assets
            self.cfg = config or CorrelationMetricConfig()
            self._observations: List[List[float]] = []
            self._prev: Optional[List[float]] = None
            self._log_rets: List[List[float]] = []
            self._metric: Optional[List[List[float]]] = None

        def update(self, prices: List[float]) -> Optional[List[List[float]]]:
            if len(prices) != self.n_assets:
                return None
            if self._prev is None:
                self._prev = list(prices)
                return None
            lr = []
            for i in range(self.n_assets):
                if self._prev[i] > 0 and prices[i] > 0:
                    lr.append(math.log(prices[i] / self._prev[i]))
                else:
                    lr.append(0.0)
            self._log_rets.append(lr)
            if len(self._log_rets) > self.cfg.window_size:
                self._log_rets.pop(0)
            self._prev = list(prices)
            self._metric = self._build_metric()
            return self._metric

        def _build_metric(self) -> Optional[List[List[float]]]:
            n = self.n_assets
            rows = self._log_rets
            if len(rows) < 2:
                return None
            # Sample covariance.
            means = [sum(r[i] for r in rows) / len(rows) for i in range(n)]
            cov = [[0.0] * n for _ in range(n)]
            for r in rows:
                z = [r[i] - means[i] for i in range(n)]
                for i in range(n):
                    for j in range(n):
                        cov[i][j] += z[i] * z[j]
            for i in range(n):
                for j in range(n):
                    cov[i][j] /= (len(rows) - 1)
                cov[i][i] += self.cfg.regularisation_eps
            c2 = self.cfg.c_market ** 2
            dim = n + 1
            g = [[0.0] * dim for _ in range(dim)]
            g[0][0] = -c2
            for i in range(n):
                for j in range(n):
                    g[i+1][j+1] = cov[i][j]
            return g

        def current_metric(self) -> Optional[List[List[float]]]:
            return self._metric

        def correlation_matrix(self) -> Optional[List[List[float]]]:
            if self._metric is None:
                return None
            n = self.n_assets
            g = self._metric
            cov = [[g[i+1][j+1] for j in range(n)] for i in range(n)]
            corr = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    denom = math.sqrt(cov[i][i] * cov[j][j])
                    corr[i][j] = cov[i][j] / denom if denom > 1e-12 else (1.0 if i == j else 0.0)
            return corr

        def observation_count(self) -> int:
            return len(self._log_rets)

        def reset(self) -> None:
            self._prev = None
            self._log_rets.clear()
            self._metric = None

    # ── MultiAssetInterval ────────────────────────────────────────────────────

    class MultiAssetInterval:
        """Pure-Python N-dimensional spacetime interval."""

        LIGHTLIKE_THRESHOLD: float = 1e-10

        @staticmethod
        def compute(a: "MultiAssetEvent", b: "MultiAssetEvent",
                    metric: List[List[float]]) -> Optional[float]:
            if not a.is_valid() or not b.is_valid():
                return None
            n = a.n_assets()
            if n != b.n_assets():
                return None
            dim = n + 1
            if len(metric) != dim or any(len(row) != dim for row in metric):
                return None
            dt = (b.timestamp - a.timestamp) / 1000.0
            dx = [dt] + [b.prices[i] - a.prices[i] for i in range(n)]
            ds2 = 0.0
            for i in range(dim):
                for j in range(dim):
                    ds2 += metric[i][j] * dx[i] * dx[j]
            return ds2

        @staticmethod
        def classify(ds2: float) -> str:
            if abs(ds2) <= MultiAssetInterval.LIGHTLIKE_THRESHOLD:
                return "LIGHTLIKE"
            return "TIMELIKE" if ds2 < 0 else "SPACELIKE"

    # ── MultiAssetLorentz ─────────────────────────────────────────────────────

    @dataclass
    class MultiAssetTransformResult:
        adjusted_prices:  List[float]
        beta_portfolio:   float
        gamma_portfolio:  float
        per_asset_beta:   List[float]
        per_asset_gamma:  List[float]

    class MultiAssetLorentz:
        """Pure-Python multi-asset Lorentz transform."""

        @staticmethod
        def transform(a: "MultiAssetEvent", b: "MultiAssetEvent",
                      metric: List[List[float]]) -> Optional[MultiAssetTransformResult]:
            if not a.is_valid() or not b.is_valid():
                return None
            n = a.n_assets()
            if n != b.n_assets():
                return None
            dt = (b.timestamp - a.timestamp) / 1000.0
            if abs(dt) < 1e-12:
                return None
            c = math.sqrt(abs(metric[0][0])) if metric else 1.0
            betas: List[float] = []
            gammas: List[float] = []
            for i in range(n):
                beta = max(-BETA_MAX_SAFE,
                           min(BETA_MAX_SAFE, (b.prices[i] - a.prices[i]) / (c * dt)))
                betas.append(beta)
                gammas.append(LorentzTransform.gamma(beta))
            beta_p = MultiAssetLorentz.portfolio_beta(betas, metric)
            adjusted = [gammas[i] * (b.prices[i] - a.prices[i]) for i in range(n)]
            return MultiAssetTransformResult(adjusted, beta_p,
                                              LorentzTransform.gamma(beta_p),
                                              betas, gammas)

        @staticmethod
        def portfolio_beta(betas: List[float],
                           metric: List[List[float]]) -> float:
            n = len(betas)
            if n == 0:
                return 0.0
            b2 = 0.0
            for i in range(n):
                for j in range(n):
                    b2 += metric[i+1][j+1] * betas[i] * betas[j]
            return max(0.0, min(BETA_MAX_SAFE, math.sqrt(max(0.0, b2))))

    # ── PortfolioGeodesic ─────────────────────────────────────────────────────

    @dataclass
    class GeodesicStep:
        event:       "MultiAssetEvent"
        proper_time: float
        path_length: float
        deviation:   float

    class PortfolioGeodesic:
        """Pure-Python geodesic integration for multi-asset spacetime."""

        @staticmethod
        def integrate(initial: "MultiAssetEvent",
                      four_velocity: List[float],
                      metric: List[List[float]],
                      n_steps: int,
                      dt: float) -> List[GeodesicStep]:
            if not initial.is_valid() or n_steps == 0 or abs(dt) < 1e-15:
                return []
            n = initial.n_assets()
            dim = n + 1
            if len(four_velocity) != dim:
                return []
            x = [initial.timestamp / 1000.0] + list(initial.prices)
            path: List[GeodesicStep] = []
            tau = 0.0
            arc = 0.0
            for _ in range(n_steps):
                x = [x[k] + four_velocity[k] * dt for k in range(dim)]
                tau += dt
                ds2_dot = sum(metric[i][j] * four_velocity[i] * four_velocity[j]
                              for i in range(dim) for j in range(dim))
                arc += math.sqrt(abs(ds2_dot)) * abs(dt)
                ev = MultiAssetEvent(
                    symbols=list(initial.symbols),
                    prices=list(x[1:]),
                    volumes=list(initial.volumes),
                    timestamp=int(x[0] * 1000),
                )
                path.append(GeodesicStep(ev, tau, arc, 0.0))
            return path

        @staticmethod
        def deviation_series(predicted: List[GeodesicStep],
                             actual: List["MultiAssetEvent"]) -> List[float]:
            if len(predicted) != len(actual):
                return []
            devs: List[float] = []
            for step, act in zip(predicted, actual):
                pp = step.event.prices
                ap = act.prices
                if len(pp) != len(ap):
                    devs.append(0.0)
                else:
                    devs.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(pp, ap))))
            return devs

        @staticmethod
        def portfolio_weights(steps: List[GeodesicStep],
                              four_velocity: List[float],
                              gross_exposure: float = 1.0) -> List[List[float]]:
            u_spatial = four_velocity[1:]
            l1 = sum(abs(v) for v in u_spatial)
            if l1 < 1e-12:
                return [[0.0] * len(u_spatial) for _ in steps]
            w = [v * gross_exposure / l1 for v in u_spatial]
            return [list(w) for _ in steps]


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__: str = "1.1.0"
__author__:  str = "Matthew Busel"

__all__ = [
    # Core
    "SpacetimeInterval",
    "LorentzTransform",
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "PerformanceMetrics",
    # Multi-asset
    "MultiAssetEvent",
    "CorrelationMetric",
    "CorrelationMetricConfig",
    "MultiAssetInterval",
    "MultiAssetLorentz",
    "MultiAssetTransformResult",
    "PortfolioGeodesic",
    "GeodesicStep",
    # Constants & enums
    "IntervalType",
    "SPEED_OF_LIGHT",
    "BETA_MAX_SAFE",
    "LIGHTLIKE_EPS",
    # Meta
    "__version__",
]
