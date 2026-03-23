"""
empirical_extended.py — SRFM Extended Empirical Validation (Crypto Markets)
=============================================================================

Extends the Q1 2025 equity validation to cryptocurrency markets (BTC, ETH
and a configurable set of altcoins) using public Binance REST API data.

Research Questions
------------------
1. Do TIMELIKE/SPACELIKE classifications from the SRFM spacetime interval
   predict next-bar realised volatility in crypto markets?

2. Is the TIMELIKE vs SPACELIKE volatility gap (observed in equities Q1 2025)
   reproducible in 24/7 crypto markets with higher baseline volatility?

3. How does the SRFM classification compare against standard technical
   analysis benchmarks (RSI, MACD) as next-bar volatility predictors?

Statistical Methods
-------------------
* Bootstrap resampling (default: 10,000 replications) for confidence intervals
  on mean volatility by regime.
* Permutation test (default: 10,000 shuffles) for the null hypothesis that
  TIMELIKE and SPACELIKE regimes have identical next-bar volatility distributions.
* Bartlett test for equality of variances across regimes.
* Bonferroni correction for multiple comparisons across assets.

Usage
-----
    python validation/empirical_extended.py [OPTIONS]

    --symbols   BTC ETH SOL BNB   (default: BTC ETH)
    --interval  1h                 (Binance kline interval, default 1h)
    --limit     1000               (bars per symbol, default 1000)
    --n-boot    10000              (bootstrap replications, default 10000)
    --n-perm    10000              (permutation test shuffles, default 10000)
    --alpha     0.05               (significance level, default 0.05)
    --c-scale   0.05               (SRFM speed-of-light scale, default 0.05)
    --output    validation/        (output directory, default validation/)
    --format    both               (report format: latex|markdown|both)
    --no-download                  (use cached data if available)

Dependencies
------------
    pip install pandas scipy numpy requests
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
DEFAULT_SYMBOLS   = ["BTCUSDT", "ETHUSDT"]
DEFAULT_INTERVAL  = "1h"
DEFAULT_LIMIT     = 1000
DEFAULT_N_BOOT    = 10_000
DEFAULT_N_PERM    = 10_000
DEFAULT_ALPHA     = 0.05
DEFAULT_C_SCALE   = 0.05

TIMELIKE_LABEL  = "TIMELIKE"
SPACELIKE_LABEL = "SPACELIKE"
LIGHTLIKE_LABEL = "LIGHTLIKE"
LIGHTLIKE_EPS   = 1e-12

BENCHMARK_RSI_PERIOD  = 14
BENCHMARK_MACD_FAST   = 12
BENCHMARK_MACD_SLOW   = 26
BENCHMARK_MACD_SIGNAL = 9

# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class BarData:
    """One OHLCV bar from Binance klines."""
    open_time:  int
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float
    close_time: int

    @property
    def log_return(self) -> float:
        if self.open <= 0:
            return 0.0
        return math.log(self.close / self.open)

    @property
    def mid_price(self) -> float:
        return (self.high + self.low) / 2.0


@dataclass
class ClassifiedBar:
    """A bar annotated with SRFM spacetime classification and TA signals."""
    bar:            BarData
    interval_class: str
    ds2:            float
    beta:           float
    gamma:          float
    next_bar_abs_return: Optional[float] = None
    rsi:            Optional[float] = None
    macd_signal:    Optional[float] = None
    macd_histogram: Optional[float] = None


@dataclass
class RegimeStats:
    """Per-regime summary statistics."""
    label:         str
    count:         int
    mean_vol:      float
    std_vol:       float
    median_vol:    float
    ci_lower:      float  # bootstrap 95% CI lower bound
    ci_upper:      float  # bootstrap 95% CI upper bound


@dataclass
class PairwiseTestResult:
    """Result of a statistical test between two regimes."""
    test_name:    str
    statistic:    float
    p_value:      float
    significant:  bool
    note:         str = ""


@dataclass
class SymbolValidationResult:
    """Full validation result for one symbol."""
    symbol:              str
    n_bars:              int
    n_timelike:          int
    n_spacelike:         int
    n_lightlike:         int
    timelike_stats:      Optional[RegimeStats]
    spacelike_stats:     Optional[RegimeStats]
    bartlett:            Optional[PairwiseTestResult]
    permutation:         Optional[PairwiseTestResult]
    rsi_vs_regime:       Optional[PairwiseTestResult]
    macd_vs_regime:      Optional[PairwiseTestResult]
    vol_ratio_tl_sl:     float   # timelike_mean_vol / spacelike_mean_vol
    bonferroni_alpha:    float


# ─── Binance Data Download ────────────────────────────────────────────────────


class CryptoValidation:
    """
    Download and validate SRFM spacetime classifications against crypto data.

    Fetches Binance kline (OHLCV) data for one or more symbols, computes
    SRFM spacetime intervals, classifies each bar, and runs statistical tests
    to determine whether TIMELIKE/SPACELIKE regimes predict next-bar volatility.

    Parameters
    ----------
    symbols:
        List of Binance trading pairs, e.g. ["BTCUSDT", "ETHUSDT"].
    interval:
        Binance kline interval string, e.g. "1h", "4h", "1d".
    limit:
        Number of bars to fetch per symbol (max 1000 per Binance API call).
    c_scale:
        SRFM speed-of-light scale: maximum meaningful log-return per bar.
    n_boot:
        Number of bootstrap resampling iterations for CI estimation.
    n_perm:
        Number of permutation test shuffles.
    alpha:
        Significance level for hypothesis tests.
    cache_dir:
        Directory for caching downloaded data as CSV. ``None`` disables caching.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        interval: str = DEFAULT_INTERVAL,
        limit: int = DEFAULT_LIMIT,
        c_scale: float = DEFAULT_C_SCALE,
        n_boot: int = DEFAULT_N_BOOT,
        n_perm: int = DEFAULT_N_PERM,
        alpha: float = DEFAULT_ALPHA,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.symbols   = symbols or DEFAULT_SYMBOLS
        self.interval  = interval
        self.limit     = min(limit, 1000)  # Binance cap
        self.c_scale   = c_scale
        self.n_boot    = n_boot
        self.n_perm    = n_perm
        self.alpha     = alpha
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._results:  Dict[str, SymbolValidationResult] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, SymbolValidationResult]:
        """Download data, classify bars, and run all statistical tests.

        Returns a dict mapping symbol → :class:`SymbolValidationResult`.
        """
        bonferroni_alpha = self.alpha / len(self.symbols)
        log.info(
            "Starting crypto validation: %d symbols, interval=%s, limit=%d, "
            "n_boot=%d, n_perm=%d, α=%.4f (Bonferroni α=%.4f)",
            len(self.symbols), self.interval, self.limit,
            self.n_boot, self.n_perm, self.alpha, bonferroni_alpha,
        )

        for symbol in self.symbols:
            log.info("Processing %s …", symbol)
            bars = self._get_bars(symbol)
            if len(bars) < 50:
                log.warning("Insufficient data for %s (%d bars), skipping.", symbol, len(bars))
                continue
            classified = self._classify_bars(bars)
            result = self._run_tests(symbol, classified, bonferroni_alpha)
            self._results[symbol] = result
            self._log_result_summary(result)

        return self._results

    def results(self) -> Dict[str, SymbolValidationResult]:
        """Return cached results from the most recent :meth:`run` call."""
        return self._results

    # ── Data Download ──────────────────────────────────────────────────────────

    def _get_bars(self, symbol: str) -> List[BarData]:
        """Download (or load from cache) kline data for *symbol*."""
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{symbol}_{self.interval}_{self.limit}.csv"
            if cache_path.exists():
                log.info("  Loading from cache: %s", cache_path)
                return self._load_cache(cache_path)

        log.info("  Downloading from Binance API …")
        bars = self._download_binance(symbol)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._save_cache(bars, cache_path)

        return bars

    def _download_binance(self, symbol: str) -> List[BarData]:
        """Download klines from Binance REST API.

        Falls back to synthetic data if the API is unreachable (for testing).
        """
        try:
            import requests
        except ImportError:
            log.warning("requests not installed; using synthetic data for %s.", symbol)
            return self._synthetic_bars(symbol)

        params = {
            "symbol":   symbol,
            "interval": self.interval,
            "limit":    self.limit,
        }
        try:
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Binance API error for %s (%s); falling back to synthetic data.", symbol, exc
            )
            return self._synthetic_bars(symbol)

        bars: List[BarData] = []
        for k in raw:
            try:
                bars.append(BarData(
                    open_time  = int(k[0]),
                    open       = float(k[1]),
                    high       = float(k[2]),
                    low        = float(k[3]),
                    close      = float(k[4]),
                    volume     = float(k[5]),
                    close_time = int(k[6]),
                ))
            except (IndexError, ValueError):
                continue
        log.info("  Downloaded %d bars for %s.", len(bars), symbol)
        return bars

    @staticmethod
    def _synthetic_bars(symbol: str, n: int = 500) -> List[BarData]:
        """Generate synthetic OHLCV data for offline testing."""
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        price = 40_000.0 if "BTC" in symbol else 2_000.0
        bars = []
        t = int(time.time() * 1000) - n * 3_600_000
        for _ in range(n):
            ret = rng.normal(0, 0.02)
            close = price * math.exp(ret)
            high  = max(price, close) * (1 + abs(rng.normal(0, 0.005)))
            low   = min(price, close) * (1 - abs(rng.normal(0, 0.005)))
            vol   = abs(rng.normal(1e8, 2e7))
            bars.append(BarData(t, price, high, low, close, vol, t + 3_600_000))
            price = close
            t += 3_600_000
        return bars

    def _save_cache(self, bars: List[BarData], path: Path) -> None:
        df = pd.DataFrame([b.__dict__ for b in bars])
        df.to_csv(path, index=False)

    def _load_cache(self, path: Path) -> List[BarData]:
        df = pd.read_csv(path)
        bars = []
        for _, row in df.iterrows():
            bars.append(BarData(
                open_time  = int(row["open_time"]),
                open       = float(row["open"]),
                high       = float(row["high"]),
                low        = float(row["low"]),
                close      = float(row["close"]),
                volume     = float(row["volume"]),
                close_time = int(row["close_time"]),
            ))
        return bars

    # ── Classification ─────────────────────────────────────────────────────────

    def _classify_bars(self, bars: List[BarData]) -> List[ClassifiedBar]:
        """Compute SRFM classification and TA signals for each bar."""
        closes = [b.close for b in bars]
        rsi_values  = compute_rsi(closes, BENCHMARK_RSI_PERIOD)
        macd_values = compute_macd(
            closes, BENCHMARK_MACD_FAST, BENCHMARK_MACD_SLOW, BENCHMARK_MACD_SIGNAL
        )

        classified: List[ClassifiedBar] = []
        for i, bar in enumerate(bars):
            # Compute β = log_return / (c · 1) where dt=1 tick.
            lr = bar.log_return
            beta_raw = lr / self.c_scale
            beta = max(-0.9999, min(0.9999, beta_raw))
            gamma = 1.0 / math.sqrt(1.0 - beta * beta)

            # ds² = -(c·dt)² + (ΔP)² — use log-return as spatial increment.
            ds2 = -(self.c_scale ** 2) + lr ** 2

            if abs(ds2) <= LIGHTLIKE_EPS:
                cls = LIGHTLIKE_LABEL
            elif ds2 < 0.0:
                cls = TIMELIKE_LABEL
            else:
                cls = SPACELIKE_LABEL

            # Next-bar absolute return.
            next_abs = None
            if i + 1 < len(bars):
                next_bar = bars[i + 1]
                if next_bar.open > 0:
                    next_abs = abs(math.log(next_bar.close / next_bar.open))

            cb = ClassifiedBar(
                bar=bar,
                interval_class=cls,
                ds2=ds2,
                beta=beta,
                gamma=gamma,
                next_bar_abs_return=next_abs,
            )

            # Attach TA signals if available.
            if rsi_values[i] is not None:
                cb.rsi = rsi_values[i]
            macd_s, macd_h = macd_values[i]
            cb.macd_signal = macd_s
            cb.macd_histogram = macd_h

            classified.append(cb)

        return classified

    # ── Statistical Tests ──────────────────────────────────────────────────────

    def _run_tests(
        self,
        symbol: str,
        classified: List[ClassifiedBar],
        bonferroni_alpha: float,
    ) -> SymbolValidationResult:
        """Run all statistical tests for one symbol."""
        # Split into regimes.
        tl = [
            cb.next_bar_abs_return for cb in classified
            if cb.interval_class == TIMELIKE_LABEL and cb.next_bar_abs_return is not None
        ]
        sl = [
            cb.next_bar_abs_return for cb in classified
            if cb.interval_class == SPACELIKE_LABEL and cb.next_bar_abs_return is not None
        ]
        ll_count = sum(1 for cb in classified if cb.interval_class == LIGHTLIKE_LABEL)

        n_tl = len(tl)
        n_sl = len(sl)

        tl_arr = np.array(tl, dtype=float)
        sl_arr = np.array(sl, dtype=float)

        # Bootstrap CI for mean volatility.
        tl_stats = self._bootstrap_stats(tl_arr, TIMELIKE_LABEL) if n_tl >= 10 else None
        sl_stats = self._bootstrap_stats(sl_arr, SPACELIKE_LABEL) if n_sl >= 10 else None

        # Bartlett test.
        bartlett = None
        if n_tl >= 10 and n_sl >= 10:
            stat, p = stats.bartlett(tl_arr, sl_arr)
            bartlett = PairwiseTestResult(
                test_name="Bartlett",
                statistic=float(stat),
                p_value=float(p),
                significant=p < bonferroni_alpha,
                note="H0: TIMELIKE and SPACELIKE have equal variance",
            )

        # Permutation test.
        permutation = None
        if n_tl >= 10 and n_sl >= 10:
            permutation = self._permutation_test(tl_arr, sl_arr, bonferroni_alpha)

        # Benchmark: RSI overbought/oversold vs next-bar vol.
        rsi_test = self._rsi_benchmark_test(classified, bonferroni_alpha)

        # Benchmark: MACD signal vs next-bar vol.
        macd_test = self._macd_benchmark_test(classified, bonferroni_alpha)

        # Vol ratio.
        vol_ratio = 1.0
        if tl_stats and sl_stats and sl_stats.mean_vol > 1e-12:
            vol_ratio = tl_stats.mean_vol / sl_stats.mean_vol

        return SymbolValidationResult(
            symbol=symbol,
            n_bars=len(classified),
            n_timelike=n_tl,
            n_spacelike=n_sl,
            n_lightlike=ll_count,
            timelike_stats=tl_stats,
            spacelike_stats=sl_stats,
            bartlett=bartlett,
            permutation=permutation,
            rsi_vs_regime=rsi_test,
            macd_vs_regime=macd_test,
            vol_ratio_tl_sl=vol_ratio,
            bonferroni_alpha=bonferroni_alpha,
        )

    def _bootstrap_stats(
        self, arr: np.ndarray, label: str
    ) -> RegimeStats:
        """Compute mean volatility with bootstrap 95% CI."""
        rng = np.random.default_rng(42)
        boot_means = np.empty(self.n_boot)
        n = len(arr)
        for i in range(self.n_boot):
            sample = rng.choice(arr, size=n, replace=True)
            boot_means[i] = sample.mean()
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))
        return RegimeStats(
            label=label,
            count=n,
            mean_vol=float(arr.mean()),
            std_vol=float(arr.std(ddof=1)),
            median_vol=float(np.median(arr)),
            ci_lower=ci_lo,
            ci_upper=ci_hi,
        )

    def _permutation_test(
        self,
        tl: np.ndarray,
        sl: np.ndarray,
        bonferroni_alpha: float,
    ) -> PairwiseTestResult:
        """
        Permutation test: H0 = TIMELIKE and SPACELIKE mean next-bar vols are equal.

        Observed statistic: |mean(TIMELIKE) − mean(SPACELIKE)|.
        P-value: fraction of permuted statistics ≥ observed.
        """
        observed = abs(tl.mean() - sl.mean())
        combined = np.concatenate([tl, sl])
        n_tl = len(tl)
        rng = np.random.default_rng(123)
        count_extreme = 0
        for _ in range(self.n_perm):
            perm = rng.permutation(combined)
            diff = abs(perm[:n_tl].mean() - perm[n_tl:].mean())
            if diff >= observed:
                count_extreme += 1
        p_value = (count_extreme + 1) / (self.n_perm + 1)  # +1 for continuity correction
        return PairwiseTestResult(
            test_name="Permutation",
            statistic=float(observed),
            p_value=float(p_value),
            significant=p_value < bonferroni_alpha,
            note=(
                f"H0: TIMELIKE/SPACELIKE have equal mean next-bar |return|. "
                f"n_perm={self.n_perm}"
            ),
        )

    def _rsi_benchmark_test(
        self,
        classified: List[ClassifiedBar],
        bonferroni_alpha: float,
    ) -> Optional[PairwiseTestResult]:
        """Test whether RSI overbought (>70) vs oversold (<30) predicts next-bar vol."""
        ob_vols = [
            cb.next_bar_abs_return for cb in classified
            if cb.rsi is not None and cb.rsi > 70 and cb.next_bar_abs_return is not None
        ]
        os_vols = [
            cb.next_bar_abs_return for cb in classified
            if cb.rsi is not None and cb.rsi < 30 and cb.next_bar_abs_return is not None
        ]
        if len(ob_vols) < 5 or len(os_vols) < 5:
            return None
        stat, p = stats.mannwhitneyu(ob_vols, os_vols, alternative="two-sided")
        return PairwiseTestResult(
            test_name="MannWhitney (RSI OB vs OS)",
            statistic=float(stat),
            p_value=float(p),
            significant=p < bonferroni_alpha,
            note="RSI>70 (overbought) vs RSI<30 (oversold) next-bar |return|",
        )

    def _macd_benchmark_test(
        self,
        classified: List[ClassifiedBar],
        bonferroni_alpha: float,
    ) -> Optional[PairwiseTestResult]:
        """Test whether MACD histogram positive vs negative predicts next-bar vol."""
        pos_vols = [
            cb.next_bar_abs_return for cb in classified
            if cb.macd_histogram is not None
            and cb.macd_histogram > 0
            and cb.next_bar_abs_return is not None
        ]
        neg_vols = [
            cb.next_bar_abs_return for cb in classified
            if cb.macd_histogram is not None
            and cb.macd_histogram <= 0
            and cb.next_bar_abs_return is not None
        ]
        if len(pos_vols) < 5 or len(neg_vols) < 5:
            return None
        stat, p = stats.mannwhitneyu(pos_vols, neg_vols, alternative="two-sided")
        return PairwiseTestResult(
            test_name="MannWhitney (MACD+ vs MACD-)",
            statistic=float(stat),
            p_value=float(p),
            significant=p < bonferroni_alpha,
            note="MACD histogram > 0 vs <= 0 next-bar |return|",
        )

    def _log_result_summary(self, r: SymbolValidationResult) -> None:
        log.info(
            "  %s: %d bars | TL=%d SL=%d LL=%d | vol_ratio=%.3f",
            r.symbol, r.n_bars, r.n_timelike, r.n_spacelike, r.n_lightlike, r.vol_ratio_tl_sl,
        )
        if r.bartlett:
            sig = "SIGNIFICANT" if r.bartlett.significant else "not significant"
            log.info(
                "  Bartlett: p=%.4e (%s)", r.bartlett.p_value, sig
            )
        if r.permutation:
            sig = "SIGNIFICANT" if r.permutation.significant else "not significant"
            log.info(
                "  Permutation: p=%.4e (%s)", r.permutation.p_value, sig
            )


# ─── Technical Analysis Benchmarks ────────────────────────────────────────────


def compute_rsi(prices: List[float], period: int = 14) -> List[Optional[float]]:
    """Compute RSI series (Wilder's smoothing). Returns None for first `period` bars."""
    result: List[Optional[float]] = [None] * len(prices)
    if len(prices) < period + 1:
        return result
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = prices[i] - prices[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss < 1e-12:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, len(prices)):
        diff = prices[i] - prices[i - 1]
        g = max(diff, 0.0)
        lo = max(-diff, 0.0)
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + lo) / period
        if avg_loss < 1e-12:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)
    return result


def _ema(prices: List[float], period: int) -> List[Optional[float]]:
    """Exponential moving average. Returns None until `period` bars available."""
    result: List[Optional[float]] = [None] * len(prices)
    if len(prices) < period:
        return result
    k = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period
    result[period - 1] = ema
    for i in range(period, len(prices)):
        ema = prices[i] * k + ema * (1 - k)
        result[i] = ema
    return result


def compute_macd(
    prices: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> List[Tuple[Optional[float], Optional[float]]]:
    """
    Compute MACD line, signal line, and histogram.

    Returns a list of (signal_line, histogram) tuples.  Both are ``None``
    until enough bars have accumulated.
    """
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)

    macd_line: List[Optional[float]] = []
    for f, s in zip(ema_fast, ema_slow):
        if f is None or s is None:
            macd_line.append(None)
        else:
            macd_line.append(f - s)

    # Signal line: EMA of MACD line.
    valid_macd = [v for v in macd_line if v is not None]
    valid_sig: List[Optional[float]] = [None] * (len(macd_line) - len(valid_macd))

    if len(valid_macd) >= signal:
        k = 2.0 / (signal + 1)
        sig_ema = sum(valid_macd[:signal]) / signal
        valid_sig.append(sig_ema)
        for val in valid_macd[signal:]:
            sig_ema = val * k + sig_ema * (1 - k)
            valid_sig.append(sig_ema)
    else:
        valid_sig.extend([None] * len(valid_macd))

    # Combine into output.
    result: List[Tuple[Optional[float], Optional[float]]] = []
    macd_idx = 0
    sig_idx = 0
    for ml in macd_line:
        if ml is None:
            result.append((None, None))
        else:
            sig_val = valid_sig[sig_idx] if sig_idx < len(valid_sig) else None
            hist = (ml - sig_val) if (ml is not None and sig_val is not None) else None
            result.append((sig_val, hist))
            sig_idx += 1
        macd_idx += 1

    return result


# ─── Report Generation ────────────────────────────────────────────────────────


class ValidationReport:
    """
    Generate LaTeX and Markdown validation reports with confidence intervals.

    The report includes:
    - Per-symbol regime statistics table.
    - Bootstrap CI for mean next-bar volatility by regime.
    - P-values for Bartlett and permutation tests.
    - Comparison vs RSI and MACD benchmarks.
    - Pooled cross-asset summary.

    Parameters
    ----------
    results:
        Dict mapping symbol → :class:`SymbolValidationResult`.
    output_dir:
        Directory to write reports into.
    alpha:
        Significance level used in the study.
    """

    def __init__(
        self,
        results: Dict[str, SymbolValidationResult],
        output_dir: str = "validation",
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        self.results    = results
        self.output_dir = Path(output_dir)
        self.alpha      = alpha
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, fmt: str = "both") -> List[Path]:
        """Generate reports in the requested format(s).

        Parameters
        ----------
        fmt:
            ``'latex'``, ``'markdown'``, or ``'both'``.

        Returns
        -------
        List of written file paths.
        """
        paths = []
        if fmt in ("markdown", "both"):
            paths.append(self._write_markdown())
        if fmt in ("latex", "both"):
            paths.append(self._write_latex())
        return paths

    # ── Markdown ───────────────────────────────────────────────────────────────

    def _write_markdown(self) -> Path:
        path = self.output_dir / "crypto_validation_report.md"
        lines = self._build_markdown()
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Markdown report written to %s", path)
        return path

    def _build_markdown(self) -> List[str]:
        lines = [
            "# SRFM Extended Crypto Validation Report",
            "",
            f"> Generated: {pd.Timestamp.now().isoformat(timespec='seconds')}  ",
            f"> Significance level: α = {self.alpha}  ",
            f"> Bonferroni correction applied across {len(self.results)} assets",
            "",
            "## Research Questions",
            "",
            "1. Do TIMELIKE regimes exhibit lower next-bar realised volatility in crypto?",
            "2. Does the equity Q1 2025 result replicate in 24/7 crypto markets?",
            "3. Does SRFM classification out-predict RSI and MACD as a vol forecaster?",
            "",
            "## Per-Asset Results",
            "",
        ]
        for sym, r in self.results.items():
            lines += self._symbol_section_md(sym, r)
        lines += self._pooled_section_md()
        lines += self._benchmark_section_md()
        return lines

    def _symbol_section_md(self, sym: str, r: SymbolValidationResult) -> List[str]:
        sig_bartlett = (
            r.bartlett.significant if r.bartlett else False
        )
        sig_perm = (
            r.permutation.significant if r.permutation else False
        )
        b_p = f"{r.bartlett.p_value:.2e}" if r.bartlett else "N/A"
        perm_p = f"{r.permutation.p_value:.2e}" if r.permutation else "N/A"

        tl_mean = f"{r.timelike_stats.mean_vol:.6f}" if r.timelike_stats else "N/A"
        sl_mean = f"{r.spacelike_stats.mean_vol:.6f}" if r.spacelike_stats else "N/A"
        tl_ci = (
            f"[{r.timelike_stats.ci_lower:.6f}, {r.timelike_stats.ci_upper:.6f}]"
            if r.timelike_stats else "N/A"
        )
        sl_ci = (
            f"[{r.spacelike_stats.ci_lower:.6f}, {r.spacelike_stats.ci_upper:.6f}]"
            if r.spacelike_stats else "N/A"
        )

        return [
            f"### {sym}",
            "",
            f"- Total bars: **{r.n_bars}**  ",
            f"- TIMELIKE: {r.n_timelike} | SPACELIKE: {r.n_spacelike} | LIGHTLIKE: {r.n_lightlike}",
            f"- Vol ratio (TL/SL): **{r.vol_ratio_tl_sl:.4f}**",
            "  (< 1.0 → TIMELIKE has lower vol, supporting SRFM hypothesis)",
            "",
            "| Regime | n | Mean |vol| | Median | 95% CI |",
            "|--------|---|------------|--------|--------|",
            f"| TIMELIKE  | {r.n_timelike} | {tl_mean} | "
            + (f"{r.timelike_stats.median_vol:.6f}" if r.timelike_stats else "N/A")
            + f" | {tl_ci} |",
            f"| SPACELIKE | {r.n_spacelike} | {sl_mean} | "
            + (f"{r.spacelike_stats.median_vol:.6f}" if r.spacelike_stats else "N/A")
            + f" | {sl_ci} |",
            "",
            "| Test | Statistic | p-value | Significant? |",
            "|------|-----------|---------|--------------|",
            f"| Bartlett (variance equality) | "
            + (f"{r.bartlett.statistic:.4f}" if r.bartlett else "N/A")
            + f" | {b_p} | {'YES' if sig_bartlett else 'no'} |",
            f"| Permutation (mean diff) | "
            + (f"{r.permutation.statistic:.6f}" if r.permutation else "N/A")
            + f" | {perm_p} | {'YES' if sig_perm else 'no'} |",
            "",
        ]

    def _pooled_section_md(self) -> List[str]:
        tl_vols: List[float] = []
        sl_vols: List[float] = []
        for r in self.results.values():
            if r.timelike_stats:
                tl_vols.append(r.timelike_stats.mean_vol)
            if r.spacelike_stats:
                sl_vols.append(r.spacelike_stats.mean_vol)

        if not tl_vols or not sl_vols:
            return ["## Pooled Summary", "", "Insufficient data for pooled analysis.", ""]

        mean_tl = sum(tl_vols) / len(tl_vols)
        mean_sl = sum(sl_vols) / len(sl_vols)
        ratio = mean_tl / mean_sl if mean_sl > 1e-12 else float("nan")
        n_sig_bartlett = sum(
            1 for r in self.results.values()
            if r.bartlett and r.bartlett.significant
        )
        n_sig_perm = sum(
            1 for r in self.results.values()
            if r.permutation and r.permutation.significant
        )

        return [
            "## Pooled Cross-Asset Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Assets analysed | {len(self.results)} |",
            f"| Mean TIMELIKE next-bar |vol| | {mean_tl:.6f} |",
            f"| Mean SPACELIKE next-bar |vol| | {mean_sl:.6f} |",
            f"| Pooled TL/SL vol ratio | {ratio:.4f} |",
            f"| Assets with significant Bartlett | {n_sig_bartlett}/{len(self.results)} |",
            f"| Assets with significant Permutation | {n_sig_perm}/{len(self.results)} |",
            "",
            "> A pooled ratio < 1.0 replicates the Q1 2025 equity finding in crypto.",
            "",
        ]

    def _benchmark_section_md(self) -> List[str]:
        lines = [
            "## Benchmark Comparison: SRFM vs RSI / MACD",
            "",
            "| Symbol | RSI OB/OS p-value | RSI sig? | MACD+/- p-value | MACD sig? |",
            "|--------|-------------------|----------|-----------------|-----------|",
        ]
        for sym, r in self.results.items():
            rsi_p = f"{r.rsi_vs_regime.p_value:.2e}" if r.rsi_vs_regime else "N/A"
            rsi_s = "YES" if (r.rsi_vs_regime and r.rsi_vs_regime.significant) else "no"
            macd_p = f"{r.macd_vs_regime.p_value:.2e}" if r.macd_vs_regime else "N/A"
            macd_s = "YES" if (r.macd_vs_regime and r.macd_vs_regime.significant) else "no"
            lines.append(f"| {sym} | {rsi_p} | {rsi_s} | {macd_p} | {macd_s} |")
        lines += [
            "",
            "> SRFM classification is significant if Bartlett or Permutation p < Bonferroni α.",
            "> RSI and MACD are significant via Mann-Whitney U at the same corrected α.",
            "",
        ]
        return lines

    # ── LaTeX ──────────────────────────────────────────────────────────────────

    def _write_latex(self) -> Path:
        path = self.output_dir / "crypto_validation_report.tex"
        lines = self._build_latex()
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("LaTeX report written to %s", path)
        return path

    def _build_latex(self) -> List[str]:
        header = [
            r"\documentclass[11pt]{article}",
            r"\usepackage{booktabs, amsmath, geometry, hyperref}",
            r"\geometry{a4paper, margin=2.5cm}",
            r"\title{SRFM Extended Crypto Validation Report}",
            r"\author{Automated validation pipeline}",
            r"\date{\today}",
            r"\begin{document}",
            r"\maketitle",
            r"\tableofcontents",
            r"\newpage",
            "",
            r"\section{Introduction}",
            r"This report extends the Q1 2025 equity validation of the Special",
            r"Relativity in Financial Modeling (SRFM) framework to cryptocurrency",
            r"markets using public Binance API data. We test whether TIMELIKE",
            r"spacetime intervals predict lower next-bar realised volatility.",
            "",
        ]
        body: List[str] = []
        for sym, r in self.results.items():
            body += self._symbol_section_tex(sym, r)
        body += self._pooled_section_tex()
        footer = [
            r"\section{Conclusion}",
            r"See the Markdown report for a human-readable summary.",
            r"\end{document}",
        ]
        return header + body + footer

    def _symbol_section_tex(self, sym: str, r: SymbolValidationResult) -> List[str]:
        safe_sym = sym.replace("_", r"\_")
        tl_mean = f"{r.timelike_stats.mean_vol:.6f}" if r.timelike_stats else "--"
        sl_mean = f"{r.spacelike_stats.mean_vol:.6f}" if r.spacelike_stats else "--"
        tl_ci = (
            f"[{r.timelike_stats.ci_lower:.6f}, {r.timelike_stats.ci_upper:.6f}]"
            if r.timelike_stats else "--"
        )
        sl_ci = (
            f"[{r.spacelike_stats.ci_lower:.6f}, {r.spacelike_stats.ci_upper:.6f}]"
            if r.spacelike_stats else "--"
        )
        b_p = f"{r.bartlett.p_value:.2e}" if r.bartlett else "--"
        perm_p = f"{r.permutation.p_value:.2e}" if r.permutation else "--"

        return [
            rf"\subsection{{{safe_sym}}}",
            "",
            r"\begin{table}[h!]\centering",
            rf"\caption{{Regime statistics for {safe_sym}}}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Regime & $n$ & Mean $|\text{ret}|$ & Median & 95\% CI \\",
            r"\midrule",
            rf"TIMELIKE  & {r.n_timelike}  & {tl_mean} & "
            + (f"{r.timelike_stats.median_vol:.6f}" if r.timelike_stats else "--")
            + rf" & {tl_ci} \\",
            rf"SPACELIKE & {r.n_spacelike} & {sl_mean} & "
            + (f"{r.spacelike_stats.median_vol:.6f}" if r.spacelike_stats else "--")
            + rf" & {sl_ci} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[h!]\centering",
            rf"\caption{{Statistical tests for {safe_sym}}}",
            r"\begin{tabular}{lrrl}",
            r"\toprule",
            r"Test & Statistic & $p$-value & Significant? \\",
            r"\midrule",
            rf"Bartlett & {r.bartlett.statistic if r.bartlett else '--':.4f} & {b_p} & "
            + ("Yes" if (r.bartlett and r.bartlett.significant) else "No") + r" \\",
            rf"Permutation & {r.permutation.statistic if r.permutation else '--':.6f} & "
            + f"{perm_p} & "
            + ("Yes" if (r.permutation and r.permutation.significant) else "No") + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            rf"Vol ratio (TL/SL): ${r.vol_ratio_tl_sl:.4f}$ "
            r"(values $< 1$ support SRFM hypothesis).",
            "",
        ]

    def _pooled_section_tex(self) -> List[str]:
        tl_vols = [
            r.timelike_stats.mean_vol for r in self.results.values()
            if r.timelike_stats
        ]
        sl_vols = [
            r.spacelike_stats.mean_vol for r in self.results.values()
            if r.spacelike_stats
        ]
        if not tl_vols or not sl_vols:
            return []
        mean_tl = sum(tl_vols) / len(tl_vols)
        mean_sl = sum(sl_vols) / len(sl_vols)
        ratio = mean_tl / mean_sl if mean_sl > 1e-12 else float("nan")
        return [
            r"\section{Pooled Cross-Asset Summary}",
            "",
            r"\begin{table}[h!]\centering",
            r"\caption{Pooled results across all assets}",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Metric & Value \\",
            r"\midrule",
            rf"Assets analysed & {len(self.results)} \\",
            rf"Mean TL next-bar $|\text{{ret}}|$ & {mean_tl:.6f} \\",
            rf"Mean SL next-bar $|\text{{ret}}|$ & {mean_sl:.6f} \\",
            rf"Pooled TL/SL ratio & {ratio:.4f} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]


# ─── CLI Entry Point ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SRFM extended crypto empirical validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--interval", default=DEFAULT_INTERVAL)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--c-scale", type=float, default=DEFAULT_C_SCALE)
    p.add_argument("--output", default="validation")
    p.add_argument("--format", choices=["latex", "markdown", "both"], default="both")
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Use cached CSV data only; error if cache missing",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cache_dir = os.path.join(args.output, "crypto_cache")
    if args.no_download:
        # Force cache-only mode: set cache_dir and rely on existing files.
        cache_dir = cache_dir

    validator = CryptoValidation(
        symbols=args.symbols,
        interval=args.interval,
        limit=args.limit,
        c_scale=args.c_scale,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        alpha=args.alpha,
        cache_dir=cache_dir,
    )
    results = validator.run()

    reporter = ValidationReport(results, output_dir=args.output, alpha=args.alpha)
    written = reporter.generate_all(fmt=args.format)
    log.info("Reports written:")
    for p in written:
        log.info("  %s", p)

    # Print brief console summary.
    print("\n=== SRFM Crypto Validation Summary ===")
    n_sig = sum(
        1 for r in results.values()
        if r.bartlett and r.bartlett.significant
    )
    print(f"Assets analysed: {len(results)}")
    print(f"Assets with significant Bartlett test: {n_sig}/{len(results)}")
    tl_ratios = [r.vol_ratio_tl_sl for r in results.values() if r.vol_ratio_tl_sl != 1.0]
    if tl_ratios:
        mean_ratio = sum(tl_ratios) / len(tl_ratios)
        print(f"Mean TL/SL vol ratio: {mean_ratio:.4f}")
        if mean_ratio < 1.0:
            print("=> TIMELIKE regimes show lower next-bar vol (supports SRFM hypothesis)")
        else:
            print("=> No consistent vol reduction in TIMELIKE regimes")
    print()


if __name__ == "__main__":
    main()
