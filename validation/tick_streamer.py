"""
Real-Time Tick Streamer
========================
Connects to market data feeds and generates real-time SRFM signals:
- Bar-by-bar TIMELIKE/SPACELIKE classification
- Rolling momentum signal
- Alert on lightlike boundary crossings (market regime change signal)

Data sources supported:
- Yahoo Finance (via yfinance - free, delayed)
- Simulated (for testing without API access)
- Generic WebSocket feed (implement your own connector)

Usage
-----
    python validation/tick_streamer.py

    # Or import and use programmatically:
    import asyncio
    from validation.tick_streamer import run_streaming_demo
    asyncio.run(run_streaming_demo(duration_secs=30.0))

Dependencies
------------
    pip install numpy
    pip install yfinance    # optional — for live Yahoo Finance feed
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Callable, Deque, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Core data structures ──────────────────────────────────────────────────────


@dataclass
class Tick:
    """A single market tick event."""

    symbol: str
    timestamp: float    # Unix timestamp (seconds)
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class BarSignal:
    """Completed OHLCV bar with SRFM-derived signals."""

    symbol: str
    timestamp: float    # Bar open timestamp (Unix seconds)
    open: float
    high: float
    low: float
    close: float
    volume: float

    # SRFM signals
    beta: float                 # Normalised price velocity |dp| / (c * dt)
    interval_type: str          # "TIMELIKE" | "LIGHTLIKE" | "SPACELIKE"
    spacetime_interval: float   # ds^2 = dp^2 - (c*dt)^2
    momentum: float             # Rolling momentum signal (dimensionless)

    # Alert flags
    regime_change: bool         # True on TIMELIKE <-> SPACELIKE transition
    lightlike_crossing: bool    # True when |beta - 1.0| < threshold


# ─── Simulated tick feed ───────────────────────────────────────────────────────


class SimulatedTickFeed:
    """
    Generates realistic synthetic tick data for testing.

    The price model uses a regime-switching process:
    - Calm regime: small fractional moves (TIMELIKE-dominant)
    - Volatile regime: large fractional moves (SPACELIKE-dominant)
    Regimes switch randomly with probability p_switch per bar.

    Parameters
    ----------
    symbol:
        Ticker label for emitted ticks.
    initial_price:
        Starting price level.
    volatility:
        Fractional price standard deviation per tick (base level).
    tick_interval_ms:
        Milliseconds between synthetic ticks.
    c_financial:
        Financial speed of light (must match the processor's value).
    seed:
        Random seed for reproducibility; None for non-deterministic.
    """

    _CALM_VOL_FACTOR = 0.4      # calm regime vol = base_vol * factor
    _VOLATILE_VOL_FACTOR = 2.5  # volatile regime vol = base_vol * factor
    _P_REGIME_SWITCH = 0.02     # probability of switching regime per tick

    def __init__(
        self,
        symbol: str = "BTC/USD",
        initial_price: float = 50_000.0,
        volatility: float = 0.001,
        tick_interval_ms: int = 100,
        c_financial: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.symbol = symbol
        self.price = initial_price
        self.volatility = volatility
        self.tick_interval_ms = tick_interval_ms
        self.c_financial = c_financial
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._in_volatile_regime = False
        self._tick_count = 0

    def _next_price(self) -> Tuple[float, float]:
        """Return (new_price, volume) for the next tick."""
        # Regime switching
        if self._rng.random() < self._P_REGIME_SWITCH:
            self._in_volatile_regime = not self._in_volatile_regime

        factor = (
            self._VOLATILE_VOL_FACTOR
            if self._in_volatile_regime
            else self._CALM_VOL_FACTOR
        )
        sigma = self.volatility * factor

        # Fat-tailed shock: mix of normal and occasional jump
        if self._rng.random() < 0.02:  # 2% chance of jump
            jump_size = self._np_rng.exponential(sigma * 5)
            shock = jump_size * (1 if self._rng.random() > 0.5 else -1)
        else:
            shock = float(self._np_rng.normal(0.0, sigma))

        # Drift: very small mean reversion
        drift = -0.0001 * (self.price - 50_000.0) / 50_000.0
        self.price = max(0.01, self.price * (1.0 + drift + shock))

        volume = float(self._np_rng.exponential(100.0) * (1.0 + factor))
        return self.price, volume

    async def stream(self) -> AsyncIterator[Tick]:
        """
        Stream ticks indefinitely as an async generator.

        Yields one Tick per tick_interval_ms, simulating real-time arrival.
        Use `async for tick in feed.stream()` in an async context.
        """
        interval_sec = self.tick_interval_ms / 1000.0
        start_time = time.time()

        while True:
            self._tick_count += 1
            price, volume = self._next_price()
            spread = price * 0.0001
            tick = Tick(
                symbol=self.symbol,
                timestamp=time.time(),
                price=price,
                volume=volume,
                bid=price - spread / 2,
                ask=price + spread / 2,
            )
            yield tick
            await asyncio.sleep(interval_sec)


# ─── Bar accumulator ───────────────────────────────────────────────────────────


@dataclass
class _BarAccumulator:
    """Accumulates ticks into a single OHLCV bar."""

    open_time: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    total_volume: float
    tick_count: int = 0

    @classmethod
    def start(cls, tick: Tick) -> "_BarAccumulator":
        return cls(
            open_time=tick.timestamp,
            open_price=tick.price,
            high_price=tick.price,
            low_price=tick.price,
            close_price=tick.price,
            total_volume=tick.volume,
            tick_count=1,
        )

    def update(self, tick: Tick) -> None:
        self.high_price = max(self.high_price, tick.price)
        self.low_price = min(self.low_price, tick.price)
        self.close_price = tick.price
        self.total_volume += tick.volume
        self.tick_count += 1


# ─── SRFM tick processor ───────────────────────────────────────────────────────


class SRFMTickProcessor:
    """
    Processes ticks into bars and generates SRFM signals.

    For each completed bar the processor:
    1. Computes beta = |dp| / (c * dt)
    2. Classifies the bar as TIMELIKE, LIGHTLIKE, or SPACELIKE
    3. Computes ds^2 = dp^2 - (c*dt)^2
    4. Derives a rolling momentum signal from the last 5 bars
    5. Raises alert flags for regime changes and lightlike crossings

    Parameters
    ----------
    bar_period_secs:
        Duration of one bar in seconds.
    c_financial:
        Financial speed of light.
    signal_callback:
        Optional callable invoked with each BarSignal as it is generated.
    """

    _LIGHTLIKE_THRESHOLD: float = 0.01  # |beta - 1| < threshold => LIGHTLIKE
    _MOMENTUM_WINDOW: int = 5

    def __init__(
        self,
        bar_period_secs: float = 60.0,
        c_financial: float = 0.1,
        signal_callback: Optional[Callable[[BarSignal], None]] = None,
    ) -> None:
        self.bar_period_secs = bar_period_secs
        self.c = c_financial
        self.signal_callback = signal_callback

        self._current_bar: Optional[_BarAccumulator] = None
        self._bar_history: Deque[BarSignal] = deque(maxlen=100)
        self._beta_history: Deque[float] = deque(maxlen=self._MOMENTUM_WINDOW)
        self._prev_interval_type: Optional[str] = None
        self._total_bars: int = 0

    def process_tick(self, tick: Tick) -> Optional[BarSignal]:
        """
        Ingest a single tick and return a BarSignal if a bar completes.

        Parameters
        ----------
        tick:
            The incoming market tick.

        Returns
        -------
        BarSignal if a new bar was completed; None otherwise.
        """
        if self._current_bar is None:
            self._current_bar = _BarAccumulator.start(tick)
            return None

        elapsed = tick.timestamp - self._current_bar.open_time

        if elapsed < self.bar_period_secs:
            self._current_bar.update(tick)
            return None

        # Bar complete — build signal
        bar = self._current_bar
        self._current_bar = _BarAccumulator.start(tick)

        dp = bar.close_price - bar.open_price
        dt = max(elapsed, 1e-6)

        interval_type, ds_squared = self._classify_bar(dp, dt)
        beta = abs(dp) / (self.c * dt)
        momentum = self._compute_momentum(beta)

        regime_change = (
            self._prev_interval_type is not None
            and self._prev_interval_type != interval_type
            and interval_type != "LIGHTLIKE"
            and self._prev_interval_type != "LIGHTLIKE"
        )
        lightlike_crossing = abs(beta - 1.0) < self._LIGHTLIKE_THRESHOLD

        signal = BarSignal(
            symbol=tick.symbol,
            timestamp=bar.open_time,
            open=bar.open_price,
            high=bar.high_price,
            low=bar.low_price,
            close=bar.close_price,
            volume=bar.total_volume,
            beta=beta,
            interval_type=interval_type,
            spacetime_interval=ds_squared,
            momentum=momentum,
            regime_change=regime_change,
            lightlike_crossing=lightlike_crossing,
        )

        self._beta_history.append(beta)
        self._bar_history.append(signal)
        self._prev_interval_type = interval_type
        self._total_bars += 1

        if self.signal_callback is not None:
            self.signal_callback(signal)

        return signal

    def _classify_bar(self, dp: float, dt: float) -> Tuple[str, float]:
        """
        Classify a bar and compute the spacetime interval.

        Parameters
        ----------
        dp:
            Price change (signed).
        dt:
            Bar duration in seconds.

        Returns
        -------
        Tuple of (interval_type_string, ds_squared).
        """
        dp2 = dp ** 2
        c_dt_2 = (self.c * dt) ** 2
        ds2 = dp2 - c_dt_2

        beta = math.sqrt(dp2) / (self.c * dt) if dt > 0 else 0.0

        if abs(beta - 1.0) < self._LIGHTLIKE_THRESHOLD:
            return "LIGHTLIKE", ds2
        if ds2 < 0.0:
            return "TIMELIKE", ds2
        return "SPACELIKE", ds2

    def _compute_momentum(self, current_beta: float) -> float:
        """
        Rolling momentum signal using the past N bars.

        The signal is the exponentially weighted mean of beta values, sign-
        corrected by the direction of the most recent price move (positive
        beta = upward pressure).

        Returns a value in (-inf, +inf); positive => upward momentum.
        """
        if len(self._beta_history) == 0:
            return 0.0

        betas = np.array(list(self._beta_history) + [current_beta])
        # Exponential weights — more recent bars count more
        weights = np.exp(np.linspace(-1.0, 0.0, len(betas)))
        weights /= weights.sum()
        return float(np.dot(weights, betas))

    @property
    def bar_count(self) -> int:
        return self._total_bars

    @property
    def recent_signals(self) -> List[BarSignal]:
        return list(self._bar_history)

    def timelike_fraction(self, last_n: int = 20) -> float:
        """Fraction of the last N bars classified TIMELIKE."""
        signals = list(self._bar_history)[-last_n:]
        if not signals:
            return 0.0
        return sum(1 for s in signals if s.interval_type == "TIMELIKE") / len(signals)


# ─── Yahoo Finance live feed ───────────────────────────────────────────────────


class YahooFinanceFeed:
    """
    Fetch recent 1-minute bars from Yahoo Finance as a pseudo-stream.

    Requires: pip install yfinance

    This is a polling-based feed that downloads the last N minutes of
    1-minute OHLCV data and replays it as ticks, advancing the clock
    at the specified rate.

    Parameters
    ----------
    symbol:
        Yahoo Finance ticker symbol (e.g. "AAPL", "BTC-USD").
    lookback_mins:
        Number of 1-minute bars to fetch on each poll.
    poll_interval_secs:
        How often to re-fetch from Yahoo Finance (seconds).
    """

    def __init__(
        self,
        symbol: str = "AAPL",
        lookback_mins: int = 60,
        poll_interval_secs: float = 30.0,
    ) -> None:
        self.symbol = symbol
        self.lookback_mins = lookback_mins
        self.poll_interval_secs = poll_interval_secs

    async def stream(self) -> AsyncIterator[Tick]:
        """Stream ticks from Yahoo Finance 1-minute OHLCV bars."""
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "yfinance is required for YahooFinanceFeed. "
                "Install with: pip install yfinance"
            )

        seen_timestamps: set = set()

        while True:
            try:
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(period="1d", interval="1m")
                if df.empty:
                    log.warning("YahooFinanceFeed: empty DataFrame for %s", self.symbol)
                    await asyncio.sleep(self.poll_interval_secs)
                    continue

                for ts, row in df.iterrows():
                    ts_unix = float(ts.timestamp())
                    if ts_unix in seen_timestamps:
                        continue
                    seen_timestamps.add(ts_unix)

                    # Emit close-price tick
                    yield Tick(
                        symbol=self.symbol,
                        timestamp=ts_unix,
                        price=float(row["Close"]),
                        volume=float(row["Volume"]),
                        bid=float(row["Low"]),
                        ask=float(row["High"]),
                    )
                    await asyncio.sleep(0.01)

            except Exception as exc:  # noqa: BLE001
                log.error("YahooFinanceFeed error: %s", exc)

            await asyncio.sleep(self.poll_interval_secs)


# ─── Streaming demo ─────────────────────────────────────────────────────────────


async def run_streaming_demo(
    symbol: str = "BTC/USD",
    duration_secs: float = 60.0,
    print_signals: bool = True,
    bar_period_secs: float = 5.0,
    c_financial: float = 0.1,
    tick_interval_ms: int = 100,
) -> List[BarSignal]:
    """
    Demo: stream simulated ticks and print SRFM bar signals.

    Parameters
    ----------
    symbol:
        Ticker label displayed in output.
    duration_secs:
        How many seconds of simulated time to run.
    print_signals:
        If True, print each BarSignal to stdout as it is generated.
    bar_period_secs:
        Bar duration in seconds (shorter = more bars in the demo).
    c_financial:
        Financial speed of light for SRFM classification.
    tick_interval_ms:
        Simulated milliseconds between ticks.

    Returns
    -------
    List of all BarSignal objects generated during the demo.
    """
    feed = SimulatedTickFeed(
        symbol=symbol,
        initial_price=50_000.0,
        volatility=0.002,
        tick_interval_ms=tick_interval_ms,
        c_financial=c_financial,
        seed=123,
    )

    signals_collected: List[BarSignal] = []

    def on_signal(sig: BarSignal) -> None:
        signals_collected.append(sig)
        if print_signals:
            _print_bar_signal(sig)

    processor = SRFMTickProcessor(
        bar_period_secs=bar_period_secs,
        c_financial=c_financial,
        signal_callback=on_signal,
    )

    print()
    print("=" * 72)
    print(f"  SRFM Real-Time Tick Streamer — {symbol}")
    print(f"  Bar period: {bar_period_secs}s  |  c = {c_financial}  |  Duration: {duration_secs}s")
    print("=" * 72)
    if print_signals:
        print(
            f"  {'Time':>8}  {'Type':>10}  {'Beta':>8}  {'ds^2':>12}  "
            f"{'Momentum':>10}  {'Alerts'}"
        )
        print("  " + "-" * 68)

    deadline = time.time() + duration_secs

    async for tick in feed.stream():
        if time.time() >= deadline:
            break
        processor.process_tick(tick)

    print()
    print(f"  Stream ended. Bars generated: {processor.bar_count}")

    if processor.bar_count > 0:
        tf = processor.timelike_fraction(last_n=processor.bar_count)
        all_sigs = processor.recent_signals
        regime_changes = sum(1 for s in all_sigs if s.regime_change)
        lightlike_crossings = sum(1 for s in all_sigs if s.lightlike_crossing)

        print(f"  TIMELIKE fraction    : {tf:.3f}  ({tf*100:.1f}%)")
        print(f"  Regime changes       : {regime_changes}")
        print(f"  Lightlike crossings  : {lightlike_crossings}")

        type_counts: dict[str, int] = {"TIMELIKE": 0, "LIGHTLIKE": 0, "SPACELIKE": 0}
        for s in all_sigs:
            type_counts[s.interval_type] = type_counts.get(s.interval_type, 0) + 1
        total = sum(type_counts.values())
        for itype, cnt in type_counts.items():
            pct = 100.0 * cnt / total if total > 0 else 0.0
            bar = "#" * int(pct / 2)
            print(f"  {itype:<12} : {cnt:4d} bars  ({pct:5.1f}%)  {bar}")

    print("=" * 72)
    return signals_collected


def _print_bar_signal(sig: BarSignal) -> None:
    """Print a single BarSignal with ANSI colour coding."""
    _TIMELIKE_COLOR = "\033[92m"   # green
    _LIGHTLIKE_COLOR = "\033[93m"  # yellow
    _SPACELIKE_COLOR = "\033[91m"  # red
    _ALERT_COLOR = "\033[95m"      # magenta
    _RESET = "\033[0m"

    color = {
        "TIMELIKE": _TIMELIKE_COLOR,
        "LIGHTLIKE": _LIGHTLIKE_COLOR,
        "SPACELIKE": _SPACELIKE_COLOR,
    }.get(sig.interval_type, _RESET)

    ts_str = datetime.fromtimestamp(sig.timestamp).strftime("%H:%M:%S")
    itype_padded = f"{sig.interval_type:>10}"

    alerts = []
    if sig.regime_change:
        alerts.append(f"{_ALERT_COLOR}REGIME_CHANGE{_RESET}")
    if sig.lightlike_crossing:
        alerts.append(f"{_LIGHTLIKE_COLOR}LIGHTLIKE_CROSS{_RESET}")
    alert_str = "  ".join(alerts)

    print(
        f"  {ts_str:>8}  "
        f"{color}{itype_padded}{_RESET}  "
        f"{sig.beta:>8.4f}  "
        f"{sig.spacetime_interval:>12.4f}  "
        f"{sig.momentum:>10.4f}  "
        f"{alert_str}"
    )


# ─── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    asyncio.run(
        run_streaming_demo(
            symbol="BTC/USD",
            duration_secs=60.0,
            print_signals=True,
            bar_period_secs=5.0,
            c_financial=0.1,
            tick_interval_ms=50,
        )
    )
