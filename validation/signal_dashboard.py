"""
SRFM Signal Dashboard
======================
Real-time terminal display showing:
- Current bar classification (TIMELIKE/LIGHTLIKE/SPACELIKE) with colour coding
- Rolling fraction of TIMELIKE bars (momentum quality indicator)
- Spacetime interval time series (last 20 bars as sparkline)
- Beta (price velocity) meter
- Regime change alerts
- Portfolio weights (if optimizer connected)

Requires no external dependencies — uses only ANSI escape codes.

Usage
-----
    python validation/signal_dashboard.py

    # Or run the integrated demo (streams + dashboard together):
    python validation/signal_dashboard.py --demo

Dependencies
------------
    pip install numpy    # for sparkline numeric processing
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np


# ─── ANSI helpers ──────────────────────────────────────────────────────────────

_ESC = "\033["
_RESET = "\033[0m"

# Colours
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_WHITE = "\033[97m"
_DIM = "\033[2m"
_BOLD = "\033[1m"

# Cursor / screen control
_CLEAR_SCREEN = "\033[2J\033[H"
_CLEAR_LINE = "\033[2K\r"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_SAVE_CURSOR = "\033[s"
_RESTORE_CURSOR = "\033[u"


def _move_to(row: int, col: int) -> str:
    return f"\033[{row};{col}H"


def _color_for_interval(interval_type: str) -> str:
    return {
        "TIMELIKE": _GREEN,
        "LIGHTLIKE": _YELLOW,
        "SPACELIKE": _RED,
    }.get(interval_type, _WHITE)


# ─── Dashboard state ───────────────────────────────────────────────────────────


@dataclass
class DashboardState:
    """Mutable state for a single symbol's dashboard panel."""

    symbol: str
    current_price: float = 0.0
    prev_price: float = 0.0
    current_beta: float = 0.0
    interval_type: str = "UNKNOWN"
    timelike_fraction: float = 0.0
    bar_count: int = 0
    signal_history: Deque = field(default_factory=lambda: deque(maxlen=20))
    alerts: List[str] = field(default_factory=list)
    last_update: float = 0.0
    portfolio_weight: Optional[float] = None


# ─── Main dashboard class ──────────────────────────────────────────────────────


class SRFMDashboard:
    """
    ANSI terminal dashboard for real-time SRFM monitoring.

    Renders a compact panel per symbol showing:
    - Live price and interval type (colour-coded)
    - Beta velocity meter
    - Spacetime interval sparkline (last 20 bars)
    - TIMELIKE fraction progress bar
    - Recent alerts (regime changes, lightlike crossings)
    - Optional portfolio weight

    Parameters
    ----------
    symbols:
        List of ticker symbols to monitor.
    max_alerts:
        Maximum number of recent alerts to retain per symbol.
    """

    # Sparkline characters (ascending intensity)
    _SPARK_CHARS = " ▁▂▃▄▅▆▇█"
    _BETA_METER_WIDTH = 30
    _TF_BAR_WIDTH = 25
    _PANEL_WIDTH = 72

    def __init__(self, symbols: List[str], max_alerts: int = 5) -> None:
        self.symbols = symbols
        self.max_alerts = max_alerts
        self.states: Dict[str, DashboardState] = {
            s: DashboardState(symbol=s) for s in symbols
        }
        self._render_count = 0
        self._start_time = time.time()

    # ── Public update API ─────────────────────────────────────────────────────

    def update(self, signal) -> None:
        """
        Update dashboard state from a BarSignal.

        Parameters
        ----------
        signal:
            A BarSignal (or any object with the relevant attributes:
            symbol, close, beta, interval_type, spacetime_interval,
            momentum, regime_change, lightlike_crossing).
        """
        symbol = getattr(signal, "symbol", None)
        if symbol not in self.states:
            self.states[symbol] = DashboardState(symbol=symbol)

        state = self.states[symbol]
        state.prev_price = state.current_price
        state.current_price = float(getattr(signal, "close", state.current_price))
        state.current_beta = float(getattr(signal, "beta", 0.0))
        state.interval_type = str(getattr(signal, "interval_type", "UNKNOWN"))
        state.bar_count += 1
        state.signal_history.append(signal)
        state.last_update = time.time()

        # Update rolling TIMELIKE fraction
        history = list(state.signal_history)
        if history:
            state.timelike_fraction = sum(
                1 for s in history if getattr(s, "interval_type", "") == "TIMELIKE"
            ) / len(history)

        # Collect alerts
        if getattr(signal, "regime_change", False):
            state.alerts.append(f"REGIME CHANGE -> {state.interval_type}")
        if getattr(signal, "lightlike_crossing", False):
            state.alerts.append("LIGHTLIKE CROSSING (beta ~ 1.0)")
        # Trim to max
        state.alerts = state.alerts[-self.max_alerts:]

    def update_weight(self, symbol: str, weight: float) -> None:
        """Set the portfolio weight for a given symbol."""
        if symbol in self.states:
            self.states[symbol].portfolio_weight = weight

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self) -> None:
        """Redraw the full dashboard to stdout."""
        self._render_count += 1
        uptime = time.time() - self._start_time
        lines: List[str] = []

        # Header
        lines.append(_CLEAR_SCREEN)
        lines.append(
            f"{_BOLD}{_CYAN}"
            f"  SRFM Signal Dashboard"
            f"{_RESET}"
            f"  {_DIM}[uptime: {uptime:.0f}s | renders: {self._render_count}]{_RESET}"
        )
        lines.append(f"  {_DIM}{'─' * (self._PANEL_WIDTH - 2)}{_RESET}")

        for symbol in self.symbols:
            state = self.states.get(symbol)
            if state is None:
                continue
            lines.extend(self._render_panel(state))
            lines.append("")

        lines.append(
            f"  {_DIM}Press Ctrl+C to exit.{_RESET}"
        )

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    def _render_panel(self, state: DashboardState) -> List[str]:
        """Render a single symbol panel as a list of lines."""
        lines: List[str] = []
        color = _color_for_interval(state.interval_type)
        pw = self._PANEL_WIDTH

        # Panel border
        lines.append(f"  {_BOLD}┌{'─' * (pw - 4)}┐{_RESET}")

        # Row 1: symbol, price, interval type
        price_change = state.current_price - state.prev_price
        price_arrow = "▲" if price_change >= 0 else "▼"
        price_color = _GREEN if price_change >= 0 else _RED
        interval_label = f"{color}{_BOLD}{state.interval_type:^12}{_RESET}"
        price_str = (
            f"{price_color}{price_arrow} {state.current_price:>10,.2f}{_RESET}"
        )
        bar_count_str = f"{_DIM}bars: {state.bar_count:>5}{_RESET}"
        line = (
            f"  │ {_BOLD}{_CYAN}{state.symbol:<8}{_RESET} "
            f"{price_str}  {interval_label}  {bar_count_str}"
        )
        lines.append(self._pad_panel_line(line, pw))

        # Row 2: beta meter
        beta_meter = self.beta_meter(state.current_beta)
        beta_line = f"  │  beta={state.current_beta:6.4f}  {beta_meter}"
        lines.append(self._pad_panel_line(beta_line, pw))

        # Row 3: TIMELIKE fraction bar
        tf_bar = self._timelike_fraction_bar(state.timelike_fraction)
        tf_pct = f"{state.timelike_fraction * 100:5.1f}%"
        tf_line = f"  │  TIMELIKE {tf_pct}  {tf_bar}"
        lines.append(self._pad_panel_line(tf_line, pw))

        # Row 4: sparkline of spacetime intervals
        ds2_values = [
            float(getattr(s, "spacetime_interval", 0.0))
            for s in state.signal_history
        ]
        if ds2_values:
            spark = self.sparkline(ds2_values)
            spark_label = "ds² " + spark
        else:
            spark_label = "ds² (no data yet)"
        spark_line = f"  │  {spark_label}"
        lines.append(self._pad_panel_line(spark_line, pw))

        # Row 5: portfolio weight (if set)
        if state.portfolio_weight is not None:
            weight_bar_len = int(state.portfolio_weight * 20)
            weight_bar = _CYAN + "█" * weight_bar_len + _DIM + "░" * (20 - weight_bar_len) + _RESET
            w_line = f"  │  weight={state.portfolio_weight:.3f}  {weight_bar}"
            lines.append(self._pad_panel_line(w_line, pw))

        # Row 6+: alerts
        if state.alerts:
            for alert in state.alerts[-3:]:
                alert_line = f"  │  {_MAGENTA}! {alert}{_RESET}"
                lines.append(self._pad_panel_line(alert_line, pw))
        else:
            no_alert_line = f"  │  {_DIM}  (no recent alerts){_RESET}"
            lines.append(self._pad_panel_line(no_alert_line, pw))

        # Panel close
        lines.append(f"  {_BOLD}└{'─' * (pw - 4)}┘{_RESET}")
        return lines

    def _pad_panel_line(self, line: str, panel_width: int) -> str:
        """Pad a panel line to align the right border (ignoring ANSI codes)."""
        visible_len = _strip_ansi_len(line)
        pad_needed = (panel_width - 1) - visible_len
        if pad_needed > 0:
            return line + " " * pad_needed + _BOLD + "│" + _RESET
        return line + _BOLD + "│" + _RESET

    # ── Sparkline ─────────────────────────────────────────────────────────────

    def sparkline(self, values: List[float]) -> str:
        """
        Render a Unicode sparkline from a list of numeric values.

        Negative values (TIMELIKE) are shown in green; positive (SPACELIKE)
        in red. The magnitude is mapped to block height.

        Parameters
        ----------
        values:
            List of ds^2 values (negative = TIMELIKE, positive = SPACELIKE).

        Returns
        -------
        ANSI-coloured sparkline string.
        """
        if not values:
            return ""

        arr = np.array(values, dtype=float)

        # Separate magnitude for height; sign for colour
        abs_arr = np.abs(arr)
        v_min = float(abs_arr.min())
        v_max = float(abs_arr.max())
        v_range = v_max - v_min if v_max > v_min else 1.0

        chars = self._SPARK_CHARS
        n_levels = len(chars) - 1  # exclude the space char

        result_chars: List[str] = []
        for val, abs_val in zip(arr, abs_arr):
            norm = (abs_val - v_min) / v_range
            idx = min(int(norm * n_levels) + 1, n_levels)  # +1 to skip space at 0
            ch = chars[idx]
            color = _GREEN if val < 0 else (_YELLOW if abs(val) < 1e-6 else _RED)
            result_chars.append(f"{color}{ch}{_RESET}")

        return "".join(result_chars)

    # ── Beta meter ────────────────────────────────────────────────────────────

    def beta_meter(self, beta: float) -> str:
        """
        Render a horizontal velocity meter for the current beta value.

        The meter is divided into three zones:
        - [0, 0.9)   green  — TIMELIKE
        - [0.9, 1.1) yellow — approaching lightcone
        - [1.1, inf) red    — SPACELIKE

        The meter cap is at beta = 2.0.

        Parameters
        ----------
        beta:
            Current normalised price velocity (dimensionless).

        Returns
        -------
        ANSI-coloured bar string.
        """
        w = self._BETA_METER_WIDTH
        beta_capped = min(beta, 2.0)
        fill_count = int((beta_capped / 2.0) * w)
        fill_count = max(0, min(fill_count, w))

        # Color transition positions
        tl_end = int(0.45 * w)   # 0.9 / 2.0 * w
        ll_end = int(0.55 * w)   # 1.1 / 2.0 * w

        bar_chars: List[str] = []
        for i in range(w):
            if i < fill_count:
                if i < tl_end:
                    bar_chars.append(f"{_GREEN}█{_RESET}")
                elif i < ll_end:
                    bar_chars.append(f"{_YELLOW}█{_RESET}")
                else:
                    bar_chars.append(f"{_RED}█{_RESET}")
            else:
                bar_chars.append(f"{_DIM}░{_RESET}")

        # Lightcone marker at position tl_end
        bar_str = "".join(bar_chars)

        return f"[{bar_str}] {_DIM}c{_RESET}"

    # ── TIMELIKE fraction bar ─────────────────────────────────────────────────

    def _timelike_fraction_bar(self, fraction: float) -> str:
        """Render a progress bar representing TIMELIKE fraction."""
        w = self._TF_BAR_WIDTH
        filled = int(fraction * w)
        filled = max(0, min(filled, w))
        bar = (
            _GREEN + "█" * filled
            + _DIM + "░" * (w - filled)
            + _RESET
        )
        return f"[{bar}]"

    # ── Screen utilities ──────────────────────────────────────────────────────

    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        sys.stdout.write(_CLEAR_SCREEN)
        sys.stdout.flush()

    def hide_cursor(self) -> None:
        sys.stdout.write(_HIDE_CURSOR)
        sys.stdout.flush()

    def show_cursor(self) -> None:
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()


# ─── ANSI strip helper ─────────────────────────────────────────────────────────


def _strip_ansi_len(s: str) -> int:
    """Return the visible (printable) length of a string with ANSI codes."""
    import re
    ansi_escape = re.compile(r"\033\[[0-9;]*[mABCDEFGHJKSTfhilmnprsu]")
    return len(ansi_escape.sub("", s))


# ─── Integrated demo ───────────────────────────────────────────────────────────


async def run_dashboard_demo(
    symbols: Optional[List[str]] = None,
    duration_secs: float = 60.0,
    bar_period_secs: float = 3.0,
    c_financial: float = 0.1,
    refresh_hz: float = 2.0,
) -> None:
    """
    Run the signal dashboard with simulated tick feeds.

    Launches one SimulatedTickFeed per symbol and routes BarSignal events
    into the SRFMDashboard, refreshing the display at refresh_hz Hz.

    Parameters
    ----------
    symbols:
        Symbols to display. Defaults to ["BTC/USD", "ETH/USD", "SPY"].
    duration_secs:
        How long to run the demo.
    bar_period_secs:
        Bar period for bar assembly.
    c_financial:
        Financial speed of light.
    refresh_hz:
        Dashboard refresh rate in Hz.
    """
    # Import locally to avoid circular if this module is imported elsewhere
    from tick_streamer import SimulatedTickFeed, SRFMTickProcessor  # noqa: PLC0415

    if symbols is None:
        symbols = ["BTC/USD", "ETH/USD", "SPY"]

    dashboard = SRFMDashboard(symbols=symbols)
    dashboard.hide_cursor()

    # Per-symbol feeds and processors
    feeds = {}
    processors = {}
    initial_prices = {"BTC/USD": 50_000.0, "ETH/USD": 3_000.0, "SPY": 480.0}

    for sym in symbols:
        feeds[sym] = SimulatedTickFeed(
            symbol=sym,
            initial_price=initial_prices.get(sym, 100.0),
            volatility=0.002,
            tick_interval_ms=50,
            c_financial=c_financial,
            seed=hash(sym) % (2**32),
        )

        def make_callback(s: str):
            def cb(sig):
                dashboard.update(sig)
            return cb

        processors[sym] = SRFMTickProcessor(
            bar_period_secs=bar_period_secs,
            c_financial=c_financial,
            signal_callback=make_callback(sym),
        )

    deadline = time.time() + duration_secs
    refresh_interval = 1.0 / refresh_hz

    async def feed_task(sym: str):
        async for tick in feeds[sym].stream():
            if time.time() >= deadline:
                break
            processors[sym].process_tick(tick)

    async def render_task():
        while time.time() < deadline:
            dashboard.render()
            await asyncio.sleep(refresh_interval)

    try:
        tasks = [asyncio.create_task(feed_task(s)) for s in symbols]
        tasks.append(asyncio.create_task(render_task()))
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        dashboard.show_cursor()
        dashboard.clear_screen()

    print("\nDashboard demo complete.")
    for sym in symbols:
        p = processors[sym]
        tf = p.timelike_fraction(last_n=p.bar_count or 1)
        print(f"  {sym:<12}  bars={p.bar_count:4d}  TIMELIKE={tf:.3f}")


# ─── Standalone demo without external imports ──────────────────────────────────


@dataclass
class _MockBarSignal:
    """Minimal BarSignal mock for standalone dashboard demo."""
    symbol: str
    close: float
    beta: float
    interval_type: str
    spacetime_interval: float
    momentum: float
    regime_change: bool
    lightlike_crossing: bool


async def run_standalone_demo(
    symbols: Optional[List[str]] = None,
    duration_secs: float = 30.0,
    refresh_hz: float = 2.0,
) -> None:
    """
    Run the dashboard with purely synthesised signals — no imports from
    tick_streamer required.  Used when the module is invoked directly.
    """
    if symbols is None:
        symbols = ["BTC/USD", "ETH/USD", "SPY"]

    dashboard = SRFMDashboard(symbols=symbols)
    dashboard.hide_cursor()

    rng = np.random.default_rng(42)
    prices = {s: p for s, p in zip(symbols, [50_000.0, 3_000.0, 480.0])}
    C = 0.1
    c_dt = C * 60.0  # c * bar_period
    interval_types_list = ["TIMELIKE", "SPACELIKE", "LIGHTLIKE"]
    prev_interval: dict = {s: "TIMELIKE" for s in symbols}

    deadline = time.time() + duration_secs
    refresh_interval = 1.0 / refresh_hz

    async def generate_signals():
        while time.time() < deadline:
            for sym in symbols:
                # Regime-switching price model
                sigma = rng.choice([c_dt * 0.4, c_dt * 1.8], p=[0.65, 0.35])
                dp = float(rng.normal(0, sigma))
                prices[sym] = max(0.01, prices[sym] * (1.0 + dp / prices[sym]))
                beta = abs(dp) / c_dt
                ds2 = dp ** 2 - c_dt ** 2
                if abs(beta - 1.0) < 0.05:
                    itype = "LIGHTLIKE"
                elif ds2 < 0:
                    itype = "TIMELIKE"
                else:
                    itype = "SPACELIKE"

                regime_change = (
                    prev_interval[sym] != itype
                    and itype != "LIGHTLIKE"
                    and prev_interval[sym] != "LIGHTLIKE"
                )
                lightlike_crossing = abs(beta - 1.0) < 0.05
                prev_interval[sym] = itype

                sig = _MockBarSignal(
                    symbol=sym,
                    close=prices[sym],
                    beta=beta,
                    interval_type=itype,
                    spacetime_interval=ds2,
                    momentum=float(rng.normal(beta * 0.5, 0.02)),
                    regime_change=regime_change,
                    lightlike_crossing=lightlike_crossing,
                )
                dashboard.update(sig)
            await asyncio.sleep(0.5)

    async def render_loop():
        while time.time() < deadline:
            dashboard.render()
            await asyncio.sleep(refresh_interval)

    try:
        await asyncio.gather(
            asyncio.create_task(generate_signals()),
            asyncio.create_task(render_loop()),
            return_exceptions=True,
        )
    finally:
        dashboard.show_cursor()
        dashboard.clear_screen()

    print("\nDashboard demo complete.")


# ─── Entry point ───────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SRFM Signal Dashboard — real-time terminal monitoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run with integrated tick_streamer (requires tick_streamer.py in path)",
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USD", "ETH/USD", "SPY"],
        help="Symbols to monitor",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Demo duration in seconds",
    )
    p.add_argument(
        "--refresh-hz",
        type=float,
        default=2.0,
        help="Dashboard refresh rate in Hz",
    )
    p.add_argument(
        "--bar-period",
        type=float,
        default=3.0,
        help="Bar period in seconds (demo mode only)",
    )
    p.add_argument(
        "--c",
        type=float,
        default=0.1,
        help="Financial speed of light",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    try:
        if args.demo:
            asyncio.run(
                run_dashboard_demo(
                    symbols=args.symbols,
                    duration_secs=args.duration,
                    bar_period_secs=args.bar_period,
                    c_financial=args.c,
                    refresh_hz=args.refresh_hz,
                )
            )
        else:
            asyncio.run(
                run_standalone_demo(
                    symbols=args.symbols,
                    duration_secs=args.duration,
                    refresh_hz=args.refresh_hz,
                )
            )
    except KeyboardInterrupt:
        sys.stdout.write(_SHOW_CURSOR)
        print("\nInterrupted.")
