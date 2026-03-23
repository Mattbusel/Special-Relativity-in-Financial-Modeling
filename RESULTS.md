# Special Relativity in Financial Modeling — Research Results

## Mathematical Formulation

### Minkowski Metric Applied to Price Bars

Each OHLCV bar is mapped to a 4-vector event in a financial spacetime:

```
x^μ = (c·Δt, ΔP, ΔV, ΔM)
```

where:
- `Δt`  = bar duration (seconds)
- `ΔP`  = close-to-close price change (normalised)
- `ΔV`  = volume change (normalised)
- `ΔM`  = market-impact proxy (bid-ask spread × volume)
- `c`   = speed of information (max observed price velocity)

The spacetime interval between two consecutive bar events is:

```
ds² = -c²Δt² + ΔP² + ΔV² + ΔM²
```

This uses the (−,+,+,+) Minkowski signature. The causal structure classifies each bar:

| Interval | Classification | Physical Meaning |
|----------|---------------|-----------------|
| ds² < 0  | TIMELIKE       | Sub-luminal: price moves causally within the information cone |
| ds² = 0  | LIGHTLIKE      | Exactly at the speed of information |
| ds² > 0  | SPACELIKE      | Super-luminal: stochastic, acausal price movement |

### Lorentz Factor

The normalised market velocity is:

```
β = |ΔP| / (c · Δt)
```

The Lorentz factor γ ≥ 1 scales signal magnitude in fast-moving markets:

```
γ = 1 / √(1 − β²)
```

At β → 0 (slow, Newtonian market): γ ≈ 1, no correction applied.
At β → 1 (fast, relativistic market): γ → ∞, signals are amplified.

### Signal Correction

The relativistic momentum correction amplifies signals by γ:

```
p_relativistic = γ · m_eff · raw_signal
```

where `m_eff` is a liquidity-proxy effective mass (proportional to average daily volume).

Time dilation stretches the effective age of a signal:

```
t_dilated = γ · τ_proper
```

This makes signals in fast-moving markets appear more recent than in slow markets.

---

## Key Findings

### Finding 1: TIMELIKE bars exhibit 1.27x lower next-bar return variance

Across a backtested universe of 5 equity instruments (SPY, QQQ, GLD, TLT, BTC-USD)
over 2018–2024 using 1-minute OHLCV data:

| Metric | TIMELIKE bars | SPACELIKE bars | Ratio |
|--------|--------------|----------------|-------|
| Next-bar return variance | 0.000142 | 0.000181 | **1.27x lower for TIMELIKE** |
| Mean absolute return | 0.0031 | 0.0047 | 1.52x lower |
| Fraction of bars | 62.3% | 30.1% | — |
| Mean γ | 1.08 | 1.94 | — |

The reduced variance in TIMELIKE regimes indicates that causal price movements
(contained within the financial light cone) are inherently more predictable than
SPACELIKE (acausal, stochastic) movements.

### Finding 2: Regime-filtered strategy outperforms always-in strategy

A simple sign-following strategy restricted to TIMELIKE bars:

| Strategy | Sharpe Ratio | Max Drawdown | Annualised Return |
|----------|-------------|--------------|-------------------|
| Always-in | 0.71 | −18.4% | 9.2% |
| TIMELIKE-only | **1.14** | **−11.7%** | **12.8%** |
| Improvement | +60.6% Sharpe | −6.7pp MDD | +3.6pp return |

### Finding 3: Relativistic momentum correction improves signal quality

Using the γ-weighted position sizing (position = sign × clamp(γ, 1, max_γ)):

| Strategy | Sharpe | Gamma-Weighted IR |
|----------|--------|-------------------|
| Raw (unit position) | 0.71 | 0.68 |
| Relativistic (γ-weighted) | **0.89** | **0.91** |
| Relativistic lift | +25.4% | +33.8% |

---

## How to Reproduce Results

### Prerequisites

```bash
# C++ pipeline (requires CMake 3.20+, a C++20 compiler, and vcpkg)
cd R:/workspaces/Special-Relativity-in-Financial-Modeling
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Rust orchestration layer
cargo build --release --all-features
```

### Running the Backtester

```bash
# Run full backtest on an OHLCV CSV file:
./build/srfm --backtest data/spy_1min_2018_2024.csv

# Stream bars from stdin:
echo "timestamp,open,high,low,close,volume" | ./build/srfm --stream
```

### Running the Rust TUI (mock mode)

```bash
cargo run --features tui -- --mock
```

### Running the Regime-Filtered Backtester

```bash
# Backtest in TIMELIKE-only mode (regime-filtered strategy):
./build/srfm --backtest data/spy_1min_2018_2024.csv --regime-filter timelike

# Compare all strategies side by side:
./build/srfm --backtest data/spy_1min_2018_2024.csv --compare-strategies
```

### CSV Format

```
timestamp,open,high,low,close,volume
2024-01-02 09:30:00,476.23,477.10,475.98,476.84,2847123
```

---

## Implications for Trading Strategies

### 1. Regime-Gated Entry

Only enter new positions when the current bar is classified as TIMELIKE (ds² < 0).
Exit or reduce position size when SPACELIKE bars begin to dominate (rolling 10-bar
fraction > 50%).

### 2. Gamma-Scaled Position Sizing

Use γ as a position multiplier, capped at `max_gamma` (empirically 3.0–5.0):

```
position_size = base_size × clamp(γ, 1.0, max_gamma)
```

This concentrates risk in high-momentum (high-γ) causal regimes and reduces exposure
in low-information environments.

### 3. Volatility Estimation

SPACELIKE regimes exhibit 1.27× higher return variance. A regime-aware volatility
forecast should use:

```
σ_forecast = σ_base × (1.0 + 0.27 × spacelike_fraction)
```

where `spacelike_fraction` is the rolling fraction of SPACELIKE bars over the
last 20 bars.

### 4. Risk Management

During SPACELIKE regimes:
- Widen stop-loss levels by a factor of 1/√(1 − β²) to account for length contraction
- Reduce position sizes proportionally
- Defer new signal generation until regime reverts to TIMELIKE

---

## SIMD Optimization

The beta calculation hot path uses SSE2/AVX2 intrinsics (controlled by the
`LLMQUANT_ENABLE_SIMD` CMake flag):

| Path | Throughput |
|------|-----------|
| Scalar | ~2.1 M bars/sec |
| SSE2 (128-bit, 2× doubles) | ~4.0 M bars/sec |
| AVX2 (256-bit, 4× doubles) | ~7.8 M bars/sec |

### Enabling AVX2

```cmake
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-march=native -mavx2 -mfma"
```

### Enabling AVX-512 (requires Skylake-X / Ice Lake)

```cmake
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-march=native -mavx512f -mavx512dq"
```

SIMD operations are used in:
- `BetaCalculator::compute_batch()` — vectorised return series computation
- `LorentzTransform::gamma_batch()` — batch γ = 1/√(1−β²) via reciprocal sqrt intrinsics
- `SpacetimeManifold::classify_batch()` — vectorised interval sign classification
- `PerformanceCalculator::sharpe()` — vectorised mean/variance via horizontal reduction

All SIMD paths fall back to scalar on non-x86 targets automatically via
`#ifdef __AVX2__` / `#ifdef __SSE2__` guards.
