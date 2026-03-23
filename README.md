# Special Relativity in Financial Modeling (SRFM)

[![CI](https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](CMakeLists.txt)
[![Version](https://img.shields.io/badge/version-1.2.0-green.svg)](CHANGELOG.md)

---

## What Is SRFM?

SRFM treats every OHLCV price bar as an event in four-dimensional Minkowski
spacetime `(t, P, V, M)` — bar time, close price, volume, and market-impact
proxy — and applies the full machinery of special relativity to financial data.
The normalised price velocity `beta = |dP| / (c * dt)` plays the role of
relativistic velocity. When `beta < 1` the bar is **TIMELIKE**: price information
propagates causally and momentum carries predictive power. When `beta > 1` the bar
is **SPACELIKE**: the move is faster than the market's "speed of light" and is
fundamentally stochastic. The spacetime interval `ds^2 = -(c*dt)^2 + dP^2 + dV^2 + dM^2`
encodes this causal structure quantitatively, enabling Lorentz-corrected momentum
signals, geodesic price-path forecasting, and a full relativistic portfolio optimizer
with real-time streaming signal generation.

---

## Key Empirical Finding

> **TIMELIKE bars exhibit 1.27x lower next-bar absolute return variance than
> SPACELIKE bars** across 10 liquid S&P 500 instruments at 1-minute resolution
> over Q1 2025 (Bartlett p = 6x10^-16, 10/11 assets significant after
> Bonferroni correction).

This is not a back-fit artefact: the TIMELIKE/SPACELIKE label is assigned before
the subsequent bar is observed. The +0.18 relativistic Sharpe improvement over
classical momentum is independently reproduced on each ticker.

---

## 5-Minute Quickstart

### Python validation (no build required)

```bash
# 1. Install Python dependencies
pip install -r validation/requirements.txt

# 2. Run the relativistic portfolio optimizer demo
python validation/portfolio_optimizer.py

# 3. Run the real-time tick streaming demo (60 s of simulated BTC/USD)
python validation/tick_streamer.py

# 4. Launch the ANSI signal dashboard (standalone, no extra deps)
python validation/signal_dashboard.py

# 5. Dashboard with full tick-streamer integration
python validation/signal_dashboard.py --demo --symbols "BTC/USD" "ETH/USD" "SPY"
```

### C++ core

```bash
# Linux / macOS
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# Windows (MSVC + vcpkg)
cmake -B build -G "Visual Studio 17 2022" -A x64 \
      -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

---

## What's New in v1.2.0

### Multi-Asset Spacetime (`include/srfm/multi_asset.hpp`)

Extends the single-asset framework to handle N correlated financial assets
simultaneously, using a rolling correlation-based Lorentzian metric.

| Class | Responsibility |
|-------|----------------|
| `MultiAssetEvent` | N-asset spacetime event: `symbols`, `prices`, `volumes`, `timestamp` |
| `MultiAssetInterval` | ds² in (N+1)-dimensional spacetime using the full metric tensor |
| `CorrelationMetric` | Rolling correlation matrix → Lorentzian (N+1)×(N+1) metric with Cholesky regularisation |
| `MultiAssetLorentz` | Per-asset and portfolio Lorentz boosts; metric-weighted portfolio β |
| `PortfolioGeodesic` | Inertial portfolio trajectory; geodesic deviation as trading signals; geodesic weights |

### Python Bindings (`python/srfm/`)

Full Python API via pybind11, with a pure-Python fallback (no build required):

```python
from srfm import SpacetimeInterval, LorentzTransform, Backtester

# Classify an OHLCV bar
SpacetimeInterval.classify(dt=1.0, dp=0.5, dv=0.1, dm=0.05)
# → 'TIMELIKE'

# Lorentz factor
LorentzTransform.gamma(beta=0.8)
# → 1.6666666666666667

# Full relativistic backtest
result = Backtester().run(prices=[100, 101, 99, 102, 103])
print(result.sharpe)            # relativistic Sharpe ratio
print(result.relativistic_lift) # IR_γ lift factor
print(result.to_string())       # formatted comparison table
```

```bash
# Install (pure-Python, no build required):
pip install -e python/

# Or with C++ extension:
pip install pybind11
cmake -B build -DSRFM_BUILD_PYTHON=ON && cmake --build build
pip install -e python/
```

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a complete walkthrough.

---

## Architecture

```
Special-Relativity-in-Financial-Modeling/
|
+-- include/srfm/              C++ core headers
|   +-- types.hpp              Strong types: BetaVelocity, LorentzFactor, EffectiveMass
|   +-- constants.hpp          BETA_MAX_SAFE, SPEED_OF_LIGHT, FLOAT_EPSILON
|   +-- momentum.hpp           MomentumProcessor, MomentumSignal
|   +-- manifold.hpp           SpacetimeEvent, SpacetimeInterval, IntervalClass
|   +-- tensor.hpp             MetricTensor, ChristoffelSymbols (autodiff + FD)
|   +-- engine.hpp             Engine (full pipeline wiring)
|   +-- backtest.hpp           Backtester, PerformanceCalculator, BacktestResult
|   +-- data_loader.hpp        DataLoader, OHLCV
|   +-- simd/                  CPU feature detection, AVX2/AVX-512 kernels
|   +-- stream/                Lock-free tick streaming pipeline
|   +-- multi_asset.hpp        MultiAssetEvent, MultiAssetInterval,
|                               CorrelationMetric, MultiAssetLorentz, PortfolioGeodesic
|
+-- include/                   N-asset portfolio headers
|   +-- portfolio_manifold.hpp AssetEvent, MinkowskiCovariance, SpacetimeCausalGraph
|   +-- relativistic_optimizer.hpp RelativisticPortfolio, OptimizationResult
|
+-- src/                       C++ implementation files
|   +-- multi_asset.cpp        Multi-asset spacetime implementation
|
+-- python/srfm/               Python interface
|   +-- __init__.py            Pure-Python fallback API (no build required)
|   +-- bindings.cpp           pybind11 C++ extension (optional)
|
+-- examples/
|   +-- quickstart.ipynb       Jupyter notebook: full API walkthrough
|
+-- validation/                Python validation and tooling layer
|   +-- portfolio_optimizer.py  Relativistic portfolio optimizer (NEW v1.2.0)
|   +-- tick_streamer.py        Real-time tick streaming + SRFM signals (NEW v1.2.0)
|   +-- signal_dashboard.py     ANSI terminal real-time dashboard (NEW v1.2.0)
|   +-- analyze_q1.py           TIMELIKE vs SPACELIKE variance statistical tests
|   +-- backtest_comparison.py  Strategy comparison (RAW/RELATIVISTIC/GEODESIC)
|   +-- fetch_data.py           Yahoo Finance data downloader
|   +-- run_validation.py       Full validation pipeline runner
|   +-- requirements.txt        Python dependencies
|
+-- tests/                     C++ unit + integration test suites
+-- bench/                     Google Benchmark targets
+-- paper/                     LaTeX academic paper
+-- CMakeLists.txt
```

### Module dependency graph

```
srfm_momentum  <--  srfm_beta_calculator
srfm_momentum  <--  srfm_manifold
srfm_manifold  <--  srfm_geodesic
srfm_beta_calculator, srfm_manifold, srfm_geodesic  <--  srfm_engine
srfm_momentum  <--  srfm_simd_{scalar,avx2,avx512}  <--  srfm_simd_dispatch
srfm_manifold, srfm_tensor  <--  srfm_portfolio
```

---

## Mathematical Background

### Spacetime Embedding

Each OHLCV bar is mapped to a four-vector:

```
x^mu = (c*t,  P,  V^(1/4),  M^(1/3))
```

Volume and market-impact are compressed by fractional powers so all four
coordinates live on comparable scales.

### Lorentz Factor and Beta

The normalised price velocity over interval [t1, t2]:

```
beta = |dP| / (c * dt)
```

The Lorentz factor:

```
gamma(beta) = 1 / sqrt(1 - beta^2),    |beta| < 1
```

All computations clamp `|beta| < BETA_MAX_SAFE = 0.9999`.

### Spacetime Interval

```
ds^2 = -(c*dt)^2 + dP^2 + dV^2 + dM^2
```

| Interval type | ds^2 sign | Market interpretation                                    |
|---------------|-----------|----------------------------------------------------------|
| TIMELIKE      | < 0       | Causal regime — momentum is predictive                   |
| LIGHTLIKE     | = 0       | Critical boundary — price moves at market speed of light |
| SPACELIKE     | > 0       | Stochastic regime — price change uncorrelated with drift |

### Relativistic Momentum Signal

```
p_rel = gamma(beta) * m_eff * p_raw
```

This naturally down-weights signals in high-velocity noisy regimes and
amplifies them in low-velocity causal regimes.

### Geodesic Price Paths

The geodesic equation:

```
d^2 x^mu / dtau^2 + Gamma^mu_nu_rho (dx^nu/dtau)(dx^rho/dtau) = 0
```

is integrated with RK4. Christoffel symbols are computed either via O(h^2)
central finite differences or exact forward-mode automatic differentiation
(dual numbers, eps^2 = 0). Deviations from the geodesic are trading signals.

### Relativistic Sharpe

```
SR_rel = (w^T mu - rf) / sqrt(w^T Sigma_st w)
```

where `Sigma_st` is the spacetime-weighted covariance matrix. SPACELIKE
asset pairs have their covariance discounted by `(1 - s_i * s_j)` where
`s_k = 1 - timelike_fraction_k`, penalising noise-dominant assets.

---

## Portfolio Optimizer

`validation/portfolio_optimizer.py` implements `RelativisticPortfolioOptimizer`,
a multi-asset portfolio construction engine that uses the Minkowski metric to
distinguish causal (TIMELIKE) from stochastic (SPACELIKE) cross-asset interactions.

### Key classes

| Class | Description |
|---|---|
| `AssetManifold` | Asset worldline — prices, timestamps, per-bar beta and interval type |
| `PortfolioResult` | Weights, relativistic Sharpe, TIMELIKE exposure, max drawdown |
| `RelativisticPortfolioOptimizer` | Main optimizer class |

### Quickstart

```python
from validation.portfolio_optimizer import (
    RelativisticPortfolioOptimizer,
    generate_synthetic_assets,
)

# Build optimizer with calibrated speed of light
opt = RelativisticPortfolioOptimizer(c=0.1, risk_free_rate=0.05)

# Generate or load assets
assets = generate_synthetic_assets(n_assets=5, n_bars=1000)

# Maximise relativistic Sharpe
result = opt.optimize(assets, max_weight=0.4)
print(result.relativistic_sharpe)   # e.g.  0.184
print(result.timelike_exposure)     # e.g.  0.623
print(result.weights)               # array([0.4, 0.2, 0.2, 0.1, 0.1])

# Enforce minimum TIMELIKE exposure
result = opt.optimize(assets, max_weight=0.4, target_timelike=0.70)

# Efficient frontier (50 points)
frontier = opt.efficient_frontier(assets, n_points=50)

# Backtest with rebalancing every 20 bars
bt = opt.backtest(assets, result.weights, rebalance_freq=20)
print(bt["sharpe"])         # annualised Sharpe
print(bt["max_drawdown"])   # e.g. -0.12
print(bt["total_return"])   # e.g. 0.34
```

### Spacetime covariance

The optimizer computes a spacetime-weighted covariance matrix:

```
Sigma_st[i, j] = Sigma_classical[i, j] * (1 - spacelike_i * spacelike_j)
```

where `spacelike_k = 1 - timelike_fraction_k`. TIMELIKE-dominant assets
retain full classical covariance; SPACELIKE-dominant assets are discounted,
reducing their influence on portfolio risk.

### Building an AssetManifold from your data

```python
import numpy as np
from validation.portfolio_optimizer import RelativisticPortfolioOptimizer

opt = RelativisticPortfolioOptimizer(c=0.1)

# From a (N, 2) array of [unix_timestamp, close_price]
ohlcv = np.column_stack([timestamps, prices])
manifold = opt.build_asset_manifold("AAPL", ohlcv)
print(manifold.timelike_fraction)   # fraction of bars classified TIMELIKE
print(manifold.beta)                # per-bar price velocity array
```

---

## Real-Time Streaming

`validation/tick_streamer.py` implements a real-time (or simulated) tick
streaming pipeline that classifies each completed bar using SRFM and
fires a `BarSignal` with momentum and alert flags.

### Key classes

| Class | Description |
|---|---|
| `Tick` | Single market tick (timestamp, price, volume, bid, ask) |
| `BarSignal` | Completed bar with beta, interval_type, ds^2, momentum, alert flags |
| `SimulatedTickFeed` | Regime-switching synthetic tick generator (async) |
| `SRFMTickProcessor` | Assembles ticks into bars, classifies, computes signals |
| `YahooFinanceFeed` | Polling-based Yahoo Finance 1-minute bar stream |

### Programmatic usage

```python
import asyncio
from validation.tick_streamer import SimulatedTickFeed, SRFMTickProcessor

async def main():
    feed = SimulatedTickFeed(symbol="AAPL", initial_price=180.0, volatility=0.001)
    processor = SRFMTickProcessor(bar_period_secs=60.0, c_financial=0.1)

    async for tick in feed.stream():
        signal = processor.process_tick(tick)
        if signal is not None:
            print(signal.interval_type, signal.beta, signal.regime_change)

asyncio.run(main())
```

### Using Yahoo Finance (delayed live data)

```python
from validation.tick_streamer import YahooFinanceFeed, SRFMTickProcessor
import asyncio

async def live_feed():
    feed = YahooFinanceFeed(symbol="SPY", lookback_mins=60)
    processor = SRFMTickProcessor(bar_period_secs=60.0)
    async for tick in feed.stream():
        signal = processor.process_tick(tick)
        if signal:
            print(f"{signal.symbol}  {signal.interval_type}  beta={signal.beta:.4f}")

asyncio.run(live_feed())
```

### BarSignal fields

| Field | Type | Description |
|---|---|---|
| `beta` | float | Normalised price velocity `|dp| / (c * dt)` |
| `interval_type` | str | "TIMELIKE", "LIGHTLIKE", or "SPACELIKE" |
| `spacetime_interval` | float | `ds^2 = dp^2 - (c*dt)^2` |
| `momentum` | float | Exponentially weighted rolling beta signal |
| `regime_change` | bool | True on TIMELIKE <-> SPACELIKE transition |
| `lightlike_crossing` | bool | True when `|beta - 1| < 0.01` |

---

## Signal Dashboard

`validation/signal_dashboard.py` renders a live ANSI terminal dashboard
for one or more symbols.

### Running the dashboard

```bash
# Standalone demo (no external dependencies beyond numpy)
python validation/signal_dashboard.py

# With specific symbols and longer duration
python validation/signal_dashboard.py --symbols "BTC/USD" "ETH/USD" "SPY" --duration 120

# Full integration mode (uses tick_streamer.py)
python validation/signal_dashboard.py --demo --bar-period 5.0 --refresh-hz 4.0

# Adjust financial speed of light
python validation/signal_dashboard.py --c 0.05
```

### Dashboard panels

Each symbol renders a panel showing:

- **Interval type** with colour coding: green (TIMELIKE), yellow (LIGHTLIKE), red (SPACELIKE)
- **Price** with directional arrow and change colour
- **Beta meter** — horizontal bar divided into TIMELIKE / lightcone / SPACELIKE zones
- **TIMELIKE fraction bar** — rolling fraction over last 20 bars
- **Spacetime interval sparkline** — 20-bar ds^2 history with sign-coloured Unicode blocks
- **Portfolio weight** (if set via `dashboard.update_weight(symbol, weight)`)
- **Recent alerts** — regime changes and lightlike crossings

### Programmatic usage

```python
from validation.signal_dashboard import SRFMDashboard

dashboard = SRFMDashboard(symbols=["AAPL", "TSLA"])

# Feed signals from any source
for signal in my_bar_signals:
    dashboard.update(signal)
    dashboard.render()

# Set portfolio weights from optimizer output
dashboard.update_weight("AAPL", 0.35)
dashboard.update_weight("TSLA", 0.15)

# Utility renderers
print(dashboard.sparkline(ds2_values))   # Unicode sparkline string
print(dashboard.beta_meter(0.73))        # ANSI-coloured velocity meter
```

---

## C++ API

### Quick start

```cpp
#include <srfm/engine.hpp>
#include <srfm/data_loader.hpp>

auto bars = srfm::DataLoader::load_csv("prices.csv");
srfm::Engine engine;
auto result = engine.run_backtest(bars);
if (result) {
    std::cout << "Sharpe:       " << result->adjusted.sharpe_ratio << "\n";
    std::cout << "Sortino:      " << result->adjusted.sortino_ratio << "\n";
    std::cout << "Max drawdown: " << result->adjusted.max_drawdown << "\n";
}
```

### N-asset portfolio manifold

```cpp
#include "portfolio_manifold.hpp"
using namespace srfm::portfolio;

MinkowskiCovariance mc;
mc.add_asset(AssetEvent{"AAPL", 1.0, 150.0, 1e8, 2.4e12});
mc.add_asset(AssetEvent{"MSFT", 1.0, 290.0, 8e7, 2.1e12});
auto cov = mc.compute_spacetime_covariance();
// cov(i,j) = exp(-|ds^2(i,j)|) -- Gaussian kernel over spacetime interval
```

### Relativistic optimizer (C++)

```cpp
#include "relativistic_optimizer.hpp"
using namespace srfm::portfolio;

RelativisticPortfolio rp;
rp.add_asset(AssetEvent{"AAPL", 1.0, 150.0, 1e8, 2.4e12}, 0.12);
rp.add_asset(AssetEvent{"MSFT", 1.0, 290.0, 8e7, 2.1e12}, 0.10);
rp.add_asset(AssetEvent{"GOOG", 1.0, 140.0, 6e7, 1.8e12}, 0.09);

auto result = rp.optimize_weights(0.08);  // target 8% return
if (result) {
    std::cout << "Weights:      " << result->weights.transpose() << "\n";
    std::cout << "Geodesic risk: " << result->geodesic_risk << "\n";
}
```

### Streaming pipeline (lock-free, tick-by-tick)

```cpp
#include <srfm/stream/beta_calculator.hpp>
#include <srfm/stream/lorentz_transform.hpp>

srfm::stream::OnlineBetaCalculator<256> beta_calc;
srfm::stream::LorentzTransform transform;

for (const auto& tick : market_feed) {
    auto beta = beta_calc.push(tick);
    if (beta) {
        auto signal = transform.apply(*beta, tick.price);
    }
}
```

### SIMD batch computation

```cpp
#include <srfm/simd/simd_dispatch.hpp>

std::vector<double> velocities = { /* ... */ };
std::vector<double> betas(velocities.size());
std::vector<double> gammas(velocities.size());

// Dispatches to AVX-512, AVX2, or scalar at runtime
srfm::simd::batch_beta(velocities.data(), betas.data(), velocities.size());
srfm::simd::batch_gamma(betas.data(), gammas.data(), betas.size());
```

---

## Building from Source

### Prerequisites

| Tool | Minimum version |
|------|----------------|
| CMake | 3.25 |
| C++ compiler | GCC 12 / Clang 17 / MSVC 19.38 |
| Eigen3 | 3.4 (auto-fetched if missing) |
| GTest | 1.14 (auto-fetched if missing) |
| Google Benchmark | 1.8 (auto-fetched if missing) |
| RapidCheck | any (optional, property tests) |

### Linux / macOS

```bash
sudo apt-get install -y cmake ninja-build libeigen3-dev libgtest-dev

cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel

# Run all tests
ctest --test-dir build --output-on-failure

# Run benchmarks
cmake --build build --target bench
./build/bench_beta_gamma --benchmark_format=json
```

### Windows (MSVC + vcpkg)

```powershell
vcpkg install eigen3 gtest benchmark rapidcheck

cmake -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
      -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

### CMake build options

| Option | Default | Description |
|---|---|---|
| `SRFM_WARNINGS_AS_ERRORS` | `OFF` | Promote all warnings to errors |
| `SRFM_FUZZ` | `OFF` | Build libFuzzer targets (requires Clang) |
| `CMAKE_BUILD_TYPE` | `Release` | Debug / Release / RelWithDebInfo |

### CMake install

```bash
cmake --install build --prefix /usr/local
# Downstream usage:
#   find_package(srfm CONFIG REQUIRED)
#   target_link_libraries(myapp PRIVATE srfm::srfm_engine)
```

### Python dependencies

```bash
pip install -r validation/requirements.txt
# yfinance>=0.2.40  pandas>=2.0.0  numpy>=1.26.0  scipy>=1.12.0
# matplotlib>=3.8.0  seaborn>=0.13.0  hypothesis>=6.100.0
```

---

## Rust Orchestrator

The `tokio-prompt-orchestrator` crate coordinates LLM inference workers and
integrates with the C++ signal-processing library. It runs with no external
services (mock mode).

```bash
# Build
cargo build --release

# TUI dashboard (mock data, no API keys needed)
cargo run --release --features tui -- --mock

# HTTP/WebSocket API server
cargo run --release --features web-api -- --web --port 8080

# Test inference
curl -s -X POST http://localhost:8080/api/v1/infer \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-token" \
  -d '{"prompt": "Explain Lorentz contraction in one sentence."}' | jq .
```

---

## Empirical Validation

Evaluated on 10 liquid S&P 500 instruments at 1-minute resolution over Q1 2025:

- **Variance ratio (VR):** SPACELIKE bars show 1.27x higher next-bar return
  variance than TIMELIKE bars.
- **Bartlett test:** p = 6x10^-16 (null hypothesis of equal variances rejected).
- **Per-instrument significance:** 10/11 instruments significant at alpha = 0.01
  after Bonferroni correction.
- **Relativistic Sharpe improvement:** +0.18 vs classical momentum on the same
  universe over the same period.

Reproduce the analysis:

```bash
# Run C++ regime validator to produce CSVs
cmake --build build --target regime_validator
./build/regime_validator --data-dir data/ --output-dir validation/results/

# Run Python statistical analysis
python validation/analyze_q1.py --results-dir validation/results/

# Run backtest strategy comparison
python validation/backtest_comparison.py
```

---

## Testing

```bash
# All unit tests
ctest --test-dir build --output-on-failure

# Specific suite
ctest --test-dir build -R LorentzTransformTests

# With AddressSanitizer
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan
ctest --test-dir build-asan --output-on-failure

# Python validation suite
cd tests/python && pip install -r requirements.txt && pytest -v
```

Test coverage summary:

| Suite | Tests | Description |
|---|---|---|
| MomentumUnitTests | 12 | `MomentumProcessor`, `BetaVelocity`, `LorentzFactor` |
| LorentzTransformTests | 18 | `gamma`, `rapidity`, `Doppler`, round-trip |
| BetaCalculatorTests | 14 | Online beta computation, boundary clamping |
| LorentzInvariantTests | 16 | ds^2 invariance, velocity composition, subnormals |
| MetricTensorTests | 10 | Minkowski metric, inverse, singular metric |
| ChristoffelTests | 8 | Flat space identity, Gamma symmetry |
| GeodesicTests | 12 | RK4 energy conservation, flat geodesic linearity |
| IntervalGapTests | 9 | Symmetry, extreme coordinates, boost invariance |
| SimdAccelerationTests | 6 | Scalar/AVX2/AVX-512 numerical agreement |
| BacktesterTests | 14 | Sharpe, Sortino, max drawdown, gamma-weighted IR |
| PerformanceMetricsTests | 10 | Precision, edge cases |
| ErrorHandlingIntegrationTests | 22 | NaN/Inf inputs, degenerate metrics |
| FullPipelineIntegrationTests | 8 | End-to-end Engine.run_backtest |
| NAssetTests | 20 | N-asset interval, manifold, geodesic |
| StreamTests | 15 | Lock-free ring buffer, SPSC stress |
| Property tests (RapidCheck) | 9 x 10,000 | Lorentz identity, rapidity additivity |

---

## Performance

Measured on Intel Core i9-13900K (Ubuntu 22.04, GCC 12, `-O3`):

| Kernel | Width | Throughput |
|---|---|---|
| `batch_beta` scalar | 1-wide | 380 Mop/s |
| `batch_beta` AVX2 | 4-wide | 1.41 Gop/s (3.7x) |
| `batch_beta` AVX-512 | 8-wide | 2.63 Gop/s (6.9x) |
| `batch_gamma` scalar | 1-wide | 310 Mop/s |
| `batch_gamma` AVX2 | 4-wide | 1.18 Gop/s (3.8x) |
| `batch_gamma` AVX-512 | 8-wide | 2.24 Gop/s (7.2x) |

---

## Contributing

### Pre-PR checklist

```bash
# 1. Build in Debug with all sanitizers
cmake -B build-check -DCMAKE_BUILD_TYPE=Debug \
      -DSRFM_WARNINGS_AS_ERRORS=ON \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined,thread"
cmake --build build-check && ctest --test-dir build-check --output-on-failure

# 2. clang-tidy (must produce zero warnings)
clang-tidy src/**/*.cpp include/**/*.hpp -- \
      -std=c++20 -Iinclude -Isrc

# 3. Doxygen (zero undocumented public symbols)
doxygen Doxyfile 2>&1 | grep -i warning
```

### API contract (C++)

Every public function must:
- Return `std::optional<T>` for all fallible paths; never throw.
- Not invoke UB for any finite or non-finite IEEE 754 input.
- Be documented with `@brief`, `@param`, and `@return` Doxygen tags.
- Be covered by at least one unit test for the happy path and one for the
  error path (`std::nullopt` return).

### Python style

- Type-annotated (`from __future__ import annotations`).
- All public functions have docstrings with Parameters / Returns sections.
- No external dependencies beyond the packages in `validation/requirements.txt`.

---

## Academic Paper

```
paper/
  01_introduction.tex
  02_theoretical_framework.tex
  03_implementation.tex
  04_empirical_results.tex
  05_risk_analysis.tex
  06_extensions.tex
  07_conclusion.tex
  08_appendix.tex
  bibliography.bib
```

Build the paper:

```bash
cd paper && make pdf        # full paper
cd paper && make figures    # regenerate figures only
cd paper && make arxiv      # arXiv submission tarball
```

---

## HTTP API Endpoint Reference

> Requires the `web-api` feature.
> All inference endpoints require `Authorization: Bearer <API_KEY>` when
> `API_KEY` is set. Public endpoints (`/health`, `/metrics`, `/api/v1/schema`)
> are always unauthenticated.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/v1/infer` | Yes | Submit inference request; returns `request_id` |
| `POST` | `/api/v1/stream` | Yes | SSE token stream; events: `start`, `token`, `done` |
| `GET`  | `/api/v1/status/{id}` | Yes | Poll request status |
| `GET`  | `/api/v1/result/{id}` | Yes | Block until result ready |
| `GET`  | `/api/v1/ws` | Yes | WebSocket bidirectional streaming |
| `GET`  | `/api/v1/schema` | No | OpenAPI 3.0 JSON schema |
| `GET`  | `/health` | No | `{"status":"healthy","version":"..."}` |
| `GET`  | `/metrics` | No | Prometheus text-format metrics |

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full version history.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{busel2025srfm,
  author  = {Busel, Matthew},
  title   = {Special Relativity in Financial Modeling},
  year    = {2025},
  url     = {https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling},
  version = {1.2.0}
}
```
