# Special Relativity in Financial Modeling (SRFM)

[![CI](https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](CMakeLists.txt)
[![Version](https://img.shields.io/badge/version-1.1.0-green.svg)](CHANGELOG.md)

---

## What Is This?

SRFM treats financial price series as trajectories in a four-dimensional
Minkowski spacetime, embedding every OHLCV bar as a spacetime event
`(t, P, V, M)` where `t` is bar time, `P` is close price, `V` is volume,
and `M` is market-impact proxy. This allows the tools of special relativity —
Lorentz transforms, spacetime intervals, geodesic equations — to be applied
directly to market data.

The core insight is that price-velocity (`β = ΔP / (c · Δt)`) plays the role
of relativistic velocity, and the spacetime interval
`ds² = −c²dt² + dP² + dV² + dM²`
carries causal information about the market microstructure regime:

| Interval type | ds² sign | Market interpretation                        |
|---------------|----------|----------------------------------------------|
| TIMELIKE      | < 0      | Causal regime — price change propagates at sub-light speed; momentum is predictive |
| LIGHTLIKE     | = 0      | Critical boundary — price moves exactly at market speed of light |
| SPACELIKE     | > 0      | Stochastic regime — faster-than-light separation; bars are decorrelated |

Q1 2025 empirical result: TIMELIKE bars exhibit **1.27x lower next-bar absolute
return variance** than SPACELIKE bars across 10 liquid instruments
(Bartlett p = 6×10⁻¹⁶, 10/11 assets significant after Bonferroni correction).

---

## Mathematical Background

### Spacetime Embedding

An OHLCV bar is mapped to a four-vector:

```
x^μ = (c·t,  P,  V^(1/4),  M^(1/3))
```

The superscript normalization of `V` and `M` compresses the dynamic range so
that all four coordinates live on similar scales.

### Lorentz Factor and Beta

The normalised price velocity over an interval `[t₁, t₂]` is:

```
β = ΔP / (c · Δt)
```

where `c` is the calibrated "financial speed of light" (default: 1.0).
The Lorentz factor is:

```
γ(β) = 1 / √(1 − β²),    |β| < 1
```

All computations clamp `|β| < BETA_MAX_SAFE = 0.9999` to avoid numerical
divergence near the light cone.

### Spacetime Interval

```
ds² = −c²(Δt)² + (ΔP)² + (ΔV)² + (ΔM)²
```

Classification:
- `ds² < 0` → TIMELIKE
- `ds² = 0` → LIGHTLIKE  (within `|ds²| < 1e-12`)
- `ds² > 0` → SPACELIKE

### Relativistic Momentum Signal

The raw financial momentum `p_raw` is corrected by the Lorentz factor:

```
p_rel = γ(β) · m_eff · p_raw
```

where `m_eff` is the effective mass (position size or volatility proxy). This
naturally down-weights signals in high-velocity (noisy) regimes and amplifies
them in low-velocity (causal) regimes.

### Geodesic Price Paths

The geodesic equation in curved spacetime:

```
d²x^μ/dτ² + Γ^μ_νρ (dx^ν/dτ)(dx^ρ/dτ) = 0
```

is integrated numerically using RK4 with adaptive step control. Christoffel
symbols `Γ^μ_νρ` are computed via O(h²) central finite differences applied to
the metric tensor `g_μν`. The geodesic path represents the "natural" price
trajectory in the absence of external forces — deviations from it are trading
signals.

### Metric Tensor

The default metric is a perturbed Minkowski metric:

```
g_μν = diag(−1, 1+ε_P, 1+ε_V, 1+ε_M)
```

where the perturbations `ε` are fitted to local market data. The inverse
`g^μν` is computed analytically for diagonal metrics and via LU decomposition
for general ones; singular metrics return `std::nullopt`.

---

## N-Asset Portfolio Manifold

The `portfolio_manifold` module extends the 4D spacetime framework to a full
N-asset portfolio analysis layer.

### AssetEvent

Each asset is embedded as a 4-vector `(t, P, V, M)`:

```cpp
#include "portfolio_manifold.hpp"
using namespace srfm::portfolio;

AssetEvent aapl{"AAPL", 1.0, 150.0, 1e8, 2.4e12};
AssetEvent msft{"MSFT", 1.0, 290.0, 8e7, 2.1e12};
```

### MinkowskiCovariance — NxN spacetime interval covariance

```cpp
MinkowskiCovariance mc;
mc.add_asset(aapl);
mc.add_asset(msft);
auto cov = mc.compute_spacetime_covariance();
// cov(i,j) = exp(-|ds²(i,j)|)  — Gaussian kernel over spacetime interval
// Diagonal = 1.0; off-diagonal in (0, 1]
```

The (i,j) entry is `exp(−|ds²(i,j)|)` where `ds²` is the Minkowski interval:

```
ds²(i,j) = −c²·Δt² + ΔP² + ΔV² + ΔM²
```

| Interval type | ds² sign | Kernel value | Interpretation |
|---------------|----------|--------------|----------------|
| TIMELIKE      | < 0      | → 0          | Causal separation — large lead-lag gap |
| LIGHTLIKE     | ≈ 0      | ≈ 1          | Maximum covariance at the light cone |
| SPACELIKE     | > 0      | → 0          | Stochastic, acausal separation |

### SpacetimeCausalGraph — directed causal influence graph

```cpp
auto graph = SpacetimeCausalGraph::build(mc);
// Edge (i→j) exists iff ds²(i,j) < −threshold (TIMELIKE)
bool aapl_leads_msft = graph->has_edge(0, 1);
int aapl_out_degree  = graph->out_degree(0);
```

---

## Relativistic Portfolio Optimization

The `relativistic_optimizer` module reformulates Markowitz optimization as a
geodesic problem on the financial manifold.

### Key differences from classical Markowitz

| Classical | Relativistic |
|-----------|-------------|
| Risk = `w^T Σ w` (Euclidean variance) | Risk = geodesic distance `w^T Σ_st w` (spacetime metric) |
| Returns = `μ` | Returns = `γ(β) · μ` (Lorentz-corrected) |
| Solver: quadratic programming | Solver: projected gradient descent on simplex |

### Usage

```cpp
#include "relativistic_optimizer.hpp"
using namespace srfm::portfolio;

RelativisticPortfolio rp;
rp.add_asset(AssetEvent{"AAPL", 1.0, 150.0, 1e8, 2.4e12}, 0.12);
rp.add_asset(AssetEvent{"MSFT", 1.0, 290.0, 8e7, 2.1e12}, 0.10);
rp.add_asset(AssetEvent{"GOOG", 1.0, 140.0, 6e7, 1.8e12}, 0.09);

auto result = rp.optimize_weights(0.08);  // target 8% return
if (result) {
    std::cout << "Weights: " << result->weights.transpose() << "\n";
    std::cout << "Geodesic risk: " << result->geodesic_risk << "\n";
    std::cout << "Expected return: " << result->expected_return << "\n";
}
```

### Gamma-weighted returns

For each asset i, the Lorentz-corrected return is:

```
μ_rel_i = γ(β_i) · μ_i,   γ(β) = 1 / √(1 − β²)
β_i = |P_i| / (c · |t_i|)
```

High-velocity assets (large β, noisy regime) have returns amplified by γ > 1,
increasing pressure on the optimizer to reduce their weight.

---

## Technical Improvements

### Autodiff Christoffel Symbols (Task 3)

`ChristoffelSymbolsDual` replaces O(h²) central finite differences with exact
**forward-mode automatic differentiation** via dual numbers (`ε² = 0`).

```cpp
// Dual-number metric function (evaluate at DualSpacetimePoint):
auto dual_fn = [](const DualSpacetimePoint& xd) -> DualMetricMatrix { ... };

MetricTensor base(metric_fn);
ChristoffelSymbolsDual cs(base, dual_fn);
auto gamma = cs.compute(x);  // exact derivatives, no step-size tuning
```

Benefits over finite differences:

| Property | Finite differences | Dual numbers |
|---|---|---|
| Metric evaluations per Γ | 8 (2 per direction) | 4 (1 per direction) |
| Truncation error | O(h²) | 0 (machine epsilon only) |
| Step-size sensitivity | Yes — requires tuning | None |
| Works for non-smooth metrics | No | Yes |

### Metric Singularity — Tikhonov Regularization (Task 4)

`MetricTensor::inverse()` previously returned `std::nullopt` for singular
metrics, silently zeroing out all Christoffel symbols. Now it applies
**Tikhonov regularization** before giving up:

```
g_reg = g + λI,   λ = 1e-10
```

A `stderr` warning is emitted whenever regularization is applied:

```
[srfm::tensor::MetricTensor::inverse] WARNING: singular metric detected at
x = (…); applying Tikhonov regularization λ = 1.00e-10
```

This recovers geodesic integration for degenerate market configurations (e.g.
zero-volatility assets, perfectly correlated pairs) without silently discarding
curvature information.

---

## What's New in v1.2.0

### Multi-Asset Spacetime (`include/srfm/multi_asset.hpp`)

Extends the single-asset framework to handle N correlated financial assets
simultaneously with a rolling correlation-based Lorentzian metric.

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
from srfm import MultiAssetEvent, CorrelationMetric, PortfolioGeodesic

# Interval classification
SpacetimeInterval.classify(dt=1.0, dp=0.5, dv=0.1, dm=0.05)
# → 'TIMELIKE'

# Lorentz factor
LorentzTransform.gamma(beta=0.8)
# → 1.6666666666666667

# Full backtest
bt = Backtester()
result = bt.run(prices=[100, 101, 99, 102, 103, 100, 104])
print(result.sharpe)           # relativistic Sharpe ratio
print(result.relativistic_lift)# IR_γ lift factor
print(result.to_string())      # formatted comparison table
```

**Install (no build required):**
```bash
pip install -e python/
```

**Install with C++ extension:**
```bash
pip install pybind11
cmake -B build -DSRFM_BUILD_PYTHON=ON
cmake --build build
pip install -e python/
```

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a complete walkthrough.

---

## Architecture

```
include/srfm/
  types.hpp          — Strong types: BetaVelocity, LorentzFactor, EffectiveMass
  constants.hpp      — BETA_MAX_SAFE, SPEED_OF_LIGHT, FLOAT_EPSILON
  momentum.hpp       — MomentumProcessor, MomentumSignal
  manifold.hpp       — SpacetimeEvent, SpacetimeInterval, IntervalClass
  tensor.hpp         — MetricTensor, DualNumber, ChristoffelSymbols,
                       ChristoffelSymbolsDual, MetricMatrix
  engine.hpp         — Engine (full pipeline wiring)
  backtest.hpp       — Backtester, PerformanceCalculator, BacktestResult
  data_loader.hpp    — DataLoader, OHLCV
  normalizer.hpp     — CoordinateNormalizer
  geodesic_signal.hpp    — GeodesicSignal
  geodesic_strategy.hpp  — GeodesicStrategy
  multi_asset.hpp    — MultiAssetEvent, MultiAssetInterval,
                       CorrelationMetric, MultiAssetLorentz, PortfolioGeodesic

include/
  portfolio_manifold.hpp      — AssetEvent, MinkowskiCovariance,
                                SpacetimeCausalGraph (N-asset portfolio layer)
  relativistic_optimizer.hpp  — RelativisticPortfolio, OptimizerConfig,
                                OptimizationResult

python/srfm/
  __init__.py        — Pure-Python fallback API (NumPy-backed)
  bindings.cpp       — pybind11 C++ extension (optional, for performance)

examples/
  quickstart.ipynb   — Jupyter notebook: interval classification, backtest,
                       multi-asset spacetime, portfolio geodesics

  simd/
    cpu_features.hpp     — detect_simd_level(), SimdLevel enum
    simd_dispatch.hpp    — batch_beta_scalar/avx2/avx512
  stream/
    tick.hpp             — Tick, TickValidator
    beta_calculator.hpp  — OnlineBetaCalculator<N> (lock-free streaming)
    lorentz_transform.hpp — streaming Lorentz transform
    spacetime_manifold.hpp — streaming spacetime embedding
    signal_processor.hpp — StreamSignalProcessor
    stream_signal.hpp    — StreamSignal
  manifold/
    n_asset_interval.hpp — N-asset spacetime interval
  tensor/
    n_asset_manifold.hpp — N-asset metric and Christoffel computation

src/
  momentum/          — MomentumProcessor implementation
  beta_calculator/   — BetaCalculator implementation
  manifold/          — SpacetimeMarketManifold implementation
  geodesic/          — GeodesicSolver (RK4) implementation
  engine/            — Engine implementation
  tensor/
    christoffel_dual.cpp — ChristoffelSymbolsDual (dual-number autodiff)
  simd/              — Scalar, AVX2, AVX-512F kernels + runtime dispatch
  portfolio_manifold.cpp    — MinkowskiCovariance, SpacetimeCausalGraph
  relativistic_optimizer.cpp — RelativisticPortfolio
```

Dependency graph (no cycles):

```
srfm_momentum  ←  srfm_beta_calculator
srfm_momentum  ←  srfm_manifold
srfm_manifold  ←  srfm_geodesic
srfm_beta_calculator, srfm_manifold, srfm_geodesic  ←  srfm_engine
srfm_momentum  ←  srfm_simd_{scalar,avx2,avx512}  ←  srfm_simd_dispatch
srfm_manifold, srfm_tensor  ←  srfm_portfolio
```

---

## Building

### Prerequisites

| Tool | Minimum version |
|------|----------------|
| CMake | 3.25 |
| C++ compiler | GCC 12 / Clang 17 / MSVC 19.38 |
| Eigen3 | 3.4 (optional, enables edge-case and Lorentz tests) |
| GTest | 1.14 (auto-fetched if not found) |
| Google Benchmark | 1.8 (auto-fetched if not found) |
| RapidCheck | any (optional, enables property tests) |

### Linux / macOS

```bash
# Install system dependencies (Ubuntu example)
sudo apt-get install -y cmake ninja-build libeigen3-dev libgtest-dev

# Configure and build
cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel

# Run all tests
ctest --test-dir build --output-on-failure

# Run benchmarks
cmake --build build --target bench
```

### Windows (MSVC)

```powershell
# Using vcpkg for dependencies
vcpkg install eigen3 gtest benchmark rapidcheck

cmake -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
      -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

### Build options

| CMake option | Default | Description |
|---|---|---|
| `SRFM_WARNINGS_AS_ERRORS` | `OFF` | Promote all warnings to errors |
| `SRFM_FUZZ` | `OFF` | Build libFuzzer targets (requires Clang) |
| `CMAKE_BUILD_TYPE` | `Release` | Debug / Release / RelWithDebInfo |

### CMake install

```bash
cmake --install build --prefix /usr/local
# Downstream CMakeLists.txt:
#   find_package(srfm CONFIG REQUIRED)
#   target_link_libraries(myapp PRIVATE srfm::srfm_engine)
```

---

## Usage

### C++ API — quick start

```cpp
#include <srfm/engine.hpp>
#include <srfm/data_loader.hpp>

// Load bars from CSV
auto bars = srfm::DataLoader::load_csv("prices.csv");
if (bars.empty()) { /* handle error */ }

// Run the full pipeline
srfm::Engine engine;
auto result = engine.run_backtest(bars);
if (!result) { /* insufficient data or degenerate series */ }

// Access performance metrics
std::cout << "Sharpe: " << result->adjusted.sharpe_ratio << "\n";
std::cout << "Sortino: " << result->adjusted.sortino_ratio << "\n";
std::cout << "Max drawdown: " << result->adjusted.max_drawdown << "\n";
```

### Manual pipeline composition

```cpp
#include <srfm/manifold.hpp>
#include <srfm/momentum.hpp>
#include <srfm/tensor.hpp>

using namespace srfm;

// Classify a single bar transition
SpacetimeEvent prev{0.0, 100.0, 1e6, 0.0};
SpacetimeEvent curr{1.0, 100.5, 1.1e6, 0.0};

auto ds2 = manifold::SpacetimeInterval::compute(prev, curr);
if (ds2 && *ds2 < 0.0) {
    // TIMELIKE: apply relativistic momentum correction
    momentum::MomentumSignal sig{0.005, types::BetaVelocity::make(0.005), 1.0};
    auto corrected = momentum::MomentumProcessor::process(sig);
}
```

### Streaming mode (lock-free, tick-by-tick)

```cpp
#include <srfm/stream/beta_calculator.hpp>
#include <srfm/stream/lorentz_transform.hpp>

srfm::stream::OnlineBetaCalculator<256> beta_calc;
srfm::stream::LorentzTransform transform;

// Push ticks as they arrive (thread-safe, lock-free)
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

// Automatically dispatches to AVX-512, AVX2, or scalar
// depending on runtime CPU feature detection
std::vector<double> velocities = { /* ... */ };
std::vector<double> betas(velocities.size());
std::vector<double> gammas(velocities.size());

srfm::simd::batch_beta(velocities.data(), betas.data(), velocities.size());
srfm::simd::batch_gamma(betas.data(), gammas.data(), betas.size());
```

---

## Rust Orchestrator — Quick Start (zero config)

The `tokio-prompt-orchestrator` crate sits above the C++ signal-processing
library and coordinates LLM inference workers. It compiles and runs with no
external services needed (mock mode).

### 1. Build

```bash
# Library + CLI (no optional features required)
cargo build --release

# With the Terminal UI dashboard
cargo build --release --features tui

# With the HTTP/WebSocket API server
cargo build --release --features web-api
```

### 2. Run the TUI dashboard (mock data, no API keys needed)

```bash
cargo run --release --features tui -- --mock
```

The dashboard shows a 2-minute scripted story: warmup → load ramp → failure →
circuit half-open → recovery → steady state, then loops.

### 3. Start the HTTP API server

```bash
# Optional: set an API key (omit for open access, development only)
export API_KEY=my-secret-token

cargo run --release --features web-api -- --web --port 8080
```

### 4. Send a test inference request

```bash
curl -s -X POST http://localhost:8080/api/v1/infer \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-token" \
  -d '{"prompt": "Explain Lorentz contraction in one sentence."}' | jq .
```

### 5. Use the EchoWorker in your own code (no keys required)

```rust
use tokio_prompt_orchestrator::{EchoWorker, ModelWorker};

#[tokio::main]
async fn main() {
    let worker = EchoWorker;
    let tokens = worker.infer("Hello, world!").await.unwrap();
    println!("{}", tokens.join(""));
}
```

---

## HTTP API Endpoint Reference

> Requires the `web-api` feature.
> All inference endpoints require `Authorization: Bearer <API_KEY>` when
> `API_KEY` is set. Public endpoints (`/health`, `/metrics`, `/api/v1/schema`)
> are always unauthenticated.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/v1/infer` | Yes | Submit an inference request; returns a `request_id` immediately |
| `POST` | `/api/v1/stream` | Yes | SSE token stream; events: `start`, `token`, `done` |
| `GET`  | `/api/v1/status/{id}` | Yes | Poll request status (`pending`, `processing`, `completed`, `failed`, `timeout`) |
| `GET`  | `/api/v1/result/{id}` | Yes | Block until the result is ready (or timeout) |
| `GET`  | `/api/v1/ws` | Yes | WebSocket upgrade — bidirectional streaming; 1 MB message limit, 30 s ping |
| `GET`  | `/api/v1/schema` | No | OpenAPI 3.0 JSON schema |
| `GET`  | `/health` | No | `{"status":"healthy","version":"…"}` |
| `GET`  | `/metrics` | No | Prometheus text-format metrics |

### Request body (`POST /api/v1/infer` and `/api/v1/stream`)

```json
{
  "prompt": "string (required)",
  "session_id": "string (optional, generated if absent)",
  "metadata": { "key": "value" },
  "stream": false
}
```

### Response body

```json
{
  "request_id": "uuid",
  "status": "processing",
  "result": "string (present when completed)",
  "error": "string (present when failed)"
}
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | *(none)* | Bearer token; unset = auth disabled (warning logged) |
| `ALLOWED_ORIGINS` | *(wildcard)* | Comma-separated CORS origins |
| `OPENAI_API_KEY` | — | Required for `OpenAiWorker` |
| `ANTHROPIC_API_KEY` | — | Required for `AnthropicWorker` |
| `LLAMA_CPP_URL` | `http://localhost:8080` | llama.cpp server URL |
| `VLLM_URL` | `http://localhost:8000` | vLLM server URL |
| `RUST_LOG` | `info` | Tracing log filter (e.g. `debug`, `tokio_prompt_orchestrator=trace`) |

### WebSocket message format

Send JSON matching the `POST /api/v1/infer` body. The server replies with:

```json
{ "request_id": "uuid", "status": "processing" }
// ... then when complete:
{ "request_id": "uuid", "status": "completed", "result": "…" }
```

Rate limit: 60 messages per minute per connection.

### Rate Limits

| Endpoint | Limit | Window | Scope |
|----------|-------|--------|-------|
| `POST /api/v1/infer` | 60 requests | 60 seconds | Per IP address |
| `WS /api/v1/ws` | 60 messages | 60 seconds | Per connection |
| `POST /api/v1/stream` | 10 concurrent connections | — | Per server |

Rate limit response headers (on `POST /api/v1/infer`):

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests allowed in the window |
| `X-RateLimit-Remaining` | Requests remaining in the current window |
| `X-RateLimit-Reset` | Unix timestamp when the window resets |

When a rate limit is exceeded the server responds with HTTP `429 Too Many Requests` and a structured error body:

```json
{"error": {"code": "too_many_connections", "message": "SSE connection limit reached"}}
```

The SSE connection limit (default: 10) is configurable at startup.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full version history.

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

# Property-based tests (requires RapidCheck)
ctest --test-dir build -R "prop_"

# Python validation suite
cd tests/python && pip install -r requirements.txt && pytest -v
```

Test coverage summary:

| Suite | Tests | Description |
|---|---|---|
| MomentumUnitTests | 12 | `MomentumProcessor`, `BetaVelocity`, `LorentzFactor` |
| LorentzTransformTests | 18 | `gamma`, `rapidity`, `Doppler`, round-trip |
| BetaCalculatorTests | 14 | Online `β` computation, boundary clamping |
| LorentzInvariantTests | 16 | ds² invariance, velocity composition, subnormals |
| MetricTensorTests | 10 | Minkowski metric, inverse, singular metric |
| ChristoffelTests | 8 | Flat space identity, symmetry `Γ^μ_νρ = Γ^μ_ρν` |
| GeodesicTests | 12 | RK4 energy conservation, flat geodesic linearity |
| IntervalGapTests | 9 | Symmetry, extreme coordinates, boost invariance |
| SimdAccelerationTests | 6 | Scalar/AVX2/AVX-512 numerical agreement |
| BacktesterTests | 14 | Sharpe, Sortino, max drawdown, γ-weighted IR |
| PerformanceMetricsTests | 10 | Precision, edge cases |
| ErrorHandlingIntegrationTests | 22 | NaN/Inf inputs, length mismatches, degenerate metrics |
| FullPipelineIntegrationTests | 8 | End-to-end Engine.run_backtest |
| NAssetTests | 20 | N-asset interval, manifold, geodesic |
| StreamTests | 15 | Lock-free ring buffer, tick validation, SPSC stress |
| Property tests (RapidCheck) | 9 × 10,000 | Lorentz identity, rapidity additivity, subluminality |

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

Run benchmarks locally:

```bash
cmake --build build --target bench
./build/bench_beta_gamma --benchmark_format=json
```

---

## Empirical Validation (Q1 2025)

Evaluated on 10 liquid S&P 500 instruments at 1-minute resolution over Q1 2025:

- **Variance ratio (VR):** SPACELIKE bars show 1.27x higher next-bar return
  variance than TIMELIKE bars.
- **Bartlett test:** p = 6×10⁻¹⁶ (null hypothesis of equal variances rejected).
- **Per-instrument significance:** 10/11 instruments significant at α = 0.01
  after Bonferroni correction.
- **Relativistic Sharpe improvement:** +0.18 vs classical momentum on the same
  universe over the same period.

Full methodology and results are in the companion paper (see below).

---

## Academic Paper

The mathematical foundations and empirical results are documented in a
full-length academic paper:

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

## API Reference

Full Doxygen-generated API documentation is built in CI and available as a
GitHub Actions artifact on every `main` push. To build locally:

```bash
doxygen Doxyfile
# Output: docs/api/html/index.html
```

Key namespaces:

| Namespace | Content |
|---|---|
| `srfm::types` | Strong types (`BetaVelocity`, `LorentzFactor`, `EffectiveMass`) |
| `srfm::constants` | Physical and numerical constants |
| `srfm::momentum` | `MomentumProcessor`, `MomentumSignal` |
| `srfm::manifold` | `SpacetimeInterval`, `SpacetimeEvent`, `IntervalClass` |
| `srfm::tensor` | `MetricTensor`, `ChristoffelSymbols`, `GeodesicSolver` |
| `srfm::core` | `Engine`, `BarData`, `BetaVelocity` |
| `srfm::backtest` | `Backtester`, `PerformanceCalculator`, `BacktestResult` |
| `srfm::simd` | `detect_simd_level()`, `batch_beta`, `batch_gamma` |
| `srfm::stream` | Lock-free streaming pipeline |

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

### API contract

Every public function must satisfy:
- Returns `std::optional<T>` for all fallible paths; never throws.
- Does not invoke UB for any finite or non-finite IEEE 754 input.
- Is documented with `@brief`, `@param`, and `@return` Doxygen tags.
- Is covered by at least one unit test exercising the happy path and at least
  one test exercising the error path (`std::nullopt` return).

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{busel2025srfm,
  author  = {Busel, Matthew},
  title   = {Special Relativity in Financial Modeling},
  year    = {2025},
  url     = {https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling},
  version = {1.1.0}
}
```
