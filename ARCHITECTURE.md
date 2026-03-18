# Architecture

## Overview

SRFM is a layered C++20 library. Each layer depends only on layers below it.
There are no circular dependencies. All inter-module communication uses value
types or `std::optional`; no raw pointers cross module boundaries.

```
┌─────────────────────────────────────────────┐
│               Engine  (src/engine/)          │  CLI entry point, full pipeline
│   DataLoader  (include/srfm/data_loader.hpp) │  CSV ingestion
└────────────────────┬────────────────────────┘
                     │
        ┌────────────┴──────────────┐
        │                           │
┌───────▼────────┐         ┌────────▼───────┐
│  Backtester    │         │  GeodesicSolver│  RK4 integration
│(src/backtest/) │         │(src/geodesic/) │
└───────┬────────┘         └────────┬───────┘
        │                           │
        └───────────┬───────────────┘
                    │
     ┌──────────────▼──────────────────┐
     │   SpacetimeManifold             │  Christoffel symbols, regime classify
     │   MetricTensor                  │  4x4 g_uv, inverse, validity
     │   (src/manifold/)               │
     └──────────────┬──────────────────┘
                    │
     ┌──────────────▼──────────────────┐
     │   BetaCalculator                │  Price -> beta (with/without look-ahead)
     │   LorentzTransform              │  gamma, time dilation, momentum, rapidity
     │   (src/lorentz/)                │
     └──────────────┬──────────────────┘
                    │
     ┌──────────────▼──────────────────┐
     │   Momentum processor            │  p_rel = gamma * m_eff * v_market
     │   BetaVelocity, LorentzFactor   │  Strong types with construction validation
     │   EffectiveMass                 │
     │   (src/momentum/)               │
     └──────────────┬──────────────────┘
                    │
     ┌──────────────▼──────────────────┐
     │   SIMD acceleration             │  Batch beta/gamma, runtime dispatch
     │   (src/simd/, include/srfm/simd)│  Scalar / SSE4.2 / AVX2 / AVX-512F
     └─────────────────────────────────┘
```

## Module Contracts

### `src/momentum/` -- Momentum-Velocity Signal Processor

Defines the foundational strong types used throughout the library:

| Type | Invariant |
|------|-----------|
| `BetaVelocity` | `isfinite(v)` and `abs(v) < BETA_MAX_SAFE` (0.9999) |
| `LorentzFactor` | `v >= 1.0` and `isfinite(v)` |
| `EffectiveMass` | `v > 0.0` and `isfinite(v)` |

Construction-time validation via factory functions (`::make()`). All fallible
paths return `std::optional`. Zero exceptions. Zero raw pointers.

### `src/lorentz/` -- Lorentz Transform Engine

Pure-static class `LorentzTransform` providing:

- `gamma(beta)` -- Lorentz factor 1/sqrt(1 - b^2)
- `dilateTime(tau, beta)` -- t = gamma * tau
- `applyMomentumCorrection(raw, beta, m_eff)` -- p_rel = gamma * m_eff * raw
- `composeVelocities(b1, b2)` -- (b1+b2)/(1+b1*b2), guaranteed subluminal
- `rapidity(beta)` -- phi = atanh(beta), additive under composition
- `contractLength(L0, beta)` -- L = L0/gamma
- `totalEnergy(beta, m_eff, c)` -- E = gamma * m_eff * c^2

`BetaCalculator` maps raw OHLCV data to `BetaVelocity`, with both
offline (look-ahead safe for backtesting) and online (causal, streaming-safe)
variants.

### `src/manifold/` -- Spacetime Market Manifold

- `MetricTensor` -- 4x4 symmetric tensor g_{uv}, sign convention (-,+,+,+).
  `is_valid()` guards all downstream consumers. `inverse_diagonal()` for the
  diagonal fast path.
- `SpacetimeManifold` -- classifies events as `Newtonian`, `Relativistic`,
  `HighGamma`, or `Subluminal` based on the proxy-beta derived from event
  coordinates. Computes all 64 Christoffel symbols via finite differences.

### `src/geodesic/` -- RK4 Geodesic Integrator

`GeodesicSolver::solve()` integrates the geodesic equation:

```
d^2 x^lambda / d tau^2 + Gamma^lambda_mu_nu (dx^mu/dtau)(dx^nu/dtau) = 0
```

Safety guarantees enforced in code:

- `steps` clamped to [1, 100000]
- `dt` clamped to [1e-8, 1.0]
- Returns `nullopt` if any intermediate state becomes non-finite

### `src/engine/` -- Pipeline Orchestrator

`Engine::process(string_view)` drives the full pipeline from raw bytes to
`PipelineResult`. Accepts arbitrary byte sequences (primary fuzz surface).

### `src/backtest/` -- Relativistic Backtester

Reports Sharpe, Sortino, max drawdown, and gamma-weighted information ratio.
Feeds all signals through Lorentz corrections before strategy evaluation.
Benchmarks relativistic-adjusted signals against raw signals.

### `src/simd/` -- SIMD Acceleration

Runtime-dispatched batch computation of beta and gamma. Dispatch selects the
highest available SIMD level at startup:

```
detect_simd_level() -> SCALAR | SSE42 | AVX2 | AVX512F
```

Each kernel (`beta_scalar`, `beta_avx2`, `beta_avx512`, etc.) is compiled
with the appropriate ISA flags in isolation. The dispatch layer links them all.

## Data Flow (Backtest Mode)

```
CSV file
  |
  v
DataLoader::load_csv()
  -> vector<OhlcvBar>
  |
  v
BetaCalculator::fromPriceVelocityOnline()
  -> vector<BetaVelocity>  (causal, no look-ahead)
  |
  v
LorentzTransform::gamma(), rapidity(), ...
  -> vector<BetaVelocityResult>
  |
  v
SpacetimeManifold::process()
  -> Regime (Newtonian / Relativistic / HighGamma / Subluminal)
  |
  v
GeodesicSolver::solve()
  -> GeodesicState  (natural price path in curved space)
  |
  v
RelativisticSignalProcessor::process()
  -> vector<RelativisticSignal>  (p_rel = gamma * m_eff * v)
  |
  v
Backtester::run()
  -> BacktestResult (Sharpe, Sortino, MDD, gamma-weighted IR)
```

## Design Rules

1. **No raw pointers** in any public API. All ownership expressed via value
   types, `std::unique_ptr`, or `std::shared_ptr`.
2. **No exceptions** propagate across module boundaries. All fallible operations
   return `std::optional` or signal via `bool`.
3. **All public methods are `noexcept`**. Internal helpers may throw only if
   they are not reachable from a `noexcept` context.
4. **Stateless processors**. `RelativisticSignalProcessor`, `SpacetimeManifold`,
   `GeodesicSolver`, and `Engine` are stateless and thread-safe.
5. **Strict layering**. Lower layers never `#include` headers from higher layers.
6. **Input validation at the boundary**. Every public factory function checks
   for NaN, infinity, zero, and out-of-range values before constructing a type.

## Test Strategy

| Category | Location | Coverage |
|----------|----------|----------|
| Unit tests | `tests/momentum/`, `tests/lorentz/`, `tests/manifold/` | Per-function happy path + boundary + invalid inputs |
| Integration tests | `tests/integration/` | Full CSV-to-signal pipeline |
| Property tests | `test/property/` (RapidCheck) | 17 properties x 10 000 inputs |
| Fuzz targets | `fuzz/` | 4 libFuzzer targets (Engine, BetaCalculator, Manifold, Geodesic) |
| SIMD tests | `tests/momentum/test_simd.cpp` | Scalar vs AVX2 vs AVX-512 numerical identity |
| Sanitiser runs | CI (`sanitisers` job) | ASAN + UBSAN + TSAN on every push |

## File Layout

```
include/srfm/        Public headers (installed with the library)
  constants.hpp      Physical and financial constants
  types.hpp          Shared strong types (SpacetimePoint, FourVelocity, ...)
  engine.hpp         Full pipeline engine public header
  manifold.hpp       Manifold public header
  simd/              SIMD dispatch public headers

src/
  momentum/          BetaVelocity, LorentzFactor, EffectiveMass, processor
  lorentz/           LorentzTransform, BetaCalculator (offline + online)
  manifold/          SpacetimeManifold, MetricTensor
  geodesic/          GeodesicSolver (RK4)
  engine/            Pipeline Engine
  backtest/          Backtester, performance metrics, geodesic strategy
  tensor/            N-asset metric tensor, Christoffel (n-asset), geodesic (n-asset)
  simd/              Scalar / AVX2 / AVX-512 kernels + dispatch

tests/               Unit and integration tests
test/property/       RapidCheck property tests
fuzz/                libFuzzer fuzz targets
bench/               Google Benchmark microbenchmarks
```
