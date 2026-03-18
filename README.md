# Special Relativity in Financial Modeling

[![CI](https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling/actions/workflows/ci.yml)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Compiler: GCC 12 | Clang 17](https://img.shields.io/badge/compiler-GCC%2012%20%7C%20Clang%2017-lightgrey)](https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling/actions)

A research-grade C++20 library applying the mathematical machinery of special
relativity to financial signal processing. Lorentz transforms, spacetime interval
classification, relativistic momentum correction, and geodesic price paths --
built as a rigorous quantitative framework, not a metaphor.

Accompanied by a formal academic paper (LaTeX, arXiv-ready) with full
mathematical derivations and Q1 2025 empirical validation on S&P 500 1-minute
OHLCV data.

---

## Features

| Capability | Description |
|---|---|
| Beta calculator | Maps OHLCV price velocity to normalised `beta` with online (causal) and offline variants |
| Lorentz transforms | `gamma`, time dilation, relativistic momentum, velocity composition, rapidity, Doppler factor |
| Spacetime classifier | Classifies market bars as TIMELIKE, SPACELIKE, or LIGHTLIKE |
| Tensor calculus | 4x4 metric tensor, 64 Christoffel symbols via O(h^2) finite differences |
| Geodesic solver | RK4 integration of the geodesic equation in curved market space |
| Relativistic backtester | Sharpe, Sortino, max drawdown, gamma-weighted information ratio |
| SIMD acceleration | AVX2 (4-wide) and AVX-512F (8-wide) batch beta/gamma with runtime dispatch |
| Fuzz-hardened | 4 libFuzzer targets, ASAN/UBSAN/TSAN clean on every push |
| Property-tested | 17 properties x 10 000 random inputs via RapidCheck |

---

## The Core Idea

Classical financial models treat time as a flat, uniform backdrop. Price moves
from t0 to t1 with no regard for the velocity of the market doing the moving.

Special relativity offers a different frame. When a market moves fast -- when
`beta = v_market / c_market` approaches 1 -- the geometry of the signal changes.
Time dilates. Momentum amplifies. The causal structure of price information bends.

This library operationalises that geometry:

- `beta` (market velocity): normalised rate of price change, `beta = |dP/dt| / max_observed_velocity`
- `gamma` (Lorentz factor): `1 / sqrt(1 - beta^2)` -- weights signals in fast markets more heavily
- `ds^2` (spacetime interval): classifies market regime: timelike (causal, predictable) vs spacelike (stochastic, decorrelated)
- `g_uv` (metric tensor): models multi-asset covariance as curved spacetime; Christoffel symbols capture correlation drift; geodesics trace the natural price path

In the Newtonian limit (beta -> 0, gamma -> 1), every transform reduces to its
classical analog. The framework is a strict generalisation, not a replacement.

---

## Empirical Results (Q1 2025)

Validated on S&P 500 1-minute OHLCV bars, Q1 2025:

| Result | Value |
|--------|-------|
| Variance ratio VR = sigma^2(SPACELIKE) / sigma^2(TIMELIKE) | **1.27x** |
| Bartlett test p-value (variance equality null) | **6 x 10^-16** |
| Assets showing directional VR > 1 | **10 / 11** |
| Assets significant at 5% (Bonferroni-corrected) | **10 / 11** |

SPACELIKE bars exhibit 27% higher return variance than TIMELIKE bars. The
separation is significant at the 10^-16 level. The spacetime interval classifier
discriminates empirically distinct market states.

---

## Math Background

### Lorentz Factor

```
gamma = 1 / sqrt(1 - beta^2),   beta in [0, 1)
```

`gamma = 1` in calm markets. `gamma -> inf` as market velocity approaches
`c_market`. In the Newtonian limit (`beta -> 0`): `gamma ~= 1 + beta^2/2`.

### Relativistic Velocity Addition

```
beta_total = (beta_1 + beta_2) / (1 + beta_1 * beta_2)
```

Composing two market velocities never produces a superluminal result.

### Rapidity

```
phi = atanh(beta)
```

Additive under velocity composition: `phi(beta_1 + beta_2) = phi_1 + phi_2`.
A more natural parameter for compounding momentum signals than `beta` itself.

### Spacetime Interval

```
ds^2 = -c^2 * dt^2 + dP^2 + dV^2 + dM^2
```

`ds^2 < 0`: TIMELIKE -- market is in the causal regime, past predicts future.
`ds^2 > 0`: SPACELIKE -- market is stochastic, signals decorrelated.
`ds^2 = 0`: LIGHTLIKE -- critical transition between regimes.

### Geodesic Equation

```
d^2 x^lambda / d tau^2 + Gamma^lambda_mu_nu (dx^mu/dtau)(dx^nu/dtau) = 0
```

Natural price path in curved market space. Deviation from geodesic signals an
externally driven price move. Solved numerically via 4th-order Runge-Kutta.

---

## Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| CMake | >= 3.25 | Build system |
| C++ compiler | GCC 12+ or Clang 17+ (C++20) | Compilation |
| Eigen3 | >= 3.4 | Linear algebra (metric tensor) |
| Google Test | >= 1.13 | Unit tests (optional, via vcpkg) |
| RapidCheck | any | Property tests (optional, via vcpkg) |
| Google Benchmark | >= 1.8 | Microbenchmarks (fetched automatically) |

---

## Build Instructions

### Linux / WSL2 (recommended)

```bash
# Install system dependencies
sudo apt-get install cmake ninja-build gcc-12 g++-12

# Install library dependencies via vcpkg
vcpkg install eigen3 rapidcheck

# Configure and build
cmake -S . -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"

cmake --build build --parallel

# Run all tests
ctest --test-dir build --output-on-failure --parallel 4
```

### Windows (MSVC / Visual Studio 2022)

Open a **x64 Native Tools Command Prompt for VS 2022**:

```bat
vcpkg install eigen3:x64-windows rapidcheck:x64-windows

cmake -S . -B build ^
  -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"

cmake --build build --config Release --parallel

ctest --test-dir build -C Release --output-on-failure
```

### Build Options

| CMake option | Default | Description |
|---|---|---|
| `SRFM_FUZZ` | `OFF` | Build libFuzzer fuzz targets (requires Clang) |
| `CMAKE_BUILD_TYPE` | `Release` | Debug / Release / RelWithDebInfo |

---

## Quickstart Example

```cpp
#include "momentum/momentum.hpp"
#include "lorentz/lorentz_transform.hpp"

using namespace srfm::momentum;
using namespace srfm::lorentz;

int main() {
    // Construct a validated market velocity (beta = 0.6c)
    auto beta_opt = BetaVelocity::make(0.6);
    if (!beta_opt) return 1;   // invalid input -- returns nullopt, never throws

    // Compute Lorentz factor: gamma = 1/sqrt(1 - 0.36) = 1.25
    auto gamma_opt = lorentz_gamma(*beta_opt);
    if (!gamma_opt) return 1;

    // Apply relativistic momentum correction: p_rel = gamma * m_eff * raw_signal
    auto m_eff = EffectiveMass::make(2.0).value();   // ADV-based liquidity proxy
    auto result = apply_momentum_correction(100.0, *beta_opt, m_eff);
    // result->first == 250.0  (gamma=1.25, m_eff=2, raw=100)

    // Compose two market velocities -- result is always subluminal
    auto b1 = BetaVelocity::make(0.5).value();
    auto b2 = BetaVelocity::make(0.5).value();
    auto composed = compose_velocities(b1, b2);
    // composed->value() == 0.8  (not 1.0 -- relativistic addition)
}
```

---

## Running Tests

```bash
cd build
ctest --output-on-failure

# Run only the momentum unit tests
./test_momentum

# Run only the SIMD acceleration tests
./test_simd
```

Expected output:

```
SRFM Momentum Signal Processor -- Unit Tests
============================================
[PASS] BetaVelocity            (12 assertions)
[PASS] EffectiveMass           (10 assertions)
[PASS] lorentz_gamma           (15 assertions)
...
All tests passed.
```

---

## Architecture Overview

Six modules with strict layering and no circular dependencies:

```
Engine (pipeline orchestrator)
  |
  +-- Backtester (Sharpe, Sortino, MDD, IR)
  |
  +-- GeodesicSolver (RK4 integration)
  |
  +-- SpacetimeManifold (Christoffel symbols, regime classification)
  |
  +-- LorentzTransform / BetaCalculator
  |
  +-- Momentum processor (BetaVelocity, LorentzFactor, EffectiveMass)
       |
       +-- SIMD kernels (Scalar / AVX2 / AVX-512F, runtime dispatch)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full module contract, data flow
diagram, and design rules.

---

## Test Coverage

| Module | Production LOC | Test LOC | Ratio |
|--------|---------------|----------|-------|
| Lorentz + Beta | 530 | 777 | 1.47:1 |
| Manifold | 385 | 467 | 1.21:1 |
| Momentum | 63 | 252 | 4.00:1 |
| Tensor / Geodesic | 456 | 1,121 | 2.46:1 |
| Backtest | 412 | 757 | 1.84:1 |
| Stream + SIMD + Integration | 1,730 | 4,603 | 2.66:1 |
| **Global** | **6,058** | **7,977** | **1.317:1** |

Test categories: Newtonian limit recovery, relativistic regime amplification,
invalid input (NaN/infinity/zero/negative), rapidity additivity, Doppler
reciprocity, flat-spacetime geodesic preservation, Christoffel symmetry,
analytic curved-metric validation, timelike/spacelike regime classification,
full pipeline end-to-end, property tests (17 properties x 10k inputs), 4
libFuzzer targets.

---

## CI/CD

GitHub Actions runs on every push and pull request:

| Job | Description |
|-----|-------------|
| `linux (gcc-12)` | CMake configure, build (C++20), ctest, clang-format check |
| `linux (clang-17)` | CMake configure, build (C++20), ctest, clang-format check |
| `windows (MSVC)` | CMake configure, build, ctest |
| `sanitisers` | ASAN + UBSAN on Clang-17 |
| `tsan` | Thread sanitiser on Clang-17 |

---

## API Documentation

Generate HTML docs locally:

```bash
doxygen Doxyfile
# Output: docs/api/html/index.html
```

Requires Doxygen >= 1.9.

---

## Contributing

1. Fork the repository and create a feature branch from `main`.
2. Follow the module contracts in [ARCHITECTURE.md](ARCHITECTURE.md):
   - No raw pointers in public APIs.
   - All fallible operations return `std::optional`.
   - All public methods are `noexcept`.
   - Input validation at every public boundary (reject NaN, infinity, out-of-range values).
3. Add tests for every new function. Maintain a test:production LOC ratio >= 1.0.
4. Ensure the build passes with `-Wall -Wextra -Wpedantic -Werror`.
5. Run `clang-format --style=file -i <changed files>` before committing.
6. Update [CHANGELOG.md](CHANGELOG.md) under `[Unreleased]`.
7. Open a pull request. CI must be green before merge.

---

## Security

Three pre-production security issues were identified and fixed during adversarial
hardening (AGT-13):

| Issue | Severity | Status |
|-------|----------|--------|
| Division by zero in Christoffel computation on degenerate metric | High | Fixed -- `MetricTensor::is_valid()` guards all paths |
| Unbounded loop in `GeodesicSolver::solve()` with adversarial `steps` input | High | Fixed -- steps clamped to [1, 100000], dt to [1e-8, 1.0] |
| Silent precision loss in `BetaCalculator` at boundary beta -> BETA_MAX_SAFE | Medium | Fixed -- clamped to `BETA_MAX_SAFE - 1e-7` |

All three mitigations are covered by dedicated fuzz targets in `fuzz/`.

---

## Research Paper

A formal academic paper accompanies this implementation, targeting arXiv
(q-fin.CP -- Computational Finance). Full derivations of beta, gamma, rapidity,
Doppler factor, spacetime interval, Christoffel symbols, geodesic equation, and
Jacobi field deviation. Q1/Q2 2025 empirical results.

Build the paper:

```bash
cd paper && make pdf        # Full 3-pass LaTeX compile + BibTeX
cd paper && make figures    # Regenerate all 8 figures from Python
cd paper && make arxiv      # Build arXiv submission tarball
```

Requires: `pdflatex`, `bibtex`, Python >= 3.10 with `matplotlib numpy scipy`.

---

## License

MIT. See [LICENSE](LICENSE).
