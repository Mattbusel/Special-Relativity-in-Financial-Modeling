# Special Relativity in Financial Modeling (SRFM)

[![CI](https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](CMakeLists.txt)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](CHANGELOG.md)

## Executive Summary

**For practitioners:** This library classifies every market bar as TIMELIKE (causal, historically predictable) or SPACELIKE (stochastic, decorrelated) using the spacetime interval from special relativity. SPACELIKE bars exhibit 27% higher next-bar return variance than TIMELIKE bars (Bartlett p = 6e-16, 10/11 S&P 500 instruments significant after Bonferroni correction). The relativistic momentum signal outperforms classical momentum because it down-weights signals in high-velocity (noisy) market regimes and amplifies them in low-velocity (causal) regimes.

**For engineers:** Six C++20 modules, zero circular dependencies, every fallible path returns `std::optional` with no exceptions and no undefined behavior. 332+ tests, 17 RapidCheck properties at 10,000 inputs each, four libFuzzer targets. ASAN/UBSAN/TSAN clean under CI. CMake 3.25+, vcpkg, GTest, Google Benchmark. Zero compiler warnings under `-Wall -Wextra -Werror`.

A C++20 library and Python validation suite that applies Lorentz geometry from
special relativity to financial signal processing. Price paths are treated as
trajectories in a (1+1)-dimensional spacetime, enabling causal regime
classification, relativistic momentum filtering, and geodesic-based strategy
generation.

Q1 2025 empirical results: TIMELIKE regimes exhibit 1.27x lower next-bar
absolute return variance than SPACELIKE regimes (Bartlett p = 6e-16) across
10 liquid instruments.

---

## Signal Flow

```
[Price Series] -> [Lorentz Transform] -> [Spacetime Classification]
                                             (SPACELIKE / TIMELIKE / LIGHTLIKE)
                                                   |
                                          [Regime-Based Strategy]
```

---

## Architecture

```
srfm_momentum          -- core Lorentz transforms, beta/gamma, rapidity
  |
  +-- srfm_beta_calculator  -- price-derived beta velocity with clamping
  |
  +-- srfm_manifold         -- spacetime manifold, metric tensor, Christoffel
  |     |                      symbols, interval classification
  |     +-- srfm_geodesic   -- RK4 geodesic solver for price path curvature
  |
  +-- srfm_engine           -- full CSV-to-signal pipeline (Engine class)
```

SIMD acceleration (runtime dispatch):

```
srfm_simd_dispatch
  +-- srfm_simd_scalar   -- reference scalar kernels
  +-- srfm_simd_avx2     -- AVX2 4-wide beta/gamma batch computation
  +-- srfm_simd_avx512   -- AVX-512F 8-wide beta/gamma batch computation
```

Python package `srfm-validation` (in `validation/`) orchestrates the empirical
validation pipeline: data download, regime classification via the C++ binaries,
Q1 statistical analysis, figure generation, and backtest comparison.

Key source locations:

| Path | Contents |
|------|----------|
| `include/` | Public C++ headers (`types.hpp`, `momentum.hpp`, `manifold.hpp`, `tensor.hpp`, `engine.hpp`) |
| `src/` | Implementation files mirroring `include/` structure |
| `tests/` | CTest-wired unit and property-based tests |
| `fuzz/` | libFuzzer fuzz targets (build with `-DSRFM_FUZZ=ON`) |
| `bench/` | Google Benchmark suite for SIMD throughput |
| `validation/` | Python orchestration and analysis scripts |
| `paper/` | LaTeX source for the arXiv-ready research paper |
| `docs/` | Doxygen HTML output (`docs/api/html/index.html`) |

---

## Prerequisites

**C++ library:**

- CMake >= 3.25
- A C++20-capable compiler: GCC 12+, Clang 17+, or MSVC 2022
- (Optional) RapidCheck via vcpkg for property-based tests: `vcpkg install rapidcheck`
- (Optional) Doxygen 1.9+ for API documentation

**Python validation suite:**

- Python >= 3.10
- `pip install -e ".[dev]"` from the repo root

---

## Quickstart

### Build and test (Linux / macOS)

```bash
# Configure
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build all targets
cmake --build build --parallel

# Run the full test suite
ctest --test-dir build --output-on-failure

# Run benchmarks
cmake --build build --target bench
```

### Build and test (Windows, MSVC)

```bat
cmake -S . -B build
cmake --build build --config RelWithDebInfo --parallel
ctest --test-dir build -C RelWithDebInfo --output-on-failure
```

### Install as a CMake package

```bash
cmake --install build --prefix /usr/local
# Downstream CMakeLists.txt:
#   find_package(srfm REQUIRED)
#   target_link_libraries(my_target PRIVATE srfm::srfm_engine)
```

### Generate API documentation

```bash
doxygen Doxyfile
# Open docs/api/html/index.html in a browser
```

### Run the Python validation suite

```bash
pip install -e ".[dev]"
pytest                            # run all Python tests

# Full empirical pipeline (requires compiled C++ binaries in build/):
python validation/run_validation.py \
    --binary-regime   build/regime_validator \
    --binary-backtest build/backtest_runner
```

---

## Mathematical Background

The spacetime interval between two market events separated by time dt and price
displacement dp is:

    ds^2 = -c^2 dt^2 + dp^2

where c is a volatility-calibrated speed-of-information constant.

- **TIMELIKE** (ds^2 < 0): causal, information-bounded regime; empirically
  exhibits lower subsequent volatility.
- **SPACELIKE** (ds^2 > 0): acausal, momentum-dominated regime.
- **LIGHTLIKE** (ds^2 = 0): critical boundary.

The financial beta velocity is beta = |dp/dt| / c. The Lorentz factor
gamma = 1 / sqrt(1 - beta^2) weights a relativistic momentum signal. Geodesics
on a curved metric (induced by the local Christoffel symbols) define curvature-
adjusted trading strategies.

**Key equations:**

    gamma = 1 / sqrt(1 - beta^2)
    p_rel = gamma * m_eff * v_market
    phi   = atanh(beta)   (rapidity — additive under composition)
    beta_composed = (beta_1 + beta_2) / (1 + beta_1 * beta_2)

See [`paper/main.pdf`](paper/main.pdf) for the full derivation with proofs and
empirical validation. Build it locally:

```bash
cd paper && make pdf       # Full 3-pass LaTeX compile + BibTeX
cd paper && make figures   # Regenerate all 8 figures from Python
cd paper && make arxiv     # Build arXiv submission tarball
```

See `BENCHMARKS.md` for SIMD throughput figures.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for build requirements, sanitizer and
fuzz instructions, and the step-by-step guide for adding a new manifold
geometry.

---

## Version tagging

After every release commit, tag the repository and push both the commit and the
tag:

```bash
# Replace X.Y.Z with the new version (e.g., 1.0.0)
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main vX.Y.Z
```

GitHub Actions will run the full CI matrix against the tagged commit. The tag
is used by `srfmConfigVersion.cmake` for downstream `find_package` version
checks.

---

## License

MIT. See [LICENSE](LICENSE).
