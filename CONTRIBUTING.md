# Contributing to Special-Relativity-in-Financial-Modeling

Thank you for contributing to this research-grade C++ library. This guide covers build requirements, build and test instructions, how to add new manifold geometries, and how to compile the accompanying LaTeX paper.

---

## Build Requirements

| Requirement | Minimum Version | Notes |
|---|---|---|
| **CMake** | 3.25 | `cmake --version` |
| **Ninja** | 1.11 | Recommended generator (`-G Ninja`) |
| **GCC** | 12 or **Clang** 17 | Full C++20 required; both compilers tested in CI |
| **vcpkg** | latest | Set `VCPKG_ROOT` environment variable |
| **Eigen3** | 3.4+ | Matrix algebra for MetricTensor and geodesics (`vcpkg install eigen3`) |
| **fmt** | 10+ | Formatted output (`vcpkg install fmt`) |
| **nlohmann_json** | 3.11+ | JSON I/O (`vcpkg install nlohmann-json`) |
| **GTest** | 1.14+ | Unit and integration tests (`vcpkg install gtest`) |
| **RapidCheck** | optional | Property-based tests (`vcpkg install rapidcheck`) |
| **Google Benchmark** | optional | Fetched automatically via FetchContent if not found |

For the research paper:

| Requirement | Notes |
|---|---|
| **pdflatex** | Part of TeX Live or MiKTeX |
| **bibtex** | Bundled with most TeX distributions |
| **Python ≥ 3.10** | For figure generation scripts |
| **matplotlib, numpy, scipy** | `pip install matplotlib numpy scipy` |

---

## Build and Test Instructions

### Linux / WSL2 (recommended)

```bash
sudo apt install cmake ninja-build gcc-12 libgtest-dev
vcpkg install eigen3 fmt nlohmann-json rapidcheck

cmake -S . -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

cmake --build build

ctest --test-dir build --output-on-failure
```

### Windows (Visual Studio 2022)

Open an **x64 Native Tools Command Prompt for VS 2022**:

```powershell
vcpkg install eigen3:x64-windows fmt:x64-windows nlohmann-json:x64-windows gtest:x64-windows rapidcheck:x64-windows

cmake -S . -B build -G "Visual Studio 17 2022" `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

cmake --build build --config Release

ctest --test-dir build -C Release --output-on-failure
```

### Build with Sanitizers (Clang)

```bash
cmake -S . -B build-asan \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

cmake --build build-asan
ctest --test-dir build-asan --output-on-failure
```

### Build Fuzz Targets (Clang only)

```bash
cmake -S . -B build-fuzz \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DSRFM_FUZZ=ON \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

cmake --build build-fuzz --target fuzz_beta_calculator fuzz_manifold fuzz_geodesic fuzz_engine
```

Run a fuzz target (example):

```bash
mkdir -p corpus/beta
./build-fuzz/fuzz_beta_calculator corpus/beta -max_total_time=60
```

### Run Property-Based Tests

Property tests require RapidCheck. With RapidCheck installed:

```bash
# Run all 17 properties, 10,000 inputs each
RC_PARAMS="max_success=10000" ctest --test-dir build -R "^prop_" -V
```

### Run Benchmarks

```bash
cmake --build build --target bench
# Or directly:
./build/bench_beta_gamma --benchmark_format=json | tee bench/BENCHMARK_RESULTS.json
```

---

## How to Add a New Manifold Geometry

The market manifold is defined by a `MetricTensor` in `include/srfm/tensor.hpp`. Follow these steps to add a new geometry (for example, a metric encoding stochastic volatility correlations):

### Step 1 — Design the Metric Function

A `MetricTensor` is constructed from a `MetricFunction`:

```cpp
using MetricFunction = std::function<MetricMatrix(const SpacetimePoint&)>;
```

Design a function `g: SpacetimePoint → MetricMatrix` that encodes your geometry. The signature convention is `(−, +, +, +)`: component `g(0,0)` is timelike (negative scale), components `g(1..3, 1..3)` are spacelike (positive definite block).

### Step 2 — Add a Factory Method to MetricTensor

Open `include/srfm/tensor.hpp` and add a declaration:

```cpp
/**
 * @brief Construct a stochastic-volatility metric from a Heston-style covariance.
 *
 * @param time_scale   Scale of the time dimension.
 * @param spot_vol     Spot asset volatility.
 * @param vol_of_vol   Volatility of variance (ν in Heston notation).
 * @param correlation  Spot-vol correlation ρ ∈ (−1, 1).
 * @return Configured MetricTensor.
 */
static MetricTensor make_heston(double time_scale,
                                double spot_vol,
                                double vol_of_vol,
                                double correlation);
```

Implement it in `src/tensor/tensor.cpp`:

```cpp
MetricTensor MetricTensor::make_heston(double time_scale,
                                        double spot_vol,
                                        double vol_of_vol,
                                        double correlation) {
    return MetricTensor([=](const SpacetimePoint&) -> MetricMatrix {
        MetricMatrix g = MetricMatrix::Zero();
        g(0, 0) = -(time_scale * time_scale);
        g(1, 1) = spot_vol * spot_vol;
        g(2, 2) = vol_of_vol * vol_of_vol;
        g(1, 2) = g(2, 1) = correlation * spot_vol * vol_of_vol;
        g(3, 3) = 1.0;  // momentum coordinate: unit scale
        return g;
    });
}
```

### Step 3 — Validate the Metric

Add a unit test in `tests/tensor/` (mirror the test file pattern of existing tests):

```cpp
TEST(MetricTensorTest, HestonIsLorentzian) {
    auto g = srfm::tensor::MetricTensor::make_heston(1.0, 0.2, 0.3, -0.5);
    srfm::SpacetimePoint origin = srfm::SpacetimePoint::Zero();
    EXPECT_TRUE(g.is_lorentzian(origin));
}

TEST(MetricTensorTest, HestonIsInvertible) {
    auto g = srfm::tensor::MetricTensor::make_heston(1.0, 0.2, 0.3, -0.5);
    srfm::SpacetimePoint origin = srfm::SpacetimePoint::Zero();
    EXPECT_TRUE(g.inverse(origin).has_value());
}
```

### Step 4 — Verify Flat-Spacetime Geodesic Preservation

In flat spacetime (all Christoffel symbols zero), geodesics must be straight lines. Add a property test:

```cpp
rc::check("Heston flat geodesic preserves velocity", []() {
    auto g   = srfm::tensor::MetricTensor::make_heston(1.0, 0.2, 0.1, 0.0);
    auto sol = srfm::tensor::GeodesicSolver(g, 0.01);
    // ...velocity at final step should match initial velocity within 1e-6
});
```

### Step 5 — Integrate with the Engine (Optional)

If the new geometry is intended as a default, update `Engine::run_backtest` in `src/engine/engine.cpp` to accept a `MetricTensor` parameter and pass your new factory as the default.

### Step 6 — Run All Tests and Benchmarks

```bash
cmake --build build && ctest --test-dir build --output-on-failure
cmake --build build --target bench
```

Zero test failures required. No new compiler warnings under `-Wall -Wextra -Werror`.

---

## Running the LaTeX Paper

The formal research paper lives in `paper/`. It is a multi-file LaTeX document with a custom style (`srfm.sty`) and Python-generated figures.

### Generate Figures

```bash
cd paper
python figures/gen_all.py
# With backtester output data:
python figures/gen_all.py --data-dir /path/to/results/
```

Requires Python ≥ 3.10 with `matplotlib`, `numpy`, `scipy`.

### Compile the Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or using the Makefile:

```bash
cd paper && make pdf
```

Output: `paper/main.pdf` — the full research paper with all figures embedded.

### Build the arXiv Submission Tarball

```bash
cd paper && make arxiv
```

---

## Pull Request Checklist

- [ ] `ctest --test-dir build --output-on-failure` passes with zero failures
- [ ] No new compiler warnings under `-Wall -Wextra -Wpedantic -Werror`
- [ ] New public functions have Doxygen `/** @brief ... @param ... @return ... */` comments
- [ ] New geometry includes a factory method test and a flat-geodesic preservation test
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`
- [ ] `VERSION` file updated if a breaking API change was made
