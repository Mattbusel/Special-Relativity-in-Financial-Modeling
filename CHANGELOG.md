# Changelog

All notable changes to SRFM (Special Relativity in Financial Modeling) are
documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `tests/momentum/test_lorentz_invariants.cpp`: New test suite covering:
  - Lorentz interval invariance (`ds²` preserved under boosts at 7 velocities).
  - Spacetime velocity norm invariance along flat geodesics.
  - Velocity composition subluminality for a full (β₁, β₂) parameter grid.
  - Rapidity additivity with tight tolerance (EPS_MED = 1e-7).
  - Round-trip boost identity: `apply_momentum_correction` then
    `inverse_transform` recovers the original signal for all (β, m_eff, raw).
  - Newtonian expansion: γ ≈ 1 + β²/2 for |β| ≪ 1.
  - Subnormal (denormal) beta inputs.
  - Very large financial values (1e12 signals stay finite).
  - Very small financial values (subnormal signals handled without NaN).
  - `std::numeric_limits` boundary probes (max, lowest, epsilon, NaN, Inf).
  - Lorentz factor monotonicity and γ(-β) == γ(β) symmetry.
  - Velocity composition associativity.
- `CMakeLists.txt`: `test_lorentz_invariants` target wired into CTest.
- CI (`ci.yml`): Valgrind memory check step in `linux (gcc-12)` job.
- CI (`ci.yml`): Doxygen HTML generation step with artifact upload.
- README.md: paper build commands (`make pdf`, `make figures`, `make arxiv`),
  expanded Mathematical Background with key equations, Contributing section
  with API contract checklist and pre-PR commands.

---

## [1.0.0] - 2026-03-18

### Added
- Version bump to 1.0.0: first stable public release.
- README.md fully rewritten with architecture overview, module dependency
  graph, mathematical background, quickstart (Linux, Windows, CMake install),
  Python validation suite instructions, and version tag instructions.
- `pyproject.toml` version bumped to 1.0.0; classifier changed to
  "Development Status :: 5 - Production/Stable".
- `CMakeLists.txt` version bumped to 1.0.0.
- `Doxyfile` PROJECT_NUMBER bumped to 1.0.0.
- CI: `python-tests` job added to `.github/workflows/ci.yml`, running the
  pytest suite in `tests/python/` against Python 3.11 on Ubuntu.

### Changed
- All version identifiers aligned to 1.0.0 across CMakeLists.txt, Doxyfile,
  and pyproject.toml.

---

## [Unreleased items from gap-filling pass]

### Added
- `tests/manifold/test_interval_gaps.cpp`: Gap-filling tests covering:
  - SPACELIKE/TIMELIKE classification symmetry (swapping event direction).
  - Numerical stability with extreme coordinate values (1e12 price, 1e18 volume, 1e-12 precision).
  - Lorentz invariance: ds2 verified invariant under 1D boost at multiple velocities.
- `Doxyfile`: Doxygen configuration for generating HTML API documentation from `src/` and `include/`.
- `CMakeLists.txt`: `test_interval_gaps` target guarded by `GTest_FOUND`; `find_package(GTest CONFIG QUIET)` added.
- `CHANGELOG.md`: This file.

---

## [0.2.0] - 2026-03-01

### Added
- AGT-13: Adversarial hardening: three security issues identified and fixed.
  - Division by zero in Christoffel computation on degenerate metric (High).
  - Unbounded loop in GeodesicSolver with adversarial steps input (High).
  - Silent precision loss in BetaCalculator at boundary beta near BETA_MAX_SAFE (Medium).
- AGT-08: AVX-512 SIMD acceleration stubs (beta_avx2, beta_avx512, gamma_avx2, gamma_avx512).
- Property tests: 9 RapidCheck properties, 10,000 inputs each.
- Formal academic paper (LaTeX, arXiv-ready, sections 01 through 08, bibliography).
- Q1 2025 empirical validation: VR = 1.27x, Bartlett p = 6e-16, 10/11 assets significant.

### Changed
- CMakeLists.txt: version bumped from 0.1.0 to 0.2.0; install targets added.

---

## [0.1.0] - 2025-11-02

### Added
- Core Lorentz engine: BetaCalculator, LorentzTransform (gamma, rapidity, Doppler).
- SpacetimeMarketManifold: OHLCV to (t, P, V, M) spacetime embedding.
- SpacetimeInterval: ds2 computation and TIMELIKE/LIGHTLIKE/SPACELIKE classification.
- MomentumProcessor: relativistic momentum correction p_rel = gamma * m_eff * v.
- MetricTensor and ChristoffelSymbols: O(h2) central finite-difference Christoffel computation.
- GeodesicSolver: RK4 numerical integration of the geodesic equation.
- Backtester: Sharpe, Sortino, max drawdown, and gamma-weighted information ratio.
- Engine: single class wiring the full pipeline.
- DataLoader: CSV ingestion with per-row validation.
- CLI: --backtest and --stream modes.
- Interactive dashboard (viz/) using React + Vite.
- GitHub Actions CI: 4-compiler matrix, ASAN, UBSAN, TSAN, performance regression.

[Unreleased]: https://github.com/mattbusel/Special-Relativity-in-Financial-Modeling/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mattbusel/Special-Relativity-in-Financial-Modeling/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mattbusel/Special-Relativity-in-Financial-Modeling/releases/tag/v0.1.0
