# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.2.0] - 2026-03-17

### Added
- Production-readiness pass: `[[nodiscard]]` on all public functions whose
  return values must not be discarded.
- `noexcept` specifications verified and documented on all
  performance-critical paths.
- Doxygen doc comments added to every public function and class across all
  headers (`@brief`, `@param`, `@return`).
- Doxyfile updated to version 0.2.0 and wired into CI (doxygen generation
  step in the `linux` workflow).
- CI: Valgrind memory-check step added to the `linux (gcc-12)` matrix cell.
- New test suite `test_lorentz_invariants.cpp` covering:
  - Lorentz interval invariance (`ds²` preserved under boosts).
  - Spacetime norm invariance (`u^μ u_μ` conserved along geodesics).
  - Velocity composition subluminality for the full parameter grid.
  - Rapidity additivity with tight tolerance.
  - Round-trip boost identity `Λ(β)·Λ(−β) = I`.
  - Numerical edge cases: subnormal inputs, very large/small financial values,
    `std::numeric_limits` boundary probes.
- CMakeLists.txt version bumped to 0.2.0; `test_lorentz_invariants` wired
  into CTest.
- CHANGELOG.md entry for 0.2.0.
- README.md: expanded Architecture section, Quickstart command block,
  Mathematical Background section, Contributing section, and paper reference.

### Changed
- CMakeLists.txt `VERSION` updated from `0.1.0` to `0.2.0`.
- Doxyfile `PROJECT_NUMBER` updated from `0.1.0` to `0.2.0`.

---

## [0.1.0-unreleased-items]

### Added
- CMake install targets and package config files (`srfmConfig.cmake`,
  `srfmConfigVersion.cmake`, `srfmTargets.cmake`) so library targets can be
  consumed downstream via `find_package(srfm)`.
- `cmake/srfmConfig.cmake.in` template for package configuration.
- CI: clang-tidy-17 static analysis step (clang-17 matrix cell).
- CI: cppcheck static analysis step (gcc-12 matrix cell).
- CI: `actions/cache@v4` caching of CMake build artifacts keyed on source hash
  for both Linux and Windows jobs.
- CI: upgraded runner images from `ubuntu-22.04` to `ubuntu-latest`.
- Fixed badge URLs in README.md: replaced `username` placeholder with `Mattbusel`.
- `CONTRIBUTING.md`: full build requirements, build/test/sanitizer/fuzz instructions,
  step-by-step guide for adding a new manifold geometry, LaTeX paper build instructions.
- `VERSION` file at the repo root containing `1.0.0`.
- `docs/Doxyfile`: Doxygen configuration targeting `src/` and `include/`, output to `docs/api/`.
- Doxygen for generating HTML API documentation via `doxygen Doxyfile`.
- `.github/workflows/ci.yml`: complete CI matrix covering cmake configure, C++20
  build, ctest, and clang-format lint on Ubuntu and Windows.
- Edge-case test coverage: NaN/infinity inputs, velocity >= c, zero time
  intervals, degenerate metric tensors.
- `CHANGELOG.md` (this file).
- Three new RapidCheck property tests: `prop_interval_invariant` (Lorentz
  interval invariance), `prop_metric_positive_definite` (metric tensor
  signature), `prop_christoffel_symmetry` (lower-index symmetry Γ^λ_μν = Γ^λ_νμ).
- Rust crate `tokio-prompt-orchestrator` (`Cargo.toml`, `src/lib.rs`):
  version 0.1.0 with `web-api`, `mcp`, `tui` feature flags.
- Python package `srfm-validation` (`pyproject.toml`): version 0.1.0 with
  numpy/pandas/scipy dependencies and pytest configuration.
- Sphinx documentation configuration (`docs/conf.py`) with autodoc, napoleon,
  mathjax, and myst_parser extensions.
- Python pytest suite (`tests/python/`): 18 tests covering `analyze_q1.py`
  statistical helpers and `run_validation.py` orchestration.
- CI `python-tests` job running pytest on Ubuntu with Python 3.11.

### Changed
- README.md fully rewritten: CI/standard badges, feature table, math background,
  prerequisites, build instructions, quickstart, architecture overview, and
  contributing guide.
- ARCHITECTURE.md updated to reflect the SRFM C++ module graph (previously
  contained documentation for an unrelated Rust project).

### Fixed
- ARCHITECTURE.md content was stale and described a different project; replaced
  with accurate SRFM C++ architecture documentation.

---

## [0.3.0] - 2026-03-01

### Added
- Adversarial hardening pass (AGT-13): fuzz targets (`fuzz/`), ASAN/TSAN/MSAN
  CI jobs, and three security fixes.
- Security fix: division-by-zero guard in Christoffel computation on degenerate
  metric (`MetricTensor::is_valid()` guards all paths).
- Security fix: `GeodesicSolver::solve()` now clamps `steps` to [1, 100000] and
  `dt` to [1e-8, 1.0], preventing adversarial unbounded loops.
- Security fix: `BetaCalculator` clamps result to `BETA_MAX_SAFE - 1e-7` at the
  boundary to eliminate silent precision loss.
- Property-based test suite: 17 properties x 10 000 random inputs via
  RapidCheck.
- `HARDENING_REPORT.md` documenting all AGT-13 findings.

### Changed
- All public API functions now return `std::optional` instead of asserting or
  calling `abort()` on invalid inputs.
- `RelativisticSignalProcessor` is now fully stateless and `noexcept`.

---

## [0.2.0] - 2026-01-15

### Added
- Tensor calculus module: `MetricTensor`, Christoffel symbols (64 coefficients),
  RK4 geodesic solver.
- `GeodesicSolver` validated against flat-spacetime straight-line property and
  analytic Bernoulli solution on a curved metric.
- Relativistic backtester: Sharpe, Sortino, max drawdown, gamma-weighted
  information ratio.
- AVX2 and AVX-512F SIMD acceleration for batch beta/gamma computation with
  runtime dispatch.
- Interactive dashboard (`viz/`) with real-time beta/gamma slider.
- Formal research paper (LaTeX, arXiv-ready) with Q1 2025 empirical validation.

### Changed
- Engine wired into a single `Engine` class for the full CSV-to-signal pipeline.
- `DataLoader` skips malformed rows without crashing.

---

## [0.1.0] - 2025-11-01

### Added
- Initial release: Lorentz transform engine, beta calculator, spacetime interval
  classifier, relativistic momentum processor.
- `BetaVelocity`, `LorentzFactor`, `EffectiveMass` strong types with
  construction-time validation.
- Spacetime interval classification: TIMELIKE / SPACELIKE / LIGHTLIKE.
- Unit tests for all core modules.
- Q1 2025 empirical results: variance ratio 1.27x, Bartlett p = 6e-16.

---

[Unreleased]: https://github.com/username/Special-Relativity-in-Financial-Modeling/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/username/Special-Relativity-in-Financial-Modeling/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/username/Special-Relativity-in-Financial-Modeling/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/username/Special-Relativity-in-Financial-Modeling/releases/tag/v0.1.0
