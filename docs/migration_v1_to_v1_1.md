# Migration Guide: v1.0.0 → v1.1.0

This guide covers what changed between the v1.0.0 and v1.1.0 releases of SRFM
and what you need to do to upgrade an existing integration.

---

## Summary

v1.1.0 is a backwards-compatible feature release. There are **no breaking
changes to any public C++ API, Rust API, or Python binding**. All code that
compiled and ran correctly under v1.0.0 continues to do so under v1.1.0
without modification.

The release adds new test targets, CMake configuration options, CI jobs, and
documentation. The CMake package version was bumped from 1.0.0 to 1.1.0.

---

## Breaking Changes

None. All public APIs are source- and ABI-compatible with v1.0.0.

---

## New APIs Added in v1.1.0

### CMake install: `find_package(srfm CONFIG REQUIRED)`

v1.1.0 ships a CMake package config file (`cmake/srfmConfig.cmake.in`).
Downstream projects can now consume SRFM as an installed CMake package without
copying headers manually:

```cmake
find_package(srfm CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE srfm::srfm_engine)
```

After installing SRFM (`cmake --install build --prefix /usr/local`), set
`CMAKE_PREFIX_PATH` to the install prefix so CMake can locate the package:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/usr/local
```

This API did not exist in v1.0.0, which required manual `include_directories`
and `target_link_libraries` with absolute paths.

### New CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `SRFM_WARNINGS_AS_ERRORS` | `OFF` | Treat all compiler warnings as errors (enabled automatically in CI). |
| `CMAKE_EXPORT_COMPILE_COMMANDS` | `ON` | Generate `compile_commands.json` for clang-tidy and IDE tooling. |

These options have no effect on the compiled library ABI. They only affect
how warnings are reported during your own build.

### New test executables registered with CTest

The following test executables are now automatically registered with CTest.
In v1.0.0 they existed but were not wired into `add_test()`, so `ctest` would
not run them automatically.

- `test_lorentz_transform`
- `test_beta_calculator`
- `test_online_beta`
- `test_backtester`
- `test_performance_metrics`
- `test_gamma_sizing`
- `test_metrics_precision`
- `test_full_pipeline`
- `test_error_handling`
- `test_metric_tensor`
- `test_christoffel`
- `test_geodesic`
- `test_n_asset`
- `test_stream`

**Action required:** If you previously ran these test binaries by hand, you
can now run the full suite with a single `ctest` invocation:

```bash
ctest --test-dir build --output-on-failure --parallel 4
```

### New integration test suite (`tests/integration/test_error_handling.cpp`)

22 new integration tests verify that every public API surface correctly handles
IEEE 754 special values (NaN, ±Inf, denormals) without crashing or producing
undefined behaviour. These tests serve as a regression guard.

No changes to your code are needed to benefit from this coverage, but if you
are implementing custom subclasses or wrappers, consider adding equivalent
IEEE 754 tests for your own code.

---

## Deprecated APIs

None. No API was deprecated between v1.0.0 and v1.1.0.

---

## How to Upgrade

### Step 1: Update the version references in your build

If you vendor SRFM or pin it to a tag, update to `v1.1.0`:

```bash
# git submodule
git submodule set-url <path> https://github.com/mattbusel/Special-Relativity-in-Financial-Modeling
git submodule update --remote

# direct clone
git fetch && git checkout v1.1.0
```

If your downstream `CMakeLists.txt` hard-codes a version requirement, update
it:

```cmake
# Before
find_package(srfm 1.0.0 CONFIG REQUIRED)

# After
find_package(srfm 1.1.0 CONFIG REQUIRED)
```

### Step 2: Re-run CMake configure

A clean reconfigure is recommended after any version bump:

```bash
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --parallel
```

### Step 3: Run the expanded test suite

```bash
ctest --test-dir build --output-on-failure --parallel 4
```

All 14 test executables listed above should now appear and pass. If any
previously-passing test fails after the upgrade, file an issue.

### Step 4: (Optional) Adopt the new CMake package interface

If you were previously linking against SRFM via absolute paths, migrate to the
installed package interface:

**Before (v1.0.0 style):**
```cmake
include_directories(/path/to/srfm/include)
target_link_libraries(my_target PRIVATE /path/to/srfm/build/libsrfm_engine.a)
```

**After (v1.1.0 style):**
```cmake
find_package(srfm 1.1.0 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE srfm::srfm_engine)
```

This ensures you automatically pick up the correct include directories,
compile definitions, and transitive dependencies.

---

## Version Reference

| Field | v1.0.0 | v1.1.0 |
|-------|--------|--------|
| `CMakeLists.txt` version | 1.0.0 | 1.1.0 |
| `Doxyfile` PROJECT_NUMBER | 1.0.0 | 1.1.0 |
| `pyproject.toml` version | 1.0.0 | 1.1.0 |
| CTest-registered executables | 0 | 14 |
| CMake package config | absent | present |
| `SRFM_WARNINGS_AS_ERRORS` option | absent | present |

---

## See Also

- `CHANGELOG.md` — complete list of all changes.
- `BUILD.md` — canonical build and test instructions.
- `ARCHITECTURE.md` — module contracts and design rules.
