# Build & Verification Guide — SRFM (Special Relativity in Financial Modeling)

## Prerequisites

| Tool | Minimum version |
|------|-----------------|
| CMake | 3.25 |
| C++ compiler | GCC 12 / Clang 16 / Clang 17 / MSVC 2022 (C++20 required) |
| vcpkg | any recent |
| fmt | via `vcpkg install fmt` |
| GTest | via `vcpkg install gtest` (optional, for tests) |
| RapidCheck | via `vcpkg install rapidcheck` (optional, for property tests) |
| Doxygen | 1.9+ (optional, for API docs) |

Set the `VCPKG_ROOT` environment variable to your vcpkg installation directory.

---

## Build (Linux / macOS)

```bash
git clone https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling
cd Special-Relativity-in-Financial-Modeling

# Configure (Release)
cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Build all targets
cmake --build build --parallel

# Run benchmarks (the C++ project builds libraries; the CLI is the Rust binary)
./build/bench/bench_beta_gamma --benchmark_format=json
# For the Rust CLI:
cargo run --release -- --help
```

## Build (Windows, MSVC)

```powershell
cmake -S . -B build `
      -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts/buildsystems/vcpkg.cmake"
cmake --build build --config RelWithDebInfo --parallel

.\build\RelWithDebInfo\srfm.exe --help
```

---

## Run Tests

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Expected output:

```
Test project .../build
      Start  1: MomentumTests
 1/12 Test  #1: MomentumTests ...................... Passed
      Start  2: ManifoldTests
 2/12 Test  #2: ManifoldTests ...................... Passed
...
100% tests passed, 0 tests failed out of 12
```

---

## Run Benchmarks

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSRFM_BUILD_BENCH=ON \
      -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --parallel
./build/bench/bench_beta_gamma
```

---

## Code Quality Checks

### clang-format

```bash
find src include -name "*.cpp" -o -name "*.hpp" | xargs clang-format --dry-run -Werror
```

To auto-format:

```bash
find src include -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i
```

### clang-tidy

```bash
cmake -S . -B build-tidy -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
clang-tidy -p build-tidy src/**/*.cpp
```

### cppcheck

```bash
cppcheck --enable=all --std=c++20 --error-exitcode=1 src/
```

---

## Generate API Documentation

```bash
doxygen Doxyfile
# Open docs/api/html/index.html in a browser
xdg-open docs/api/html/index.html   # Linux
open docs/api/html/index.html       # macOS
```

---

## Sanitizer Builds

```bash
# Address Sanitizer
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" \
      -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build-asan && ctest --test-dir build-asan --output-on-failure

# Thread Sanitizer
cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=thread" \
      -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build-tsan && ctest --test-dir build-tsan --output-on-failure
```

The CI scripts in `ci/` also wrap ASAN, MSAN, TSAN, and Valgrind.

---

## Install as CMake Package

```bash
cmake --install build --prefix /usr/local
```

Downstream `CMakeLists.txt`:

```cmake
find_package(srfm REQUIRED)
target_link_libraries(my_target PRIVATE srfm::srfm_engine)
```

---

## Verification Checklist

- [ ] `cmake -S . -B build` succeeds with no errors
- [ ] `cmake --build build` succeeds
- [ ] `ctest --test-dir build --output-on-failure` — all tests pass
- [ ] `cargo run --release -- --help` prints usage (Rust CLI)
- [ ] `./build/bench/bench_beta_gamma` runs benchmarks (C++ benchmark binary)
- [ ] `clang-format --dry-run` passes (no style violations)
- [ ] Doxygen generates without `WARN_IF_UNDOCUMENTED` warnings

---

## Troubleshooting

### `fmt/core.h: No such file or directory`

Install fmt via vcpkg:

```bash
$VCPKG_ROOT/vcpkg install fmt
```

Confirm `CMAKE_TOOLCHAIN_FILE` points to your vcpkg installation.

### `error: no matching function for call to 'BetaCalculator::...'`

Ensure you are on the `main` branch and headers in `include/` match the implementation in `src/`.

### Tests fail with `BETA_MAX_SAFE` boundary errors

The test fixture uses the value from `include/srfm/constants.hpp`. If you changed `BETA_MAX_SAFE`, update the test baseline in `tests/beta_calculator/`.

### `ctest` cannot find test executables

Ensure you built with `-DCMAKE_BUILD_TYPE=Debug` and that `GTest_FOUND` is true:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=... \
      -DCMAKE_VERBOSE_MAKEFILE=ON
```
