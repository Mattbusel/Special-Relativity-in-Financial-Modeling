#!/usr/bin/env bash
# ci/run_asan.sh — Build and run test suite with AddressSanitizer + UBSan
#
# Usage:
#   ./ci/run_asan.sh [build_dir]
#
# Prerequisites:
#   - GCC ≥ 12 or Clang ≥ 17 with ASAN support
#   - CMake ≥ 3.25
#   - vcpkg with RapidCheck installed
#
# Exit codes:
#   0 — All tests pass, zero ASAN/UBSAN errors
#   1 — Build failure or test failures or sanitizer errors

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${1:-${REPO_ROOT}/build/asan}"

echo "═══════════════════════════════════════════════════════════"
echo "  SRFM C++ — AddressSanitizer + UndefinedBehaviorSanitizer"
echo "  Build dir: ${BUILD_DIR}"
echo "═══════════════════════════════════════════════════════════"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with ASAN + UBSAN
cmake "${REPO_ROOT}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
    -DSRFM_FUZZ=OFF

# Build all targets
cmake --build . --parallel "$(nproc 2>/dev/null || echo 4)"

echo ""
echo "── Running test suite under ASAN+UBSAN ──────────────────────"

# Environment for ASAN: abort on first error
export ASAN_OPTIONS="abort_on_error=1:detect_leaks=1:check_initialization_order=1:strict_string_checks=1:detect_stack_use_after_return=1"
export UBSAN_OPTIONS="abort_on_error=1:print_stacktrace=1:halt_on_error=1"

# Run property tests with reduced count under sanitizers (still thorough)
export RC_PARAMS="max_success=1000"

ctest --output-on-failure --parallel "$(nproc 2>/dev/null || echo 4)"

echo ""
echo "✓ ASAN+UBSAN: PASS — Zero sanitizer errors"
