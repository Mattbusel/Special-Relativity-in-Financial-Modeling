#!/usr/bin/env bash
# ci/run_tsan.sh — Build and run test suite with ThreadSanitizer
#
# Usage:
#   ./ci/run_tsan.sh [build_dir]
#
# Focus:
#   Verifies that the stateless C++ classes (RelativisticSignalProcessor,
#   SpacetimeManifold, GeodesicSolver, Engine) are truly thread-safe
#   when called from multiple threads concurrently.
#
# Note: TSAN and ASAN are mutually exclusive.  Run in separate CI jobs.
#
# Exit codes:
#   0 — Zero data races detected
#   1 — Build failure or data races found

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${1:-${REPO_ROOT}/build/tsan}"

echo "═══════════════════════════════════════════════════════════"
echo "  SRFM C++ — ThreadSanitizer"
echo "  Build dir: ${BUILD_DIR}"
echo "═══════════════════════════════════════════════════════════"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with TSAN
cmake "${REPO_ROOT}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
    -DSRFM_FUZZ=OFF

cmake --build . --parallel "$(nproc 2>/dev/null || echo 4)"

echo ""
echo "── Running stream/concurrency tests under TSAN ──────────────"

export TSAN_OPTIONS="abort_on_error=1:second_deadlock_stack=1:halt_on_error=1"
export RC_PARAMS="max_success=500"

# Run tests; TSAN will abort if a data race is detected
ctest --output-on-failure --parallel 1  # sequential to let TSAN observe races

echo ""
echo "✓ TSAN: PASS — Zero data races detected"
