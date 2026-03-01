#!/usr/bin/env bash
# ci/run_msan.sh — Build and run test suite with MemorySanitizer (Clang only)
#
# Usage:
#   ./ci/run_msan.sh [build_dir]
#
# Prerequisites:
#   - Clang ≥ 17 (MSAN is Clang-only; GCC does not support it)
#   - All dependencies (including libstdc++) must be MSAN-instrumented.
#     For simplicity we use the system allocator and only instrument our code.
#   - Set CC=clang CXX=clang++ before running.
#
# Exit codes:
#   0 — Zero uninitialized reads detected
#   1 — Build failure or MSAN errors

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${1:-${REPO_ROOT}/build/msan}"

# Verify Clang is being used
if ! command -v clang++ &>/dev/null; then
    echo "ERROR: clang++ not found. MSAN requires Clang ≥ 17." >&2
    exit 1
fi

CLANG_VERSION=$(clang++ --version | grep -oP 'clang version \K[0-9]+' | head -1)
if [[ "${CLANG_VERSION:-0}" -lt 17 ]]; then
    echo "ERROR: Clang ${CLANG_VERSION} found; MSAN requires Clang ≥ 17." >&2
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  SRFM C++ — MemorySanitizer (Clang ${CLANG_VERSION})"
echo "  Build dir: ${BUILD_DIR}"
echo "═══════════════════════════════════════════════════════════"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with MSAN
cmake "${REPO_ROOT}" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="-fsanitize=memory -fno-omit-frame-pointer -g -fsanitize-memory-track-origins=2" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=memory" \
    -DSRFM_FUZZ=OFF

cmake --build . --parallel "$(nproc 2>/dev/null || echo 4)"

echo ""
echo "── Running test suite under MSAN ────────────────────────────"

export MSAN_OPTIONS="abort_on_error=1:print_stats=1:halt_on_error=1"
export RC_PARAMS="max_success=500"

ctest --output-on-failure --parallel "$(nproc 2>/dev/null || echo 4)"

echo ""
echo "✓ MSAN: PASS — Zero uninitialized reads detected"
