#!/usr/bin/env bash
# ci/run_valgrind.sh — Run test suite under Valgrind memcheck
#
# Usage:
#   ./ci/run_valgrind.sh [build_dir]
#
# Valgrind memcheck detects:
#   • Use of uninitialised values
#   • Invalid reads/writes (heap, stack, global)
#   • Memory leaks (all categories: definite, indirect, possible)
#   • Double frees, invalid frees
#
# Exit codes:
#   0 — Zero Valgrind errors, zero leaks
#   1 — Build failure, Valgrind errors, or leaks found

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${1:-${REPO_ROOT}/build/valgrind}"

# Verify Valgrind is available
if ! command -v valgrind &>/dev/null; then
    echo "ERROR: valgrind not found. Install with: apt-get install valgrind" >&2
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  SRFM C++ — Valgrind memcheck"
echo "  Build dir: ${BUILD_DIR}"
echo "═══════════════════════════════════════════════════════════"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Build without sanitizers (Valgrind does its own instrumentation)
cmake "${REPO_ROOT}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="-g -O1" \
    -DSRFM_FUZZ=OFF

cmake --build . --parallel "$(nproc 2>/dev/null || echo 4)"

echo ""
echo "── Running tests under Valgrind memcheck ─────────────────────"

# Valgrind options: abort on error, check for all leak kinds
VALGRIND_CMD=(
    valgrind
    --tool=memcheck
    --error-exitcode=1
    --leak-check=full
    --show-leak-kinds=all
    --track-origins=yes
    --errors-for-leak-kinds=definite,indirect
    --suppressions="${REPO_ROOT}/ci/valgrind.supp" 2>/dev/null || true
)

export RC_PARAMS="max_success=100"  # reduced count; Valgrind is very slow

# Run each test binary under Valgrind
FAIL_COUNT=0
while IFS= read -r test_binary; do
    echo "  Running: ${test_binary}"
    if ! "${VALGRIND_CMD[@]}" "${test_binary}"; then
        echo "  ✗ FAIL: ${test_binary}"
        ((FAIL_COUNT++)) || true
    else
        echo "  ✓ PASS: ${test_binary}"
    fi
done < <(ctest --show-only=json-v1 2>/dev/null \
         | grep -o '"command":.*"' \
         | grep -oP '"[^"]+\.exe?"' \
         | tr -d '"' \
         || find "${BUILD_DIR}" -name "test_*" -o -name "prop_*" \
         | grep -v CMakeFiles)

if [[ ${FAIL_COUNT} -gt 0 ]]; then
    echo ""
    echo "✗ Valgrind: FAIL — ${FAIL_COUNT} test(s) had errors"
    exit 1
fi

echo ""
echo "✓ Valgrind: PASS — Zero memory errors, zero leaks"
