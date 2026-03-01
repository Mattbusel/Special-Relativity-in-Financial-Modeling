#!/usr/bin/env bash
# ── build_n_asset.sh ───────────────────────────────────────────────────────────
# Bash shim: delegates to build_n_asset.ps1 via PowerShell.
#
# On Windows (Git Bash / WSL), runs:
#   pwsh -File scripts/build_n_asset.ps1
#
# On Linux/macOS with g++ and Eigen present, falls back to a native g++ build.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Prefer PowerShell (Windows primary environment) ───────────────────────────

if command -v pwsh &>/dev/null; then
    echo "=== build_n_asset.sh: delegating to PowerShell ==="
    pwsh -File "$SCRIPT_DIR/build_n_asset.ps1"
    exit $?
fi

# ── Fallback: native g++ build (Linux/macOS) ──────────────────────────────────

echo "=== build_n_asset.sh: pwsh not found, attempting g++ build ==="

EIGEN_INC="$REPO_ROOT/third_party/eigen"
BUILD_DIR="$REPO_ROOT/build/n_asset"
TESTS_DIR="$REPO_ROOT/tests/n_asset"

mkdir -p "$BUILD_DIR"

SRC_MANIFOLD="$REPO_ROOT/src/tensor/n_asset_manifold.cpp"
SRC_CHRISTOFFEL="$REPO_ROOT/src/tensor/christoffel_n.cpp"
SRC_GEODESIC="$REPO_ROOT/src/tensor/geodesic_n.cpp"
SRC_INTERVAL="$REPO_ROOT/src/manifold/n_asset_interval.cpp"
SRC_ENGINE="$REPO_ROOT/src/engine/n_asset_engine.cpp"

FLAGS="-std=c++20 -O2 -Wall -I$REPO_ROOT -I$EIGEN_INC -DEIGEN_MPL2_ONLY"

run_test() {
    local name="$1"
    local main="$2"
    shift 2
    local deps=("$@")
    local exe="$BUILD_DIR/$name"

    echo ""
    echo "── Building $name ──"
    # shellcheck disable=SC2086
    g++ $FLAGS "$main" "${deps[@]}" -o "$exe"
    echo "Running $exe ..."
    "$exe"
}

PASS=0
FAIL=0

run_test test_n_asset_manifold \
    "$TESTS_DIR/test_n_asset_manifold.cpp" \
    "$SRC_MANIFOLD" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

run_test test_christoffel_n \
    "$TESTS_DIR/test_christoffel_n.cpp" \
    "$SRC_MANIFOLD" "$SRC_CHRISTOFFEL" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

run_test test_geodesic_n \
    "$TESTS_DIR/test_geodesic_n.cpp" \
    "$SRC_MANIFOLD" "$SRC_CHRISTOFFEL" "$SRC_GEODESIC" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

run_test test_n_asset_interval \
    "$TESTS_DIR/test_n_asset_interval.cpp" \
    "$SRC_MANIFOLD" "$SRC_INTERVAL" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

run_test test_n_asset_engine \
    "$TESTS_DIR/test_n_asset_engine.cpp" \
    "$SRC_MANIFOLD" "$SRC_CHRISTOFFEL" "$SRC_GEODESIC" "$SRC_INTERVAL" "$SRC_ENGINE" \
    && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo ""
echo "══════════════════════════════════════"
echo "Stage 4 N-Asset Manifold — Results"
echo "Passed: $PASS  Failed: $FAIL"
echo "══════════════════════════════════════"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
