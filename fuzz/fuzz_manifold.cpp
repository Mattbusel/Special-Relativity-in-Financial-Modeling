/**
 * @file  fuzz_manifold.cpp
 * @brief libFuzzer target for SpacetimeManifold::process and christoffelSymbols
 *
 * Build:
 *   cmake -DSRFM_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++ ..
 *   cmake --build . --target fuzz_manifold
 *
 * Run for 60 seconds:
 *   ./fuzz_manifold -max_total_time=60
 *
 * Safety invariants verified on every input:
 *   1. No crash, no UB, no abort for any SpacetimeEvent values including
 *      NaN, ±Inf, ±0.0, subnormals.
 *   2. process() always returns a valid Regime or std::nullopt.  Never aborts.
 *   3. christoffelSymbols() for the flat metric always returns all zeros.
 *   4. christoffelSymbols() always returns 64 finite values (for valid metrics).
 *
 * Fuzzer strategy:
 *   Input bytes → 4 doubles → SpacetimeEvent{t, x, y, z}
 *   Exercises all IEEE 754 corner cases.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>

#include "manifold/spacetime_manifold.hpp"

using namespace srfm::manifold;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    SpacetimeManifold manifold;

    // ── Test 1: SpacetimeManifold::process with fuzzed event ──────────────────
    if (size >= 4 * sizeof(double)) {
        SpacetimeEvent event{};
        __builtin_memcpy(&event.t, data + 0 * sizeof(double), sizeof(double));
        __builtin_memcpy(&event.x, data + 1 * sizeof(double), sizeof(double));
        __builtin_memcpy(&event.y, data + 2 * sizeof(double), sizeof(double));
        __builtin_memcpy(&event.z, data + 3 * sizeof(double), sizeof(double));

        // Must not crash or abort; may return nullopt for non-finite inputs
        const auto regime = manifold.process(event);

        if (event.is_finite()) {
            // For finite inputs, must always return a valid regime
            assert(regime.has_value());
        }

        if (regime.has_value()) {
            // Regime must be one of the defined enum values
            const int r = static_cast<int>(*regime);
            assert(r >= 0 && r <= 3);
        }
    }

    // ── Test 2: christoffelSymbols on fuzzed metric ────────────────────────────
    if (size >= static_cast<size_t>(DIM * DIM) * sizeof(double)) {
        MetricTensor fuzz_metric{};
        for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
                double val{};
                const size_t offset =
                    static_cast<size_t>(i * DIM + j) * sizeof(double);
                __builtin_memcpy(&val, data + offset, sizeof(double));
                fuzz_metric.g[static_cast<size_t>(i)][static_cast<size_t>(j)] = val;
            }
        }

        // Must not crash; may return all-zeros for invalid metric
        const auto christoffel = manifold.christoffelSymbols(fuzz_metric);

        // All 64 entries must be finite (we return zeros for invalid metric)
        for (std::size_t k = 0; k < static_cast<std::size_t>(NUM_CHRISTOFFEL); ++k) {
            assert(std::isfinite(christoffel[k]));
        }
    }

    // ── Test 3: flat metric always gives zero Christoffels ────────────────────
    {
        const MetricTensor flat = MetricTensor::minkowski();
        const auto christoffel = manifold.christoffelSymbols(flat);
        for (std::size_t k = 0; k < static_cast<std::size_t>(NUM_CHRISTOFFEL); ++k) {
            assert(std::isfinite(christoffel[k]));
            assert(std::abs(christoffel[k]) < 1e-8);
        }
    }

    return 0;
}
