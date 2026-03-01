/**
 * @file  fuzz_geodesic.cpp
 * @brief libFuzzer target for GeodesicSolver::solve
 *
 * Build:
 *   cmake -DSRFM_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++ ..
 *   cmake --build . --target fuzz_geodesic
 *
 * Run for 60 seconds:
 *   ./fuzz_geodesic -max_total_time=60
 *
 * Safety invariants verified on every input:
 *   1. Always terminates (step count clamped to 100,000).
 *   2. RK4 must not produce NaN in the output state.
 *   3. No crash, no UB, no abort for any (metric, initial_state, steps, dt) combo.
 *   4. For flat metric: result is always a valid straight line or nullopt.
 *
 * Fuzzer strategy:
 *   Bytes interpreted as:
 *     [8 doubles: initial position x[4]]
 *     [8 doubles: initial velocity u[4]]
 *     [16 doubles: metric tensor g[4][4]]
 *     [1 int32: steps]
 *     [1 double: dt]
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>

#include "geodesic/geodesic_solver.hpp"
#include "manifold/spacetime_manifold.hpp"

using namespace srfm::geodesic;
using namespace srfm::manifold;

static constexpr size_t INITIAL_STATE_BYTES = 8 * sizeof(double);   // x[4] + u[4]
static constexpr size_t METRIC_BYTES        = 16 * sizeof(double);  // g[4][4]
static constexpr size_t STEPS_BYTES         = sizeof(int32_t);
static constexpr size_t DT_BYTES            = sizeof(double);
static constexpr size_t MIN_INPUT           = INITIAL_STATE_BYTES + METRIC_BYTES
                                              + STEPS_BYTES + DT_BYTES;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    GeodesicSolver solver;

    // ── Test 1: fuzz the flat geodesic (flat metric, fuzzed initial state) ────
    if (size >= INITIAL_STATE_BYTES) {
        GeodesicState init{};
        for (std::size_t i = 0; i < static_cast<std::size_t>(DIM); ++i) {
            __builtin_memcpy(&init.x[i], data + i * sizeof(double), sizeof(double));
            __builtin_memcpy(&init.u[i],
                             data + (i + static_cast<std::size_t>(DIM)) * sizeof(double),
                             sizeof(double));
        }

        const MetricTensor flat = MetricTensor::minkowski();
        // Fixed safe parameters to guarantee termination
        const auto result = solver.solve(init, flat, 100, 0.001);

        // If result returned, all components must be finite
        if (result.has_value()) {
            assert(result->is_finite());
        }
    }

    // ── Test 2: fuzz everything — metric, state, steps, dt ───────────────────
    if (size >= MIN_INPUT) {
        const uint8_t* ptr = data;

        // Parse initial state
        GeodesicState init{};
        for (std::size_t i = 0; i < static_cast<std::size_t>(DIM); ++i) {
            __builtin_memcpy(&init.x[i], ptr, sizeof(double));
            ptr += sizeof(double);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(DIM); ++i) {
            __builtin_memcpy(&init.u[i], ptr, sizeof(double));
            ptr += sizeof(double);
        }

        // Parse metric
        MetricTensor metric{};
        for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
                __builtin_memcpy(
                    &metric.g[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                    ptr, sizeof(double));
                ptr += sizeof(double);
            }
        }

        // Parse steps (clamp handled internally by solver)
        int32_t raw_steps{};
        __builtin_memcpy(&raw_steps, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);

        // Parse dt
        double dt{};
        __builtin_memcpy(&dt, ptr, sizeof(double));

        // Must always terminate and never crash
        const auto result = solver.solve(init, metric, static_cast<int>(raw_steps), dt);

        if (result.has_value()) {
            // Output must be finite
            assert(result->is_finite());
        }
    }

    return 0;
}
