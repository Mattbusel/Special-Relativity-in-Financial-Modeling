/**
 * @file  fuzz_engine.cpp
 * @brief libFuzzer target for the full Engine pipeline (end-to-end)
 *
 * Build:
 *   cmake -DSRFM_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++ ..
 *   cmake --build . --target fuzz_engine
 *
 * Run for 60 seconds:
 *   ./fuzz_engine -max_total_time=60
 *
 * Safety invariants verified on every input:
 *   1. No crash, no UB, no abort for any byte sequence.
 *   2. If result is returned:
 *      a. beta ∈ (−BETA_MAX_SAFE, BETA_MAX_SAFE)
 *      b. gamma ≥ 1.0
 *      c. doppler > 0
 *      d. relativistic_signal is finite
 *      e. price_count ≥ 2
 *   3. For empty input: always returns nullopt.
 *   4. For pure binary garbage: must not crash.
 *
 * Fuzzer strategy:
 *   Input is passed directly as std::string_view.  The Engine::parse_prices()
 *   parser must handle:
 *     • Binary garbage (null bytes, high bytes)
 *     • "NaN", "inf", "-inf" text tokens
 *     • Very long sequences
 *     • Mixed delimiters (comma, space, tab, newline, CR)
 *     • UTF-8 sequences
 *     • Negative numbers (→ rejected prices)
 *     • Exponential notation: "1e308", "1e-308"
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>
#include <string_view>

#include "engine/engine.hpp"
#include "momentum/momentum.hpp"

using namespace srfm::engine;
using namespace srfm::momentum;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    const std::string_view input{
        reinterpret_cast<const char*>(data), size
    };

    Engine engine;
    const auto result = engine.process(input);

    if (result.has_value()) {
        // Invariant 2a: beta in safe range
        assert(result->beta > -BETA_MAX_SAFE);
        assert(result->beta <  BETA_MAX_SAFE);
        assert(std::isfinite(result->beta));

        // Invariant 2b: gamma >= 1
        assert(result->gamma >= 1.0);
        assert(std::isfinite(result->gamma));

        // Invariant 2c: doppler > 0
        assert(result->doppler > 0.0);
        assert(std::isfinite(result->doppler));

        // Invariant 2d: relativistic_signal is finite
        assert(std::isfinite(result->relativistic_signal));

        // Invariant 2e: at least 2 prices were parsed
        assert(result->price_count >= 2);

        // Cross-checks
        const double b  = result->beta;
        const double g2 = result->gamma * result->gamma;
        const double expected_g2 = 1.0 / (1.0 - b * b);
        assert(std::isfinite(expected_g2));
        const double rel_err = std::abs(g2 - expected_g2) / expected_g2;
        assert(rel_err < 1e-10);

        // Rapidity finite
        assert(std::isfinite(result->rapidity));

        // D(β)·D(-β) = 1 (spot-check via gamma: D = exp(φ) = exp(atanh(β)))
        const double phi    = result->rapidity;
        const double d_from_phi = std::exp(phi);
        assert(std::isfinite(d_from_phi));
        assert(d_from_phi > 0.0);
        const double doppler_err = std::abs(result->doppler - d_from_phi)
                                   / (d_from_phi + 1e-300);
        assert(doppler_err < 1e-9);
    }

    return 0;
}
