/**
 * @file  fuzz_beta_calculator.cpp
 * @brief libFuzzer target for BetaCalculator::fromPriceVelocityOnline
 *
 * Build:
 *   cmake -DSRFM_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++ ..
 *   cmake --build . --target fuzz_beta_calculator
 *
 * Run for 60 seconds:
 *   ./fuzz_beta_calculator -max_total_time=60
 *
 * Safety invariants verified on every input:
 *   1. No crash, no UB, no abort.
 *   2. If result is returned, beta ∈ (−BETA_MAX_SAFE, BETA_MAX_SAFE).
 *   3. If result is returned, gamma ≥ 1.0.
 *   4. If result is returned, doppler > 0.
 *   5. The function always returns within finite time.
 *
 * Fuzzer strategy:
 *   The input bytes are interpreted as raw doubles via memcpy.
 *   This exercises all IEEE 754 bit patterns including:
 *     • NaN (quiet and signalling), ±Inf, ±0, denormals
 *     • Negative prices (rejected as invalid)
 *     • Extremely large/small prices
 *     • Sequences shorter than 2 prices (nullopt expected)
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>

#include "beta_calculator/beta_calculator.hpp"

using namespace srfm::beta_calculator;
using namespace srfm::momentum;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Interpret input bytes as a sequence of doubles
    const size_t n_doubles = size / sizeof(double);
    std::vector<double> prices;
    prices.reserve(n_doubles + 1);

    for (size_t i = 0; i < n_doubles; ++i) {
        double val{};
        __builtin_memcpy(&val, data + i * sizeof(double), sizeof(double));
        prices.push_back(val);
    }

    // Also try a c_market value from the trailing byte, if available
    double c_market = 1.0;
    if ((size % sizeof(double)) > 0) {
        const uint8_t last_byte = data[size - 1u];
        // Map [0, 255] → [0.01, 25.6]
        c_market = (static_cast<double>(last_byte) + 1.0) * 0.1;
    }

    BetaCalculator calc;
    const auto result = calc.fromPriceVelocityOnline(prices, c_market);

    if (result.has_value()) {
        // Safety invariant 2: beta in safe range
        assert(result->beta > -BETA_MAX_SAFE);
        assert(result->beta <  BETA_MAX_SAFE);
        assert(std::isfinite(result->beta));

        // Safety invariant 3: gamma >= 1
        assert(result->gamma >= 1.0);
        assert(std::isfinite(result->gamma));

        // Safety invariant 4: doppler > 0 and finite
        assert(result->doppler > 0.0);
        assert(std::isfinite(result->doppler));

        // Rapidity must be finite
        assert(std::isfinite(result->rapidity));

        // Cross-check: gamma^2 ≈ 1/(1-beta^2)
        const double b  = result->beta;
        const double g2 = result->gamma * result->gamma;
        const double expected = 1.0 / (1.0 - b * b);
        assert(std::isfinite(expected));
        const double rel_err = std::abs(g2 - expected) / expected;
        assert(rel_err < 1e-10);
    }

    return 0;
}
