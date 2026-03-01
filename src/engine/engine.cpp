/**
 * @file  engine.cpp
 * @brief Full SRFM pipeline engine implementation (AGT-13 / SRFM).
 *
 * See engine.hpp for the full module contract.
 */

#include "engine.hpp"

#include "../beta_calculator/beta_calculator.hpp"
#include "../geodesic/geodesic_solver.hpp"
#include "../manifold/spacetime_manifold.hpp"
#include "../momentum/momentum.hpp"

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstring>

namespace srfm::engine {

// ── Price parsing ─────────────────────────────────────────────────────────────

std::vector<double>
Engine::parse_prices(std::string_view data) const noexcept {
    std::vector<double> prices;
    prices.reserve(64);

    const char* ptr = data.data();
    const char* end = ptr + data.size();

    while (ptr < end) {
        // Skip non-numeric leading characters (delimiters and garbage)
        while (ptr < end && !std::isdigit(static_cast<unsigned char>(*ptr))
                         && *ptr != '-' && *ptr != '+' && *ptr != '.') {
            ++ptr;
        }
        if (ptr >= end) break;

        // Attempt to parse a double
        double value = 0.0;
        auto [next_ptr, ec] = std::from_chars(ptr, end, value);
        if (ec == std::errc{} && std::isfinite(value) && value > 0.0) {
            prices.push_back(value);
        }
        // Advance past the attempted token (at least one char to avoid infinite loop)
        if (next_ptr == ptr) {
            ++ptr;
        } else {
            ptr = next_ptr;
        }
    }

    return prices;
}

// ── Engine::process ───────────────────────────────────────────────────────────

std::optional<PipelineResult>
Engine::process(std::string_view data) const noexcept {
    // Step 1: Parse prices
    const auto prices = parse_prices(data);
    if (prices.size() < 2) return std::nullopt;

    // Step 2: BetaCalculator — compute β from streaming prices
    beta_calculator::BetaCalculator calc;
    auto beta_result = calc.fromPriceVelocityOnline(prices, 1.0);
    if (!beta_result) return std::nullopt;

    // Step 3: SpacetimeManifold — classify regime using mean price as event x
    manifold::SpacetimeManifold mfld;
    double mean_price = 0.0;
    for (const double p : prices) mean_price += p;
    mean_price /= static_cast<double>(prices.size());

    manifold::SpacetimeEvent event{
        static_cast<double>(prices.size()), // t = time index
        beta_result->beta,                  // x = beta (velocity proxy)
        mean_price,                         // y = mean price
        beta_result->gamma - 1.0            // z = Lorentz excess
    };

    auto regime_opt = mfld.process(event);
    if (!regime_opt) return std::nullopt;

    // Step 4: GeodesicSolver — integrate one geodesic in flat spacetime
    // (verifies the solver doesn't crash; result is a straight line in flat space)
    geodesic::GeodesicSolver solver;
    const auto flat_metric = manifold::MetricTensor::minkowski();
    geodesic::GeodesicState init_state{};
    init_state.x[0] = event.t;
    init_state.x[1] = event.x;
    init_state.x[2] = event.y > 0.0 ? std::log(event.y) : 0.0;
    init_state.x[3] = event.z;
    init_state.u[0] = 1.0; // proper-time flow
    init_state.u[1] = beta_result->beta;
    init_state.u[2] = 0.0;
    init_state.u[3] = 0.0;

    // Use 10 steps — enough to verify no crash, not enough to be slow
    const auto geo_result = solver.solve(init_state, flat_metric, 10, 0.01);
    if (!geo_result) return std::nullopt;

    // Step 5: RelativisticSignalProcessor — compute γ-corrected signal
    momentum::RelativisticSignalProcessor proc;
    auto bv = momentum::BetaVelocity::make(beta_result->beta);
    if (!bv) return std::nullopt;
    auto meff = momentum::EffectiveMass::make(beta_result->gamma); // use γ as m_eff proxy
    if (!meff) {
        // gamma is always >= 1.0, so this shouldn't fail; but be safe
        meff = momentum::EffectiveMass::make(1.0);
        if (!meff) return std::nullopt;
    }

    const momentum::RawSignal raw_sig{mean_price};
    auto sig_result = proc.process_one(raw_sig, *bv, *meff);
    if (!sig_result) return std::nullopt;

    return PipelineResult{
        beta_result->beta,
        beta_result->gamma,
        beta_result->rapidity,
        beta_result->doppler,
        *regime_opt,
        sig_result->adjusted_value,
        prices.size()
    };
}

} // namespace srfm::engine
