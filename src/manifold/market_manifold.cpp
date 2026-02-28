/// @file src/manifold/market_manifold.cpp
/// @brief MarketManifold — high-level financial spacetime manifold (AGT-06 stub).

#include "srfm/manifold.hpp"

#include <algorithm>
#include <cmath>

namespace srfm::manifold {

// ─── MarketManifold::classify ─────────────────────────────────────────────────

std::optional<IntervalType>
MarketManifold::classify(const SpacetimeEvent& a,
                          const SpacetimeEvent& b) noexcept {
    auto ds2 = SpacetimeInterval::compute(a, b);
    if (!ds2) {
        return std::nullopt;
    }
    return SpacetimeInterval::classify(*ds2);
}

// ─── MarketManifold::beta ─────────────────────────────────────────────────────

std::optional<double>
MarketManifold::beta(const SpacetimeEvent& a,
                      const SpacetimeEvent& b,
                      double c_market) noexcept {
    if (!std::isfinite(a.time) || !std::isfinite(b.time) ||
        !std::isfinite(a.price) || !std::isfinite(b.price) ||
        !std::isfinite(a.volume) || !std::isfinite(b.volume) ||
        !std::isfinite(a.momentum) || !std::isfinite(b.momentum) ||
        !std::isfinite(c_market) || c_market <= 0.0) {
        return std::nullopt;
    }

    const double dt = b.time - a.time;
    if (std::abs(dt) < constants::FLOAT_EPSILON) {
        // Simultaneous events — velocity undefined.
        return std::nullopt;
    }

    const double dp = b.price    - a.price;
    const double dv = b.volume   - a.volume;
    const double dm = b.momentum - a.momentum;

    // |Δspace| / (c · |Δtime|)  →  β in [0, ∞)
    const double spatial_speed = std::sqrt(dp*dp + dv*dv + dm*dm) / std::abs(dt);
    const double raw_beta      = spatial_speed / c_market;

    // Clamp to valid range [0, BETA_MAX_SAFE).
    const double safe_beta = std::min(raw_beta, constants::BETA_MAX_SAFE - 1e-15);
    return safe_beta;
}

// ─── MarketManifold::is_causal ────────────────────────────────────────────────

bool
MarketManifold::is_causal(const SpacetimeEvent& a,
                            const SpacetimeEvent& b) noexcept {
    auto result = classify(a, b);
    if (!result) {
        return false;
    }
    // Timelike and lightlike trajectories are both causal.
    return (*result == IntervalType::Timelike ||
            *result == IntervalType::Lightlike);
}

}  // namespace srfm::manifold
