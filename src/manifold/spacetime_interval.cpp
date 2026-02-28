/// @file src/manifold/spacetime_interval.cpp
/// @brief SpacetimeInterval — financial Minkowski interval ds² (AGT-06 stub).

#include "srfm/manifold.hpp"

#include <algorithm>
#include <cmath>

namespace srfm::manifold {

// ─── to_string ────────────────────────────────────────────────────────────────

const char* to_string(IntervalType t) noexcept {
    switch (t) {
        case IntervalType::Timelike:  return "Timelike";
        case IntervalType::Lightlike: return "Lightlike";
        case IntervalType::Spacelike: return "Spacelike";
    }
    return "Unknown";
}

// ─── SpacetimeInterval::compute ───────────────────────────────────────────────

std::optional<double>
SpacetimeInterval::compute(const SpacetimeEvent& a,
                            const SpacetimeEvent& b,
                            double c_market) noexcept {
    // Guard: all coordinates must be finite.
    if (!std::isfinite(a.time)     || !std::isfinite(b.time)     ||
        !std::isfinite(a.price)    || !std::isfinite(b.price)    ||
        !std::isfinite(a.volume)   || !std::isfinite(b.volume)   ||
        !std::isfinite(a.momentum) || !std::isfinite(b.momentum) ||
        !std::isfinite(c_market)   || c_market <= 0.0) {
        return std::nullopt;
    }

    const double dt = b.time     - a.time;
    const double dp = b.price    - a.price;
    const double dv = b.volume   - a.volume;
    const double dm = b.momentum - a.momentum;

    // ds² = −c²·Δt² + ΔP² + ΔV² + ΔM²
    // Minkowski signature (−,+,+,+):
    //   − time component: causal separation dominates → timelike
    //   + spatial components: spatial separation dominates → spacelike
    const double time_term    = c_market * c_market * dt * dt;
    const double spatial_term = dp * dp + dv * dv + dm * dm;

    return spatial_term - time_term;
}

// ─── SpacetimeInterval::classify ─────────────────────────────────────────────

IntervalType
SpacetimeInterval::classify(double interval_squared) noexcept {
    if (std::abs(interval_squared) <= constants::FLOAT_EPSILON) {
        return IntervalType::Lightlike;
    }
    return (interval_squared < 0.0) ? IntervalType::Timelike
                                    : IntervalType::Spacelike;
}

}  // namespace srfm::manifold
