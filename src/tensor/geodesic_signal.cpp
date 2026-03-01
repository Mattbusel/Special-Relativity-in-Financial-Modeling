/// @file src/tensor/geodesic_signal.cpp
/// @brief Implementation of GeodesicDeviationCalculator — AGT-07.
///
/// Integrates the geodesic equation from the first bar and computes the
/// Euclidean spatial deviation between the actual and predicted market position
/// at each subsequent bar.
///
/// The core loop:
///   1. Convert events → SpacetimePoints
///   2. Estimate u₀ from the first displacement
///   3. Integrate geodesic for (n−1) steps (one step per bar gap)
///   4. For bar i: deviation = (actual[1:3] − geodesic[1:3]).norm()

#include "srfm/geodesic_signal.hpp"
#include "srfm/constants.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace srfm::tensor {

// ─── Construction ─────────────────────────────────────────────────────────────

GeodesicDeviationCalculator::GeodesicDeviationCalculator(
    const MetricTensor& metric,
    double step_size) noexcept
    : metric_(metric)
    , solver_(metric, step_size)
{}

// ─── Private Helpers ──────────────────────────────────────────────────────────

SpacetimePoint GeodesicDeviationCalculator::to_point(
    const manifold::SpacetimeEvent& ev) noexcept
{
    SpacetimePoint p;
    p[0] = ev.time;
    p[1] = ev.price;
    p[2] = ev.volume;
    p[3] = ev.momentum;
    return p;
}

FourVelocity GeodesicDeviationCalculator::estimate_velocity(
    const SpacetimePoint& p0,
    const SpacetimePoint& p1) noexcept
{
    FourVelocity u = p1 - p0;

    // Guard against zero or non-finite displacement
    double norm = u.norm();
    if (!std::isfinite(norm) || norm < std::numeric_limits<double>::epsilon() * 100.0) {
        // Fall back to canonical timelike direction (along τ axis)
        u = FourVelocity::Zero();
        u[0] = 1.0;
        return u;
    }

    return u / norm;
}

double GeodesicDeviationCalculator::spatial_deviation(
    const SpacetimePoint& actual,
    const SpacetimePoint& geodesic) noexcept
{
    // Only spatial components [1, 2, 3]
    double d1 = actual[1] - geodesic[1];
    double d2 = actual[2] - geodesic[2];
    double d3 = actual[3] - geodesic[3];

    double sq = d1 * d1 + d2 * d2 + d3 * d3;
    if (!std::isfinite(sq)) {
        return 0.0;
    }
    return std::sqrt(sq);
}

// ─── compute ─────────────────────────────────────────────────────────────────

std::vector<GeodesicSignal>
GeodesicDeviationCalculator::compute(
    const std::vector<manifold::SpacetimeEvent>& events) const noexcept
{
    const std::size_t n = events.size();

    if (n == 0) {
        return {};
    }

    if (n == 1) {
        return {GeodesicSignal{0.0, 0.0, true}};
    }

    // ── Convert events to SpacetimePoints ─────────────────────────────────────
    std::vector<SpacetimePoint> actual_points;
    actual_points.reserve(n);
    for (const auto& ev : events) {
        actual_points.push_back(to_point(ev));
    }

    // ── Guard: check first two points are finite ───────────────────────────────
    auto all_finite = [](const SpacetimePoint& p) -> bool {
        for (int i = 0; i < SPACETIME_DIM; ++i) {
            if (!std::isfinite(p[i])) return false;
        }
        return true;
    };

    if (!all_finite(actual_points[0]) || !all_finite(actual_points[1])) {
        // Return all-invalid signals
        std::vector<GeodesicSignal> result(n, GeodesicSignal{0.0, 0.0, false});
        return result;
    }

    // ── Estimate initial four-velocity ────────────────────────────────────────
    FourVelocity u0 = estimate_velocity(actual_points[0], actual_points[1]);

    // ── Integrate geodesic for (n-1) steps ────────────────────────────────────
    //
    // solver_.integrate returns n states: [x₀, x₁, ..., x_{n-1}]
    // where x_i is the geodesic prediction at proper time i * step_size.
    std::vector<GeodesicState> trajectory =
        solver_.integrate(actual_points[0], u0, static_cast<int>(n) - 1);

    // trajectory.size() == n  (initial state + n-1 steps)

    // ── Build output signals ──────────────────────────────────────────────────
    std::vector<GeodesicSignal> result;
    result.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        const SpacetimePoint& actual   = actual_points[i];
        const SpacetimePoint& geodesic = trajectory[i].position;

        bool valid = all_finite(actual) && all_finite(geodesic);

        double deviation = 0.0;
        if (valid) {
            deviation = spatial_deviation(actual, geodesic);
            if (!std::isfinite(deviation)) {
                deviation = 0.0;
                valid = false;
            }
        }

        double proper_time = static_cast<double>(i) * constants::DEFAULT_GEODESIC_STEP;

        result.push_back(GeodesicSignal{deviation, proper_time, valid});
    }

    return result;
}

} // namespace srfm::tensor
