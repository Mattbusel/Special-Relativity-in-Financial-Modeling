#pragma once

/// @file include/srfm/geodesic_signal.hpp
/// @brief Geodesic Deviation Signal — AGT-07 public API.
///
/// # Module: Geodesic Deviation
///
/// ## Responsibility
/// Compute, for each bar in a price series, the Euclidean distance between the
/// actual market position in financial spacetime and the position predicted by
/// the geodesic equation integrated from the series start.
///
///   geodesic_deviation_i = ||x_actual_i[1:3] − x_geodesic_i[1:3]||₂
///
/// where components [1], [2], [3] are the spatial (price, volume, momentum)
/// dimensions of the SpacetimePoint.
///
/// ## Physical Interpretation
/// The geodesic traces the "natural" price path: free-fall through the curved
/// financial spacetime determined by the asset covariance geometry.  A large
/// deviation from the geodesic indicates the market is being pulled away from
/// its natural trajectory by external forces (news, liquidity shocks).
/// Mean-reversion hypothesis: large deviations should be followed by a return
/// toward the geodesic (reversion to the natural path).
///
/// ## Guarantees
/// - All methods are noexcept
/// - Returns GeodesicSignal{0.0, 0.0, false} for pathological inputs
/// - First bar always returns deviation = 0.0 (reference point)
/// - No dynamic allocation beyond the returned vector

#include "srfm/tensor.hpp"
#include "srfm/manifold.hpp"

#include <vector>

namespace srfm::tensor {

// ─── GeodesicSignal ───────────────────────────────────────────────────────────

/// Per-bar geodesic deviation signal.
///
/// Produced by GeodesicDeviationCalculator::compute for each bar in a price
/// series.  The deviation measures how far the actual market position is from
/// the geodesic-predicted position in the spatial subspace.
struct GeodesicSignal {
    double geodesic_deviation; ///< ||x_actual[1:3] − x_geodesic[1:3]||₂
    double proper_time;        ///< Proper time τ at this bar (bar_index × step_size)
    bool   is_valid;           ///< false if geodesic integration failed for this bar
};

// ─── GeodesicDeviationCalculator ─────────────────────────────────────────────

/// Computes geodesic deviation for a sequence of market events.
///
/// Algorithm
/// ---------
/// 1. Convert the sequence of SpacetimeEvent objects into SpacetimePoints.
/// 2. Estimate an initial four-velocity u₀ from the first two events.
/// 3. Integrate the geodesic equation from x₀ for (n−1) steps via RK4.
/// 4. At each bar i, compute spatial deviation:
///      dev_i = (actual_spatial_i − geodesic_spatial_i).norm()
///    where spatial components are indices [1, 2, 3].
///
/// # Example
/// ```cpp
/// auto metric = srfm::tensor::MetricTensor::make_minkowski(1.0, 0.2);
/// srfm::tensor::GeodesicDeviationCalculator calc(metric);
///
/// std::vector<srfm::manifold::SpacetimeEvent> events = ...;
/// auto signals = calc.compute(events);
///
/// for (std::size_t i = 0; i < signals.size(); ++i) {
///     if (signals[i].is_valid) {
///         fmt::print("bar {} deviation = {:.4f}\n", i, signals[i].geodesic_deviation);
///     }
/// }
/// ```
///
/// # Panics
/// This class never panics.
class GeodesicDeviationCalculator {
public:
    /// Construct with a metric tensor and RK4 proper-time step.
    ///
    /// # Arguments
    /// * `metric`    — Position-dependent metric encoding market covariance
    /// * `step_size` — Proper-time step dτ for geodesic integration
    explicit GeodesicDeviationCalculator(
        const MetricTensor& metric,
        double step_size = constants::DEFAULT_GEODESIC_STEP) noexcept;

    /// Compute geodesic deviation for a sequence of spacetime events.
    ///
    /// # Arguments
    /// * `events` — Ordered sequence of market observations as SpacetimeEvents.
    ///              Must be in chronological order (ascending time coordinate).
    ///
    /// # Returns
    /// Vector of GeodesicSignal, one per input event:
    ///   - events[0]: deviation = 0.0 (reference point), is_valid = true
    ///   - events[i]: actual deviation from geodesic at step i
    ///
    /// Returns an empty vector if events is empty.
    /// Returns a single {0.0, 0.0, true} if events has exactly one element.
    [[nodiscard]] std::vector<GeodesicSignal>
    compute(const std::vector<manifold::SpacetimeEvent>& events) const noexcept;

private:
    /// Convert a SpacetimeEvent to an Eigen SpacetimePoint.
    [[nodiscard]] static SpacetimePoint
    to_point(const manifold::SpacetimeEvent& ev) noexcept;

    /// Estimate initial four-velocity from the displacement between two events.
    /// Returns the unit-normalised tangent vector, or the canonical basis e₀
    /// if the displacement is degenerate (zero or non-finite).
    [[nodiscard]] static FourVelocity
    estimate_velocity(const SpacetimePoint& p0,
                      const SpacetimePoint& p1) noexcept;

    /// Spatial norm of the deviation (components [1,2,3] only).
    [[nodiscard]] static double
    spatial_deviation(const SpacetimePoint& actual,
                      const SpacetimePoint& geodesic) noexcept;

    const MetricTensor& metric_;
    GeodesicSolver      solver_;
};

} // namespace srfm::tensor
