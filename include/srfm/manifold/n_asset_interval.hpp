#pragma once
/**
 * @file  n_asset_interval.hpp
 * @brief Spacetime interval computations for N-asset events.
 *
 * Module:  include/srfm/manifold/
 * Stage:   4 — N-Asset Manifold
 *
 * ## Responsibility
 * Provide spacetime event types and compute the Lorentzian interval
 *
 *   ds² = g_μν Δx^μ Δx^ν
 *
 * between pairs of N-asset events, classifying them as TIMELIKE, SPACELIKE,
 * or LIGHTLIKE based on the sign and magnitude of ds².
 *
 * ## Classification Thresholds
 * - |ds²| < 1e-10  → LIGHTLIKE
 * - ds² < 0        → TIMELIKE
 * - ds² > 0        → SPACELIKE
 *
 * ## Guarantees
 * - All public methods are noexcept.
 * - All fallible operations return std::optional.
 * - No raw pointers.
 *
 * ## NOT Responsible For
 * - Causality enforcement.
 * - Geodesic distance (see geodesic_n.hpp).
 */

#include "../tensor/n_asset_manifold.hpp"

#include <optional>
#include <span>
#include <vector>
#include <cmath>
#include <third_party/eigen/Eigen/Dense>

namespace srfm::manifold {

using srfm::tensor::NAssetManifold;

// ── Threshold ─────────────────────────────────────────────────────────────────

/// |ds²| below this value is classified as LIGHTLIKE.
inline constexpr double LIGHTLIKE_THRESHOLD = 1e-10;

// ── Event type ────────────────────────────────────────────────────────────────

/**
 * @brief A spacetime event: a moment in time with N asset prices.
 */
struct NAssetEvent {
    double          t;       ///< Time coordinate.
    Eigen::VectorXd prices;  ///< N asset price coordinates.

    /**
     * @brief Concatenate (t, prices) into a single (N+1)-vector.
     *
     * @return Vector [t, prices[0], prices[1], ..., prices[N-1]].
     */
    [[nodiscard]] Eigen::VectorXd to_coords() const noexcept;

    /**
     * @brief Factory: validate and construct an NAssetEvent.
     *
     * Validates that prices is non-empty.
     *
     * @param t       Time coordinate.
     * @param prices  Asset price vector (must be non-empty).
     * @return        Constructed event, or std::nullopt if prices is empty.
     */
    [[nodiscard]] static std::optional<NAssetEvent>
    make(double t, Eigen::VectorXd prices) noexcept;
};

// ── Interval type ─────────────────────────────────────────────────────────────

/// Classification of a Lorentzian spacetime interval.
enum class IntervalType {
    TIMELIKE,   ///< ds² < 0 (causal separation; price change < time*c_market).
    SPACELIKE,  ///< ds² > 0 (space-like; price change > time*c_market).
    LIGHTLIKE,  ///< |ds²| < threshold (on the light cone).
};

// ── Interval result ───────────────────────────────────────────────────────────

/**
 * @brief Result of a single interval computation.
 */
struct IntervalResult {
    double       ds_sq;      ///< Squared interval ds².
    IntervalType type;       ///< TIMELIKE, SPACELIKE, or LIGHTLIKE.
    double       magnitude;  ///< sqrt(|ds²|).
};

// ── Interval computation ──────────────────────────────────────────────────────

/**
 * @brief Computes spacetime intervals between N-asset events.
 *
 * Stateless: all methods are const and noexcept.
 */
class NAssetInterval {
public:
    /// Default constructor.
    NAssetInterval() noexcept = default;

    /**
     * @brief Compute ds² = g_μν Δx^μ Δx^ν between two events.
     *
     * Δx = b.to_coords() - a.to_coords().
     *
     * @param a        First event.
     * @param b        Second event.
     * @param manifold The NAssetManifold providing the metric.
     * @return         IntervalResult, or std::nullopt on dimension mismatch.
     */
    [[nodiscard]] std::optional<IntervalResult>
    compute(const NAssetEvent& a,
            const NAssetEvent& b,
            const NAssetManifold& manifold) const noexcept;

    /**
     * @brief Compute intervals from a reference event to many events.
     *
     * Equivalent to calling compute(ref, events[i], manifold) for each i.
     *
     * @param ref      Reference event.
     * @param events   Span of target events.
     * @param manifold The NAssetManifold.
     * @return         Vector of IntervalResults (same length as events),
     *                 or std::nullopt if any individual computation fails.
     */
    [[nodiscard]] std::optional<std::vector<IntervalResult>>
    batch_from_reference(const NAssetEvent&        ref,
                         std::span<const NAssetEvent> events,
                         const NAssetManifold&     manifold) const noexcept;

private:
    /**
     * @brief Classify a ds² value.
     *
     * @param ds_sq  Squared interval.
     * @return       LIGHTLIKE if |ds_sq| < LIGHTLIKE_THRESHOLD,
     *               TIMELIKE if ds_sq < 0, SPACELIKE otherwise.
     */
    [[nodiscard]] static IntervalType classify(double ds_sq) noexcept;
};

// ── Inline helpers ────────────────────────────────────────────────────────────

inline IntervalType NAssetInterval::classify(double ds_sq) noexcept {
    if (std::abs(ds_sq) < LIGHTLIKE_THRESHOLD) {
        return IntervalType::LIGHTLIKE;
    }
    return ds_sq < 0.0 ? IntervalType::TIMELIKE : IntervalType::SPACELIKE;
}

} // namespace srfm::manifold
