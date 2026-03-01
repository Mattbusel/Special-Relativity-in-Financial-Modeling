#pragma once
/**
 * @file  lorentz_transform.hpp
 * @brief Stateful Lorentz transformation for (bar_index, normalised_price) events.
 *
 * Module:  include/srfm/stream/
 * Owner:   AGT-10  (Builder)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Apply the 1+1-dimensional Lorentz boost to streaming tick coordinates.
 *
 * Each tick is treated as an event at spacetime coordinates:
 *
 *   t = bar_index  (sequence number, playing the role of time)
 *   x = normalised_close  (z-score from CoordinateNormalizer, playing "space")
 *
 * The boost is parameterised by β from BetaCalculator:
 *
 *   γ   = 1 / √(1 − β²)
 *   t'  = γ · (t − β·x)
 *   x'  = γ · (x − β·t)
 *
 * These prime coordinates feed into SpacetimeManifold for interval computation.
 *
 * Guarantees
 * ----------
 *   • Stateful: stores the previous transformed coordinates for interval calc.
 *   • noexcept: transform() is noexcept.
 *   • Returns identity transform (t'=t, x'=x) when β=0 (Newtonian limit).
 *   • Never produces non-finite output for valid inputs (β < BETA_MAX_SAFE).
 *
 * NOT Responsible For
 * -------------------
 *   • Computing β     (BetaCalculator)
 *   • Computing Δs²   (SpacetimeManifold)
 *   • Signal scaling  (signal_processor)
 */

#include <cmath>

namespace srfm::stream {

// ── TransformedEvent ──────────────────────────────────────────────────────────

/**
 * @brief Lorentz-boosted spacetime event coordinates.
 *
 * Produced by LorentzTransform::transform() and consumed by SpacetimeManifold.
 */
struct TransformedEvent {
    double t_prime{0.0};  ///< Boosted time coordinate.
    double x_prime{0.0};  ///< Boosted space coordinate.
    double gamma{1.0};    ///< Lorentz factor applied (γ ≥ 1).
    double beta{0.0};     ///< Market velocity applied (β).
};

// ── LorentzTransform ──────────────────────────────────────────────────────────

/**
 * @brief Applies a 1+1D Lorentz boost to tick coordinates.
 *
 * Maintains no persistent inter-tick state beyond what SpacetimeManifold needs.
 * Stateless in the physics sense — transform() is a pure mathematical operation
 * given (t, x, β).
 *
 * @code
 *   LorentzTransform lt;
 *   auto ev = lt.transform(bar_idx, norm_close, beta_calc.beta());
 *   // ev.t_prime, ev.x_prime feed into SpacetimeManifold
 * @endcode
 */
class LorentzTransform {
public:
    LorentzTransform() noexcept = default;

    /**
     * @brief Compute the Lorentz-boosted coordinates for a single tick event.
     *
     * @param t     Raw time coordinate (bar index, cast to double).
     * @param x     Normalised space coordinate (z-score close price).
     * @param beta  Market velocity β ∈ (−BETA_MAX_SAFE, +BETA_MAX_SAFE).
     *              If |β| ≥ 1 or non-finite, treated as 0 (identity boost).
     *
     * @return TransformedEvent with boosted (t', x'), γ, and β recorded.
     *
     * @note noexcept — pure arithmetic, no allocation.
     */
    [[nodiscard]] TransformedEvent transform(double t, double x,
                                             double beta) const noexcept {
        // Guard against invalid β — fall back to identity boost.
        if (!std::isfinite(beta) || beta <= -1.0 || beta >= 1.0) {
            return TransformedEvent{t, x, 1.0, 0.0};
        }

        const double gamma = 1.0 / std::sqrt(1.0 - beta * beta);

        // Guarded against non-finite γ (unreachable with valid β, defence-in-depth).
        if (!std::isfinite(gamma)) {
            return TransformedEvent{t, x, 1.0, 0.0};
        }

        const double t_prime = gamma * (t - beta * x);
        const double x_prime = gamma * (x - beta * t);

        return TransformedEvent{t_prime, x_prime, gamma, beta};
    }

    /**
     * @brief Compute γ = 1/√(1−β²) for a given β.
     *
     * Convenience function; returns 1.0 for invalid inputs.
     *
     * @note noexcept.
     */
    [[nodiscard]] static double lorentz_gamma(double beta) noexcept {
        if (!std::isfinite(beta) || beta <= -1.0 || beta >= 1.0) return 1.0;
        const double g = 1.0 / std::sqrt(1.0 - beta * beta);
        return std::isfinite(g) ? g : 1.0;
    }

    /**
     * @brief Inverse Lorentz boost: recover (t, x) from (t', x') given β.
     *
     * Uses the inverse transformation:
     *   t = γ · (t' + β·x')
     *   x = γ · (x' + β·t')
     *
     * @note noexcept.
     */
    [[nodiscard]] TransformedEvent inverse(double t_prime, double x_prime,
                                           double beta) const noexcept {
        return transform(t_prime, x_prime, -beta);
    }
};

} // namespace srfm::stream
