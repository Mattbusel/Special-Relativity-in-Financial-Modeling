#pragma once
/**
 * @file  spacetime_manifold.hpp
 * @brief Minkowski spacetime manifold tracker — interval and regime classifier.
 *
 * Module:  include/srfm/stream/
 * Owner:   AGT-10  (Builder)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Maintain a running record of Lorentz-transformed tick coordinates and compute
 * the Minkowski spacetime interval between consecutive events:
 *
 *   Δs² = Δt'² − Δx'²   (c = 1 in normalised units)
 *
 * Classify the interval into a Regime (TIMELIKE / LIGHTLIKE / SPACELIKE) and
 * use it to modulate the relativistic signal strength.
 *
 * Spacetime interval interpretation
 * ----------------------------------
 *   Δs² > +ε  → TIMELIKE  — causally connected; signal is amplified by Δs.
 *   Δs² < −ε  → SPACELIKE — acausally separated; signal is attenuated.
 *  |Δs²| ≤ ε  → LIGHTLIKE — on the light cone; signal passes through unscaled.
 *
 * The output `signal` field is:
 *   TIMELIKE  : +√|Δs²|
 *   SPACELIKE : −√|Δs²|
 *   LIGHTLIKE :  0.0
 *
 * Guarantees
 * ----------
 *   • O(1) per tick: stores only the previous transformed event.
 *   • noexcept: all methods are noexcept.
 *   • Safe first tick: returns LIGHTLIKE / 0.0 when no previous event exists.
 *   • Finite: output signal is always finite for finite Lorentz inputs.
 *
 * NOT Responsible For
 * -------------------
 *   • Computing (t', x')  (LorentzTransform)
 *   • Scaling by γ·m_eff  (SignalProcessor combines all stages)
 *   • Persistence         (in-memory state only)
 */

#include <cmath>
#include "stream_signal.hpp"

namespace srfm::stream {

// ── ManifoldResult ────────────────────────────────────────────────────────────

/**
 * @brief Result of a single manifold update — interval, regime, and signal.
 */
struct ManifoldResult {
    double ds2{0.0};              ///< Minkowski interval Δs² = Δt'² − Δx'².
    Regime regime{Regime::LIGHTLIKE}; ///< Spacetime classification.
    double signal{0.0};           ///< ±√|Δs²| (signed by regime).
};

// ── SpacetimeManifold ─────────────────────────────────────────────────────────

/**
 * @brief Minkowski interval computer for sequential Lorentz-boosted events.
 *
 * @code
 *   SpacetimeManifold manifold;
 *   auto result = manifold.update(ev.t_prime, ev.x_prime);
 *   // result.regime → TIMELIKE / LIGHTLIKE / SPACELIKE
 *   // result.signal → signed sqrt of interval
 * @endcode
 */
class SpacetimeManifold {
public:
    /**
     * @brief Construct with a custom lightlike tolerance.
     *
     * @param epsilon  Absolute |Δs²| threshold for LIGHTLIKE classification.
     *                 Defaults to 1e-6 (appropriate for normalised z-score coords).
     */
    explicit SpacetimeManifold(double epsilon = 1e-6) noexcept
        : epsilon_{epsilon > 0.0 ? epsilon : 1e-6}
    {}

    // ── State mutation ─────────────────────────────────────────────────────────

    /**
     * @brief Record a new transformed event and compute the interval from the last.
     *
     * On the first call (no previous event), returns LIGHTLIKE with signal = 0.
     *
     * @param t_prime  Boosted time coordinate from LorentzTransform.
     * @param x_prime  Boosted space coordinate from LorentzTransform.
     * @return ManifoldResult with ds2, regime, and signal for this step.
     *
     * @note noexcept.
     */
    [[nodiscard]] ManifoldResult update(double t_prime,
                                        double x_prime) noexcept {
        ManifoldResult result;

        if (!has_prev_) {
            // First event — no interval to compute.
            prev_t_prime_ = t_prime;
            prev_x_prime_ = x_prime;
            has_prev_     = true;
            result.ds2    = 0.0;
            result.regime = Regime::LIGHTLIKE;
            result.signal = 0.0;
            return result;
        }

        // Δt' and Δx' between consecutive events.
        const double dt = t_prime - prev_t_prime_;
        const double dx = x_prime - prev_x_prime_;

        // Minkowski interval (c = 1).
        const double ds2 = dt * dt - dx * dx;
        result.ds2 = ds2;

        // Classify.
        if (ds2 > epsilon_) {
            result.regime = Regime::TIMELIKE;
            result.signal = std::sqrt(ds2);
        } else if (ds2 < -epsilon_) {
            result.regime = Regime::SPACELIKE;
            result.signal = -std::sqrt(-ds2);
        } else {
            result.regime = Regime::LIGHTLIKE;
            result.signal = 0.0;
        }

        // Slide the window.
        prev_t_prime_ = t_prime;
        prev_x_prime_ = x_prime;

        return result;
    }

    // ── Accessors ──────────────────────────────────────────────────────────────

    /// Whether at least one event has been ingested.
    [[nodiscard]] bool has_previous() const noexcept { return has_prev_; }

    /// Previous boosted time coordinate.
    [[nodiscard]] double prev_t_prime() const noexcept { return prev_t_prime_; }

    /// Previous boosted space coordinate.
    [[nodiscard]] double prev_x_prime() const noexcept { return prev_x_prime_; }

    /// Lightlike tolerance ε.
    [[nodiscard]] double epsilon() const noexcept { return epsilon_; }

    // ── Reset ──────────────────────────────────────────────────────────────────

    /**
     * @brief Reset all state as if no events have been seen.
     *
     * Preserves the configured epsilon.
     */
    void reset() noexcept {
        has_prev_     = false;
        prev_t_prime_ = 0.0;
        prev_x_prime_ = 0.0;
    }

    // ── Stateless helper ──────────────────────────────────────────────────────

    /**
     * @brief Compute the Minkowski interval for an arbitrary pair of events.
     *
     * Does not mutate any state.
     *
     * @param dt  Time separation Δt'.
     * @param dx  Space separation Δx'.
     * @return Minkowski interval Δs² = Δt'² − Δx'².
     * @note noexcept — pure arithmetic.
     */
    [[nodiscard]] static double interval(double dt, double dx) noexcept {
        return dt * dt - dx * dx;
    }

    /**
     * @brief Classify a precomputed interval value.
     *
     * @param ds2      Minkowski interval value.
     * @param epsilon  Lightlike tolerance.
     * @return Regime classification.
     * @note noexcept.
     */
    [[nodiscard]] static Regime classify(double ds2,
                                         double epsilon = 1e-6) noexcept {
        if (ds2 >  epsilon) return Regime::TIMELIKE;
        if (ds2 < -epsilon) return Regime::SPACELIKE;
        return Regime::LIGHTLIKE;
    }

private:
    double epsilon_;              ///< Lightlike tolerance.
    bool   has_prev_{false};      ///< Whether a previous event has been stored.
    double prev_t_prime_{0.0};    ///< Previous boosted time coordinate.
    double prev_x_prime_{0.0};    ///< Previous boosted space coordinate.
};

} // namespace srfm::stream
