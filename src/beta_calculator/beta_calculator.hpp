#pragma once
/**
 * @file  beta_calculator.hpp
 * @brief Online BetaVelocity calculator from streaming price data (AGT-13 / SRFM)
 *
 * Module:  src/beta_calculator/
 * Owner:   AGT-13  (Adversarial hardening)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Compute the relativistic β (normalised market velocity) from a stream of
 * price observations:
 *
 *   v_market = Δprice / Δtime  (raw price velocity)
 *   β        = v_market / c_market  (normalised, |β| < BETA_MAX_SAFE)
 *   φ        = atanh(β)  (rapidity — additive under Lorentz boosts)
 *   D(β)     = √((1+β)/(1−β))  (relativistic Doppler factor)
 *
 * Design Constraints
 * ------------------
 *   • All fallible operations return std::optional (no exceptions).
 *   • All public methods are noexcept.
 *   • No raw pointers in the public API.
 *   • Thread-safe: stateless free functions; BetaCalculator is const-callable.
 *
 * NOT Responsible For
 * -------------------
 *   • Sourcing price data (caller provides std::vector<double>)
 *   • Persistence or cross-session state
 *   • Non-normalised velocity units (caller provides c_market)
 */

#include <cmath>
#include <optional>
#include <vector>

#include "../momentum/momentum.hpp"

namespace srfm::beta_calculator {

using momentum::BetaVelocity;
using momentum::BETA_MAX_SAFE;

// ── BetaVelocityResult ────────────────────────────────────────────────────────

/**
 * @brief Computed relativistic quantities for a given β.
 *
 * All values are derived from a single validated BetaVelocity.
 */
struct BetaVelocityResult {
    double beta{0.0};      ///< Normalised market velocity β ∈ (−BETA_MAX_SAFE, BETA_MAX_SAFE)
    double gamma{1.0};     ///< Lorentz factor γ = 1/√(1−β²) ≥ 1
    double rapidity{0.0};  ///< φ = atanh(β)  (additive under composition)
    double doppler{1.0};   ///< D(β) = √((1+β)/(1−β))  (Doppler factor > 0)
};

// ── Free-function physics kernels ─────────────────────────────────────────────

/**
 * @brief Compute rapidity φ = atanh(β).
 *
 * Rapidity is additive under relativistic velocity composition:
 *   φ(β₁ ⊕ β₂) = φ(β₁) + φ(β₂)
 *
 * @return std::nullopt if β is non-finite or |β| ≥ BETA_MAX_SAFE.
 */
[[nodiscard]] std::optional<double>
rapidity(BetaVelocity beta) noexcept;

/**
 * @brief Compute relativistic Doppler factor D(β) = √((1+β)/(1−β)).
 *
 * Invariant: D(β) · D(−β) = 1.0  for all valid β.
 *
 * @return std::nullopt if result is non-finite.
 */
[[nodiscard]] std::optional<double>
doppler_factor(BetaVelocity beta) noexcept;

/**
 * @brief Compute full BetaVelocityResult for a given β value.
 *
 * Convenience wrapper: γ + φ + D all computed and validated together.
 *
 * @return std::nullopt if any sub-computation fails.
 */
[[nodiscard]] std::optional<BetaVelocityResult>
full_beta_result(double beta_value) noexcept;

// ── BetaCalculator ────────────────────────────────────────────────────────────

/**
 * @brief Stateless online calculator for market β velocity.
 *
 * "Online" means the calculation consumes a sequence of price observations
 * and computes a single representative β for the whole window.  The
 * representative β is the normalised mean log-return velocity.
 *
 * @example
 * @code
 *   std::vector<double> prices = {100.0, 100.5, 101.0, 100.8};
 *   BetaCalculator calc;
 *   auto result = calc.fromPriceVelocityOnline(prices, 1.0);
 *   // result->beta ≈ 0.003 (tiny, normal market)
 * @endcode
 */
class BetaCalculator {
public:
    BetaCalculator() noexcept = default;

    /**
     * @brief Compute BetaVelocityResult from a streaming price series.
     *
     * Algorithm:
     *   1. Compute log-return velocities: v_i = ln(p_{i+1}/p_i) per time step.
     *   2. Compute mean velocity: v̄ = mean(v_i).
     *   3. Normalise: β = clamp(v̄ / c_market, −BETA_MAX_SAFE + ε, BETA_MAX_SAFE − ε).
     *   4. Compute derived quantities (γ, φ, D).
     *
     * @param prices    Sequence of ≥2 positive, finite price observations.
     * @param c_market  Market "speed of light" (normalisation constant > 0).
     *                  Defaults to 1.0 (prices already in normalised units).
     * @return BetaVelocityResult, or std::nullopt if inputs are invalid.
     */
    [[nodiscard]] std::optional<BetaVelocityResult>
    fromPriceVelocityOnline(const std::vector<double>& prices,
                            double c_market = 1.0) const noexcept;
};

} // namespace srfm::beta_calculator
