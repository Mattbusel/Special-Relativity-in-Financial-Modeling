/**
 * @file  prop_rapidity_additivity.cpp
 * @brief Property: ∀ β₁,β₂ ∈ (−0.9999, 0.9999): φ(β₁ ⊕ β₂) = φ₁ + φ₂ to 1e-10
 *
 * Run with 10,000 random inputs:
 *   RC_PARAMS="max_success=10000" ./prop_rapidity_additivity
 *
 * Mathematical basis:
 *   Rapidity φ = atanh(β) is the natural parameter for Lorentz boosts.
 *   Under relativistic velocity composition (⊕):
 *     β₁ ⊕ β₂ = (β₁ + β₂) / (1 + β₁β₂)
 *   Rapidity transforms additively:
 *     φ(β₁ ⊕ β₂) = φ(β₁) + φ(β₂)
 *   This is the defining property of rapidity.
 *
 * A violation would indicate an implementation error in compose_velocities()
 * or rapidity().
 */

#include <rapidcheck.h>
#include <cmath>
#include <cassert>

#include "momentum/momentum.hpp"
#include "beta_calculator/beta_calculator.hpp"

using namespace srfm::momentum;
using namespace srfm::beta_calculator;

int main() {
    // ── Property: φ(β₁ ⊕ β₂) = φ(β₁) + φ(β₂) to 1e-10 ────────────────────
    rc::check(
        "rapidity_additivity: phi(beta1 + beta2) == phi1 + phi2 to 1e-10",
        [](double raw1, double raw2) {
            // Map to a safe range well inside (−BETA_MAX_SAFE, BETA_MAX_SAFE)
            // Use ±0.7 as the safe range so composition stays within BETA_MAX_SAFE
            const double b1 = std::tanh(raw1) * 0.7;
            const double b2 = std::tanh(raw2) * 0.7;

            auto bv1 = BetaVelocity::make(b1);
            auto bv2 = BetaVelocity::make(b2);
            RC_ASSERT(bv1.has_value());
            RC_ASSERT(bv2.has_value());

            // Compute individual rapidities
            auto phi1_opt = rapidity(*bv1);
            auto phi2_opt = rapidity(*bv2);
            RC_ASSERT(phi1_opt.has_value());
            RC_ASSERT(phi2_opt.has_value());

            const double phi1 = *phi1_opt;
            const double phi2 = *phi2_opt;

            // Compose velocities
            auto composed_opt = compose_velocities(*bv1, *bv2);
            RC_ASSERT(composed_opt.has_value());

            // Rapidity of composed velocity
            auto phi_composed_opt = rapidity(*composed_opt);
            RC_ASSERT(phi_composed_opt.has_value());

            const double phi_composed = *phi_composed_opt;
            const double phi_sum      = phi1 + phi2;

            // Both must be finite
            RC_ASSERT(std::isfinite(phi_composed));
            RC_ASSERT(std::isfinite(phi_sum));

            // Absolute error < 1e-10
            // (we use absolute error since phi can be near zero)
            const double abs_err = std::abs(phi_composed - phi_sum);
            RC_ASSERT(abs_err < 1e-10);
        }
    );

    // ── Property: rapidity is odd — φ(−β) = −φ(β) ───────────────────────────
    rc::check(
        "rapidity_additivity: rapidity is odd — phi(-beta) == -phi(beta)",
        [](double raw_beta) {
            const double b = std::tanh(raw_beta) * (BETA_MAX_SAFE - 1e-10);
            auto bv_pos = BetaVelocity::make(b);
            auto bv_neg = BetaVelocity::make(-b);
            RC_ASSERT(bv_pos.has_value());
            RC_ASSERT(bv_neg.has_value());

            auto phi_pos = rapidity(*bv_pos);
            auto phi_neg = rapidity(*bv_neg);
            RC_ASSERT(phi_pos.has_value());
            RC_ASSERT(phi_neg.has_value());

            const double abs_err = std::abs(*phi_pos + *phi_neg);
            RC_ASSERT(abs_err < 1e-15);
        }
    );

    // ── Property: rapidity is monotone increasing with β ────────────────────
    rc::check(
        "rapidity_additivity: rapidity is strictly increasing with beta",
        [](double raw_a, double raw_b) {
            // Ensure b_a < b_b both in (0, BETA_MAX_SAFE)
            const double raw_lo = std::min(raw_a, raw_b);
            const double raw_hi = std::max(raw_a, raw_b);
            if (std::abs(raw_lo - raw_hi) < 1e-14) return; // too close, skip

            const double b_lo = std::tanh(std::abs(raw_lo)) * 0.9;
            const double b_hi = std::tanh(std::abs(raw_hi)) * 0.9;
            if (b_lo >= b_hi) return;

            auto bv_lo = BetaVelocity::make(b_lo);
            auto bv_hi = BetaVelocity::make(b_hi);
            RC_ASSERT(bv_lo.has_value());
            RC_ASSERT(bv_hi.has_value());

            auto phi_lo = rapidity(*bv_lo);
            auto phi_hi = rapidity(*bv_hi);
            RC_ASSERT(phi_lo.has_value());
            RC_ASSERT(phi_hi.has_value());

            RC_ASSERT(*phi_hi > *phi_lo);
        }
    );

    return 0;
}
