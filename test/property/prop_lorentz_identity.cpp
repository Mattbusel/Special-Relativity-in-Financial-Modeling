/**
 * @file  prop_lorentz_identity.cpp
 * @brief Property: ∀ β ∈ [0, 0.9999): γ(β)² = 1/(1−β²)  to 1e-12
 *
 * Run with 10,000 random inputs:
 *   RC_PARAMS="max_success=10000" ./prop_lorentz_identity
 *
 * Mathematical basis:
 *   γ(β) = 1/√(1−β²)
 *   γ²   = 1/(1−β²)
 *
 * This identity must hold to 1e-12 relative error for all sub-luminal β.
 * A violation would indicate catastrophic cancellation or a sign error in
 * the sqrt implementation on the target hardware.
 */

#include <rapidcheck.h>
#include <cmath>
#include <cassert>
#include <cstdlib>

#include "momentum/momentum.hpp"

using namespace srfm::momentum;

int main() {
    // ── Property 1: γ² = 1/(1−β²) for β ∈ [0, BETA_MAX_SAFE) ────────────────
    rc::check(
        "lorentz_identity: gamma(beta)^2 == 1/(1 - beta^2) to 1e-12",
        [](double raw_beta) {
            // Map arbitrary double to a valid β in [0, BETA_MAX_SAFE)
            // Use sigmoid-like clamping: tanh maps R → (−1,1), scale to safe range
            const double beta_val = std::tanh(raw_beta) * (BETA_MAX_SAFE - 1e-10);

            auto bv = BetaVelocity::make(beta_val);
            RC_ASSERT(bv.has_value());

            auto gamma_opt = lorentz_gamma(*bv);
            RC_ASSERT(gamma_opt.has_value());

            const double gamma = gamma_opt->value();
            const double b     = bv->value();
            const double lhs   = gamma * gamma;
            const double rhs   = 1.0 / (1.0 - b * b);

            // Both must be finite
            RC_ASSERT(std::isfinite(lhs));
            RC_ASSERT(std::isfinite(rhs));
            RC_ASSERT(rhs > 0.0);

            // Relative error < 1e-12
            const double rel_err = std::abs(lhs - rhs) / rhs;
            RC_ASSERT(rel_err < 1e-12);
        }
    );

    // ── Property 2: γ(−β) = γ(β) (Lorentz factor is even in β) ──────────────
    rc::check(
        "lorentz_identity: gamma is even — gamma(-beta) == gamma(beta)",
        [](double raw_beta) {
            const double beta_val = std::tanh(raw_beta) * (BETA_MAX_SAFE - 1e-10);
            const double beta_neg = -beta_val;

            auto bv_pos = BetaVelocity::make(beta_val);
            auto bv_neg = BetaVelocity::make(beta_neg);
            RC_ASSERT(bv_pos.has_value());
            RC_ASSERT(bv_neg.has_value());

            auto g_pos = lorentz_gamma(*bv_pos);
            auto g_neg = lorentz_gamma(*bv_neg);
            RC_ASSERT(g_pos.has_value());
            RC_ASSERT(g_neg.has_value());

            const double rel_err = std::abs(g_pos->value() - g_neg->value())
                                   / g_pos->value();
            RC_ASSERT(rel_err < 1e-15);
        }
    );

    // ── Property 3: γ ≥ 1 for all valid β ────────────────────────────────────
    rc::check(
        "lorentz_identity: gamma >= 1 for all valid beta",
        [](double raw_beta) {
            const double beta_val = std::tanh(raw_beta) * (BETA_MAX_SAFE - 1e-10);
            auto bv = BetaVelocity::make(beta_val);
            RC_ASSERT(bv.has_value());
            auto gamma_opt = lorentz_gamma(*bv);
            RC_ASSERT(gamma_opt.has_value());
            RC_ASSERT(gamma_opt->value() >= 1.0);
        }
    );

    return 0;
}
