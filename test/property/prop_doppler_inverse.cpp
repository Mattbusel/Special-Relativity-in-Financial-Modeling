/**
 * @file  prop_doppler_inverse.cpp
 * @brief Property: ∀ β: D(β) · D(−β) = 1  to 1e-12
 *
 * Run with 10,000 random inputs:
 *   RC_PARAMS="max_success=10000" ./prop_doppler_inverse
 *
 * Mathematical basis:
 *   D(β) = √((1+β)/(1−β))   (relativistic Doppler factor)
 *   D(−β) = √((1−β)/(1+β)) = 1/D(β)
 *   Therefore D(β) · D(−β) = 1 exactly.
 *
 * This identity is critical for correctness of blue-shift / red-shift
 * round-trips in the financial signal processing pipeline.
 *
 * Failure modes this test guards against:
 *   • Sign error in doppler_factor() formula
 *   • Numerical cancellation near β = 0
 *   • sqrt() returning NaN for near-zero arguments
 */

#include <rapidcheck.h>
#include <cmath>

#include "momentum/momentum.hpp"
#include "beta_calculator/beta_calculator.hpp"

using namespace srfm::momentum;
using namespace srfm::beta_calculator;

int main() {
    // ── Property 1: D(β) · D(−β) = 1 to 1e-12 ──────────────────────────────
    rc::check(
        "doppler_inverse: D(beta) * D(-beta) == 1 to 1e-12",
        [](double raw_beta) {
            const double b = std::tanh(raw_beta) * (BETA_MAX_SAFE - 1e-10);

            auto bv_pos = BetaVelocity::make(b);
            auto bv_neg = BetaVelocity::make(-b);
            RC_ASSERT(bv_pos.has_value());
            RC_ASSERT(bv_neg.has_value());

            auto d_pos_opt = doppler_factor(*bv_pos);
            auto d_neg_opt = doppler_factor(*bv_neg);
            RC_ASSERT(d_pos_opt.has_value());
            RC_ASSERT(d_neg_opt.has_value());

            const double product = *d_pos_opt * *d_neg_opt;
            RC_ASSERT(std::isfinite(product));

            const double abs_err = std::abs(product - 1.0);
            RC_ASSERT(abs_err < 1e-12);
        }
    );

    // ── Property 2: D(β) > 0 for all valid β ────────────────────────────────
    rc::check(
        "doppler_inverse: D(beta) > 0 for all valid beta",
        [](double raw_beta) {
            const double b = std::tanh(raw_beta) * (BETA_MAX_SAFE - 1e-10);
            auto bv = BetaVelocity::make(b);
            RC_ASSERT(bv.has_value());
            auto d = doppler_factor(*bv);
            RC_ASSERT(d.has_value());
            RC_ASSERT(*d > 0.0);
            RC_ASSERT(std::isfinite(*d));
        }
    );

    // ── Property 3: D(β) ≥ 1 for β ≥ 0 (blueshift) ─────────────────────────
    rc::check(
        "doppler_inverse: D(beta) >= 1 for beta >= 0",
        [](double raw_beta) {
            // Use only non-negative beta
            const double b = std::tanh(std::abs(raw_beta)) * (BETA_MAX_SAFE - 1e-10);
            auto bv = BetaVelocity::make(b);
            RC_ASSERT(bv.has_value());
            auto d = doppler_factor(*bv);
            RC_ASSERT(d.has_value());
            RC_ASSERT(*d >= 1.0 - 1e-15);  // allow tiny FP slack at β=0 boundary
        }
    );

    // ── Property 4: D(β) ≤ 1 for β ≤ 0 (redshift) ──────────────────────────
    rc::check(
        "doppler_inverse: D(beta) <= 1 for beta <= 0",
        [](double raw_beta) {
            const double b = -std::tanh(std::abs(raw_beta)) * (BETA_MAX_SAFE - 1e-10);
            auto bv = BetaVelocity::make(b);
            RC_ASSERT(bv.has_value());
            auto d = doppler_factor(*bv);
            RC_ASSERT(d.has_value());
            RC_ASSERT(*d <= 1.0 + 1e-15);
        }
    );

    // ── Property 5: D(β) = exp(φ) where φ = atanh(β) ───────────────────────
    rc::check(
        "doppler_inverse: D(beta) == exp(rapidity(beta)) to 1e-12",
        [](double raw_beta) {
            const double b = std::tanh(raw_beta) * (BETA_MAX_SAFE - 1e-10);
            auto bv = BetaVelocity::make(b);
            RC_ASSERT(bv.has_value());

            auto d_opt   = doppler_factor(*bv);
            auto phi_opt = rapidity(*bv);
            RC_ASSERT(d_opt.has_value());
            RC_ASSERT(phi_opt.has_value());

            const double exp_phi = std::exp(*phi_opt);
            RC_ASSERT(std::isfinite(exp_phi));

            const double rel_err = std::abs(*d_opt - exp_phi)
                                   / (std::abs(exp_phi) + 1e-300);
            RC_ASSERT(rel_err < 1e-12);
        }
    );

    return 0;
}
