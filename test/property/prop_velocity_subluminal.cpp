/**
 * @file  prop_velocity_subluminal.cpp
 * @brief Property: ∀ β₁,β₂ ∈ (−1,1): |β₁ ⊕ β₂| < 1 (never superluminal)
 *
 * Run with 10,000 random inputs:
 *   RC_PARAMS="max_success=10000" ./prop_velocity_subluminal
 *
 * Mathematical basis:
 *   β₁ ⊕ β₂ = (β₁ + β₂) / (1 + β₁β₂)
 *
 *   For |β₁|, |β₂| < 1:
 *     |β₁ ⊕ β₂| < 1  (the relativistic composition preserves sub-luminal speed)
 *
 * This is perhaps the most important safety property of the pipeline:
 *   a composed β ≥ 1 would cause γ → ∞, producing infinite signals and
 *   crashing downstream calculations.
 *
 * Note: compose_velocities() additionally enforces |result| < BETA_MAX_SAFE,
 * so violations caught by this test indicate a logic error in BetaVelocity::make()
 * or compose_velocities().
 */

#include <rapidcheck.h>
#include <cmath>

#include "momentum/momentum.hpp"

using namespace srfm::momentum;

int main() {
    // ── Property 1: |β₁ ⊕ β₂| < 1 ──────────────────────────────────────────
    rc::check(
        "velocity_subluminal: |compose(b1, b2)| < 1 always",
        [](double raw1, double raw2) {
            // Map to full range just inside BETA_MAX_SAFE
            const double b1 = std::tanh(raw1) * (BETA_MAX_SAFE - 1e-10);
            const double b2 = std::tanh(raw2) * (BETA_MAX_SAFE - 1e-10);

            auto bv1 = BetaVelocity::make(b1);
            auto bv2 = BetaVelocity::make(b2);
            RC_ASSERT(bv1.has_value());
            RC_ASSERT(bv2.has_value());

            auto composed = compose_velocities(*bv1, *bv2);
            // If compose returns nullopt, the composed value was ≥ BETA_MAX_SAFE.
            // We still need to verify it's not ≥ 1 by computing manually.
            const double manual = (b1 + b2) / (1.0 + b1 * b2);
            RC_ASSERT(std::isfinite(manual));
            RC_ASSERT(std::abs(manual) < 1.0);

            // If compose_velocities returned a value, it must be within safe range
            if (composed.has_value()) {
                RC_ASSERT(std::abs(composed->value()) < BETA_MAX_SAFE);
                RC_ASSERT(std::abs(composed->value()) < 1.0);
            }
        }
    );

    // ── Property 2: composition is commutative |β₁ ⊕ β₂| = |β₂ ⊕ β₁| ──────
    rc::check(
        "velocity_subluminal: composition is commutative",
        [](double raw1, double raw2) {
            const double b1 = std::tanh(raw1) * 0.7;
            const double b2 = std::tanh(raw2) * 0.7;

            auto bv1 = BetaVelocity::make(b1);
            auto bv2 = BetaVelocity::make(b2);
            RC_ASSERT(bv1.has_value());
            RC_ASSERT(bv2.has_value());

            auto c12 = compose_velocities(*bv1, *bv2);
            auto c21 = compose_velocities(*bv2, *bv1);

            // Both must either succeed or fail together
            RC_ASSERT(c12.has_value() == c21.has_value());
            if (c12.has_value()) {
                const double abs_err = std::abs(c12->value() - c21->value());
                RC_ASSERT(abs_err < 1e-15);
            }
        }
    );

    // ── Property 3: β ⊕ 0 = β (identity element) ───────────────────────────
    rc::check(
        "velocity_subluminal: zero is identity element for composition",
        [](double raw_beta) {
            const double b = std::tanh(raw_beta) * (BETA_MAX_SAFE - 1e-10);
            auto bv  = BetaVelocity::make(b);
            auto bv0 = BetaVelocity::make(0.0);
            RC_ASSERT(bv.has_value());
            RC_ASSERT(bv0.has_value());

            auto composed = compose_velocities(*bv, *bv0);
            RC_ASSERT(composed.has_value());

            const double abs_err = std::abs(composed->value() - b);
            RC_ASSERT(abs_err < 1e-14);
        }
    );

    // ── Property 4: β ⊕ (−β) = 0 (inverse element) ─────────────────────────
    rc::check(
        "velocity_subluminal: beta composed with -beta == 0",
        [](double raw_beta) {
            const double b = std::tanh(raw_beta) * (BETA_MAX_SAFE - 1e-10);
            auto bv_pos = BetaVelocity::make(b);
            auto bv_neg = BetaVelocity::make(-b);
            RC_ASSERT(bv_pos.has_value());
            RC_ASSERT(bv_neg.has_value());

            auto composed = compose_velocities(*bv_pos, *bv_neg);
            RC_ASSERT(composed.has_value());
            RC_ASSERT(std::abs(composed->value()) < 1e-14);
        }
    );

    // ── Property 5: γ always finite after valid composition ─────────────────
    rc::check(
        "velocity_subluminal: gamma is finite after composition",
        [](double raw1, double raw2) {
            const double b1 = std::tanh(raw1) * 0.8;
            const double b2 = std::tanh(raw2) * 0.8;

            auto bv1 = BetaVelocity::make(b1);
            auto bv2 = BetaVelocity::make(b2);
            RC_ASSERT(bv1.has_value());
            RC_ASSERT(bv2.has_value());

            auto composed = compose_velocities(*bv1, *bv2);
            RC_ASSERT(composed.has_value());  // 0.8+0.8 composed safely

            auto gamma_opt = lorentz_gamma(*composed);
            RC_ASSERT(gamma_opt.has_value());
            RC_ASSERT(std::isfinite(gamma_opt->value()));
            RC_ASSERT(gamma_opt->value() >= 1.0);
        }
    );

    return 0;
}
