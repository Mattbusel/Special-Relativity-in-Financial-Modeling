/**
 * @file  test_lorentz_invariants.cpp
 * @brief Tests for Lorentz transform invariants, spacetime interval
 *        preservation, and numerical edge cases in financial calculations.
 *
 * Test categories
 * ---------------
 *   test_spacetime_interval_invariance  — ds² preserved under Lorentz boost
 *   test_velocity_norm_invariance       — u^μ u_μ conserved (time-like geodesic)
 *   test_composition_subluminality_grid — velocity addition always sub-luminal
 *   test_rapidity_additivity_strict     — φ(β₁⊕β₂) = φ₁+φ₂ tight tolerance
 *   test_round_trip_boost_identity      — boost then inverse-boost recovers signal
 *   test_gamma_newtonian_expansion      — γ ≈ 1 + β²/2 for |β| ≪ 1
 *   test_numerical_subnormal_inputs     — subnormal (denormal) beta values
 *   test_very_large_financial_values    — large raw signals stay finite
 *   test_very_small_financial_values    — tiny positive raw signals stay finite
 *   test_stdlib_boundary_probes         — numeric_limits boundary values
 *   test_lorentz_factor_monotonicity    — γ strictly monotone in |β|
 *   test_beta_velocity_symmetry         — γ(-β) == γ(β)
 *   test_compose_associativity          — (β₁⊕β₂)⊕β₃ ≈ β₁⊕(β₂⊕β₃)
 */

#include "srfm_test.hpp"
#include "momentum/momentum.hpp"
#include "manifold/spacetime_manifold.hpp"
#include "geodesic/geodesic_solver.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

using namespace srfm::momentum;
using namespace srfm::manifold;
using namespace srfm::geodesic;

static constexpr double EPS      = 1e-9;
static constexpr double EPS_MED  = 1e-7;
static constexpr double EPS_LOOSE = 1e-4;

// =============================================================================
// Lorentz interval invariance
//
// Under a Lorentz boost with velocity β, the spacetime interval
//   ds² = −c²dt² + dx² + dy² + dz²
// is invariant.  In the boosted frame (c=1):
//   dt' = γ(dt − β·dx)
//   dx' = γ(dx − β·dt)
//   dy' = dy,   dz' = dz
// We verify ds'² == ds².
// =============================================================================

static void test_spacetime_interval_invariance() {
    // Reference event separation in lab frame
    const double dt_lab = 2.0;
    const double dx_lab = 0.5;
    const double dy_lab = 0.1;
    const double dz_lab = 0.3;

    // ds² in lab frame  (signature −,+,+,+, c=1)
    const double ds2_lab = -(dt_lab * dt_lab)
                         +  (dx_lab * dx_lab)
                         +  (dy_lab * dy_lab)
                         +  (dz_lab * dz_lab);

    for (double b_val : {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99}) {
        auto bv = BetaVelocity::make(b_val);
        if (!bv) continue;

        auto gamma_opt = lorentz_gamma(*bv);
        SRFM_HAS_VALUE(gamma_opt);
        const double g = gamma_opt->value();
        const double b = bv->value();

        // Boosted separations
        const double dt_prime = g * (dt_lab - b * dx_lab);
        const double dx_prime = g * (dx_lab - b * dt_lab);
        const double dy_prime = dy_lab;
        const double dz_prime = dz_lab;

        // ds² in boosted frame
        const double ds2_prime = -(dt_prime * dt_prime)
                               +  (dx_prime * dx_prime)
                               +  (dy_prime * dy_prime)
                               +  (dz_prime * dz_prime);

        // Invariant: ds² must match
        SRFM_CHECK_NEAR(ds2_prime, ds2_lab, EPS_MED);
    }
}

// =============================================================================
// Spacetime norm invariance along flat geodesic
//
// On the flat Minkowski metric, u^μ u_μ = g_{μν} u^μ u^ν.
// For a geodesic with u = (1, 0, 0, 0) this equals −1.
// The RK4 integrator must conserve this norm throughout the trajectory.
// =============================================================================

static void test_velocity_norm_invariance() {
    GeodesicSolver solver;
    auto flat = MetricTensor::minkowski();

    // Initial 4-velocity: normalised time-like trajectory
    GeodesicState init{};
    init.x = {0.0, 0.0, 0.0, 0.0};
    init.u = {1.0, 0.0, 0.0, 0.0};   // u^μ u_μ = g_{00}·1² = -1

    auto final_state = solver.solve(init, flat, 200, 0.01);
    SRFM_HAS_VALUE(final_state);

    // Compute u^μ u_μ = −u0² + u1² + u2² + u3²  (Minkowski)
    const auto& u = final_state->u;
    const double norm = -(u[0] * u[0]) + (u[1] * u[1]) + (u[2] * u[2]) + (u[3] * u[3]);

    // Must equal initial norm −1.0 to machine precision
    SRFM_CHECK_NEAR(norm, -1.0, EPS_MED);

    // Test with a mixed-direction 4-velocity
    GeodesicState init2{};
    init2.x = {0.0, 1.0, 0.5, 0.0};
    init2.u = {1.5, 0.3, 0.1, 0.0};
    const double norm0 = -(init2.u[0]*init2.u[0])
                       +  (init2.u[1]*init2.u[1])
                       +  (init2.u[2]*init2.u[2])
                       +  (init2.u[3]*init2.u[3]);

    auto final2 = solver.solve(init2, flat, 100, 0.005);
    SRFM_HAS_VALUE(final2);
    const auto& u2 = final2->u;
    const double norm_final = -(u2[0]*u2[0]) + (u2[1]*u2[1]) + (u2[2]*u2[2]) + (u2[3]*u2[3]);
    SRFM_CHECK_NEAR(norm_final, norm0, EPS_MED);
}

// =============================================================================
// Velocity composition subluminality — full grid
//
// For every pair (β₁, β₂) in a grid, β₁⊕β₂ must satisfy |result| < 1.
// =============================================================================

static void test_composition_subluminality_grid() {
    const std::vector<double> grid = {
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999
    };

    for (double b1_val : grid) {
        for (double b2_val : grid) {
            auto b1 = BetaVelocity::make(b1_val);
            auto b2 = BetaVelocity::make(b2_val);
            if (!b1 || !b2) continue;

            auto composed = compose_velocities(*b1, *b2);
            // Composition of two sub-luminal velocities is always sub-luminal.
            // The result may be nullopt only if composed >= BETA_MAX_SAFE.
            if (composed) {
                SRFM_CHECK(std::abs(composed->value()) < 1.0);
                SRFM_CHECK(std::isfinite(composed->value()));
            }

            // Negative β₂
            auto b2_neg = BetaVelocity::make(-b2_val);
            if (b2_neg) {
                auto composed_neg = compose_velocities(*b1, *b2_neg);
                if (composed_neg) {
                    SRFM_CHECK(std::abs(composed_neg->value()) < 1.0);
                    SRFM_CHECK(std::isfinite(composed_neg->value()));
                }
            }
        }
    }
}

// =============================================================================
// Rapidity additivity (strict tolerance)
//
// φ(β₁⊕β₂) = φ(β₁) + φ(β₂) = atanh(β₁) + atanh(β₂)
// Must hold to EPS_MED for representative pairs.
// =============================================================================

static void test_rapidity_additivity_strict() {
    struct Pair { double b1; double b2; };
    const std::array<Pair, 6> pairs{{
        {0.0,  0.5},
        {0.1,  0.2},
        {0.3,  0.4},
        {0.5,  0.3},
        {0.2,  0.6},
        {0.05, 0.05},
    }};

    for (auto [b1_val, b2_val] : pairs) {
        auto b1 = BetaVelocity::make(b1_val);
        auto b2 = BetaVelocity::make(b2_val);
        SRFM_HAS_VALUE(b1);
        SRFM_HAS_VALUE(b2);

        const double phi1 = std::atanh(b1_val);
        const double phi2 = std::atanh(b2_val);

        auto composed = compose_velocities(*b1, *b2);
        SRFM_HAS_VALUE(composed);

        const double phi_composed = std::atanh(composed->value());

        // atanh(β₁⊕β₂) == atanh(β₁) + atanh(β₂)
        SRFM_CHECK_NEAR(phi_composed, phi1 + phi2, EPS_MED);
    }
}

// =============================================================================
// Round-trip boost identity
//
// apply_momentum_correction followed by inverse_transform must recover the
// original raw signal for a range of (β, m_eff) combinations.
// =============================================================================

static void test_round_trip_boost_identity() {
    const std::vector<double> betas   = {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99};
    const std::vector<double> m_effs  = {0.5, 1.0, 2.0, 10.0};
    const std::vector<double> signals = {-1000.0, -1.0, 0.0, 1.0, 100.0, 1e6};

    for (double b_val : betas) {
        auto bv = BetaVelocity::make(b_val);
        if (!bv) continue;

        for (double m_val : m_effs) {
            auto m = EffectiveMass::make(m_val);
            if (!m) continue;

            for (double raw : signals) {
                // Forward: p_rel = γ · m_eff · raw
                auto corrected = apply_momentum_correction(raw, *bv, *m);
                SRFM_HAS_VALUE(corrected);

                // The corrected value is γ·m_eff·raw.
                // Inverse-transform divides by γ, recovering m_eff·raw.
                auto recovered_gamma_part = inverse_transform(corrected->first, *bv);
                SRFM_HAS_VALUE(recovered_gamma_part);

                // recovered_gamma_part == m_eff * raw
                const double expected = m_val * raw;
                SRFM_CHECK_NEAR(*recovered_gamma_part, expected, EPS_MED);

                // Result is always finite
                SRFM_CHECK(std::isfinite(*recovered_gamma_part));
            }
        }
    }
}

// =============================================================================
// Newtonian expansion: γ ≈ 1 + β²/2 for |β| ≪ 1
// =============================================================================

static void test_gamma_newtonian_expansion() {
    // For small β, the Taylor expansion gives γ ≈ 1 + β²/2 + 3β⁴/8 + ...
    for (double b_val : {0.001, 0.01, 0.05, 0.1}) {
        auto bv = BetaVelocity::make(b_val);
        SRFM_HAS_VALUE(bv);

        auto g_opt = lorentz_gamma(*bv);
        SRFM_HAS_VALUE(g_opt);

        const double b    = b_val;
        const double g    = g_opt->value();
        const double approx = 1.0 + 0.5 * b * b;

        // Relative error from the leading-order Newtonian approximation
        const double rel_err = std::abs(g - approx) / g;
        // For b <= 0.1 the next term (3/8)β⁴ ~ 3.75e-5; relative error < 1e-3
        SRFM_CHECK(rel_err < 0.001);

        // γ ≥ 1 always
        SRFM_CHECK(g >= 1.0);
    }
}

// =============================================================================
// Subnormal (denormal) beta inputs
//
// std::numeric_limits<double>::denorm_min() is the smallest positive
// subnormal.  BetaVelocity::make should accept it (it is finite and < BETA_MAX_SAFE)
// and produce a γ indistinguishable from 1.0 at machine precision.
// =============================================================================

static void test_numerical_subnormal_inputs() {
    const double subnormal = std::numeric_limits<double>::denorm_min();

    auto bv = BetaVelocity::make(subnormal);
    SRFM_HAS_VALUE(bv);

    auto g = lorentz_gamma(*bv);
    SRFM_HAS_VALUE(g);

    // γ must be exactly 1.0 (within floating-point tolerance)
    SRFM_CHECK_NEAR(g->value(), 1.0, EPS);

    // Negative subnormal
    auto bv_neg = BetaVelocity::make(-subnormal);
    SRFM_HAS_VALUE(bv_neg);
    auto g_neg = lorentz_gamma(*bv_neg);
    SRFM_HAS_VALUE(g_neg);
    SRFM_CHECK_NEAR(g_neg->value(), 1.0, EPS);

    // Minimum normalised positive double
    const double min_norm = std::numeric_limits<double>::min();
    auto bv_min = BetaVelocity::make(min_norm);
    SRFM_HAS_VALUE(bv_min);
    auto g_min = lorentz_gamma(*bv_min);
    SRFM_HAS_VALUE(g_min);
    SRFM_CHECK_NEAR(g_min->value(), 1.0, EPS);
}

// =============================================================================
// Very large raw financial signal values
//
// apply_momentum_correction with large raw signals must produce finite output
// (not overflow to Inf) for all valid (β, m_eff).
// =============================================================================

static void test_very_large_financial_values() {
    // Maximum finite double divided by a safety margin so γ·m_eff·raw doesn't
    // overflow.  γ_max < 71 at β=0.9999-ε, m_eff_max = reasonable bound.
    const double large_raw = 1e12;   // representative large ADJ price signal

    for (double b_val : {0.0, 0.5, 0.9, 0.99}) {
        auto bv = BetaVelocity::make(b_val);
        if (!bv) continue;

        auto m = EffectiveMass::make(1.0);
        if (!m) continue;

        auto result = apply_momentum_correction(large_raw, *bv, *m);
        SRFM_HAS_VALUE(result);
        SRFM_CHECK(std::isfinite(result->first));
        SRFM_CHECK(result->first >= large_raw);  // γ ≥ 1 → amplification
    }

    // Very large m_eff
    auto bv = BetaVelocity::make(0.5).value();
    auto m_large = EffectiveMass::make(1e6);
    SRFM_HAS_VALUE(m_large);
    auto r = apply_momentum_correction(1.0, bv, *m_large);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(std::isfinite(r->first));
    SRFM_CHECK(r->first > 0.0);
}

// =============================================================================
// Very small positive raw financial signal values
//
// Tiny positive signals (subnormal territory) must not produce NaN or
// negative output after γ-correction.
// =============================================================================

static void test_very_small_financial_values() {
    const double tiny = std::numeric_limits<double>::min();       // smallest normal
    const double denorm = std::numeric_limits<double>::denorm_min();

    auto bv = BetaVelocity::make(0.5).value();
    auto m1 = EffectiveMass::make(1.0).value();

    auto r_tiny = apply_momentum_correction(tiny, bv, m1);
    SRFM_HAS_VALUE(r_tiny);
    SRFM_CHECK(std::isfinite(r_tiny->first));
    SRFM_CHECK(r_tiny->first > 0.0);

    auto r_denorm = apply_momentum_correction(denorm, bv, m1);
    SRFM_HAS_VALUE(r_denorm);
    SRFM_CHECK(std::isfinite(r_denorm->first));
    // γ·denorm_min may underflow to 0; only check it is not negative or NaN
    SRFM_CHECK(r_denorm->first >= 0.0);
    SRFM_CHECK(!std::isnan(r_denorm->first));

    // Batch with alternating tiny and normal values
    RelativisticSignalProcessor proc;
    std::array<RawSignal, 4> mixed{{
        {tiny}, {1.0}, {denorm}, {-1.0}
    }};
    auto out = proc.process(mixed, bv, m1);
    SRFM_HAS_VALUE(out);
    SRFM_CHECK(out->size() == 4);
    for (const auto& sig : *out) {
        SRFM_CHECK(!std::isnan(sig.adjusted_value));
    }
}

// =============================================================================
// std::numeric_limits boundary probes
//
// Explicitly probe values at the boundaries of the double type.
// =============================================================================

static void test_stdlib_boundary_probes() {
    using L = std::numeric_limits<double>;

    // max() is finite and >> 1 so should be rejected as superluminal
    SRFM_NO_VALUE(BetaVelocity::make(L::max()));
    SRFM_NO_VALUE(BetaVelocity::make(-L::max()));

    // lowest() == -max()
    SRFM_NO_VALUE(BetaVelocity::make(L::lowest()));

    // epsilon() is a valid small β
    auto bv_eps = BetaVelocity::make(L::epsilon());
    SRFM_HAS_VALUE(bv_eps);
    auto g_eps = lorentz_gamma(*bv_eps);
    SRFM_HAS_VALUE(g_eps);
    SRFM_CHECK_NEAR(g_eps->value(), 1.0, EPS);

    // quiet_NaN and signaling_NaN must be rejected
    SRFM_NO_VALUE(BetaVelocity::make(L::quiet_NaN()));
    SRFM_NO_VALUE(BetaVelocity::make(L::signaling_NaN()));

    // infinity variants must be rejected
    SRFM_NO_VALUE(BetaVelocity::make(L::infinity()));
    SRFM_NO_VALUE(BetaVelocity::make(-L::infinity()));

    // EffectiveMass with max()
    SRFM_NO_VALUE(EffectiveMass::make(L::infinity()));
    SRFM_NO_VALUE(EffectiveMass::make(L::quiet_NaN()));
    SRFM_HAS_VALUE(EffectiveMass::make(L::max()));   // max() is finite and > 0

    // apply_momentum_correction with max() m_eff and tiny beta: overflow to Inf?
    // We only require it not to return nullopt and not to be NaN.
    auto bv0 = BetaVelocity::make(0.0).value();
    auto m_max = EffectiveMass::make(L::max());
    if (m_max) {
        auto r = apply_momentum_correction(1.0, bv0, *m_max);
        SRFM_HAS_VALUE(r);
        SRFM_CHECK(!std::isnan(r->first));
        // May legitimately be Inf (max() * 1.0 * 1.0 = max())
        SRFM_CHECK(r->first > 0.0);
    }
}

// =============================================================================
// γ strictly monotone in |β|
//
// For 0 ≤ β₁ < β₂ < BETA_MAX_SAFE: γ(β₁) < γ(β₂).
// =============================================================================

static void test_lorentz_factor_monotonicity() {
    const std::vector<double> betas = {
        0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, BETA_MAX_SAFE - 1e-5
    };

    double prev_gamma = 0.0;
    for (double b_val : betas) {
        auto bv = BetaVelocity::make(b_val);
        if (!bv) continue;
        auto g = lorentz_gamma(*bv);
        SRFM_HAS_VALUE(g);
        SRFM_CHECK(g->value() > prev_gamma);
        prev_gamma = g->value();
    }
}

// =============================================================================
// γ symmetry: γ(β) == γ(−β)
// =============================================================================

static void test_beta_velocity_symmetry() {
    for (double b_val : {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99}) {
        auto bv_pos = BetaVelocity::make( b_val);
        auto bv_neg = BetaVelocity::make(-b_val);
        if (!bv_pos || !bv_neg) continue;

        auto g_pos = lorentz_gamma(*bv_pos);
        auto g_neg = lorentz_gamma(*bv_neg);
        SRFM_HAS_VALUE(g_pos);
        SRFM_HAS_VALUE(g_neg);

        // γ(−β) == γ(β) because γ depends only on β²
        SRFM_CHECK_NEAR(g_pos->value(), g_neg->value(), EPS);
    }
}

// =============================================================================
// Velocity composition associativity
//
// (β₁⊕β₂)⊕β₃ ≈ β₁⊕(β₂⊕β₃) (associativity holds for relativistic addition)
// =============================================================================

static void test_compose_associativity() {
    struct Triple { double b1, b2, b3; };
    const std::array<Triple, 4> triples{{
        {0.1, 0.2, 0.3},
        {0.3, 0.3, 0.3},
        {0.0, 0.5, 0.0},
        {0.1, 0.1, 0.1},
    }};

    for (auto [b1_val, b2_val, b3_val] : triples) {
        auto b1 = BetaVelocity::make(b1_val);
        auto b2 = BetaVelocity::make(b2_val);
        auto b3 = BetaVelocity::make(b3_val);
        if (!b1 || !b2 || !b3) continue;

        // Left-associative: (b1⊕b2)⊕b3
        auto left12 = compose_velocities(*b1, *b2);
        if (!left12) continue;
        auto left = compose_velocities(*left12, *b3);

        // Right-associative: b1⊕(b2⊕b3)
        auto right23 = compose_velocities(*b2, *b3);
        if (!right23) continue;
        auto right = compose_velocities(*b1, *right23);

        if (!left || !right) continue;

        SRFM_CHECK_NEAR(left->value(), right->value(), EPS_LOOSE);
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::printf("SRFM Lorentz Invariants and Numerical Edge Case Tests\n");
    std::printf("======================================================\n");

    SRFM_SUITE("spacetime interval invariance",       test_spacetime_interval_invariance);
    SRFM_SUITE("velocity norm invariance (geodesic)", test_velocity_norm_invariance);
    SRFM_SUITE("composition subluminality grid",      test_composition_subluminality_grid);
    SRFM_SUITE("rapidity additivity (strict)",        test_rapidity_additivity_strict);
    SRFM_SUITE("round-trip boost identity",           test_round_trip_boost_identity);
    SRFM_SUITE("gamma Newtonian expansion",           test_gamma_newtonian_expansion);
    SRFM_SUITE("subnormal inputs",                    test_numerical_subnormal_inputs);
    SRFM_SUITE("very large financial values",         test_very_large_financial_values);
    SRFM_SUITE("very small financial values",         test_very_small_financial_values);
    SRFM_SUITE("stdlib boundary probes",              test_stdlib_boundary_probes);
    SRFM_SUITE("gamma monotonicity",                  test_lorentz_factor_monotonicity);
    SRFM_SUITE("gamma symmetry beta/-beta",           test_beta_velocity_symmetry);
    SRFM_SUITE("composition associativity",           test_compose_associativity);

    return srfm_test::report();
}
