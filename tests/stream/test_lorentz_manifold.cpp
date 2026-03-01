/**
 * @file  test_lorentz_manifold.cpp
 * @brief Unit tests for LorentzTransform and SpacetimeManifold.
 *
 * Test structure — LorentzTransform:
 *   test_lorentz_identity_at_zero_beta         — β=0 is identity
 *   test_lorentz_known_boost_values            — analytical check at β=0.6
 *   test_lorentz_negative_beta                 — β<0 swaps directions
 *   test_lorentz_gamma_formula                 — lorentz_gamma() matches formula
 *   test_lorentz_invalid_beta_treated_as_zero  — |β|≥1 falls back to identity
 *   test_lorentz_inverse_round_trip            — transform then inverse recovers
 *   test_lorentz_gamma_monotone                — γ increases with |β|
 *
 * Test structure — SpacetimeManifold:
 *   test_manifold_first_event_lightlike        — first call returns LIGHTLIKE/0
 *   test_manifold_timelike_interval            — Δt > Δx → TIMELIKE
 *   test_manifold_spacelike_interval           — Δx > Δt → SPACELIKE
 *   test_manifold_lightlike_interval           — Δt ≈ Δx → LIGHTLIKE
 *   test_manifold_signal_sign                  — TIMELIKE>0, SPACELIKE<0, LIGHT=0
 *   test_manifold_reset                        — reset() clears state
 *   test_manifold_static_interval              — static interval() helper
 *   test_manifold_static_classify              — static classify() helper
 *   test_manifold_has_previous                 — has_previous() semantics
 */

#include "srfm_stream_test.hpp"
#include "../../include/srfm/stream/lorentz_transform.hpp"
#include "../../include/srfm/stream/spacetime_manifold.hpp"

#include <cmath>
#include <limits>

using namespace srfm::stream;

static constexpr double EPS       = 1e-9;
static constexpr double EPS_LOOSE = 1e-6;

// ═════════════════════════════════════════════════════════════════════════════
// LorentzTransform — identity at β = 0
// ═════════════════════════════════════════════════════════════════════════════

static void test_lorentz_identity_at_zero_beta() {
    LorentzTransform lt;

    auto ev = lt.transform(5.0, 3.0, 0.0);

    STREAM_CHECK_NEAR(ev.t_prime, 5.0, EPS);
    STREAM_CHECK_NEAR(ev.x_prime, 3.0, EPS);
    STREAM_CHECK_NEAR(ev.gamma,   1.0, EPS);
    STREAM_CHECK_NEAR(ev.beta,    0.0, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// LorentzTransform — known boost at β = 0.6 (γ = 1.25)
// ═════════════════════════════════════════════════════════════════════════════

static void test_lorentz_known_boost_values() {
    LorentzTransform lt;

    // At β=0.6, γ=1.25.
    // t'  = γ(t − βx) = 1.25(10 − 0.6·5) = 1.25 * 7 = 8.75
    // x'  = γ(x − βt) = 1.25(5  − 0.6·10) = 1.25 * (-1) = -1.25
    auto ev = lt.transform(10.0, 5.0, 0.6);

    STREAM_CHECK_NEAR(ev.gamma,   1.25,  1e-7);
    STREAM_CHECK_NEAR(ev.beta,    0.6,   EPS);
    STREAM_CHECK_NEAR(ev.t_prime, 8.75,  1e-7);
    STREAM_CHECK_NEAR(ev.x_prime, -1.25, 1e-7);
}

// ═════════════════════════════════════════════════════════════════════════════
// LorentzTransform — negative β
// ═════════════════════════════════════════════════════════════════════════════

static void test_lorentz_negative_beta() {
    LorentzTransform lt;

    // β = -0.6, γ = 1.25.
    // t' = γ(t + 0.6x) = 1.25(10 + 3) = 16.25
    // x' = γ(x + 0.6t) = 1.25(5 + 6)  = 13.75
    auto ev = lt.transform(10.0, 5.0, -0.6);

    STREAM_CHECK_NEAR(ev.gamma,   1.25,  1e-7);
    STREAM_CHECK_NEAR(ev.beta,    -0.6,  EPS);
    STREAM_CHECK_NEAR(ev.t_prime, 16.25, 1e-7);
    STREAM_CHECK_NEAR(ev.x_prime, 13.75, 1e-7);
}

// ═════════════════════════════════════════════════════════════════════════════
// LorentzTransform — lorentz_gamma() formula
// ═════════════════════════════════════════════════════════════════════════════

static void test_lorentz_gamma_formula() {
    // γ(β=0) = 1.
    STREAM_CHECK_NEAR(LorentzTransform::lorentz_gamma(0.0), 1.0, EPS);

    // γ(β=0.6) = 1.25.
    STREAM_CHECK_NEAR(LorentzTransform::lorentz_gamma(0.6), 1.25, 1e-7);

    // γ(β=0.8) = 1/√(1−0.64) = 1/0.6 ≈ 1.6667.
    STREAM_CHECK_NEAR(LorentzTransform::lorentz_gamma(0.8),
                      1.0 / std::sqrt(0.36), 1e-7);

    // γ always ≥ 1.
    for (int i = 0; i <= 99; ++i) {
        double b = static_cast<double>(i) * 0.009999;
        STREAM_CHECK(LorentzTransform::lorentz_gamma(b) >= 1.0);
    }

    // γ(β<0) == γ(|β|).
    STREAM_CHECK_NEAR(LorentzTransform::lorentz_gamma(-0.6),
                      LorentzTransform::lorentz_gamma(0.6), EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// LorentzTransform — invalid β falls back to identity
// ═════════════════════════════════════════════════════════════════════════════

static void test_lorentz_invalid_beta_treated_as_zero() {
    LorentzTransform lt;

    // β = 1.0 → identity.
    auto ev1 = lt.transform(7.0, 3.0, 1.0);
    STREAM_CHECK_NEAR(ev1.t_prime, 7.0, EPS);
    STREAM_CHECK_NEAR(ev1.x_prime, 3.0, EPS);
    STREAM_CHECK_NEAR(ev1.gamma,   1.0, EPS);

    // β = -1.0 → identity.
    auto ev2 = lt.transform(7.0, 3.0, -1.0);
    STREAM_CHECK_NEAR(ev2.t_prime, 7.0, EPS);
    STREAM_CHECK_NEAR(ev2.x_prime, 3.0, EPS);

    // β = NaN → identity.
    auto ev3 = lt.transform(7.0, 3.0,
        std::numeric_limits<double>::quiet_NaN());
    STREAM_CHECK_NEAR(ev3.t_prime, 7.0, EPS);
    STREAM_CHECK_NEAR(ev3.x_prime, 3.0, EPS);

    // β = Inf → identity.
    auto ev4 = lt.transform(7.0, 3.0,
        std::numeric_limits<double>::infinity());
    STREAM_CHECK_NEAR(ev4.t_prime, 7.0, EPS);
    STREAM_CHECK_NEAR(ev4.x_prime, 3.0, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// LorentzTransform — inverse round-trip
// ═════════════════════════════════════════════════════════════════════════════

static void test_lorentz_inverse_round_trip() {
    LorentzTransform lt;

    const double t = 15.0, x = 7.0;
    const double beta = 0.5;

    auto fwd = lt.transform(t, x, beta);
    auto inv = lt.inverse(fwd.t_prime, fwd.x_prime, beta);

    STREAM_CHECK_NEAR(inv.t_prime, t, EPS_LOOSE);
    STREAM_CHECK_NEAR(inv.x_prime, x, EPS_LOOSE);

    // Test with negative beta.
    auto fwd2 = lt.transform(t, x, -0.7);
    auto inv2 = lt.inverse(fwd2.t_prime, fwd2.x_prime, -0.7);
    STREAM_CHECK_NEAR(inv2.t_prime, t, EPS_LOOSE);
    STREAM_CHECK_NEAR(inv2.x_prime, x, EPS_LOOSE);
}

// ═════════════════════════════════════════════════════════════════════════════
// LorentzTransform — γ is monotone increasing with |β|
// ═════════════════════════════════════════════════════════════════════════════

static void test_lorentz_gamma_monotone() {
    double prev = 0.0;
    for (int i = 0; i <= 99; ++i) {
        double b = static_cast<double>(i) * 0.009999;
        double g = LorentzTransform::lorentz_gamma(b);
        STREAM_CHECK(g >= prev);
        prev = g;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SpacetimeManifold — first event is LIGHTLIKE
// ═════════════════════════════════════════════════════════════════════════════

static void test_manifold_first_event_lightlike() {
    SpacetimeManifold m;

    auto r = m.update(5.0, 3.0);

    STREAM_CHECK(r.regime == Regime::LIGHTLIKE);
    STREAM_CHECK_NEAR(r.signal, 0.0, EPS);
    STREAM_CHECK_NEAR(r.ds2,    0.0, EPS);
    STREAM_CHECK(m.has_previous());
}

// ═════════════════════════════════════════════════════════════════════════════
// SpacetimeManifold — TIMELIKE interval (|Δt| > |Δx|)
// ═════════════════════════════════════════════════════════════════════════════

static void test_manifold_timelike_interval() {
    SpacetimeManifold m;
    (void)m.update(0.0, 0.0);  // first event — result intentionally discarded

    // Δt = 5, Δx = 2 → ds2 = 25 - 4 = 21 > 0 → TIMELIKE
    auto r = m.update(5.0, 2.0);

    STREAM_CHECK(r.regime == Regime::TIMELIKE);
    STREAM_CHECK_NEAR(r.ds2,    21.0,          EPS_LOOSE);
    STREAM_CHECK_NEAR(r.signal, std::sqrt(21.0), EPS_LOOSE);
    STREAM_CHECK(r.signal > 0.0);
}

// ═════════════════════════════════════════════════════════════════════════════
// SpacetimeManifold — SPACELIKE interval (|Δx| > |Δt|)
// ═════════════════════════════════════════════════════════════════════════════

static void test_manifold_spacelike_interval() {
    SpacetimeManifold m;
    (void)m.update(0.0, 0.0);

    // Δt = 2, Δx = 5 → ds2 = 4 - 25 = -21 < 0 → SPACELIKE
    auto r = m.update(2.0, 5.0);

    STREAM_CHECK(r.regime == Regime::SPACELIKE);
    STREAM_CHECK_NEAR(r.ds2,    -21.0,          EPS_LOOSE);
    STREAM_CHECK_NEAR(r.signal, -std::sqrt(21.0), EPS_LOOSE);
    STREAM_CHECK(r.signal < 0.0);
}

// ═════════════════════════════════════════════════════════════════════════════
// SpacetimeManifold — LIGHTLIKE (Δt ≈ Δx, within epsilon)
// ═════════════════════════════════════════════════════════════════════════════

static void test_manifold_lightlike_interval() {
    SpacetimeManifold m{1e-6}; // epsilon = 1e-6
    (void)m.update(0.0, 0.0);

    // Δt = Δx = 1.0 → ds2 = 0 → LIGHTLIKE.
    auto r = m.update(1.0, 1.0);

    STREAM_CHECK(r.regime == Regime::LIGHTLIKE);
    STREAM_CHECK_NEAR(r.signal, 0.0, EPS);

    // ds2 = 1e-7 < epsilon → LIGHTLIKE.
    SpacetimeManifold m2{1e-6};
    (void)m2.update(0.0, 0.0);
    // Δt² - Δx² = (1.0)² - (1.0 - 1e-7/2)² ≈ 1e-7
    double small_dx = std::sqrt(1.0 - 1e-7);
    auto r2 = m2.update(1.0, small_dx);
    STREAM_CHECK(r2.regime == Regime::LIGHTLIKE);
}

// ═════════════════════════════════════════════════════════════════════════════
// SpacetimeManifold — signal sign by regime
// ═════════════════════════════════════════════════════════════════════════════

static void test_manifold_signal_sign() {
    // TIMELIKE → signal > 0.
    {
        SpacetimeManifold m;
        (void)m.update(0.0, 0.0);
        auto r = m.update(10.0, 1.0);  // clearly TIMELIKE
        STREAM_CHECK(r.regime == Regime::TIMELIKE);
        STREAM_CHECK(r.signal > 0.0);
    }

    // SPACELIKE → signal < 0.
    {
        SpacetimeManifold m;
        (void)m.update(0.0, 0.0);
        auto r = m.update(1.0, 10.0);  // clearly SPACELIKE
        STREAM_CHECK(r.regime == Regime::SPACELIKE);
        STREAM_CHECK(r.signal < 0.0);
    }

    // LIGHTLIKE → signal = 0.
    {
        SpacetimeManifold m;
        (void)m.update(0.0, 0.0);
        auto r = m.update(1.0, 1.0);
        STREAM_CHECK(r.regime == Regime::LIGHTLIKE);
        STREAM_CHECK_NEAR(r.signal, 0.0, EPS);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SpacetimeManifold — reset() clears state
// ═════════════════════════════════════════════════════════════════════════════

static void test_manifold_reset() {
    SpacetimeManifold m;
    (void)m.update(5.0, 3.0);
    STREAM_CHECK(m.has_previous());

    m.reset();
    STREAM_CHECK(!m.has_previous());

    // After reset, first event is LIGHTLIKE again.
    auto r = m.update(99.0, 99.0);
    STREAM_CHECK(r.regime == Regime::LIGHTLIKE);
    STREAM_CHECK_NEAR(r.signal, 0.0, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// SpacetimeManifold — static interval() helper
// ═════════════════════════════════════════════════════════════════════════════

static void test_manifold_static_interval() {
    // ds2 = dt² - dx²
    STREAM_CHECK_NEAR(SpacetimeManifold::interval(5.0, 3.0),
                      25.0 - 9.0, EPS);
    STREAM_CHECK_NEAR(SpacetimeManifold::interval(3.0, 5.0),
                      9.0 - 25.0, EPS);
    STREAM_CHECK_NEAR(SpacetimeManifold::interval(1.0, 1.0), 0.0, EPS);
    STREAM_CHECK_NEAR(SpacetimeManifold::interval(0.0, 0.0), 0.0, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// SpacetimeManifold — static classify() helper
// ═════════════════════════════════════════════════════════════════════════════

static void test_manifold_static_classify() {
    STREAM_CHECK(SpacetimeManifold::classify(10.0)  == Regime::TIMELIKE);
    STREAM_CHECK(SpacetimeManifold::classify(-10.0) == Regime::SPACELIKE);
    STREAM_CHECK(SpacetimeManifold::classify(0.0)   == Regime::LIGHTLIKE);
    STREAM_CHECK(SpacetimeManifold::classify(5e-7, 1e-6) == Regime::LIGHTLIKE);
    STREAM_CHECK(SpacetimeManifold::classify(2e-6, 1e-6) == Regime::TIMELIKE);
    STREAM_CHECK(SpacetimeManifold::classify(-2e-6, 1e-6) == Regime::SPACELIKE);
}

// ═════════════════════════════════════════════════════════════════════════════
// SpacetimeManifold — has_previous() semantics
// ═════════════════════════════════════════════════════════════════════════════

static void test_manifold_has_previous() {
    SpacetimeManifold m;
    STREAM_CHECK(!m.has_previous());

    (void)m.update(0.0, 0.0);
    STREAM_CHECK(m.has_previous());

    // Continues to be true.
    for (int i = 1; i < 10; ++i) {
        (void)m.update(static_cast<double>(i), 0.0);
        STREAM_CHECK(m.has_previous());
    }

    m.reset();
    STREAM_CHECK(!m.has_previous());
}

// ═════════════════════════════════════════════════════════════════════════════
// LorentzTransform + SpacetimeManifold integration
// ═════════════════════════════════════════════════════════════════════════════

static void test_lorentz_manifold_integration() {
    LorentzTransform  lt;
    SpacetimeManifold mf;

    // Simulate 50 ticks at β=0.6, bar advancing by 1 each time, x by 0.3.
    const double beta = 0.6;

    for (int i = 0; i < 50; ++i) {
        double t = static_cast<double>(i);
        double x = 0.3 * t;

        auto ev = lt.transform(t, x, beta);

        STREAM_CHECK_FINITE(ev.t_prime);
        STREAM_CHECK_FINITE(ev.x_prime);
        STREAM_CHECK(ev.gamma >= 1.0);

        auto mr = mf.update(ev.t_prime, ev.x_prime);
        STREAM_CHECK_FINITE(mr.ds2);
        STREAM_CHECK_FINITE(mr.signal);

        // After first event, signal must be finite.
        if (i > 0) {
            STREAM_CHECK_FINITE(mr.signal);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("SRFM Stream — LorentzTransform + SpacetimeManifold Tests\n");
    std::printf("============================================================\n");

    // LorentzTransform tests.
    STREAM_SUITE("identity at β=0",             test_lorentz_identity_at_zero_beta);
    STREAM_SUITE("known boost at β=0.6",        test_lorentz_known_boost_values);
    STREAM_SUITE("negative β",                  test_lorentz_negative_beta);
    STREAM_SUITE("lorentz_gamma formula",       test_lorentz_gamma_formula);
    STREAM_SUITE("invalid β → identity",        test_lorentz_invalid_beta_treated_as_zero);
    STREAM_SUITE("inverse round-trip",          test_lorentz_inverse_round_trip);
    STREAM_SUITE("γ monotone increasing",       test_lorentz_gamma_monotone);

    // SpacetimeManifold tests.
    STREAM_SUITE("first event LIGHTLIKE",       test_manifold_first_event_lightlike);
    STREAM_SUITE("TIMELIKE interval",           test_manifold_timelike_interval);
    STREAM_SUITE("SPACELIKE interval",          test_manifold_spacelike_interval);
    STREAM_SUITE("LIGHTLIKE interval",          test_manifold_lightlike_interval);
    STREAM_SUITE("signal sign by regime",       test_manifold_signal_sign);
    STREAM_SUITE("reset() clears state",        test_manifold_reset);
    STREAM_SUITE("static interval()",           test_manifold_static_interval);
    STREAM_SUITE("static classify()",           test_manifold_static_classify);
    STREAM_SUITE("has_previous() semantics",    test_manifold_has_previous);

    // Integration.
    STREAM_SUITE("Lorentz+Manifold integration", test_lorentz_manifold_integration);

    return stream_test::report();
}
