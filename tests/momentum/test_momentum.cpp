/**
 * @file  test_momentum.cpp
 * @brief Unit + integration tests for src/momentum/momentum.hpp/.cpp
 *
 * Structure
 * ---------
 *   test_beta_velocity        — BetaVelocity::make validation
 *   test_effective_mass       — EffectiveMass::make + from_adv
 *   test_lorentz_gamma        — γ physics, boundary values, identities
 *   test_momentum_correction  — apply_momentum_correction
 *   test_compose_velocities   — relativistic addition
 *   test_inverse_transform    — round-trip property
 *   test_processor_batch      — RelativisticSignalProcessor::process
 *   test_processor_one        — RelativisticSignalProcessor::process_one
 *   test_integration          — full pipeline, ADV→signal
 *
 * Naming convention:  test_{what}_{condition}_{expected_outcome}
 */

#include "srfm_test.hpp"
#include "momentum/momentum.hpp"

#include <array>
#include <cmath>
#include <limits>

using namespace srfm::momentum;

// ── Helpers ───────────────────────────────────────────────────────────────────

static constexpr double EPS = 1e-9;

// ═════════════════════════════════════════════════════════════════════════════
// BetaVelocity
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_velocity() {
    // Valid: zero
    SRFM_HAS_VALUE(BetaVelocity::make(0.0));

    // Valid: small positive
    SRFM_HAS_VALUE(BetaVelocity::make(0.001));

    // Valid: small negative
    SRFM_HAS_VALUE(BetaVelocity::make(-0.5));

    // Valid: just below max safe
    SRFM_HAS_VALUE(BetaVelocity::make(0.9998));

    // Invalid: exactly BETA_MAX_SAFE
    SRFM_NO_VALUE(BetaVelocity::make(BETA_MAX_SAFE));

    // Invalid: superluminal (β = 1)
    SRFM_NO_VALUE(BetaVelocity::make(1.0));

    // Invalid: superluminal positive
    SRFM_NO_VALUE(BetaVelocity::make(1.5));

    // Invalid: superluminal negative
    SRFM_NO_VALUE(BetaVelocity::make(-1.5));

    // Invalid: NaN
    SRFM_NO_VALUE(BetaVelocity::make(std::numeric_limits<double>::quiet_NaN()));

    // Invalid: +Infinity
    SRFM_NO_VALUE(BetaVelocity::make(std::numeric_limits<double>::infinity()));

    // Invalid: -Infinity
    SRFM_NO_VALUE(BetaVelocity::make(-std::numeric_limits<double>::infinity()));

    // Value round-trips
    auto b = BetaVelocity::make(0.6);
    SRFM_HAS_VALUE(b);
    SRFM_CHECK_NEAR(b->value(), 0.6, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// EffectiveMass
// ═════════════════════════════════════════════════════════════════════════════

static void test_effective_mass() {
    // Valid values
    SRFM_HAS_VALUE(EffectiveMass::make(1.0));
    SRFM_HAS_VALUE(EffectiveMass::make(0.001));
    SRFM_HAS_VALUE(EffectiveMass::make(1e6));

    // Invalid: zero
    SRFM_NO_VALUE(EffectiveMass::make(0.0));

    // Invalid: negative
    SRFM_NO_VALUE(EffectiveMass::make(-1.0));

    // Invalid: NaN
    SRFM_NO_VALUE(EffectiveMass::make(std::numeric_limits<double>::quiet_NaN()));

    // Invalid: infinity (not finite)
    SRFM_NO_VALUE(EffectiveMass::make(std::numeric_limits<double>::infinity()));

    // from_adv: valid ratio 2x baseline
    auto m = EffectiveMass::from_adv(2'000'000.0, 1'000'000.0);
    SRFM_HAS_VALUE(m);
    SRFM_CHECK_NEAR(m->value(), 2.0, EPS);

    // from_adv: unit ratio
    auto unit = EffectiveMass::from_adv(1'000'000.0, 1'000'000.0);
    SRFM_HAS_VALUE(unit);
    SRFM_CHECK_NEAR(unit->value(), 1.0, EPS);

    // from_adv: zero ADV
    SRFM_NO_VALUE(EffectiveMass::from_adv(0.0, 1'000'000.0));

    // from_adv: zero baseline
    SRFM_NO_VALUE(EffectiveMass::from_adv(1'000'000.0, 0.0));

    // from_adv: negative ADV
    SRFM_NO_VALUE(EffectiveMass::from_adv(-500.0, 1'000'000.0));

    // from_adv: value round-trips
    SRFM_CHECK_NEAR(m->value(), 2.0, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// lorentz_gamma
// ═════════════════════════════════════════════════════════════════════════════

static void test_lorentz_gamma() {
    // Newtonian limit: β = 0 → γ = 1
    auto b0 = BetaVelocity::make(0.0).value();
    auto g0 = lorentz_gamma(b0);
    SRFM_HAS_VALUE(g0);
    SRFM_CHECK_NEAR(g0->value(), 1.0, EPS);

    // Small β: minimal correction
    auto b_small = BetaVelocity::make(0.001).value();
    auto g_small = lorentz_gamma(b_small);
    SRFM_HAS_VALUE(g_small);
    SRFM_CHECK(g_small->value() > 1.0);
    SRFM_CHECK(g_small->value() < 1.001);

    // β = 0.6 → γ = 1/√(1−0.36) = 1/0.8 = 1.25
    auto b06 = BetaVelocity::make(0.6).value();
    auto g06 = lorentz_gamma(b06);
    SRFM_HAS_VALUE(g06);
    SRFM_CHECK_NEAR(g06->value(), 1.25, 1e-7);

    // Negative β produces same γ (γ depends only on β²)
    auto b_neg = BetaVelocity::make(-0.6).value();
    auto g_neg = lorentz_gamma(b_neg);
    SRFM_HAS_VALUE(g_neg);
    SRFM_CHECK_NEAR(g06->value(), g_neg->value(), EPS);

    // Very high β: γ grows large
    auto b_high = BetaVelocity::make(0.9998).value();
    auto g_high = lorentz_gamma(b_high);
    SRFM_HAS_VALUE(g_high);
    SRFM_CHECK(g_high->value() > 50.0);

    // γ always ≥ 1 for all valid β
    for (int i = 0; i <= 99; ++i) {
        double b_val = static_cast<double>(i) * (BETA_MAX_SAFE / 100.0);
        auto bv = BetaVelocity::make(b_val).value();
        auto gv = lorentz_gamma(bv);
        SRFM_HAS_VALUE(gv);
        SRFM_CHECK(gv->value() >= 1.0);
    }

    // γ² identity: γ² = 1/(1−β²)
    auto bv = BetaVelocity::make(0.6).value();
    auto gv = lorentz_gamma(bv).value();
    double b = bv.value();
    SRFM_CHECK_NEAR(gv.value() * gv.value(), 1.0 / (1.0 - b * b), 1e-9);

    // γβ identity: γβ = β/√(1−β²)
    SRFM_CHECK_NEAR(gv.value() * b,
                    b / std::sqrt(1.0 - b * b),
                    1e-9);
}

// ═════════════════════════════════════════════════════════════════════════════
// apply_momentum_correction
// ═════════════════════════════════════════════════════════════════════════════

static void test_momentum_correction() {
    // β=0 → γ=1 → adjusted = m_eff * raw
    auto b0   = BetaVelocity::make(0.0).value();
    auto m2   = EffectiveMass::make(2.0).value();
    auto r0   = apply_momentum_correction(100.0, b0, m2);
    SRFM_HAS_VALUE(r0);
    SRFM_CHECK_NEAR(r0->first, 200.0, 1e-7);
    SRFM_CHECK_NEAR(r0->second.value(), 1.0, EPS);

    // β=0.6, m_eff=1 → γ=1.25 → adjusted=125
    auto b06 = BetaVelocity::make(0.6).value();
    auto m1  = EffectiveMass::make(1.0).value();
    auto r1  = apply_momentum_correction(100.0, b06, m1);
    SRFM_HAS_VALUE(r1);
    SRFM_CHECK_NEAR(r1->first, 125.0, 1e-7);
    SRFM_CHECK_NEAR(r1->second.value(), 1.25, 1e-7);

    // Zero raw signal → zero adjusted
    auto r2 = apply_momentum_correction(0.0, b06, m1);
    SRFM_HAS_VALUE(r2);
    SRFM_CHECK(std::abs(r2->first) < 1e-15);

    // Negative raw signal → negative adjusted
    auto r3 = apply_momentum_correction(-100.0, b06, m1);
    SRFM_HAS_VALUE(r3);
    SRFM_CHECK(r3->first < 0.0);
    SRFM_CHECK_NEAR(r3->first, -125.0, 1e-7);

    // m_eff scales proportionally (β=0 so γ=1)
    auto m5  = EffectiveMass::make(5.0).value();
    auto ra  = apply_momentum_correction(10.0, b0, m1);
    auto rb  = apply_momentum_correction(10.0, b0, m5);
    SRFM_HAS_VALUE(ra); SRFM_HAS_VALUE(rb);
    SRFM_CHECK_NEAR(rb->first / ra->first, 5.0, 1e-9);

    // adjusted == gamma * m_eff * raw (invariant)
    auto b08  = BetaVelocity::make(0.8).value();
    auto m3   = EffectiveMass::make(3.0).value();
    auto r4   = apply_momentum_correction(7.0, b08, m3);
    SRFM_HAS_VALUE(r4);
    double expected = r4->second.value() * 3.0 * 7.0;
    SRFM_CHECK_NEAR(r4->first, expected, 1e-9);
}

// ═════════════════════════════════════════════════════════════════════════════
// compose_velocities
// ═════════════════════════════════════════════════════════════════════════════

static void test_compose_velocities() {
    // 0.5c + 0.5c = (0.5+0.5)/(1+0.25) = 1.0/1.25 = 0.8
    auto b5a = BetaVelocity::make(0.5).value();
    auto b5b = BetaVelocity::make(0.5).value();
    auto composed = compose_velocities(b5a, b5b);
    SRFM_HAS_VALUE(composed);
    SRFM_CHECK_NEAR(composed->value(), 0.8, 1e-10);

    // Zero + any = any
    auto b0   = BetaVelocity::make(0.0).value();
    auto b7   = BetaVelocity::make(0.7).value();
    auto r_id = compose_velocities(b0, b7);
    SRFM_HAS_VALUE(r_id);
    SRFM_CHECK_NEAR(r_id->value(), 0.7, 1e-10);

    // Result always sub-luminal
    auto b9a = BetaVelocity::make(0.9).value();
    auto b9b = BetaVelocity::make(0.9).value();
    auto r9  = compose_velocities(b9a, b9b);
    SRFM_HAS_VALUE(r9);
    SRFM_CHECK(std::abs(r9->value()) < BETA_MAX_SAFE);

    // Commutative
    auto b3 = BetaVelocity::make(0.3).value();
    auto b4 = BetaVelocity::make(0.4).value();
    auto r_ab = compose_velocities(b3, b4);
    auto r_ba = compose_velocities(b4, b3);
    SRFM_HAS_VALUE(r_ab); SRFM_HAS_VALUE(r_ba);
    SRFM_CHECK_NEAR(r_ab->value(), r_ba->value(), 1e-15);

    // Negative + positive composition (anti-parallel)
    auto b_neg = BetaVelocity::make(-0.3).value();
    auto r_neg = compose_velocities(b_neg, b4);
    SRFM_HAS_VALUE(r_neg);
    // (-0.3 + 0.4) / (1 + (-0.3)(0.4)) = 0.1 / 0.88 ≈ 0.1136
    SRFM_CHECK_NEAR(r_neg->value(), 0.1 / (1.0 - 0.12), 1e-9);
}

// ═════════════════════════════════════════════════════════════════════════════
// inverse_transform
// ═════════════════════════════════════════════════════════════════════════════

static void test_inverse_transform() {
    // β=0: inverse of identity is identity
    auto b0  = BetaVelocity::make(0.0).value();
    auto r0  = inverse_transform(42.0, b0);
    SRFM_HAS_VALUE(r0);
    SRFM_CHECK_NEAR(*r0, 42.0, EPS);

    // Round-trip: dilate then invert recovers original
    auto b06 = BetaVelocity::make(0.6).value();
    double proper = 200.0;
    double gamma  = lorentz_gamma(b06).value().value();
    double dilated = proper * gamma;
    auto recovered = inverse_transform(dilated, b06);
    SRFM_HAS_VALUE(recovered);
    SRFM_CHECK_NEAR(*recovered, proper, 1e-8);

    // Negative value round-trips correctly
    auto b05 = BetaVelocity::make(0.5).value();
    double g5 = lorentz_gamma(b05).value().value();
    auto inv_neg = inverse_transform(-100.0 * g5, b05);
    SRFM_HAS_VALUE(inv_neg);
    SRFM_CHECK_NEAR(*inv_neg, -100.0, 1e-8);
}

// ═════════════════════════════════════════════════════════════════════════════
// RelativisticSignalProcessor — batch
// ═════════════════════════════════════════════════════════════════════════════

static void test_processor_batch() {
    RelativisticSignalProcessor proc;
    auto b06 = BetaVelocity::make(0.6).value();
    auto m1  = EffectiveMass::make(1.0).value();

    // Empty span returns empty vector
    std::array<RawSignal, 0> empty{};
    auto r_empty = proc.process(empty, b06, m1);
    SRFM_HAS_VALUE(r_empty);
    SRFM_CHECK(r_empty->empty());

    // Single signal: adjusted = γ * m_eff * raw = 1.25 * 100 = 125
    std::array<RawSignal, 1> one{{{100.0}}};
    auto r1 = proc.process(one, b06, m1);
    SRFM_HAS_VALUE(r1);
    SRFM_CHECK(r1->size() == 1);
    SRFM_CHECK_NEAR((*r1)[0].adjusted_value, 125.0, 1e-7);
    SRFM_CHECK_NEAR((*r1)[0].gamma.value(), 1.25, 1e-7);
    SRFM_CHECK_NEAR((*r1)[0].raw_value, 100.0, EPS);

    // All signals in batch share same gamma
    std::array<RawSignal, 3> three{{{10.0}, {20.0}, {30.0}}};
    auto r3 = proc.process(three, b06, m1);
    SRFM_HAS_VALUE(r3);
    SRFM_CHECK(r3->size() == 3);
    SRFM_CHECK_NEAR((*r3)[0].gamma.value(), (*r3)[1].gamma.value(), EPS);
    SRFM_CHECK_NEAR((*r3)[1].gamma.value(), (*r3)[2].gamma.value(), EPS);

    // Raw values preserved
    SRFM_CHECK_NEAR((*r3)[0].raw_value, 10.0, EPS);
    SRFM_CHECK_NEAR((*r3)[1].raw_value, 20.0, EPS);
    SRFM_CHECK_NEAR((*r3)[2].raw_value, 30.0, EPS);

    // adjusted == gamma * m_eff * raw (invariant holds for every element)
    double gval = (*r3)[0].gamma.value();
    for (const auto& sig : *r3) {
        SRFM_CHECK_NEAR(sig.adjusted_value,
                        gval * m1.value() * sig.raw_value, 1e-9);
    }

    // Zero signal → zero adjusted
    std::array<RawSignal, 1> zero_sig{{{0.0}}};
    auto rz = proc.process(zero_sig, b06, m1);
    SRFM_HAS_VALUE(rz);
    SRFM_CHECK(std::abs((*rz)[0].adjusted_value) < 1e-15);

    // Negative signal → negative adjusted
    std::array<RawSignal, 1> neg_sig{{{-100.0}}};
    auto rn = proc.process(neg_sig, b06, m1);
    SRFM_HAS_VALUE(rn);
    SRFM_CHECK((*rn)[0].adjusted_value < 0.0);
    SRFM_CHECK_NEAR((*rn)[0].adjusted_value, -125.0, 1e-7);

    // Output length matches input length
    for (std::size_t n : {0u, 1u, 5u, 50u}) {
        std::vector<RawSignal> sigs(n, {1.0});
        auto out = proc.process(sigs, b06, m1);
        SRFM_HAS_VALUE(out);
        SRFM_CHECK(out->size() == n);
    }

    // Newtonian limit (β=0, m_eff=1): adjusted == raw
    auto b0  = BetaVelocity::make(0.0).value();
    std::array<RawSignal, 2> pair{{{33.3}, {-7.0}}};
    auto r_newton = proc.process(pair, b0, m1);
    SRFM_HAS_VALUE(r_newton);
    SRFM_CHECK_NEAR((*r_newton)[0].adjusted_value, 33.3, 1e-9);
    SRFM_CHECK_NEAR((*r_newton)[1].adjusted_value, -7.0, 1e-9);

    // ADV-based m_eff: adv=2x → scale factor doubles
    auto m2 = EffectiveMass::from_adv(2'000'000.0, 1'000'000.0).value();
    std::array<RawSignal, 1> sig50{{{50.0}}};
    auto r_adv = proc.process(sig50, b0, m2); // γ=1, m_eff=2 → 100
    SRFM_HAS_VALUE(r_adv);
    SRFM_CHECK_NEAR((*r_adv)[0].adjusted_value, 100.0, 1e-9);
}

// ═════════════════════════════════════════════════════════════════════════════
// RelativisticSignalProcessor — single signal
// ═════════════════════════════════════════════════════════════════════════════

static void test_processor_one() {
    RelativisticSignalProcessor proc;
    auto b06 = BetaVelocity::make(0.6).value();
    auto m1  = EffectiveMass::make(1.0).value();

    // process_one matches process for a single element
    auto single = proc.process_one({100.0}, b06, m1);
    std::array<RawSignal, 1> arr{{{100.0}}};
    auto batch  = proc.process(arr, b06, m1);
    SRFM_HAS_VALUE(single); SRFM_HAS_VALUE(batch);
    SRFM_CHECK_NEAR(single->adjusted_value, (*batch)[0].adjusted_value, 1e-12);
    SRFM_CHECK_NEAR(single->gamma.value(),  (*batch)[0].gamma.value(),  1e-15);

    // raw_value field is preserved
    auto r = proc.process_one({77.0}, b06, m1);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK_NEAR(r->raw_value, 77.0, EPS);

    // Newtonian limit: adjusted == raw when β=0, m_eff=1
    auto b0 = BetaVelocity::make(0.0).value();
    auto r0 = proc.process_one({42.0}, b0, m1);
    SRFM_HAS_VALUE(r0);
    SRFM_CHECK_NEAR(r0->adjusted_value, 42.0, EPS);

    // Negative input preserved with correct sign
    auto rn = proc.process_one({-200.0}, b06, m1);
    SRFM_HAS_VALUE(rn);
    SRFM_CHECK(rn->adjusted_value < 0.0);
}

// ═════════════════════════════════════════════════════════════════════════════
// Integration: full pipeline  ADV → EffectiveMass → batch process
// ═════════════════════════════════════════════════════════════════════════════

static void test_integration() {
    // Scenario: liquid market, 60% speed → γ=1.25, m_eff=2 → scale=2.5
    auto beta  = BetaVelocity::make(0.6).value();
    auto m_eff = EffectiveMass::from_adv(2'000'000.0, 1'000'000.0).value();

    std::array<RawSignal, 4> raw{{{10.0}, {20.0}, {-5.0}, {0.0}}};
    RelativisticSignalProcessor proc;
    auto out = proc.process(raw, beta, m_eff);
    SRFM_HAS_VALUE(out);
    SRFM_CHECK(out->size() == 4);

    double scale = (*out)[0].gamma.value() * m_eff.value();
    SRFM_CHECK_NEAR(scale, 2.5, 1e-7);
    SRFM_CHECK_NEAR((*out)[0].adjusted_value,  25.0, 1e-7);
    SRFM_CHECK_NEAR((*out)[1].adjusted_value,  50.0, 1e-7);
    SRFM_CHECK_NEAR((*out)[2].adjusted_value, -12.5, 1e-7);
    SRFM_CHECK(std::abs((*out)[3].adjusted_value) < 1e-15);

    // Round-trip via inverse_transform (m_eff=1 so we can invert cleanly)
    auto m1     = EffectiveMass::make(1.0).value();
    auto b05    = BetaVelocity::make(0.5).value();
    auto single = proc.process_one({200.0}, b05, m1);
    SRFM_HAS_VALUE(single);
    auto recovered = inverse_transform(single->adjusted_value, b05);
    SRFM_HAS_VALUE(recovered);
    SRFM_CHECK_NEAR(*recovered, 200.0, 1e-8);

    // Composed velocity then process stays finite and sub-luminal
    auto b7a = BetaVelocity::make(0.7).value();
    auto b7b = BetaVelocity::make(0.7).value();
    auto comp = compose_velocities(b7a, b7b);
    SRFM_HAS_VALUE(comp);
    SRFM_CHECK(std::abs(comp->value()) < BETA_MAX_SAFE);
    std::array<RawSignal, 1> sig100{{{100.0}}};
    auto r_comp = proc.process(sig100, comp.value(), m1);
    SRFM_HAS_VALUE(r_comp);
    SRFM_CHECK(std::isfinite((*r_comp)[0].adjusted_value));
    SRFM_CHECK((*r_comp)[0].adjusted_value >= 100.0); // γ ≥ 1

    // γ is monotonically increasing with |β|
    std::array<double, 7> betas_arr{0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99};
    double prev_gamma = 0.0;
    for (double bv : betas_arr) {
        auto bx = BetaVelocity::make(bv).value();
        double gx = lorentz_gamma(bx).value().value();
        SRFM_CHECK(gx > prev_gamma);
        prev_gamma = gx;
    }

    // Large batch: all outputs finite and γ ≥ 1
    auto b_max = BetaVelocity::make(0.9998).value();
    auto m_small = EffectiveMass::make(0.001).value();
    std::vector<RawSignal> big(1000);
    for (int i = 0; i < 1000; ++i) big[static_cast<std::size_t>(i)] = {(i - 500) * 0.1};
    auto big_out = proc.process(big, b_max, m_small);
    SRFM_HAS_VALUE(big_out);
    SRFM_CHECK(big_out->size() == 1000);
    for (const auto& s : *big_out) {
        SRFM_CHECK(std::isfinite(s.adjusted_value));
        SRFM_CHECK(s.gamma.value() >= 1.0);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("SRFM Momentum Signal Processor — Unit Tests\n");
    std::printf("============================================\n");

    SRFM_SUITE("BetaVelocity",            test_beta_velocity);
    SRFM_SUITE("EffectiveMass",           test_effective_mass);
    SRFM_SUITE("lorentz_gamma",           test_lorentz_gamma);
    SRFM_SUITE("apply_momentum_correction", test_momentum_correction);
    SRFM_SUITE("compose_velocities",      test_compose_velocities);
    SRFM_SUITE("inverse_transform",       test_inverse_transform);
    SRFM_SUITE("Processor::process",      test_processor_batch);
    SRFM_SUITE("Processor::process_one",  test_processor_one);
    SRFM_SUITE("Integration",             test_integration);

    return srfm_test::report();
}
