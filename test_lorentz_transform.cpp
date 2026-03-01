#include <gtest/gtest.h>
#include "../../src/lorentz/lorentz_transform.hpp"
#include "srfm/constants.hpp"
#include <cmath>
#include <limits>

using namespace srfm;
using namespace srfm::lorentz;
using namespace srfm::constants;

// ─── Gamma: Newtonian Limit ───────────────────────────────────────────────────

TEST(LorentzGamma, NewtonianLimit_BetaZero) {
    // At β = 0, γ = 1 exactly (no relativistic correction)
    auto result = LorentzTransform::gamma(BetaVelocity{0.0});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 1.0, FLOAT_EPSILON);
}

TEST(LorentzGamma, NewtonianLimit_SmallBeta) {
    // At β = 0.001, γ ≈ 1.0000005 — nearly no correction
    auto result = LorentzTransform::gamma(BetaVelocity{0.001});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 1.0, 1e-5);
}

TEST(LorentzGamma, BelowNewtonianThreshold_StillValid) {
    auto result = LorentzTransform::gamma(BetaVelocity{BETA_NEWTONIAN_THRESHOLD});
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value, 1.0);
    EXPECT_NEAR(result->value, 1.0 / std::sqrt(1.0 - 0.01), FLOAT_EPSILON);
}

// ─── Gamma: Relativistic Regime ──────────────────────────────────────────────

TEST(LorentzGamma, HighBeta_CorrectValue) {
    // At β = 0.6, γ = 1.25 exactly
    auto result = LorentzTransform::gamma(BetaVelocity{0.6});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 1.25, 1e-10);
}

TEST(LorentzGamma, VeryHighBeta_GrowsLarge) {
    // At β = 0.9999, γ ≈ 70.7
    auto result = LorentzTransform::gamma(BetaVelocity{0.9999});
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value, 70.0);
    EXPECT_LT(result->value, 1000.0);  // sanity upper bound
}

TEST(LorentzGamma, AlwaysGreaterThanOrEqualOne) {
    for (double b : {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999}) {
        auto result = LorentzTransform::gamma(BetaVelocity{b});
        ASSERT_TRUE(result.has_value()) << "Failed at beta=" << b;
        EXPECT_GE(result->value, 1.0) << "gamma < 1 at beta=" << b;
    }
}

// ─── Gamma: Invalid Inputs ────────────────────────────────────────────────────

TEST(LorentzGamma, BetaEqualOne_ReturnsNullopt) {
    auto result = LorentzTransform::gamma(BetaVelocity{1.0});
    EXPECT_FALSE(result.has_value());
}

TEST(LorentzGamma, BetaGreaterThanOne_ReturnsNullopt) {
    auto result = LorentzTransform::gamma(BetaVelocity{1.5});
    EXPECT_FALSE(result.has_value());
}

TEST(LorentzGamma, BetaNaN_ReturnsNullopt) {
    auto result = LorentzTransform::gamma(BetaVelocity{std::numeric_limits<double>::quiet_NaN()});
    EXPECT_FALSE(result.has_value());
}

TEST(LorentzGamma, BetaInfinity_ReturnsNullopt) {
    auto result = LorentzTransform::gamma(BetaVelocity{std::numeric_limits<double>::infinity()});
    EXPECT_FALSE(result.has_value());
}

// ─── Time Dilation ────────────────────────────────────────────────────────────

TEST(LorentzTimeDilation, ZeroBeta_NoDilation) {
    // At β = 0, dilated time = proper time
    auto result = LorentzTransform::dilateTime(100.0, BetaVelocity{0.0});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 100.0, FLOAT_EPSILON);
}

TEST(LorentzTimeDilation, HighBeta_DilatesTime) {
    // Fast market: signal time stretches, making it heavier
    auto result = LorentzTransform::dilateTime(100.0, BetaVelocity{0.6});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 125.0, 1e-8);  // γ = 1.25, so 100 * 1.25 = 125
}

TEST(LorentzTimeDilation, NegativeProperTime_ReturnsNullopt) {
    auto result = LorentzTransform::dilateTime(-1.0, BetaVelocity{0.5});
    EXPECT_FALSE(result.has_value());
}

TEST(LorentzTimeDilation, ZeroProperTime_ReturnsZero) {
    auto result = LorentzTransform::dilateTime(0.0, BetaVelocity{0.5});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.0, FLOAT_EPSILON);
}

TEST(LorentzTimeDilation, AlwaysGreaterThanProperTime) {
    double proper = 50.0;
    for (double b : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        auto result = LorentzTransform::dilateTime(proper, BetaVelocity{b});
        ASSERT_TRUE(result.has_value());
        EXPECT_GE(*result, proper) << "Dilation should not compress time, beta=" << b;
    }
}

// ─── Momentum Correction ─────────────────────────────────────────────────────

TEST(LorentzMomentum, ZeroBeta_MassScaledOnly) {
    // At β = 0, γ = 1, so p_rel = 1 * m * v = m * raw
    auto result = LorentzTransform::applyMomentumCorrection(2.0, BetaVelocity{0.0}, 3.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->adjusted_value, 6.0, FLOAT_EPSILON);
    EXPECT_NEAR(result->gamma.value, 1.0, FLOAT_EPSILON);
    EXPECT_NEAR(result->raw_value, 2.0, FLOAT_EPSILON);
}

TEST(LorentzMomentum, HighBeta_AmplifiedMomentum) {
    // At β = 0.6, γ = 1.25
    // p_rel = 1.25 * 1.0 * 1.0 = 1.25
    auto result = LorentzTransform::applyMomentumCorrection(1.0, BetaVelocity{0.6}, 1.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->adjusted_value, 1.25, 1e-8);
}

TEST(LorentzMomentum, ZeroMass_ReturnsNullopt) {
    auto result = LorentzTransform::applyMomentumCorrection(1.0, BetaVelocity{0.5}, 0.0);
    EXPECT_FALSE(result.has_value());
}

TEST(LorentzMomentum, NegativeMass_ReturnsNullopt) {
    auto result = LorentzTransform::applyMomentumCorrection(1.0, BetaVelocity{0.5}, -1.0);
    EXPECT_FALSE(result.has_value());
}

// ─── Velocity Composition ─────────────────────────────────────────────────────

TEST(LorentzVelocityComposition, TwoHalfSpeedLights_SubLuminal) {
    // 0.5c + 0.5c in special relativity = 0.8c, not 1.0c
    auto result = LorentzTransform::composeVelocities(
        BetaVelocity{0.5}, BetaVelocity{0.5});
    EXPECT_NEAR(result.value, 0.8, 1e-10);
}

TEST(LorentzVelocityComposition, ZeroPlusAny_EqualsAny) {
    auto result = LorentzTransform::composeVelocities(
        BetaVelocity{0.0}, BetaVelocity{0.7});
    EXPECT_NEAR(result.value, 0.7, FLOAT_EPSILON);
}

TEST(LorentzVelocityComposition, AlwaysSubLuminal) {
    for (double b1 : {0.3, 0.5, 0.7, 0.9}) {
        for (double b2 : {0.3, 0.5, 0.7, 0.9}) {
            auto result = LorentzTransform::composeVelocities(
                BetaVelocity{b1}, BetaVelocity{b2});
            EXPECT_LT(result.value, 1.0)
                << "Superluminal composition at b1=" << b1 << " b2=" << b2;
        }
    }
}

TEST(LorentzVelocityComposition, Commutative) {
    auto r1 = LorentzTransform::composeVelocities(BetaVelocity{0.3}, BetaVelocity{0.6});
    auto r2 = LorentzTransform::composeVelocities(BetaVelocity{0.6}, BetaVelocity{0.3});
    EXPECT_NEAR(r1.value, r2.value, FLOAT_EPSILON);
}

// ─── Inverse Transform ────────────────────────────────────────────────────────

TEST(LorentzInverse, InverseRecoversPropperValue) {
    double proper = 42.0;
    BetaVelocity beta{0.6};

    auto dilated = LorentzTransform::dilateTime(proper, beta);
    ASSERT_TRUE(dilated.has_value());

    auto recovered = LorentzTransform::inverseTransform(*dilated, beta);
    ASSERT_TRUE(recovered.has_value());
    EXPECT_NEAR(*recovered, proper, 1e-9);
}

TEST(LorentzInverse, InvalidBeta_ReturnsNullopt) {
    auto result = LorentzTransform::inverseTransform(100.0, BetaVelocity{1.5});
    EXPECT_FALSE(result.has_value());
}

// ─── Mathematical Identity Tests ──────────────────────────────────────────────

TEST(LorentzIdentity, GammaSquaredIdentity) {
    // γ² = 1 / (1 - β²) — verify numerically
    for (double b : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        auto g = LorentzTransform::gamma(BetaVelocity{b});
        ASSERT_TRUE(g.has_value());
        double expected = 1.0 / (1.0 - b * b);
        EXPECT_NEAR(g->value * g->value, expected, 1e-8) << "At beta=" << b;
    }
}

TEST(LorentzIdentity, GammaBetaProduct) {
    // γβ = β / sqrt(1 - β²) — used in four-velocity calculations
    for (double b : {0.2, 0.4, 0.6, 0.8}) {
        auto g = LorentzTransform::gamma(BetaVelocity{b});
        ASSERT_TRUE(g.has_value());
        double gamma_beta = g->value * b;
        double expected   = b / std::sqrt(1.0 - b * b);
        EXPECT_NEAR(gamma_beta, expected, 1e-10) << "At beta=" << b;
    }
}
