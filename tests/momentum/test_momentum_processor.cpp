/// @file tests/momentum/test_momentum_processor.cpp
/// @brief Tests for MomentumProcessor — AGT-06.

#include "srfm/momentum.hpp"
#include "srfm/constants.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

using namespace srfm::momentum;
using namespace srfm::constants;
using srfm::BetaVelocity;

// ─── process: basic ───────────────────────────────────────────────────────────

TEST(MomentumProcessorProcess, NewtonianLimitGammaEqualsOne) {
    // β = 0 → γ = 1 → adjusted = 1 × mass × raw
    MomentumSignal sig{.raw_value = 2.0, .beta = BetaVelocity{0.0}, .effective_mass = 1.0};
    auto result = MomentumProcessor::process(sig);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->gamma.value, 1.0, 1e-12);
    EXPECT_NEAR(result->adjusted_value, 2.0, 1e-12);
}

TEST(MomentumProcessorProcess, RelativisticAmplification) {
    // β = 0.6 → γ = 1/√(1−0.36) = 1/√0.64 = 1.25
    MomentumSignal sig{.raw_value = 1.0, .beta = BetaVelocity{0.6}, .effective_mass = 1.0};
    auto result = MomentumProcessor::process(sig);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->gamma.value, 1.25, 1e-8);
    EXPECT_NEAR(result->adjusted_value, 1.25, 1e-8);
}

TEST(MomentumProcessorProcess, EffectiveMassScalesOutput) {
    // adjusted = γ × m × raw; m = 3 → adjusted = 1 × 3 × 1 = 3
    MomentumSignal sig{.raw_value = 1.0, .beta = BetaVelocity{0.0}, .effective_mass = 3.0};
    auto result = MomentumProcessor::process(sig);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->adjusted_value, 3.0, 1e-12);
}

TEST(MomentumProcessorProcess, NegativeRawSignalPreservesSign) {
    MomentumSignal sig{.raw_value = -2.0, .beta = BetaVelocity{0.0}, .effective_mass = 1.0};
    auto result = MomentumProcessor::process(sig);
    ASSERT_TRUE(result.has_value());
    EXPECT_LT(result->adjusted_value, 0.0);
    EXPECT_NEAR(result->raw_value, -2.0, 1e-12);
}

TEST(MomentumProcessorProcess, ZeroRawSignalGivesZeroAdjusted) {
    MomentumSignal sig{.raw_value = 0.0, .beta = BetaVelocity{0.5}, .effective_mass = 5.0};
    auto result = MomentumProcessor::process(sig);
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->adjusted_value, 0.0);
}

TEST(MomentumProcessorProcess, ReturnsNulloptForNonpositiveMass) {
    MomentumSignal sig{.raw_value = 1.0, .beta = BetaVelocity{0.0}, .effective_mass = 0.0};
    EXPECT_FALSE(MomentumProcessor::process(sig).has_value());

    sig.effective_mass = -1.0;
    EXPECT_FALSE(MomentumProcessor::process(sig).has_value());
}

TEST(MomentumProcessorProcess, ReturnsNulloptForInvalidBeta) {
    MomentumSignal sig{.raw_value = 1.0, .beta = BetaVelocity{1.0}, .effective_mass = 1.0};
    EXPECT_FALSE(MomentumProcessor::process(sig).has_value());

    sig.beta = BetaVelocity{BETA_MAX_SAFE};
    EXPECT_FALSE(MomentumProcessor::process(sig).has_value());
}

TEST(MomentumProcessorProcess, BetaIsPreservedInResult) {
    MomentumSignal sig{.raw_value = 1.0, .beta = BetaVelocity{0.5}, .effective_mass = 1.0};
    auto result = MomentumProcessor::process(sig);
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->beta.value, 0.5);
}

TEST(MomentumProcessorProcess, GammaMonotoneInBeta) {
    // γ should increase with β
    double prev_gamma = 1.0;
    for (double beta_val : {0.0, 0.1, 0.3, 0.6, 0.9, 0.99}) {
        MomentumSignal sig{.raw_value = 1.0, .beta = BetaVelocity{beta_val}, .effective_mass = 1.0};
        auto result = MomentumProcessor::process(sig);
        ASSERT_TRUE(result.has_value());
        EXPECT_GE(result->gamma.value, prev_gamma - 1e-12);
        prev_gamma = result->gamma.value;
    }
}

// ─── relativistic_momentum ───────────────────────────────────────────────────

TEST(MomentumProcessorRelMomentum, NewtonianLimitReducesToClassical) {
    // p_rel(β=0) = 1 × m × v = m·v
    auto result = MomentumProcessor::relativistic_momentum(
        BetaVelocity{0.0}, 2.0, 3.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 6.0, 1e-12);
}

TEST(MomentumProcessorRelMomentum, ReturnsNulloptForZeroMass) {
    EXPECT_FALSE(MomentumProcessor::relativistic_momentum(
        BetaVelocity{0.0}, 0.0, 1.0).has_value());
}

TEST(MomentumProcessorRelMomentum, ReturnsNulloptForInvalidBeta) {
    EXPECT_FALSE(MomentumProcessor::relativistic_momentum(
        BetaVelocity{1.5}, 1.0, 1.0).has_value());
}

TEST(MomentumProcessorRelMomentum, ReturnsNulloptForNaNSpeed) {
    EXPECT_FALSE(MomentumProcessor::relativistic_momentum(
        BetaVelocity{0.0}, 1.0,
        std::numeric_limits<double>::quiet_NaN()).has_value());
}

TEST(MomentumProcessorRelMomentum, AlwaysNonnegative) {
    // Magnitude uses abs(speed)
    for (double speed : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
        auto result = MomentumProcessor::relativistic_momentum(
            BetaVelocity{0.5}, 1.0, speed);
        ASSERT_TRUE(result.has_value());
        EXPECT_GE(*result, 0.0);
    }
}

// ─── process_series ──────────────────────────────────────────────────────────

TEST(MomentumProcessorSeries, EmptyReturnsNullopt) {
    EXPECT_FALSE(MomentumProcessor::process_series({}).has_value());
}

TEST(MomentumProcessorSeries, LengthPreserved) {
    std::vector<MomentumSignal> sigs{
        {1.0, BetaVelocity{0.0}, 1.0},
        {2.0, BetaVelocity{0.5}, 1.0},
        {3.0, BetaVelocity{0.9}, 1.0},
    };
    auto result = MomentumProcessor::process_series(sigs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), sigs.size());
}

TEST(MomentumProcessorSeries, InvalidBetaFallsBackToRaw) {
    // Beta >= BETA_MAX_SAFE → γ = 1 fallback
    std::vector<MomentumSignal> sigs{
        {5.0, BetaVelocity{2.0}, 1.0},  // invalid beta
    };
    auto result = MomentumProcessor::process_series(sigs);
    ASSERT_TRUE(result.has_value());
    // Newtonian fallback: adjusted = raw_value = 5.0
    EXPECT_NEAR((*result)[0].adjusted_value, 5.0, 1e-12);
    EXPECT_NEAR((*result)[0].gamma.value, 1.0, 1e-12);
}

TEST(MomentumProcessorSeries, ValuesMatchProcessSingle) {
    std::vector<MomentumSignal> sigs{
        {1.0, BetaVelocity{0.6}, 1.0},
        {2.0, BetaVelocity{0.8}, 2.0},
    };
    auto series = MomentumProcessor::process_series(sigs);
    ASSERT_TRUE(series.has_value());
    for (std::size_t i = 0; i < sigs.size(); ++i) {
        auto single = MomentumProcessor::process(sigs[i]);
        ASSERT_TRUE(single.has_value());
        EXPECT_NEAR((*series)[i].adjusted_value, single->adjusted_value, 1e-12);
        EXPECT_NEAR((*series)[i].gamma.value,    single->gamma.value,    1e-12);
    }
}
