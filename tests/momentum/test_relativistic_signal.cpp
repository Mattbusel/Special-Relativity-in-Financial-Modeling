/// @file tests/momentum/test_relativistic_signal.cpp
/// @brief Tests for RelativisticMomentum struct and signal properties — AGT-06.

#include "srfm/momentum.hpp"
#include "srfm/constants.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace srfm::momentum;
using namespace srfm::constants;
using srfm::BetaVelocity;

// ─── RelativisticMomentum struct ─────────────────────────────────────────────

TEST(RelativisticMomentumStruct, DefaultConstructible) {
    RelativisticMomentum rm{};
    EXPECT_DOUBLE_EQ(rm.raw_value, 0.0);
    EXPECT_DOUBLE_EQ(rm.adjusted_value, 0.0);
    EXPECT_DOUBLE_EQ(rm.gamma.value, 0.0);
    EXPECT_DOUBLE_EQ(rm.beta.value, 0.0);
}

TEST(RelativisticMomentumStruct, FieldsSetCorrectly) {
    RelativisticMomentum rm{
        .raw_value      = 3.0,
        .adjusted_value = 6.0,
        .gamma          = LorentzFactor{2.0},
        .beta           = BetaVelocity{0.866},
    };
    EXPECT_DOUBLE_EQ(rm.raw_value, 3.0);
    EXPECT_DOUBLE_EQ(rm.adjusted_value, 6.0);
    EXPECT_DOUBLE_EQ(rm.gamma.value, 2.0);
    EXPECT_NEAR(rm.beta.value, 0.866, 1e-12);
}

// ─── Signal invariants ────────────────────────────────────────────────────────

TEST(RelativisticMomentumSignal, GammaIsAlwaysAtLeastOne) {
    for (double beta_val : {0.0, 0.1, 0.5, 0.8, 0.99, 0.9998}) {
        MomentumSignal sig{.raw_value = 1.0, .beta = BetaVelocity{beta_val}, .effective_mass = 1.0};
        auto result = MomentumProcessor::process(sig);
        ASSERT_TRUE(result.has_value()) << "beta=" << beta_val;
        EXPECT_GE(result->gamma.value, 1.0) << "beta=" << beta_val;
    }
}

TEST(RelativisticMomentumSignal, AdjustedMagnitudeGeRawMagnitude) {
    // γ ≥ 1 and mass ≥ 1 → |adjusted| ≥ |raw| when mass = 1
    for (double beta_val : {0.0, 0.5, 0.9, 0.99}) {
        MomentumSignal sig{.raw_value = 2.0, .beta = BetaVelocity{beta_val}, .effective_mass = 1.0};
        auto result = MomentumProcessor::process(sig);
        ASSERT_TRUE(result.has_value());
        EXPECT_GE(std::abs(result->adjusted_value), std::abs(result->raw_value));
    }
}

TEST(RelativisticMomentumSignal, SignOfAdjustedMatchesRaw) {
    for (double raw : {-10.0, -1.0, 0.0, 1.0, 10.0}) {
        MomentumSignal sig{.raw_value = raw, .beta = BetaVelocity{0.5}, .effective_mass = 1.0};
        auto result = MomentumProcessor::process(sig);
        ASSERT_TRUE(result.has_value());
        if (raw < 0.0) {
            EXPECT_LT(result->adjusted_value, 0.0);
        } else if (raw > 0.0) {
            EXPECT_GT(result->adjusted_value, 0.0);
        } else {
            EXPECT_DOUBLE_EQ(result->adjusted_value, 0.0);
        }
    }
}

TEST(RelativisticMomentumSignal, GammaScalesWithBeta) {
    // Higher β → higher γ → higher |adjusted|
    MomentumSignal sig_low{.raw_value = 1.0, .beta = BetaVelocity{0.1}, .effective_mass = 1.0};
    MomentumSignal sig_high{.raw_value = 1.0, .beta = BetaVelocity{0.9}, .effective_mass = 1.0};
    auto low  = MomentumProcessor::process(sig_low);
    auto high = MomentumProcessor::process(sig_high);
    ASSERT_TRUE(low.has_value());
    ASSERT_TRUE(high.has_value());
    EXPECT_GT(high->gamma.value, low->gamma.value);
    EXPECT_GT(high->adjusted_value, low->adjusted_value);
}

TEST(RelativisticMomentumSignal, Beta06GivesGamma125) {
    // Well-known: β = 0.6, γ = 1.25
    MomentumSignal sig{.raw_value = 4.0, .beta = BetaVelocity{0.6}, .effective_mass = 1.0};
    auto result = MomentumProcessor::process(sig);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->gamma.value, 1.25, 1e-8);
    EXPECT_NEAR(result->adjusted_value, 5.0, 1e-8);
}

TEST(RelativisticMomentumSignal, HighBetaGivesLargeGamma) {
    // β = 0.99 → γ = 1/√(1−0.9801) = 1/√0.0199 ≈ 7.089
    MomentumSignal sig{.raw_value = 1.0, .beta = BetaVelocity{0.99}, .effective_mass = 1.0};
    auto result = MomentumProcessor::process(sig);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->gamma.value, 7.0);
}

// ─── Process series invariants ────────────────────────────────────────────────

TEST(RelativisticMomentumSeries, OutputLengthMatchesInput) {
    std::vector<MomentumSignal> sigs(50, {1.0, BetaVelocity{0.5}, 1.0});
    auto result = MomentumProcessor::process_series(sigs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 50u);
}

TEST(RelativisticMomentumSeries, SingleElementSeries) {
    std::vector<MomentumSignal> sigs{{1.0, BetaVelocity{0.0}, 1.0}};
    auto result = MomentumProcessor::process_series(sigs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 1u);
    EXPECT_NEAR((*result)[0].gamma.value, 1.0, 1e-12);
}

TEST(RelativisticMomentumSignal, RawValuePreservedThroughSeries) {
    std::vector<double> raw_vals = {-3.0, 0.0, 1.5, 7.0, -0.5};
    std::vector<MomentumSignal> sigs;
    for (double v : raw_vals) {
        sigs.push_back({v, BetaVelocity{0.5}, 1.0});
    }
    auto result = MomentumProcessor::process_series(sigs);
    ASSERT_TRUE(result.has_value());
    for (std::size_t i = 0; i < raw_vals.size(); ++i) {
        EXPECT_NEAR((*result)[i].raw_value, raw_vals[i], 1e-12);
    }
}

TEST(RelativisticMomentumSignal, Beta866GivesGammaTwo) {
    // β = √3/2 ≈ 0.866 → γ = 2.0
    const double beta_val = std::sqrt(3.0) / 2.0;
    MomentumSignal sig{.raw_value = 1.0, .beta = BetaVelocity{beta_val}, .effective_mass = 1.0};
    auto result = MomentumProcessor::process(sig);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->gamma.value, 2.0, 1e-6);
    EXPECT_NEAR(result->adjusted_value, 2.0, 1e-6);
}

TEST(RelativisticMomentumSignal, LargeMassScalesFactor) {
    const double mass = 100.0;
    MomentumSignal sig{.raw_value = 1.0, .beta = BetaVelocity{0.6}, .effective_mass = mass};
    auto result = MomentumProcessor::process(sig);
    ASSERT_TRUE(result.has_value());
    // adjusted = γ × mass × raw = 1.25 × 100 × 1 = 125
    EXPECT_NEAR(result->adjusted_value, 125.0, 1e-6);
}
