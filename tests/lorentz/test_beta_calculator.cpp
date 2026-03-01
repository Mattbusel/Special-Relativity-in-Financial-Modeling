#include <gtest/gtest.h>
#include "../../src/lorentz/beta_calculator.hpp"
#include "srfm/constants.hpp"
#include <cmath>
#include <limits>
#include <vector>

using namespace srfm;
using namespace srfm::lorentz;
using namespace srfm::constants;

// ─── fromPriceVelocity ────────────────────────────────────────────────────────

TEST(BetaCalc_FromPriceVelocity, HalfMaxVelocity_BetaIsHalf) {
    auto result = BetaCalculator::fromPriceVelocity(5.0, 10.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 0.5, FLOAT_EPSILON);
}

TEST(BetaCalc_FromPriceVelocity, EqualToMaxVelocity_ClampedBelowOne) {
    // v == max → raw β = 1.0, must be clamped to BETA_MAX_SAFE
    auto result = BetaCalculator::fromPriceVelocity(10.0, 10.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_LT(result->value, 1.0);
    EXPECT_LE(result->value, BETA_MAX_SAFE);
}

TEST(BetaCalc_FromPriceVelocity, ExceedsMaxVelocity_ClampedToSafe) {
    // v > max → raw β > 1, clamp to BETA_MAX_SAFE
    auto result = BetaCalculator::fromPriceVelocity(20.0, 10.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_LE(result->value, BETA_MAX_SAFE);
}

TEST(BetaCalc_FromPriceVelocity, ZeroVelocity_BetaIsZero) {
    auto result = BetaCalculator::fromPriceVelocity(0.0, 10.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 0.0, FLOAT_EPSILON);
}

TEST(BetaCalc_FromPriceVelocity, NegativeVelocity_BetaIsPositive) {
    // Speed is non-negative; direction is separate
    auto result = BetaCalculator::fromPriceVelocity(-5.0, 10.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 0.5, FLOAT_EPSILON);
}

TEST(BetaCalc_FromPriceVelocity, ZeroMaxVelocity_ReturnsNullopt) {
    EXPECT_FALSE(BetaCalculator::fromPriceVelocity(5.0, 0.0).has_value());
}

TEST(BetaCalc_FromPriceVelocity, NegativeMaxVelocity_ReturnsNullopt) {
    EXPECT_FALSE(BetaCalculator::fromPriceVelocity(5.0, -10.0).has_value());
}

TEST(BetaCalc_FromPriceVelocity, NaNVelocity_ReturnsNullopt) {
    EXPECT_FALSE(BetaCalculator::fromPriceVelocity(
        std::numeric_limits<double>::quiet_NaN(), 10.0).has_value());
}

TEST(BetaCalc_FromPriceVelocity, InfiniteVelocity_ClampedToSafe) {
    // +∞ velocity → β clamped to BETA_MAX_SAFE
    auto result = BetaCalculator::fromPriceVelocity(
        std::numeric_limits<double>::infinity(), 10.0);
    // Either returns nullopt or clamped value — both are acceptable
    if (result.has_value()) {
        EXPECT_LE(result->value, BETA_MAX_SAFE);
    }
}

TEST(BetaCalc_FromPriceVelocity, ResultAlwaysInValidRange) {
    for (double v : {0.0, 1.0, 5.0, 10.0, 100.0}) {
        auto result = BetaCalculator::fromPriceVelocity(v, 10.0);
        if (result.has_value()) {
            EXPECT_GE(result->value, 0.0);
            EXPECT_LT(result->value, 1.0);
        }
    }
}

// ─── fromReturn ───────────────────────────────────────────────────────────────

TEST(BetaCalc_FromReturn, HalfMaxReturn_BetaIsHalf) {
    auto result = BetaCalculator::fromReturn(0.02, 0.04);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 0.5, FLOAT_EPSILON);
}

TEST(BetaCalc_FromReturn, NegativeReturn_SameAsMagnitude) {
    auto pos = BetaCalculator::fromReturn(0.03, 0.06);
    auto neg = BetaCalculator::fromReturn(-0.03, 0.06);
    ASSERT_TRUE(pos.has_value());
    ASSERT_TRUE(neg.has_value());
    EXPECT_NEAR(pos->value, neg->value, FLOAT_EPSILON);
}

TEST(BetaCalc_FromReturn, ZeroReturn_BetaIsZero) {
    auto result = BetaCalculator::fromReturn(0.0, 0.05);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 0.0, FLOAT_EPSILON);
}

TEST(BetaCalc_FromReturn, ZeroMaxReturn_ReturnsNullopt) {
    EXPECT_FALSE(BetaCalculator::fromReturn(0.01, 0.0).has_value());
}

TEST(BetaCalc_FromReturn, ReturnExceedsMax_ClampedToSafe) {
    auto result = BetaCalculator::fromReturn(1.0, 0.05);
    if (result.has_value()) {
        EXPECT_LE(result->value, BETA_MAX_SAFE);
    }
}

// ─── meanAbsVelocity ──────────────────────────────────────────────────────────

TEST(BetaCalc_MeanAbsVelocity, ConstantSeries_ZeroVelocity) {
    std::vector<double> prices = {100.0, 100.0, 100.0, 100.0};
    auto result = BetaCalculator::meanAbsVelocity(prices, 1.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.0, FLOAT_EPSILON);
}

TEST(BetaCalc_MeanAbsVelocity, LinearRise_ConstantVelocity) {
    // p = {0, 1, 2, 3}, dt = 1 → dP/dt = 1 everywhere
    std::vector<double> prices = {0.0, 1.0, 2.0, 3.0};
    auto result = BetaCalculator::meanAbsVelocity(prices, 1.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 1.0, 1e-10);
}

TEST(BetaCalc_MeanAbsVelocity, TwoPrices_SingleStep) {
    // Only two points → forward difference: v = (p₁ − p₀) / dt
    std::vector<double> prices = {100.0, 105.0};
    auto result = BetaCalculator::meanAbsVelocity(prices, 1.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 5.0, 1e-10);
}

TEST(BetaCalc_MeanAbsVelocity, SmallerTimeDelta_HigherVelocity) {
    // Same price step but smaller dt → larger dP/dt
    std::vector<double> prices = {0.0, 10.0};
    auto v1 = BetaCalculator::meanAbsVelocity(prices, 1.0);
    auto v2 = BetaCalculator::meanAbsVelocity(prices, 0.5);
    ASSERT_TRUE(v1.has_value());
    ASSERT_TRUE(v2.has_value());
    EXPECT_NEAR(*v2, 2.0 * (*v1), 1e-10);
}

TEST(BetaCalc_MeanAbsVelocity, SinglePrice_ReturnsNullopt) {
    std::vector<double> prices = {100.0};
    EXPECT_FALSE(BetaCalculator::meanAbsVelocity(prices, 1.0).has_value());
}

TEST(BetaCalc_MeanAbsVelocity, EmptySeries_ReturnsNullopt) {
    std::vector<double> prices;
    EXPECT_FALSE(BetaCalculator::meanAbsVelocity(prices, 1.0).has_value());
}

TEST(BetaCalc_MeanAbsVelocity, ZeroTimeDelta_ReturnsNullopt) {
    std::vector<double> prices = {1.0, 2.0, 3.0};
    EXPECT_FALSE(BetaCalculator::meanAbsVelocity(prices, 0.0).has_value());
}

TEST(BetaCalc_MeanAbsVelocity, NegativeTimeDelta_ReturnsNullopt) {
    std::vector<double> prices = {1.0, 2.0, 3.0};
    EXPECT_FALSE(BetaCalculator::meanAbsVelocity(prices, -1.0).has_value());
}

TEST(BetaCalc_MeanAbsVelocity, AlwaysNonNegative) {
    std::vector<double> prices = {5.0, 3.0, 7.0, 2.0, 9.0};
    auto result = BetaCalculator::meanAbsVelocity(prices, 1.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(*result, 0.0);
}

// ─── fromRollingWindow ────────────────────────────────────────────────────────

TEST(BetaCalc_RollingWindow, WholeSeriesUsed_MatchesMeanAbsVelocity) {
    std::vector<double> prices = {10.0, 11.0, 12.0, 13.0};
    double max_v = 10.0, dt = 1.0;
    std::size_t window = prices.size();

    auto beta   = BetaCalculator::fromRollingWindow(prices, window, max_v, dt);
    auto vel    = BetaCalculator::meanAbsVelocity(prices, dt);
    ASSERT_TRUE(beta.has_value());
    ASSERT_TRUE(vel.has_value());
    EXPECT_NEAR(beta->value, std::min(*vel / max_v, BETA_MAX_SAFE), 1e-10);
}

TEST(BetaCalc_RollingWindow, WindowTooLarge_ReturnsNullopt) {
    std::vector<double> prices = {1.0, 2.0, 3.0};
    EXPECT_FALSE(BetaCalculator::fromRollingWindow(prices, 10, 5.0, 1.0).has_value());
}

TEST(BetaCalc_RollingWindow, WindowOfOne_ReturnsNullopt) {
    std::vector<double> prices = {1.0, 2.0, 3.0};
    EXPECT_FALSE(BetaCalculator::fromRollingWindow(prices, 1, 5.0, 1.0).has_value());
}

TEST(BetaCalc_RollingWindow, ZeroMaxVelocity_ReturnsNullopt) {
    std::vector<double> prices = {1.0, 2.0, 3.0};
    EXPECT_FALSE(BetaCalculator::fromRollingWindow(prices, 3, 0.0, 1.0).has_value());
}

TEST(BetaCalc_RollingWindow, ResultInValidRange) {
    std::vector<double> prices = {100.0, 101.0, 103.0, 102.0, 105.0};
    auto result = BetaCalculator::fromRollingWindow(prices, 4, 20.0, 1.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result->value, 0.0);
    EXPECT_LT(result->value, 1.0);
}

// ─── isNewtonian / isRelativistic / isValid ───────────────────────────────────

TEST(BetaCalc_Classification, ZeroBeta_IsNewtonian) {
    EXPECT_TRUE(BetaCalculator::isNewtonian(BetaVelocity{0.0}));
}

TEST(BetaCalc_Classification, JustBelowThreshold_IsNewtonian) {
    EXPECT_TRUE(BetaCalculator::isNewtonian(
        BetaVelocity{BETA_NEWTONIAN_THRESHOLD - 1e-12}));
}

TEST(BetaCalc_Classification, AtThreshold_IsRelativistic) {
    EXPECT_TRUE(BetaCalculator::isRelativistic(
        BetaVelocity{BETA_NEWTONIAN_THRESHOLD}));
}

TEST(BetaCalc_Classification, HighBeta_IsRelativistic) {
    EXPECT_TRUE(BetaCalculator::isRelativistic(BetaVelocity{0.9}));
    EXPECT_FALSE(BetaCalculator::isNewtonian(BetaVelocity{0.9}));
}

TEST(BetaCalc_Classification, SafeBeta_IsValid) {
    EXPECT_TRUE(BetaCalculator::isValid(BetaVelocity{0.5}));
    EXPECT_TRUE(BetaCalculator::isValid(BetaVelocity{0.0}));
}

TEST(BetaCalc_Classification, BetaMaxSafe_IsInvalid) {
    EXPECT_FALSE(BetaCalculator::isValid(BetaVelocity{BETA_MAX_SAFE}));
}

TEST(BetaCalc_Classification, AboveOne_IsInvalid) {
    EXPECT_FALSE(BetaCalculator::isValid(BetaVelocity{1.5}));
}

TEST(BetaCalc_Classification, NewtonianAndRelativistic_Exclusive) {
    for (double b : {0.0, 0.05, 0.1, 0.5, 0.9}) {
        bool n = BetaCalculator::isNewtonian(BetaVelocity{b});
        bool r = BetaCalculator::isRelativistic(BetaVelocity{b});
        EXPECT_NE(n, r) << "Newtonian and Relativistic not exclusive at beta=" << b;
    }
}

// ─── clamp ────────────────────────────────────────────────────────────────────

TEST(BetaCalc_Clamp, WithinRange_Unchanged) {
    auto result = BetaCalculator::clamp(0.5);
    EXPECT_NEAR(result.value, 0.5, FLOAT_EPSILON);
}

TEST(BetaCalc_Clamp, ExceedsOne_ClampedToMaxSafe) {
    auto result = BetaCalculator::clamp(2.0);
    EXPECT_NEAR(result.value, BETA_MAX_SAFE, FLOAT_EPSILON);
}

TEST(BetaCalc_Clamp, BelowNegativeOne_ClampedToNegativeMaxSafe) {
    auto result = BetaCalculator::clamp(-2.0);
    EXPECT_NEAR(result.value, -BETA_MAX_SAFE, FLOAT_EPSILON);
}

TEST(BetaCalc_Clamp, Zero_StaysZero) {
    auto result = BetaCalculator::clamp(0.0);
    EXPECT_NEAR(result.value, 0.0, FLOAT_EPSILON);
}

TEST(BetaCalc_Clamp, AlwaysProducesValidBeta) {
    for (double raw : {-10.0, -1.5, -0.5, 0.0, 0.5, 1.5, 10.0}) {
        auto result = BetaCalculator::clamp(raw);
        EXPECT_TRUE(BetaCalculator::isValid(result))
            << "clamp(" << raw << ") produced invalid beta=" << result.value;
    }
}

// ─── kineticEnergy ────────────────────────────────────────────────────────────

TEST(BetaCalc_KineticEnergy, ZeroBeta_ZeroKineticEnergy) {
    // E_k = (γ − 1)·m·c² = (1−1)·m·c² = 0
    auto result = BetaCalculator::kineticEnergy(BetaVelocity{0.0}, 5.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.0, FLOAT_EPSILON);
}

TEST(BetaCalc_KineticEnergy, Beta06_CorrectValue) {
    // γ(0.6) = 1.25 → E_k = (1.25−1)·m·c² = 0.25·m (c=1)
    auto result = BetaCalculator::kineticEnergy(BetaVelocity{0.6}, 2.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.5, 1e-8); // 0.25 × 2.0
}

TEST(BetaCalc_KineticEnergy, AlwaysNonNegative) {
    for (double b : {0.0, 0.1, 0.5, 0.9}) {
        auto result = BetaCalculator::kineticEnergy(BetaVelocity{b}, 3.0);
        ASSERT_TRUE(result.has_value());
        EXPECT_GE(*result, 0.0) << "Negative kinetic energy at beta=" << b;
    }
}

TEST(BetaCalc_KineticEnergy, ZeroMass_ReturnsNullopt) {
    EXPECT_FALSE(BetaCalculator::kineticEnergy(BetaVelocity{0.5}, 0.0).has_value());
}

TEST(BetaCalc_KineticEnergy, InvalidBeta_ReturnsNullopt) {
    EXPECT_FALSE(BetaCalculator::kineticEnergy(BetaVelocity{1.5}, 1.0).has_value());
}

TEST(BetaCalc_KineticEnergy, MonotonicallyIncreasingWithBeta) {
    double mass = 1.0;
    double prev = -1.0;
    for (double b : {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}) {
        auto result = BetaCalculator::kineticEnergy(BetaVelocity{b}, mass);
        ASSERT_TRUE(result.has_value());
        EXPECT_GE(*result, prev)
            << "Kinetic energy decreased at beta=" << b;
        prev = *result;
    }
}

// ─── dopplerFactor ────────────────────────────────────────────────────────────

TEST(BetaCalc_Doppler, ZeroBeta_DopplerIsOne) {
    // No relative motion → no Doppler shift
    auto result = BetaCalculator::dopplerFactor(BetaVelocity{0.0});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 1.0, FLOAT_EPSILON);
}

TEST(BetaCalc_Doppler, PositiveBeta_DopplerAboveOne) {
    // Market approaching: blue-shift → D > 1
    auto result = BetaCalculator::dopplerFactor(BetaVelocity{0.6});
    ASSERT_TRUE(result.has_value());
    // D = √((1.6)/(0.4)) = √4 = 2
    EXPECT_NEAR(*result, 2.0, 1e-8);
}

TEST(BetaCalc_Doppler, NegativeBeta_DopplerBelowOne) {
    // Market receding: red-shift → D < 1
    auto result = BetaCalculator::dopplerFactor(BetaVelocity{-0.6});
    ASSERT_TRUE(result.has_value());
    // D = √((1−0.6)/(1+0.6)) = √(0.4/1.6) = 0.5
    EXPECT_NEAR(*result, 0.5, 1e-8);
}

TEST(BetaCalc_Doppler, AlwaysPositive) {
    for (double b : {-0.9, -0.5, 0.0, 0.5, 0.9}) {
        auto result = BetaCalculator::dopplerFactor(BetaVelocity{b});
        ASSERT_TRUE(result.has_value());
        EXPECT_GT(*result, 0.0) << "Non-positive Doppler at beta=" << b;
    }
}

TEST(BetaCalc_Doppler, InvalidBeta_ReturnsNullopt) {
    EXPECT_FALSE(BetaCalculator::dopplerFactor(BetaVelocity{1.5}).has_value());
}

TEST(BetaCalc_Doppler, ReciprocityPlusMinusBeta) {
    // D(β) × D(−β) = 1 (approaching and receding are reciprocal)
    BetaVelocity beta{0.7};
    auto dpos = BetaCalculator::dopplerFactor(beta);
    auto dneg = BetaCalculator::dopplerFactor(BetaVelocity{-0.7});
    ASSERT_TRUE(dpos.has_value());
    ASSERT_TRUE(dneg.has_value());
    EXPECT_NEAR(*dpos * *dneg, 1.0, 1e-10);
}
