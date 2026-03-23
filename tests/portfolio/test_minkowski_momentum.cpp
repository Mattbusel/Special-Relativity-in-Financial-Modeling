/**
 * @file  tests/portfolio/test_minkowski_momentum.cpp
 * @brief GTest unit tests for srfm::minkowski_momentum — 20+ tests.
 *
 * Round 6 implementation.
 */

#include "srfm/minkowski_momentum.hpp"

#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

using namespace srfm::minkowski_momentum;

// ─── FourMomentum construction ────────────────────────────────────────────────

TEST(FourMomentumTest, FieldsStoredCorrectly) {
    FourMomentum p{1.0, 0.3, 0.2, 0.1};
    EXPECT_DOUBLE_EQ(p.energy, 1.0);
    EXPECT_DOUBLE_EQ(p.px,     0.3);
    EXPECT_DOUBLE_EQ(p.py,     0.2);
    EXPECT_DOUBLE_EQ(p.pz,     0.1);
}

TEST(FourMomentumTest, ZeroFourMomentum) {
    FourMomentum p{0.0, 0.0, 0.0, 0.0};
    EXPECT_DOUBLE_EQ(MinkowskiMomentum::invariant_mass_sq(p), 0.0);
}

// ─── MinkowskiMomentum::invariant_mass_sq ────────────────────────────────────

TEST(InvariantMassSqTest, KnownValue) {
    // m² = 1² - 0.3² - 0.4² - 0.5² = 1 - 0.09 - 0.16 - 0.25 = 0.50
    FourMomentum p{1.0, 0.3, 0.4, 0.5};
    const double m2 = MinkowskiMomentum::invariant_mass_sq(p);
    EXPECT_NEAR(m2, 0.50, 1e-12);
}

TEST(InvariantMassSqTest, TimeLike) {
    // E > |p| => m2 > 0 (well-diversified portfolio)
    FourMomentum p{2.0, 0.5, 0.5, 0.5};
    EXPECT_GT(MinkowskiMomentum::invariant_mass_sq(p), 0.0);
}

TEST(InvariantMassSqTest, SpaceLike) {
    // |p| > E => m2 < 0 (over-leveraged / concentrated)
    FourMomentum p{0.5, 2.0, 0.0, 0.0};
    EXPECT_LT(MinkowskiMomentum::invariant_mass_sq(p), 0.0);
}

TEST(InvariantMassSqTest, LightLike) {
    // m² = 0 when E² = p_x² (null four-momentum)
    FourMomentum p{1.0, 1.0, 0.0, 0.0};
    EXPECT_NEAR(MinkowskiMomentum::invariant_mass_sq(p), 0.0, 1e-12);
}

TEST(InvariantMassSqTest, LorentzInvariance) {
    // Verify that the invariant_mass_sq formula gives the same value for
    // a boosted frame. For a pure equity boost along x with beta=0.5:
    //   gamma = 1/sqrt(1-0.25) = 1/sqrt(0.75)
    //   E'  = gamma*(E - beta*px)
    //   px' = gamma*(px - beta*E)
    // m² should be unchanged.
    const double beta = 0.5;
    const double gamma = 1.0 / std::sqrt(1.0 - beta * beta);
    FourMomentum p{1.0, 0.3, 0.2, 0.1};
    FourMomentum boosted{
        gamma * (p.energy - beta * p.px),
        gamma * (p.px     - beta * p.energy),
        p.py,
        p.pz
    };
    EXPECT_NEAR(MinkowskiMomentum::invariant_mass_sq(p),
                MinkowskiMomentum::invariant_mass_sq(boosted), 1e-10);
}

// ─── MinkowskiMomentum::invariant_mass ───────────────────────────────────────

TEST(InvariantMassTest, PositiveForTimeLike) {
    FourMomentum p{2.0, 0.5, 0.5, 0.5};
    auto m = MinkowskiMomentum::invariant_mass(p);
    ASSERT_TRUE(m.has_value());
    EXPECT_GT(*m, 0.0);
}

TEST(InvariantMassTest, NegativeForSpaceLike) {
    FourMomentum p{0.5, 2.0, 0.0, 0.0};
    auto m = MinkowskiMomentum::invariant_mass(p);
    ASSERT_TRUE(m.has_value());
    EXPECT_LT(*m, 0.0);
}

TEST(InvariantMassTest, NulloptForNaN) {
    FourMomentum p{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0};
    auto m = MinkowskiMomentum::invariant_mass(p);
    EXPECT_FALSE(m.has_value());
}

TEST(InvariantMassTest, MagnitudeCorrect) {
    // m² = 1² - 0² - 0² - 0² = 1 => m = 1
    FourMomentum p{1.0, 0.0, 0.0, 0.0};
    auto m = MinkowskiMomentum::invariant_mass(p);
    ASSERT_TRUE(m.has_value());
    EXPECT_NEAR(*m, 1.0, 1e-12);
}

// ─── MinkowskiMomentum::rapidity ─────────────────────────────────────────────

TEST(RapidityTest, ZeroForZeroPx) {
    FourMomentum p{1.0, 0.0, 0.0, 0.0};
    auto y = MinkowskiMomentum::rapidity(p);
    ASSERT_TRUE(y.has_value());
    EXPECT_NEAR(*y, 0.0, 1e-12);
}

TEST(RapidityTest, PositiveForPositivePx) {
    FourMomentum p{2.0, 0.5, 0.0, 0.0};
    auto y = MinkowskiMomentum::rapidity(p);
    ASSERT_TRUE(y.has_value());
    EXPECT_GT(*y, 0.0);
}

TEST(RapidityTest, KnownValue) {
    // y = 0.5 * ln((E+px)/(E-px)) = 0.5 * ln(3/1) = 0.5 * ln(3)
    FourMomentum p{2.0, 1.0, 0.0, 0.0};
    auto y = MinkowskiMomentum::rapidity(p);
    ASSERT_TRUE(y.has_value());
    EXPECT_NEAR(*y, 0.5 * std::log(3.0), 1e-12);
}

TEST(RapidityTest, NulloptWhenPxEqualsE) {
    // E == p_x => (E - p_x) == 0 => undefined
    FourMomentum p{1.0, 1.0, 0.0, 0.0};
    auto y = MinkowskiMomentum::rapidity(p);
    EXPECT_FALSE(y.has_value());
}

TEST(RapidityTest, NulloptWhenPxExceedsE) {
    FourMomentum p{0.5, 2.0, 0.0, 0.0};
    auto y = MinkowskiMomentum::rapidity(p);
    EXPECT_FALSE(y.has_value());
}

// ─── MinkowskiMomentum::transverse_momentum ──────────────────────────────────

TEST(TransverseMomentumTest, KnownValue) {
    FourMomentum p{1.0, 0.0, 3.0, 4.0};
    EXPECT_NEAR(MinkowskiMomentum::transverse_momentum(p), 5.0, 1e-12);
}

TEST(TransverseMomentumTest, ZeroWhenNoBondOrCommodity) {
    FourMomentum p{1.0, 0.5, 0.0, 0.0};
    EXPECT_NEAR(MinkowskiMomentum::transverse_momentum(p), 0.0, 1e-12);
}

// ─── MinkowskiMomentum::spatial_magnitude ────────────────────────────────────

TEST(SpatialMagnitudeTest, ThreeFourFive) {
    // sqrt(3^2 + 4^2 + 0^2) = 5
    FourMomentum p{10.0, 3.0, 4.0, 0.0};
    EXPECT_NEAR(MinkowskiMomentum::spatial_magnitude(p), 5.0, 1e-12);
}

// ─── FourMomentumConservation::sum ───────────────────────────────────────────

TEST(FourMomentumConservationTest, SumOfTwoVectors) {
    std::vector<FourMomentum> trades = {
        {1.0, 0.2, 0.1, 0.05},
        {0.5, 0.1, 0.3, 0.10},
    };
    auto total = FourMomentumConservation::sum(trades);
    EXPECT_NEAR(total.energy, 1.5,  1e-12);
    EXPECT_NEAR(total.px,     0.3,  1e-12);
    EXPECT_NEAR(total.py,     0.4,  1e-12);
    EXPECT_NEAR(total.pz,     0.15, 1e-12);
}

TEST(FourMomentumConservationTest, EmptySpanSumsToZero) {
    std::vector<FourMomentum> trades;
    auto total = FourMomentumConservation::sum(trades);
    EXPECT_DOUBLE_EQ(total.energy, 0.0);
    EXPECT_DOUBLE_EQ(total.px,     0.0);
}

// ─── FourMomentumConservation::conserves ─────────────────────────────────────

TEST(FourMomentumConservationTest, ConservesWhenExactMatch) {
    std::vector<FourMomentum> trades = {{1.0, 0.5, 0.3, 0.2}};
    FourMomentum reference{1.0, 0.5, 0.3, 0.2};
    EXPECT_TRUE(FourMomentumConservation::conserves(trades, reference));
}

TEST(FourMomentumConservationTest, NotConservedWhenMismatch) {
    std::vector<FourMomentum> trades = {{1.0, 0.5, 0.3, 0.2}};
    FourMomentum reference{2.0, 0.5, 0.3, 0.2};
    EXPECT_FALSE(FourMomentumConservation::conserves(trades, reference));
}

TEST(FourMomentumConservationTest, ConservesWithinTolerance) {
    std::vector<FourMomentum> trades = {{1.0, 0.5, 0.3, 0.2}};
    FourMomentum reference{1.0 + 1e-10, 0.5, 0.3, 0.2};
    EXPECT_TRUE(FourMomentumConservation::conserves(trades, reference, 1e-8));
}

// ─── MomentumPortfolioOptimizer ───────────────────────────────────────────────

TEST(MomentumPortfolioOptimizerTest, OptimizeReturnsResult) {
    std::vector<double> returns = {0.10, 0.08};
    std::vector<std::array<double, 3>> exposures = {
        {0.8, 0.1, 0.1},
        {0.2, 0.7, 0.1},
    };
    auto result = MomentumPortfolioOptimizer::optimize(returns, exposures);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->weights.size(), 2u);
}

TEST(MomentumPortfolioOptimizerTest, WeightsSumToOne) {
    std::vector<double> returns = {0.10, 0.08, 0.06};
    std::vector<std::array<double, 3>> exposures = {
        {0.8, 0.1, 0.1},
        {0.3, 0.6, 0.1},
        {0.1, 0.2, 0.7},
    };
    auto result = MomentumPortfolioOptimizer::optimize(returns, exposures);
    ASSERT_TRUE(result.has_value());
    double sum = 0.0;
    for (double w : result->weights) sum += w;
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(MomentumPortfolioOptimizerTest, WeightsWithinBounds) {
    std::vector<double> returns = {0.10, 0.08};
    std::vector<std::array<double, 3>> exposures = {
        {0.8, 0.1, 0.1},
        {0.2, 0.7, 0.1},
    };
    MomentumPortfolioOptimizer::Config cfg;
    cfg.min_weight = 0.1;
    cfg.max_weight = 0.9;
    auto result = MomentumPortfolioOptimizer::optimize(returns, exposures, cfg);
    ASSERT_TRUE(result.has_value());
    for (double w : result->weights) {
        EXPECT_GE(w, cfg.min_weight - 1e-9);
        EXPECT_LE(w, cfg.max_weight + 1e-9);
    }
}

TEST(MomentumPortfolioOptimizerTest, EmptyInputReturnsNullopt) {
    std::vector<double> returns;
    std::vector<std::array<double, 3>> exposures;
    auto result = MomentumPortfolioOptimizer::optimize(returns, exposures);
    EXPECT_FALSE(result.has_value());
}

TEST(MomentumPortfolioOptimizerTest, MismatchedInputSizesReturnsNullopt) {
    std::vector<double> returns = {0.10};
    std::vector<std::array<double, 3>> exposures = {
        {0.8, 0.1, 0.1},
        {0.2, 0.7, 0.1},
    };
    auto result = MomentumPortfolioOptimizer::optimize(returns, exposures);
    EXPECT_FALSE(result.has_value());
}
