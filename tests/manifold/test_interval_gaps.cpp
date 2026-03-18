/// @file tests/manifold/test_interval_gaps.cpp
/// @brief Gap-filling tests for SpacetimeInterval.
///
/// Covers:
///   1. Symmetry of SPACELIKE/TIMELIKE classification under event swap.
///   2. Numerical stability with extreme (very large / very small) prices.
///   3. Lorentz invariance: ds2 is invariant under a Lorentz boost.

#include "srfm/manifold.hpp"
#include "srfm/constants.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

using namespace srfm::manifold;
using namespace srfm::constants;

// ─── 1. Symmetry under direction swap ─────────────────────────────────────────
// ds2(a,b) == ds2(b,a) because every spatial delta is squared.
// The CLASSIFICATION should therefore also be identical in both directions.

TEST(IntervalSymmetry, TimeLikeClassificationIsSymmetric) {
    // Pure time displacement is always Timelike.
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{2.0, 100.0, 0.0, 0.0};

    auto ab = SpacetimeInterval::compute(a, b);
    auto ba = SpacetimeInterval::compute(b, a);
    ASSERT_TRUE(ab.has_value());
    ASSERT_TRUE(ba.has_value());
    EXPECT_EQ(SpacetimeInterval::classify(*ab), IntervalType::Timelike);
    EXPECT_EQ(SpacetimeInterval::classify(*ba), IntervalType::Timelike);
    EXPECT_NEAR(*ab, *ba, 1e-12);
}

TEST(IntervalSymmetry, SpaceLikeClassificationIsSymmetric) {
    // Pure price displacement is always Spacelike.
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{0.0, 110.0, 0.0, 0.0};

    auto ab = SpacetimeInterval::compute(a, b);
    auto ba = SpacetimeInterval::compute(b, a);
    ASSERT_TRUE(ab.has_value());
    ASSERT_TRUE(ba.has_value());
    EXPECT_EQ(SpacetimeInterval::classify(*ab), IntervalType::Spacelike);
    EXPECT_EQ(SpacetimeInterval::classify(*ba), IntervalType::Spacelike);
    EXPECT_NEAR(*ab, *ba, 1e-12);
}

TEST(IntervalSymmetry, MixedRegimeIsSymmetric) {
    // Arbitrary event pair: classification must be consistent both ways.
    SpacetimeEvent a{1.0, 50.0, 1e4, 0.2};
    SpacetimeEvent b{3.0, 80.0, 2e4, 0.5};

    auto ab = SpacetimeInterval::compute(a, b);
    auto ba = SpacetimeInterval::compute(b, a);
    ASSERT_TRUE(ab.has_value());
    ASSERT_TRUE(ba.has_value());
    EXPECT_NEAR(*ab, *ba, 1e-10);
    EXPECT_EQ(SpacetimeInterval::classify(*ab), SpacetimeInterval::classify(*ba));
}

TEST(IntervalSymmetry, SwappingAllCoordinatesPreservesClass) {
    SpacetimeEvent a{0.5, 120.0, 5e5, 0.1};
    SpacetimeEvent b{1.5,  80.0, 3e5, 0.3};

    auto forward = SpacetimeInterval::compute(a, b);
    auto backward = SpacetimeInterval::compute(b, a);
    ASSERT_TRUE(forward.has_value());
    ASSERT_TRUE(backward.has_value());
    EXPECT_NEAR(*forward, *backward, 1e-10);
}

// ─── 2. Numerical stability with extreme prices ────────────────────────────────

TEST(NumericalStability, VeryLargePriceDisplacement) {
    // ΔP = 1e12 (e.g. inflation scenario or a different asset scale)
    SpacetimeEvent a{0.0, 0.0,     0.0, 0.0};
    SpacetimeEvent b{0.0, 1.0e12,  0.0, 0.0};

    auto result = SpacetimeInterval::compute(a, b);
    ASSERT_TRUE(result.has_value());
    // Pure price => Spacelike
    EXPECT_EQ(SpacetimeInterval::classify(*result), IntervalType::Spacelike);
    EXPECT_GT(*result, 0.0);
}

TEST(NumericalStability, VerySmallPriceDisplacement) {
    // ΔP = 1e-12, Δt = 0 => still Spacelike (positive)
    SpacetimeEvent a{0.0, 0.0,     0.0, 0.0};
    SpacetimeEvent b{0.0, 1.0e-12, 0.0, 0.0};

    auto result = SpacetimeInterval::compute(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(SpacetimeInterval::classify(*result), IntervalType::Spacelike);
}

TEST(NumericalStability, LargeTimeWithSmallPriceShouldBeTimelike) {
    // Δt = 1e6, ΔP = 1 => ds2 = -c^2*(1e6)^2 + 1 << 0 => Timelike
    SpacetimeEvent a{0.0,    0.0, 0.0, 0.0};
    SpacetimeEvent b{1.0e6,  1.0, 0.0, 0.0};

    auto result = SpacetimeInterval::compute(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(SpacetimeInterval::classify(*result), IntervalType::Timelike);
    EXPECT_LT(*result, 0.0);
}

TEST(NumericalStability, ExtremeVolumeCoordinate) {
    // ΔV = 1e18 (extreme volume spike) => Spacelike
    SpacetimeEvent a{0.0, 100.0, 0.0,    0.0};
    SpacetimeEvent b{0.0, 100.0, 1.0e18, 0.0};

    auto result = SpacetimeInterval::compute(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(SpacetimeInterval::classify(*result), IntervalType::Spacelike);
}

TEST(NumericalStability, NaNInputReturnsNullopt) {
    SpacetimeEvent a{0.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
    SpacetimeEvent b{1.0, 100.0, 0.0, 0.0};
    EXPECT_FALSE(SpacetimeInterval::compute(a, b).has_value());
}

TEST(NumericalStability, InfinityInputReturnsNullopt) {
    SpacetimeEvent a{std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0};
    SpacetimeEvent b{1.0, 100.0, 0.0, 0.0};
    EXPECT_FALSE(SpacetimeInterval::compute(a, b).has_value());
}

// ─── 3. Lorentz invariance of ds2 ─────────────────────────────────────────────
// Under a Lorentz boost in the price direction with velocity beta_b,
// the coordinates transform as:
//   t' = gamma*(t - beta_b * x)
//   x' = gamma*(x - beta_b * t)
// The interval ds2 = -t^2 + x^2 is invariant (in natural units, c=1).
//
// This verifies the core relativistic invariant holds numerically.

static double compute_1d_interval(double dt, double dx) {
    // ds2 = -dt^2 + dx^2 (c=1 natural units, 1-dimensional)
    return -dt * dt + dx * dx;
}

static void lorentz_boost_1d(double t, double x, double beta_b,
                               double& t_prime, double& x_prime) {
    const double gamma = 1.0 / std::sqrt(1.0 - beta_b * beta_b);
    t_prime = gamma * (t - beta_b * x);
    x_prime = gamma * (x - beta_b * t);
}

TEST(LorentzInvariance, IntervalInvariantUnderBoostedFrame) {
    // Two events in the rest frame
    const double t1 = 0.0, x1 = 0.0;
    const double t2 = 2.0, x2 = 0.5;

    const double ds2_rest = compute_1d_interval(t2 - t1, x2 - x1);

    const double beta_b = 0.3;  // boost velocity

    double t1p, x1p, t2p, x2p;
    lorentz_boost_1d(t1, x1, beta_b, t1p, x1p);
    lorentz_boost_1d(t2, x2, beta_b, t2p, x2p);

    const double ds2_boosted = compute_1d_interval(t2p - t1p, x2p - x1p);

    EXPECT_NEAR(ds2_rest, ds2_boosted, 1e-10)
        << "Spacetime interval changed under Lorentz boost: rest="
        << ds2_rest << " boosted=" << ds2_boosted;
}

TEST(LorentzInvariance, IntervalInvariantUnderMultipleBoosts) {
    const double t1 = 0.0, x1 = 0.0;
    const double t2 = 5.0, x2 = 2.0;

    const double ds2_rest = compute_1d_interval(t2 - t1, x2 - x1);

    for (double beta_b : {0.1, 0.5, 0.8, 0.95}) {
        double t1p, x1p, t2p, x2p;
        lorentz_boost_1d(t1, x1, beta_b, t1p, x1p);
        lorentz_boost_1d(t2, x2, beta_b, t2p, x2p);

        const double ds2_boosted = compute_1d_interval(t2p - t1p, x2p - x1p);
        EXPECT_NEAR(ds2_rest, ds2_boosted, 1e-8)
            << "Invariant failed at beta=" << beta_b;
    }
}

TEST(LorentzInvariance, SpacelikeIntervalStaysSpacelikeAfterBoost) {
    // Pure space displacement: ds2 > 0 (Spacelike).
    // Must remain Spacelike in any boosted frame (below the speed of light).
    const double t1 = 0.0, x1 = 0.0;
    const double t2 = 0.0, x2 = 3.0;  // ds2 = +9

    const double ds2_rest = compute_1d_interval(t2 - t1, x2 - x1);
    ASSERT_GT(ds2_rest, 0.0);

    for (double beta_b : {0.1, 0.5, 0.7}) {
        double t1p, x1p, t2p, x2p;
        lorentz_boost_1d(t1, x1, beta_b, t1p, x1p);
        lorentz_boost_1d(t2, x2, beta_b, t2p, x2p);
        const double ds2_boosted = compute_1d_interval(t2p - t1p, x2p - x1p);
        EXPECT_GT(ds2_boosted, 0.0)
            << "Spacelike interval became non-spacelike at beta=" << beta_b;
    }
}
