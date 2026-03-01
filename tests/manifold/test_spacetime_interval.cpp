/// @file tests/manifold/test_spacetime_interval.cpp
/// @brief Tests for SpacetimeInterval — AGT-06.

#include "srfm/manifold.hpp"
#include "srfm/constants.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

using namespace srfm::manifold;
using namespace srfm::constants;

// ─── compute: basic arithmetic ────────────────────────────────────────────────

TEST(SpacetimeIntervalCompute, PureTimeDisplacementIsNegative) {
    // ds² = −c²·Δt² + 0 = −1  (c=1, Δt=1)
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{1.0, 100.0, 0.0, 0.0};
    auto result = SpacetimeInterval::compute(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, -1.0, 1e-12);
}

TEST(SpacetimeIntervalCompute, PurePriceDisplacementIsPositive) {
    // ds² = 0 − 0 + ΔP² = 4  (ΔP=2, Δt=0, ΔV=0, ΔM=0)
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{0.0, 102.0, 0.0, 0.0};
    auto result = SpacetimeInterval::compute(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 4.0, 1e-12);
}

TEST(SpacetimeIntervalCompute, AllZeroDisplacementIsZero) {
    SpacetimeEvent a{1.0, 50.0, 1e5, 0.3};
    auto result = SpacetimeInterval::compute(a, a);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.0, 1e-12);
}

TEST(SpacetimeIntervalCompute, FullFourVector) {
    // ds² = ΔP² + ΔV² + ΔM² − c²Δt²
    // a=(0,0,0,0), b=(2,1,1,1)
    // ds² = 1² + 1² + 1² − 1·2² = 3 − 4 = −1
    SpacetimeEvent a{0.0, 0.0, 0.0, 0.0};
    SpacetimeEvent b{2.0, 1.0, 1.0, 1.0};
    auto result = SpacetimeInterval::compute(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, -1.0, 1e-12);
}

TEST(SpacetimeIntervalCompute, CustomSpeedOfInformation) {
    // ds² = ΔP² − c²Δt², c=2, Δt=1, ΔP=1 → 1 − 4 = −3
    SpacetimeEvent a{0.0, 0.0, 0.0, 0.0};
    SpacetimeEvent b{1.0, 1.0, 0.0, 0.0};
    auto result = SpacetimeInterval::compute(a, b, 2.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, -3.0, 1e-12);
}

TEST(SpacetimeIntervalCompute, SymmetryUnderSwap) {
    // ds²(a→b) == ds²(b→a) because all differences are squared.
    SpacetimeEvent a{0.0, 100.0, 1e6, 0.1};
    SpacetimeEvent b{3.0, 115.0, 1.1e6, 0.3};
    auto ab = SpacetimeInterval::compute(a, b);
    auto ba = SpacetimeInterval::compute(b, a);
    ASSERT_TRUE(ab.has_value());
    ASSERT_TRUE(ba.has_value());
    EXPECT_NEAR(*ab, *ba, 1e-12);
}

TEST(SpacetimeIntervalCompute, ReturnsNulloptForInfiniteCoord) {
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{std::numeric_limits<double>::infinity(), 100.0, 0.0, 0.0};
    EXPECT_FALSE(SpacetimeInterval::compute(a, b).has_value());
}

TEST(SpacetimeIntervalCompute, ReturnsNulloptForNaNCoord) {
    SpacetimeEvent a{0.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
    SpacetimeEvent b{1.0, 100.0, 0.0, 0.0};
    EXPECT_FALSE(SpacetimeInterval::compute(a, b).has_value());
}

TEST(SpacetimeIntervalCompute, ReturnsNulloptForZeroCMarket) {
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{1.0, 101.0, 0.0, 0.0};
    EXPECT_FALSE(SpacetimeInterval::compute(a, b, 0.0).has_value());
}

TEST(SpacetimeIntervalCompute, ReturnsNulloptForNegativeCMarket) {
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{1.0, 101.0, 0.0, 0.0};
    EXPECT_FALSE(SpacetimeInterval::compute(a, b, -1.0).has_value());
}

// ─── classify ─────────────────────────────────────────────────────────────────

TEST(SpacetimeIntervalClassify, NegativeIsTimelike) {
    EXPECT_EQ(SpacetimeInterval::classify(-1.0), IntervalType::Timelike);
}

TEST(SpacetimeIntervalClassify, PositiveIsSpacelike) {
    EXPECT_EQ(SpacetimeInterval::classify(1.0), IntervalType::Spacelike);
}

TEST(SpacetimeIntervalClassify, ExactZeroIsLightlike) {
    EXPECT_EQ(SpacetimeInterval::classify(0.0), IntervalType::Lightlike);
}

TEST(SpacetimeIntervalClassify, SmallNegativeWithinEpsilonIsLightlike) {
    // |ds²| ≤ FLOAT_EPSILON → lightlike
    EXPECT_EQ(SpacetimeInterval::classify(-FLOAT_EPSILON / 2.0), IntervalType::Lightlike);
}

TEST(SpacetimeIntervalClassify, SmallPositiveWithinEpsilonIsLightlike) {
    EXPECT_EQ(SpacetimeInterval::classify(FLOAT_EPSILON / 2.0), IntervalType::Lightlike);
}

TEST(SpacetimeIntervalClassify, BeyondEpsilonNegativeIsTimelike) {
    EXPECT_EQ(SpacetimeInterval::classify(-2.0 * FLOAT_EPSILON), IntervalType::Timelike);
}

TEST(SpacetimeIntervalClassify, BeyondEpsilonPositiveIsSpacelike) {
    EXPECT_EQ(SpacetimeInterval::classify(2.0 * FLOAT_EPSILON), IntervalType::Spacelike);
}

TEST(SpacetimeIntervalClassify, LargeNegativeIsTimelike) {
    EXPECT_EQ(SpacetimeInterval::classify(-1e10), IntervalType::Timelike);
}

TEST(SpacetimeIntervalClassify, LargePositiveIsSpacelike) {
    EXPECT_EQ(SpacetimeInterval::classify(1e10), IntervalType::Spacelike);
}
