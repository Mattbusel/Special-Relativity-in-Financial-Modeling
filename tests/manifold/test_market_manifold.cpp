/// @file tests/manifold/test_market_manifold.cpp
/// @brief Tests for MarketManifold — AGT-06.

#include "srfm/manifold.hpp"
#include "srfm/constants.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

using namespace srfm::manifold;
using namespace srfm::constants;

namespace {

// ─── Fixtures ─────────────────────────────────────────────────────────────────

/// Two events separated purely in time (timelike by construction).
SpacetimeEvent make_origin() { return {0.0, 100.0, 1e6, 0.0}; }
SpacetimeEvent make_time_advance(double dt) { return {dt, 100.0, 1e6, 0.0}; }
SpacetimeEvent make_spatial_jump(double dp) { return {1.0, 100.0 + dp, 1e6, 0.0}; }

}  // anonymous namespace

// ─── classify: Timelike ───────────────────────────────────────────────────────

TEST(MarketManifoldClassify, PureTimeAdvanceIsTimelike) {
    // Δx = 0, Δt > 0  →  ds² = −c²·Δt² < 0  →  Timelike
    auto result = MarketManifold::classify(make_origin(), make_time_advance(1.0));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, IntervalType::Timelike);
}

TEST(MarketManifoldClassify, LargeTimeSmallSpaceIsTimelike) {
    SpacetimeEvent a{0.0, 100.0, 1e6, 0.0};
    SpacetimeEvent b{10.0, 100.1, 1e6, 0.0};
    // Δt = 10, Δp = 0.1 → ds² = 0.01 − 100 = −99.99 < 0
    auto result = MarketManifold::classify(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, IntervalType::Timelike);
}

TEST(MarketManifoldClassify, ReturnsNulloptForNonfiniteTime) {
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{std::numeric_limits<double>::infinity(), 100.0, 0.0, 0.0};
    EXPECT_FALSE(MarketManifold::classify(a, b).has_value());
}

TEST(MarketManifoldClassify, ReturnsNulloptForNaNPrice) {
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{1.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
    EXPECT_FALSE(MarketManifold::classify(a, b).has_value());
}

// ─── classify: Spacelike ──────────────────────────────────────────────────────

TEST(MarketManifoldClassify, LargeSpaceSmallTimeIsSpacelike) {
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{0.01, 200.0, 0.0, 0.0};
    // Δp = 100, Δt = 0.01  →  ds² = 10000 − 0.0001 > 0  →  Spacelike
    auto result = MarketManifold::classify(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, IntervalType::Spacelike);
}

TEST(MarketManifoldClassify, IdenticalEventsIsLightlike) {
    // Δ everything = 0 → ds² = 0 → Lightlike
    SpacetimeEvent a{1.0, 100.0, 1e6, 0.5};
    auto result = MarketManifold::classify(a, a);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, IntervalType::Lightlike);
}

// ─── beta ─────────────────────────────────────────────────────────────────────

TEST(MarketManifoldBeta, ZeroSpatialChangeGivesZeroBeta) {
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{1.0, 100.0, 0.0, 0.0};  // no spatial change
    auto result = MarketManifold::beta(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(*result, 0.0);
}

TEST(MarketManifoldBeta, ZeroDeltaTimeReturnsNullopt) {
    SpacetimeEvent a{1.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{1.0, 200.0, 0.0, 0.0};  // Δt = 0
    EXPECT_FALSE(MarketManifold::beta(a, b).has_value());
}

TEST(MarketManifoldBeta, BetaIsClamped) {
    // Enormous spatial jump relative to time  →  raw β >> 1 → should be clamped
    SpacetimeEvent a{0.0, 0.0, 0.0, 0.0};
    SpacetimeEvent b{1e-9, 1e6, 0.0, 0.0};
    auto result = MarketManifold::beta(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_LT(*result, BETA_MAX_SAFE);
    EXPECT_GE(*result, 0.0);
}

TEST(MarketManifoldBeta, SymmetryInTime) {
    // β should be the same regardless of which event comes first.
    SpacetimeEvent a{0.0, 100.0, 1e6, 0.1};
    SpacetimeEvent b{2.0, 110.0, 1.1e6, 0.2};
    auto ab = MarketManifold::beta(a, b, 1.0);
    auto ba = MarketManifold::beta(b, a, 1.0);
    ASSERT_TRUE(ab.has_value());
    ASSERT_TRUE(ba.has_value());
    EXPECT_NEAR(*ab, *ba, 1e-12);
}

TEST(MarketManifoldBeta, NonfiniteCMarketReturnsNullopt) {
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{1.0, 101.0, 0.0, 0.0};
    EXPECT_FALSE(MarketManifold::beta(a, b, 0.0).has_value());
    EXPECT_FALSE(MarketManifold::beta(a, b, -1.0).has_value());
    EXPECT_FALSE(MarketManifold::beta(a, b,
        std::numeric_limits<double>::quiet_NaN()).has_value());
}

// ─── is_causal ────────────────────────────────────────────────────────────────

TEST(MarketManifoldIsCausal, TimelikeIsCausal) {
    EXPECT_TRUE(MarketManifold::is_causal(make_origin(), make_time_advance(1.0)));
}

TEST(MarketManifoldIsCausal, SpacelikeIsNotCausal) {
    SpacetimeEvent a{0.0, 0.0, 0.0, 0.0};
    SpacetimeEvent b{0.001, 1000.0, 0.0, 0.0};
    EXPECT_FALSE(MarketManifold::is_causal(a, b));
}

TEST(MarketManifoldIsCausal, NanInputIsNotCausal) {
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{std::numeric_limits<double>::quiet_NaN(), 100.0, 0.0, 0.0};
    EXPECT_FALSE(MarketManifold::is_causal(a, b));
}

TEST(MarketManifoldIsCausal, LightlikeIsCausal) {
    // Lightlike: ds² = 0 → classify returns Lightlike → is_causal = true
    SpacetimeEvent a{0.0, 100.0, 0.0, 0.0};
    SpacetimeEvent b{1.0, 101.0, 0.0, 0.0};
    // ds² = dp² − c²dt² = 1 − 1 = 0 when dp=1, c=1, dt=1
    EXPECT_TRUE(MarketManifold::is_causal(a, b));
}

// ─── to_string ────────────────────────────────────────────────────────────────

TEST(IntervalTypeToString, Timelike) {
    EXPECT_STREQ(to_string(IntervalType::Timelike), "Timelike");
}
TEST(IntervalTypeToString, Lightlike) {
    EXPECT_STREQ(to_string(IntervalType::Lightlike), "Lightlike");
}
TEST(IntervalTypeToString, Spacelike) {
    EXPECT_STREQ(to_string(IntervalType::Spacelike), "Spacelike");
}
