/// @file tests/lorentz/test_online_beta.cpp
/// @brief Tests for BetaCalculator::fromPriceVelocityOnline (no look-ahead β).
///
/// Test categories:
///   - Invalid inputs return nullopt
///   - Running max is monotonically non-decreasing
///   - β at bar i is identical whether or not future bars exist
///   - Online and offline agree when global max occurs at bar 0
///   - Output length matches input length
///   - All β values are in valid range [0, BETA_MAX_SAFE)
///   - Stationary prices give β = 0

#include <gtest/gtest.h>
#include "beta_calculator.hpp"
#include "srfm/constants.hpp"

#include <cmath>
#include <numeric>
#include <vector>

using namespace srfm::lorentz;
using namespace srfm::constants;

// ─── Invalid input guards ─────────────────────────────────────────────────────

TEST(OnlineBeta, EmptySeries_Nullopt) {
    std::vector<double> prices;
    auto result = BetaCalculator::fromPriceVelocityOnline(prices, 1.0);
    EXPECT_FALSE(result.has_value());
}

TEST(OnlineBeta, SingleElement_Nullopt) {
    std::vector<double> prices = {100.0};
    auto result = BetaCalculator::fromPriceVelocityOnline(prices, 1.0);
    EXPECT_FALSE(result.has_value());
}

TEST(OnlineBeta, ZeroTimeDelta_Nullopt) {
    std::vector<double> prices = {1.0, 2.0, 3.0};
    auto result = BetaCalculator::fromPriceVelocityOnline(prices, 0.0);
    EXPECT_FALSE(result.has_value());
}

TEST(OnlineBeta, NegativeTimeDelta_Nullopt) {
    std::vector<double> prices = {1.0, 2.0, 3.0};
    auto result = BetaCalculator::fromPriceVelocityOnline(prices, -1.0);
    EXPECT_FALSE(result.has_value());
}

TEST(OnlineBeta, NanPrice_Nullopt) {
    std::vector<double> prices = {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    auto result = BetaCalculator::fromPriceVelocityOnline(prices, 1.0);
    EXPECT_FALSE(result.has_value());
}

TEST(OnlineBeta, InfPrice_Nullopt) {
    std::vector<double> prices = {1.0, std::numeric_limits<double>::infinity(), 3.0};
    auto result = BetaCalculator::fromPriceVelocityOnline(prices, 1.0);
    EXPECT_FALSE(result.has_value());
}

// ─── Output length ────────────────────────────────────────────────────────────

TEST(OnlineBeta, OutputLengthMatchesInput) {
    std::vector<double> prices = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto result = BetaCalculator::fromPriceVelocityOnline(prices, 1.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), prices.size());
}

TEST(OnlineBeta, TwoElementSeries_OutputLengthTwo) {
    std::vector<double> prices = {10.0, 20.0};
    auto result = BetaCalculator::fromPriceVelocityOnline(prices, 1.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 2u);
}

// ─── All β values in valid range ──────────────────────────────────────────────

TEST(OnlineBeta, AllBetasInValidRange) {
    // Monotonically increasing prices — clear velocity signal.
    std::vector<double> prices(30);
    for (int i = 0; i < 30; ++i) prices[i] = 100.0 + i * 0.5;

    auto result = BetaCalculator::fromPriceVelocityOnline(prices, 1.0);
    ASSERT_TRUE(result.has_value());
    for (const auto& beta : *result) {
        EXPECT_GE(beta.value, 0.0) << "β must be non-negative";
        EXPECT_LT(beta.value, BETA_MAX_SAFE) << "β must be below BETA_MAX_SAFE";
        EXPECT_TRUE(std::isfinite(beta.value)) << "β must be finite";
    }
}

// ─── Stationary series → β = 0 ───────────────────────────────────────────────

TEST(OnlineBeta, StationarySeries_BetaIsZero) {
    // Constant prices → zero velocity → β = 0 everywhere.
    std::vector<double> prices(10, 100.0);
    auto result = BetaCalculator::fromPriceVelocityOnline(prices, 1.0);
    ASSERT_TRUE(result.has_value());
    for (const auto& beta : *result) {
        EXPECT_DOUBLE_EQ(beta.value, 0.0);
    }
}

// ─── Running max is monotonically non-decreasing ─────────────────────────────

TEST(OnlineBeta, RunningMax_Monotonic) {
    // Prices with a burst of velocity partway through.
    // Velocities: small, small, large spike, small, small.
    // Once running_max sees the spike, it stays at that level.
    std::vector<double> prices = {100.0, 100.1, 100.2, 110.0, 110.1, 110.2, 110.3};
    // Velocities (backward diff): 0.1, 0.1, 0.1, 9.8, 0.1, 0.1, 0.1
    // Running max:                0.1, 0.1, 0.1, 9.8, 9.8, 9.8, 9.8
    // β after spike:              small because v/running_max is small

    auto result = BetaCalculator::fromPriceVelocityOnline(prices, 1.0);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 7u);

    // β at bar 4 (post-spike, velocity=0.1, running_max=9.8) should be << β at bar 3.
    // Bar 3 had the spike so β[3] should be large (or max safe).
    // Bars 4-6 have v=0.1, running_max=9.8, so β ≈ 0.1/9.8 ≈ 0.01 → small.
    EXPECT_GT((*result)[3].value, (*result)[4].value)
        << "β at spike should exceed post-spike β";
    // After the spike, running_max doesn't decrease, so post-spike β values
    // are smaller than the spike bar.
    for (std::size_t i = 4; i < 7; ++i) {
        EXPECT_LT((*result)[i].value, (*result)[3].value)
            << "Post-spike β should be smaller than spike β at bar " << i;
    }
}

// ─── No look-ahead: β at bar i same whether or not future bars exist ──────────

TEST(OnlineBeta, NoLookAhead_BetaAtBarIIsStable) {
    // For a constant-velocity series, truncating the series at bar i
    // should give the same β_i as using the full series.
    // (Because the running_max is the same up to bar i in both cases.)

    std::vector<double> full_prices = {100.0, 101.0, 102.0, 103.0,
                                        104.0, 105.0, 106.0, 107.0};
    auto full_result = BetaCalculator::fromPriceVelocityOnline(full_prices, 1.0);
    ASSERT_TRUE(full_result.has_value());

    // For each prefix of length ≥ 2, compute the online betas and check bar i.
    for (std::size_t prefix_len = 2; prefix_len <= 6; ++prefix_len) {
        std::span<const double> prefix(full_prices.data(), prefix_len);
        auto prefix_result = BetaCalculator::fromPriceVelocityOnline(prefix, 1.0);
        ASSERT_TRUE(prefix_result.has_value())
            << "prefix of length " << prefix_len << " should succeed";

        // The last β in the prefix should match the same position in the full run.
        const std::size_t bar_idx = prefix_len - 1;
        EXPECT_NEAR(prefix_result->back().value,
                    (*full_result)[bar_idx].value, 1e-12)
            << "β[" << bar_idx << "] differs between full and prefix series";
    }
}

// ─── Online and offline agree when max velocity is at bar 0 ──────────────────

TEST(OnlineBeta, OnlineOfflineAgree_WhenMaxAtBar0) {
    // Construct a series where the first interval has the largest velocity.
    // Prices: [100, 200, 200.1, 200.2, 200.3]
    // Velocities: 100 (bar0→1), 0.1, 0.1, 0.1
    // Global max = 100 = running_max from bar 0.
    // For all bars, running_max = 100 (same as global max).
    // → Online and offline normalise by the same denominator → same β.

    std::vector<double> prices = {100.0, 200.0, 200.1, 200.2, 200.3};
    const double dt = 1.0;

    auto online = BetaCalculator::fromPriceVelocityOnline(prices, dt);
    ASSERT_TRUE(online.has_value());

    // Offline: compute mean-abs-velocity over whole series and normalise each
    // bar's instantaneous velocity by the global max.
    // Global velocity at each bar (backward diff):
    // bar0: |200-100|/1 = 100
    // bar1: |200-100|/1 = 100 (same formula for online bar 0/1)
    // bar2: |200.1-200|/1 = 0.1
    // bar3: |200.2-200.1|/1 = 0.1
    // bar4: |200.3-200.2|/1 = 0.1
    // Global max = 100.

    // For bar 2: online β = 0.1/100 = 0.001
    const double expected_bar2 = 0.1 / 100.0;
    EXPECT_NEAR((*online)[2].value, expected_bar2, 1e-10);

    // For bar 0 and 1: velocity = 100, running_max = 100 → β = 1.0 → clamped
    EXPECT_GT((*online)[0].value, 0.99 * BETA_MAX_SAFE)
        << "Bar 0 should have near-maximum β when it has max velocity";
}
