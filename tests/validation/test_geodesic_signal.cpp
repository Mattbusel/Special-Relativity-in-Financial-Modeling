/// @file tests/validation/test_geodesic_signal.cpp
/// @brief Unit & Integration Tests — GeodesicDeviationCalculator
///
/// Tests cover:
///   - Empty and single-event inputs
///   - Flat (straight-line) geodesic: deviation should remain near zero
///   - Non-zero deviation for off-geodesic actual paths
///   - is_valid flag propagation
///   - Non-finite input handling
///   - Large input stress test

#include "srfm/geodesic_signal.hpp"
#include "srfm/tensor.hpp"
#include "srfm/manifold.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace srfm;
using namespace srfm::tensor;
using namespace srfm::manifold;

// ─── Test Fixtures ────────────────────────────────────────────────────────────

namespace {

/// Build a simple Minkowski metric (flat spacetime) for testing.
MetricTensor flat_metric() {
    return MetricTensor::make_minkowski(1.0, 1.0);
}

/// Build a sequence of SpacetimeEvents on a straight spatial line.
/// price evolves linearly, volume and momentum are constant.
std::vector<SpacetimeEvent> straight_line_events(std::size_t n,
                                                   double price_step = 0.01)
{
    std::vector<SpacetimeEvent> evs;
    evs.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        evs.push_back({
            .time     = static_cast<double>(i),
            .price    = 100.0 + static_cast<double>(i) * price_step,
            .volume   = 1000.0,
            .momentum = 0.01,
        });
    }
    return evs;
}

} // anonymous namespace

// ─── Empty Input ──────────────────────────────────────────────────────────────

TEST(GeodesicSignalTest, empty_input_returns_empty_vector) {
    auto metric = flat_metric();
    GeodesicDeviationCalculator calc(metric);

    std::vector<SpacetimeEvent> empty;
    auto result = calc.compute(empty);

    EXPECT_TRUE(result.empty());
}

// ─── Single Event ─────────────────────────────────────────────────────────────

TEST(GeodesicSignalTest, single_event_returns_zero_deviation) {
    auto metric = flat_metric();
    GeodesicDeviationCalculator calc(metric);

    std::vector<SpacetimeEvent> one = {
        {.time = 0.0, .price = 100.0, .volume = 1000.0, .momentum = 0.01}
    };
    auto result = calc.compute(one);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_DOUBLE_EQ(result[0].geodesic_deviation, 0.0);
    EXPECT_DOUBLE_EQ(result[0].proper_time, 0.0);
    EXPECT_TRUE(result[0].is_valid);
}

// ─── Two Events ───────────────────────────────────────────────────────────────

TEST(GeodesicSignalTest, two_events_first_is_reference) {
    auto metric = flat_metric();
    GeodesicDeviationCalculator calc(metric);

    std::vector<SpacetimeEvent> two = {
        {.time = 0.0, .price = 100.0, .volume = 1000.0, .momentum = 0.0},
        {.time = 1.0, .price = 100.5, .volume = 1100.0, .momentum = 0.005},
    };
    auto result = calc.compute(two);

    ASSERT_EQ(result.size(), 2u);
    // First bar is always the reference point
    EXPECT_DOUBLE_EQ(result[0].geodesic_deviation, 0.0);
    EXPECT_TRUE(result[0].is_valid);
    // Second bar: geodesic was initialized FROM the first two points, so
    // the deviation at bar 1 depends on how well the geodesic extrapolation
    // matches the actual second point.
    EXPECT_TRUE(result[1].is_valid);
    EXPECT_GE(result[1].geodesic_deviation, 0.0);
}

// ─── Straight-Line Path ───────────────────────────────────────────────────────

TEST(GeodesicSignalTest, straight_line_deviations_are_finite_and_non_negative) {
    auto metric = flat_metric();
    GeodesicDeviationCalculator calc(metric);

    auto events = straight_line_events(50, 0.01);
    auto result = calc.compute(events);

    ASSERT_EQ(result.size(), 50u);

    for (std::size_t i = 0; i < result.size(); ++i) {
        SCOPED_TRACE("bar " + std::to_string(i));
        EXPECT_GE(result[i].geodesic_deviation, 0.0);
        EXPECT_TRUE(std::isfinite(result[i].geodesic_deviation));
        EXPECT_TRUE(result[i].is_valid);
    }
}

// ─── First Bar is Zero ────────────────────────────────────────────────────────

TEST(GeodesicSignalTest, first_bar_deviation_is_always_zero) {
    auto metric = flat_metric();
    GeodesicDeviationCalculator calc(metric);

    for (int seed = 0; seed < 5; ++seed) {
        auto events = straight_line_events(20, 0.005 * (seed + 1));
        auto result = calc.compute(events);
        ASSERT_FALSE(result.empty());
        EXPECT_DOUBLE_EQ(result[0].geodesic_deviation, 0.0)
            << "seed=" << seed;
    }
}

// ─── Proper Time Monotonicity ─────────────────────────────────────────────────

TEST(GeodesicSignalTest, proper_time_is_monotonically_increasing) {
    auto metric = flat_metric();
    GeodesicDeviationCalculator calc(metric);

    auto events = straight_line_events(30);
    auto result = calc.compute(events);

    for (std::size_t i = 1; i < result.size(); ++i) {
        EXPECT_GT(result[i].proper_time, result[i - 1].proper_time)
            << "proper_time not monotonic at bar " << i;
    }
}

// ─── Off-Geodesic Path Produces Non-Zero Deviation ───────────────────────────

TEST(GeodesicSignalTest, off_geodesic_path_has_nonzero_deviation) {
    auto metric = flat_metric();
    GeodesicDeviationCalculator calc(metric);

    // Start on geodesic, then diverge sharply
    std::vector<SpacetimeEvent> events;
    for (std::size_t i = 0; i < 10; ++i) {
        events.push_back({
            .time     = static_cast<double>(i),
            .price    = 100.0 + static_cast<double>(i) * 0.01,
            .volume   = 1000.0,
            .momentum = 0.0,
        });
    }
    // Sudden large price jump: drastically off the geodesic
    events.push_back({
        .time     = 10.0,
        .price    = 200.0,   // 100% jump
        .volume   = 1000.0,
        .momentum = 0.0,
    });

    auto result = calc.compute(events);
    ASSERT_EQ(result.size(), 11u);

    // The last bar (index 10) should have the largest deviation
    EXPECT_GT(result[10].geodesic_deviation, result[0].geodesic_deviation);
}

// ─── Non-Finite Input ─────────────────────────────────────────────────────────

TEST(GeodesicSignalTest, nan_first_point_returns_all_invalid) {
    auto metric = flat_metric();
    GeodesicDeviationCalculator calc(metric);

    std::vector<SpacetimeEvent> events = {
        {.time = std::numeric_limits<double>::quiet_NaN(),
         .price = 100.0, .volume = 1000.0, .momentum = 0.0},
        {.time = 1.0, .price = 100.5, .volume = 1000.0, .momentum = 0.0},
        {.time = 2.0, .price = 101.0, .volume = 1000.0, .momentum = 0.0},
    };

    auto result = calc.compute(events);
    ASSERT_EQ(result.size(), 3u);
    for (const auto& sig : result) {
        EXPECT_FALSE(sig.is_valid);
    }
}

// ─── Covariance-Based Metric ──────────────────────────────────────────────────

TEST(GeodesicSignalTest, covariance_metric_produces_finite_deviations) {
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 0.04; // σ=0.2
    auto metric = MetricTensor::make_from_covariance(1.0, cov);
    GeodesicDeviationCalculator calc(metric, 0.001);

    auto events = straight_line_events(20);
    auto result = calc.compute(events);

    ASSERT_EQ(result.size(), 20u);
    for (const auto& sig : result) {
        EXPECT_TRUE(std::isfinite(sig.geodesic_deviation));
        EXPECT_GE(sig.geodesic_deviation, 0.0);
    }
}

// ─── Large Input Stress Test ─────────────────────────────────────────────────

TEST(GeodesicSignalTest, large_input_does_not_crash) {
    auto metric = flat_metric();
    GeodesicDeviationCalculator calc(metric);

    auto events = straight_line_events(1000);
    auto result = calc.compute(events);

    EXPECT_EQ(result.size(), 1000u);
    // All should be valid (straight-line path, well-conditioned)
    std::size_t invalid_count = 0;
    for (const auto& sig : result) {
        if (!sig.is_valid) ++invalid_count;
    }
    EXPECT_EQ(invalid_count, 0u);
}

// ─── Constant Events (Zero Momentum) ─────────────────────────────────────────

TEST(GeodesicSignalTest, constant_events_handled_gracefully) {
    auto metric = flat_metric();
    GeodesicDeviationCalculator calc(metric);

    // All events identical: zero displacement, degenerate velocity
    std::vector<SpacetimeEvent> events;
    for (std::size_t i = 0; i < 10; ++i) {
        events.push_back({.time = static_cast<double>(i),
                          .price = 100.0, .volume = 1000.0, .momentum = 0.0});
    }

    // Should not crash regardless of result
    EXPECT_NO_THROW({
        auto result = calc.compute(events);
        EXPECT_EQ(result.size(), 10u);
        for (const auto& sig : result) {
            EXPECT_TRUE(std::isfinite(sig.geodesic_deviation));
            EXPECT_GE(sig.geodesic_deviation, 0.0);
        }
    });
}
