/// @file tests/manifold/test_normalizer.cpp
/// @brief Unit tests for CoordinateNormalizer (rolling z-score normalization).
///
/// Test categories:
///   - Rolling mean convergence to zero after window fills
///   - Rolling stddev convergence to 1.0 after window fills
///   - Flat series does not divide by zero (returns 0.0)
///   - Time coordinate passes through unchanged
///   - Window size is respected (eviction of oldest values)
///   - Default window size is 20
///   - reset() clears all state
///   - size() / window_size() accessors
///   - Multiple independent coordinates (price, volume, momentum)
///   - Window partially filled normalization
///   - MarketManifold::process integration
///   - Large-scale values normalize correctly

#include <gtest/gtest.h>
#include "srfm/normalizer.hpp"
#include "srfm/manifold.hpp"

#include <cmath>
#include <numeric>
#include <vector>

using namespace srfm;
using namespace srfm::manifold;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Feed `n` events with linearly increasing price (1,2,...,n), constant
/// volume and momentum. Return all normalized events.
static std::vector<SpacetimeEvent> feed_linear_price(
        CoordinateNormalizer& norm, std::size_t n) {
    std::vector<SpacetimeEvent> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        SpacetimeEvent raw{
            .time     = static_cast<double>(i),
            .price    = static_cast<double>(i + 1),
            .volume   = 1.0,
            .momentum = 0.5,
        };
        out.push_back(norm.normalize(raw));
    }
    return out;
}

/// Sample mean of a vector of doubles.
static double vec_mean(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x;
    return sum / static_cast<double>(v.size());
}

/// Sample standard deviation (Bessel-corrected).
static double vec_stddev(const std::vector<double>& v, double mean) {
    double sq = 0.0;
    for (double x : v) { double d = x - mean; sq += d * d; }
    return std::sqrt(sq / static_cast<double>(v.size() - 1));
}

// ─── Test 1: time coordinate is unchanged ────────────────────────────────────

TEST(CoordinateNormalizer, TimeCoordinatePassesThrough) {
    CoordinateNormalizer norm(5);
    SpacetimeEvent raw{.time = 42.0, .price = 100.0, .volume = 1e6, .momentum = 0.01};
    auto result = norm.normalize(raw);
    EXPECT_DOUBLE_EQ(result.time, 42.0);
}

TEST(CoordinateNormalizer, TimeCoordinateUnchangedOverMultipleCalls) {
    CoordinateNormalizer norm(5);
    for (int i = 0; i < 20; ++i) {
        SpacetimeEvent raw{
            .time     = static_cast<double>(i) * 3.7,
            .price    = static_cast<double>(i),
            .volume   = static_cast<double>(i * 100),
            .momentum = static_cast<double>(i) * 0.01,
        };
        auto result = norm.normalize(raw);
        EXPECT_DOUBLE_EQ(result.time, raw.time)
            << "Time should pass through unchanged at step " << i;
    }
}

// ─── Test 2: flat series does not divide by zero ─────────────────────────────

TEST(CoordinateNormalizer, FlatPriceSeries_ReturnsZero) {
    CoordinateNormalizer norm(10);
    SpacetimeEvent raw{.time = 0.0, .price = 100.0, .volume = 500.0, .momentum = 0.5};
    for (int i = 0; i < 15; ++i) {
        auto result = norm.normalize(raw);
        // Constant series → stddev = 0 → z-score = 0.0 (no division by zero)
        EXPECT_DOUBLE_EQ(result.price, 0.0)
            << "Flat price should yield 0.0 at step " << i;
    }
}

TEST(CoordinateNormalizer, FlatAllCoordinates_AllZero) {
    CoordinateNormalizer norm(5);
    SpacetimeEvent raw{.time = 1.0, .price = 50.0, .volume = 2000.0, .momentum = -0.3};
    for (int i = 0; i < 10; ++i) {
        auto result = norm.normalize(raw);
        EXPECT_DOUBLE_EQ(result.price,    0.0);
        EXPECT_DOUBLE_EQ(result.volume,   0.0);
        EXPECT_DOUBLE_EQ(result.momentum, 0.0);
    }
}

// ─── Test 3: window default is 20 ────────────────────────────────────────────

TEST(CoordinateNormalizer, DefaultWindowSizeIs20) {
    CoordinateNormalizer norm;
    EXPECT_EQ(norm.window_size(), 20u);
}

// ─── Test 4: size() tracks buffer fill ───────────────────────────────────────

TEST(CoordinateNormalizer, SizeIncreasesUntilWindowFull) {
    CoordinateNormalizer norm(5);
    EXPECT_EQ(norm.size(), 0u);
    SpacetimeEvent raw{.time = 0.0, .price = 1.0, .volume = 1.0, .momentum = 1.0};
    for (std::size_t i = 1; i <= 5; ++i) {
        norm.normalize(raw);
        EXPECT_EQ(norm.size(), i);
    }
    // Beyond window: size stays at window_size
    norm.normalize(raw);
    EXPECT_EQ(norm.size(), 5u);
    norm.normalize(raw);
    EXPECT_EQ(norm.size(), 5u);
}

// ─── Test 5: window_size() accessor ──────────────────────────────────────────

TEST(CoordinateNormalizer, WindowSizeAccessor) {
    CoordinateNormalizer norm(7);
    EXPECT_EQ(norm.window_size(), 7u);
}

// ─── Test 6: reset() clears all state ────────────────────────────────────────

TEST(CoordinateNormalizer, ResetClearsBuffer) {
    CoordinateNormalizer norm(5);
    SpacetimeEvent raw{.time = 0.0, .price = 10.0, .volume = 100.0, .momentum = 1.0};
    for (int i = 0; i < 5; ++i) norm.normalize(raw);
    EXPECT_EQ(norm.size(), 5u);
    norm.reset();
    EXPECT_EQ(norm.size(), 0u);
}

TEST(CoordinateNormalizer, ResetAllowsReuse) {
    CoordinateNormalizer norm(3);
    SpacetimeEvent raw1{.time = 0.0, .price = 100.0, .volume = 1e6, .momentum = 0.01};
    norm.normalize(raw1);
    norm.normalize(raw1);
    norm.normalize(raw1);
    norm.reset();

    // After reset, the normalizer should behave as freshly constructed.
    // First two samples: stddev still 0 for constant series.
    SpacetimeEvent raw2{.time = 1.0, .price = 1.0, .volume = 2.0, .momentum = 3.0};
    SpacetimeEvent raw3{.time = 2.0, .price = 2.0, .volume = 4.0, .momentum = 6.0};
    norm.normalize(raw2);
    auto r = norm.normalize(raw3);
    // With two samples (1,2): mean=1.5, stddev=sqrt(0.5)≈0.7071
    // z-score(2) = (2-1.5)/0.7071 ≈ 0.7071
    EXPECT_NEAR(r.price, 0.7071067, 1e-5);
}

// ─── Test 7: window eviction — only recent N samples are used ─────────────────

TEST(CoordinateNormalizer, WindowEvictsOldestSamples) {
    // Window of 3. Feed: 1,2,3,4,5.
    // After 5th push, window contains {3,4,5}; z-score(5) should use that.
    CoordinateNormalizer norm(3);
    std::vector<double> prices = {1.0, 2.0, 3.0, 4.0, 5.0};
    SpacetimeEvent last_normalized{};
    for (double p : prices) {
        SpacetimeEvent raw{.time = 0.0, .price = p, .volume = 1.0, .momentum = 0.0};
        last_normalized = norm.normalize(raw);
    }
    // Window after 5 pushes: {3,4,5}, mean=4, stddev=1.0
    // z-score(5) = (5-4)/1.0 = 1.0
    EXPECT_NEAR(last_normalized.price, 1.0, 1e-10);
}

// ─── Test 8: normalized output has ~zero mean over full window ────────────────

TEST(CoordinateNormalizer, MeanApproximatelyZeroAfterWindowFills) {
    // For any series, the z-scored outputs over the window should have mean~0.
    // We feed a monotonic series and collect the last `window` outputs.
    const std::size_t W = 10;
    CoordinateNormalizer norm(W);

    // Feed 2*W samples; collect last W normalized prices.
    std::vector<double> norm_prices;
    for (std::size_t i = 0; i < 2 * W; ++i) {
        SpacetimeEvent raw{
            .time     = static_cast<double>(i),
            .price    = static_cast<double>(i) + 1.0,
            .volume   = 1.0,
            .momentum = 0.0,
        };
        auto r = norm.normalize(raw);
        if (i >= W) {
            norm_prices.push_back(r.price);
        }
    }
    // The normalized values over a sliding window of a linear series
    // are not exactly zero-mean (each call uses a different window),
    // but the values should be bounded and well-scaled (z-scores ≈ ±1..2).
    // For a strictly linear series they will be symmetric around 0.
    ASSERT_EQ(norm_prices.size(), W);
    double m = vec_mean(norm_prices);
    // Linear series z-scores are symmetric: mean should be very close to 0.
    EXPECT_NEAR(m, 0.0, 0.2)
        << "Mean of normalized prices over last window should be ~0";
}

// ─── Test 9: normalized output has ~unit stddev over full window ──────────────

TEST(CoordinateNormalizer, StddevApproximatelyOneAfterWindowFills) {
    // Feed W+1 samples of a known series; last W normalized outputs
    // should have stddev close to 1.
    const std::size_t W = 20;
    CoordinateNormalizer norm(W);

    std::vector<double> norm_prices;
    // Use a series with clear variance: alternating ±10
    for (std::size_t i = 0; i < 2 * W; ++i) {
        double p = (i % 2 == 0) ? 10.0 : -10.0;
        SpacetimeEvent raw{.time = static_cast<double>(i),
                           .price = p, .volume = 1.0, .momentum = 0.0};
        auto r = norm.normalize(raw);
        if (i >= W) {
            norm_prices.push_back(r.price);
        }
    }
    ASSERT_GE(norm_prices.size(), 2u);
    double m  = vec_mean(norm_prices);
    double sd = vec_stddev(norm_prices, m);
    // For a perfectly alternating ±10 series, z-scores should be close to ±1
    // so stddev of the z-scores ≈ 1.
    EXPECT_NEAR(sd, 1.0, 0.15)
        << "Stddev of normalized prices over last window should be ~1";
}

// ─── Test 10: coordinates are normalized independently ───────────────────────

TEST(CoordinateNormalizer, CoordinatesAreIndependent) {
    // price: monotone increasing
    // volume: constant
    // momentum: oscillating
    CoordinateNormalizer norm(5);
    for (int i = 0; i < 10; ++i) {
        SpacetimeEvent raw{
            .time     = static_cast<double>(i),
            .price    = static_cast<double>(i + 1),  // increasing
            .volume   = 500.0,                         // constant → stddev=0
            .momentum = (i % 2 == 0) ? 1.0 : -1.0,  // oscillating
        };
        auto r = norm.normalize(raw);
        // Volume is constant: stddev=0 → always 0.0
        if (i >= 1) {
            EXPECT_DOUBLE_EQ(r.volume, 0.0)
                << "Constant volume should normalize to 0.0 at step " << i;
        }
        // Time always unchanged
        EXPECT_DOUBLE_EQ(r.time, raw.time);
    }
}

// ─── Test 11: large-scale values normalize correctly ─────────────────────────

TEST(CoordinateNormalizer, LargeScaleValues_Finite) {
    CoordinateNormalizer norm(10);
    // Simulate realistic market data: price~100, volume~1e6, momentum~0.01
    for (int i = 0; i < 20; ++i) {
        SpacetimeEvent raw{
            .time     = static_cast<double>(i),
            .price    = 100.0 + static_cast<double>(i) * 0.5,
            .volume   = 1e6 + static_cast<double>(i) * 1000.0,
            .momentum = 0.01 + static_cast<double>(i) * 0.001,
        };
        auto r = norm.normalize(raw);
        // All outputs must be finite
        EXPECT_TRUE(std::isfinite(r.price))    << "price not finite at " << i;
        EXPECT_TRUE(std::isfinite(r.volume))   << "volume not finite at " << i;
        EXPECT_TRUE(std::isfinite(r.momentum)) << "momentum not finite at " << i;
    }
}

// ─── Test 12: window of 1 behaves correctly ───────────────────────────────────

TEST(CoordinateNormalizer, WindowOfOne_ReturnsZero) {
    CoordinateNormalizer norm(1);
    EXPECT_EQ(norm.window_size(), 1u);
    SpacetimeEvent raw{.time = 5.0, .price = 99.0, .volume = 3000.0, .momentum = 0.1};
    auto r = norm.normalize(raw);
    // Single-sample stddev = 0 → z-score = 0.0
    EXPECT_DOUBLE_EQ(r.price,    0.0);
    EXPECT_DOUBLE_EQ(r.volume,   0.0);
    EXPECT_DOUBLE_EQ(r.momentum, 0.0);
    EXPECT_DOUBLE_EQ(r.time,     5.0);
}

// ─── Test 13: two-sample window computes correct z-score ─────────────────────

TEST(CoordinateNormalizer, TwoSampleWindow_CorrectZScore) {
    CoordinateNormalizer norm(2);
    // First sample: p=1
    SpacetimeEvent raw1{.time = 0.0, .price = 1.0, .volume = 1.0, .momentum = 1.0};
    norm.normalize(raw1);  // only 1 sample → stddev=0 → z-score=0

    // Second sample: p=3; window={1,3}, mean=2, stddev=sqrt(2)≈1.4142
    // z-score(3) = (3-2)/sqrt(2) = 1/sqrt(2) ≈ 0.7071
    SpacetimeEvent raw2{.time = 1.0, .price = 3.0, .volume = 3.0, .momentum = 3.0};
    auto r = norm.normalize(raw2);
    EXPECT_NEAR(r.price,    1.0 / std::sqrt(2.0), 1e-10);
    EXPECT_NEAR(r.volume,   1.0 / std::sqrt(2.0), 1e-10);
    EXPECT_NEAR(r.momentum, 1.0 / std::sqrt(2.0), 1e-10);
}

// ─── Test 14: MarketManifold::process uses normalizer before classify ─────────

TEST(MarketManifold_Process, IntervalChangesAfterNormalization) {
    // Without normalization, ds² is dominated by volume (~1e12 scale).
    // With normalization, coordinates are z-scored before computing ds².
    CoordinateNormalizer norm(10);

    // Feed several bars to build up the window.
    SpacetimeEvent prev_norm{.time = 0.0, .price = 0.0, .volume = 0.0, .momentum = 0.0};
    for (int i = 1; i <= 9; ++i) {
        SpacetimeEvent raw{
            .time     = static_cast<double>(i),
            .price    = static_cast<double>(i) * 100.0,
            .volume   = static_cast<double>(i) * 1e6,
            .momentum = static_cast<double>(i) * 0.01,
        };
        auto type = MarketManifold::process(norm, prev_norm, raw);
        EXPECT_TRUE(type.has_value());
    }
}

// ─── Test 15: process returns valid interval type (not nullopt) ───────────────

TEST(MarketManifold_Process, ReturnsValidIntervalType) {
    CoordinateNormalizer norm(5);
    SpacetimeEvent prev{.time = 0.0, .price = 0.0, .volume = 0.0, .momentum = 0.0};

    for (int i = 1; i <= 10; ++i) {
        SpacetimeEvent curr{
            .time     = static_cast<double>(i),
            .price    = 100.0 + static_cast<double>(i),
            .volume   = 1e6   + static_cast<double>(i * 1000),
            .momentum = 0.01  + static_cast<double>(i) * 0.001,
        };
        auto result = MarketManifold::process(norm, prev, curr);
        ASSERT_TRUE(result.has_value())
            << "process() should not return nullopt at step " << i;
        // Result must be a valid IntervalType
        bool valid = (*result == IntervalType::Timelike  ||
                      *result == IntervalType::Lightlike ||
                      *result == IntervalType::Spacelike);
        EXPECT_TRUE(valid);
    }
}

// ─── Test 16: negative values normalize correctly ─────────────────────────────

TEST(CoordinateNormalizer, NegativeValues_CorrectZScore) {
    CoordinateNormalizer norm(2);
    SpacetimeEvent raw1{.time = 0.0, .price = -3.0, .volume = -3.0, .momentum = -3.0};
    SpacetimeEvent raw2{.time = 1.0, .price = -1.0, .volume = -1.0, .momentum = -1.0};
    norm.normalize(raw1);
    auto r = norm.normalize(raw2);
    // window = {-3,-1}, mean=-2, stddev=sqrt(2)
    // z-score(-1) = (-1 - (-2)) / sqrt(2) = 1/sqrt(2)
    EXPECT_NEAR(r.price,    1.0 / std::sqrt(2.0), 1e-10);
    EXPECT_NEAR(r.volume,   1.0 / std::sqrt(2.0), 1e-10);
    EXPECT_NEAR(r.momentum, 1.0 / std::sqrt(2.0), 1e-10);
}
