/**
 * @file  test_coordinate_normalizer.cpp
 * @brief Unit tests for CoordinateNormalizer — rolling z-score normaliser.
 *
 * Test structure:
 *   test_normalizer_pre_warmup_returns_zero      — returns 0 before window full
 *   test_normalizer_constant_series_returns_zero — zero sigma → z=0
 *   test_normalizer_newtonian_limit              — single-element window trivially
 *   test_normalizer_known_mean                   — mean converges correctly
 *   test_normalizer_known_sigma                  — variance converges correctly
 *   test_normalizer_z_score_in_range             — |z| bounded by construction
 *   test_normalizer_sliding_window_evicts_old    — old values leave the window
 *   test_normalizer_large_window                 — window=200 stability
 *   test_normalizer_reset                        — reset reverts state
 *   test_normalizer_warm_up_flag                 — warmed_up() semantics
 *   test_normalizer_symmetric_series             — symmetric inputs → z≈0
 *   test_normalizer_monotonic_series             — rising series produces z > 0
 */

#include "srfm_stream_test.hpp"
#include "../../include/srfm/stream/coordinate_normalizer.hpp"

#include <cmath>
#include <vector>

using namespace srfm::stream;

static constexpr double EPS_TIGHT = 1e-9;
static constexpr double EPS_LOOSE = 1e-6;

// ═════════════════════════════════════════════════════════════════════════════
// Pre-warmup returns zero
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_pre_warmup_returns_zero() {
    CoordinateNormalizer norm{10};

    // Calling normalise() before any update: z = 0.
    STREAM_CHECK_NEAR(norm.normalise(100.0), 0.0, EPS_TIGHT);

    // After 9 updates (window=10, need 10): still 0.
    for (int i = 0; i < 9; ++i) {
        norm.update(100.0 + i);
        STREAM_CHECK_NEAR(norm.normalise(100.0 + i), 0.0, EPS_TIGHT);
    }

    STREAM_CHECK(!norm.warmed_up());
}

// ═════════════════════════════════════════════════════════════════════════════
// Constant series → sigma = 0 → z = 0
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_constant_series_returns_zero() {
    CoordinateNormalizer norm{5};

    for (int i = 0; i < 10; ++i) {
        norm.update(42.0);
    }

    // All values identical → sigma = 0 → normalise returns 0.
    STREAM_CHECK_NEAR(norm.normalise(42.0), 0.0, EPS_TIGHT);
    STREAM_CHECK_NEAR(norm.sigma(),         0.0, EPS_TIGHT);
    STREAM_CHECK_NEAR(norm.mean(),         42.0, EPS_TIGHT);
}

// ═════════════════════════════════════════════════════════════════════════════
// Known mean after window fill
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_known_mean() {
    CoordinateNormalizer norm{5};

    // Push 1, 2, 3, 4, 5 → mean = 3.
    for (int i = 1; i <= 5; ++i) norm.update(static_cast<double>(i));
    STREAM_CHECK_NEAR(norm.mean(), 3.0, EPS_TIGHT);

    // Slide: remove 1, add 6 → window = {2,3,4,5,6} → mean = 4.
    norm.update(6.0);
    STREAM_CHECK_NEAR(norm.mean(), 4.0, EPS_LOOSE);

    // Slide: remove 2, add 7 → window = {3,4,5,6,7} → mean = 5.
    norm.update(7.0);
    STREAM_CHECK_NEAR(norm.mean(), 5.0, EPS_LOOSE);
}

// ═════════════════════════════════════════════════════════════════════════════
// Known sigma after window fill
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_known_sigma() {
    CoordinateNormalizer norm{4};

    // Values: -3, -1, 1, 3 → mean = 0, population var = (9+1+1+9)/4 = 5, sigma = √5
    for (double v : {-3.0, -1.0, 1.0, 3.0}) norm.update(v);

    STREAM_CHECK_NEAR(norm.mean(),  0.0,            EPS_LOOSE);
    STREAM_CHECK_NEAR(norm.sigma(), std::sqrt(5.0), EPS_LOOSE);
}

// ═════════════════════════════════════════════════════════════════════════════
// z-score of mean value is 0
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_z_score_in_range() {
    CoordinateNormalizer norm{5};
    for (double v : {10.0, 20.0, 30.0, 40.0, 50.0}) norm.update(v);

    // Normalise the mean → z = 0.
    double mean = norm.mean();
    STREAM_CHECK_NEAR(norm.normalise(mean), 0.0, EPS_LOOSE);

    // Normalise mean + sigma → z = 1.
    double sigma = norm.sigma();
    STREAM_CHECK_NEAR(norm.normalise(mean + sigma), 1.0, EPS_LOOSE);

    // Normalise mean - sigma → z = -1.
    STREAM_CHECK_NEAR(norm.normalise(mean - sigma), -1.0, EPS_LOOSE);
}

// ═════════════════════════════════════════════════════════════════════════════
// Sliding window evicts old values
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_sliding_window_evicts_old() {
    CoordinateNormalizer norm{3};

    // Fill with 1, 1, 1 → mean = 1.
    for (int i = 0; i < 3; ++i) norm.update(1.0);
    STREAM_CHECK_NEAR(norm.mean(), 1.0, EPS_TIGHT);

    // Slide in 100, 100, 100 → should evict the 1s.
    for (int i = 0; i < 3; ++i) norm.update(100.0);
    STREAM_CHECK_NEAR(norm.mean(), 100.0, EPS_LOOSE);
    STREAM_CHECK_NEAR(norm.sigma(), 0.0,  EPS_LOOSE);
}

// ═════════════════════════════════════════════════════════════════════════════
// Large window (200) stability — mean / sigma converge
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_large_window() {
    CoordinateNormalizer norm{200};

    // Push 1000 values uniformly in [0, 1].
    for (int i = 0; i < 1000; ++i) {
        double v = (i % 200) / 200.0; // deterministic pseudo-uniform
        norm.update(v);
    }

    STREAM_CHECK(norm.warmed_up());
    STREAM_CHECK(std::isfinite(norm.mean()));
    STREAM_CHECK(std::isfinite(norm.sigma()));
    STREAM_CHECK(norm.sigma() >= 0.0);

    // All normalised values should be finite.
    for (double v : {0.0, 0.5, 1.0}) {
        STREAM_CHECK_FINITE(norm.normalise(v));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Reset reverts all state
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_reset() {
    CoordinateNormalizer norm{5};

    for (int i = 0; i < 10; ++i) norm.update(static_cast<double>(i) * 10.0);
    STREAM_CHECK(norm.warmed_up());

    norm.reset();

    STREAM_CHECK(!norm.warmed_up());
    STREAM_CHECK(norm.count()  == 0u);
    STREAM_CHECK_NEAR(norm.mean(),  0.0, EPS_TIGHT);
    STREAM_CHECK_NEAR(norm.sigma(), 0.0, EPS_TIGHT);
    STREAM_CHECK_NEAR(norm.normalise(50.0), 0.0, EPS_TIGHT);
}

// ═════════════════════════════════════════════════════════════════════════════
// warmed_up() semantics
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_warm_up_flag() {
    CoordinateNormalizer norm{4};

    STREAM_CHECK(!norm.warmed_up());
    norm.update(1.0); STREAM_CHECK(!norm.warmed_up());
    norm.update(2.0); STREAM_CHECK(!norm.warmed_up());
    norm.update(3.0); STREAM_CHECK(!norm.warmed_up());
    norm.update(4.0); STREAM_CHECK(norm.warmed_up());  // exactly window size

    // Stays warmed up forever.
    for (int i = 0; i < 100; ++i) {
        norm.update(static_cast<double>(i));
        STREAM_CHECK(norm.warmed_up());
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Symmetric series → z-score ≈ 0 for the mean
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_symmetric_series() {
    CoordinateNormalizer norm{6};

    // {-3, -2, -1, 1, 2, 3} → mean = 0.
    for (double v : {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0}) norm.update(v);

    STREAM_CHECK_NEAR(norm.mean(),          0.0, EPS_LOOSE);
    STREAM_CHECK_NEAR(norm.normalise(0.0),  0.0, EPS_LOOSE);
    STREAM_CHECK(norm.normalise(3.0) > 0.0);
    STREAM_CHECK(norm.normalise(-3.0) < 0.0);
}

// ═════════════════════════════════════════════════════════════════════════════
// Monotonic rising series → the latest value has z > 0
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_monotonic_series() {
    CoordinateNormalizer norm{5};

    // Rising: 1, 2, 3, 4, 5 → mean = 3, last close = 5 → z > 0.
    for (double v : {1.0, 2.0, 3.0, 4.0, 5.0}) norm.update(v);

    STREAM_CHECK(norm.normalise(5.0) > 0.0);
    STREAM_CHECK(norm.normalise(1.0) < 0.0);
}

// ═════════════════════════════════════════════════════════════════════════════
// count() accessor
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_count_accessor() {
    CoordinateNormalizer norm{5};

    for (std::size_t i = 1; i <= 8; ++i) {
        norm.update(static_cast<double>(i));
        // count saturates at window size.
        STREAM_CHECK(norm.count() == std::min(i, norm.window()));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Two separate windows of same size are independent (no shared state)
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_two_instances_are_independent() {
    CoordinateNormalizer a{5};
    CoordinateNormalizer b{5};

    // Feed different data to each.
    for (double v : {1.0, 2.0, 3.0, 4.0, 5.0})   a.update(v); // mean=3
    for (double v : {10.0, 20.0, 30.0, 40.0, 50.0}) b.update(v); // mean=30

    STREAM_CHECK_NEAR(a.mean(), 3.0,  EPS_LOOSE);
    STREAM_CHECK_NEAR(b.mean(), 30.0, EPS_LOOSE);
    STREAM_CHECK(std::abs(a.normalise(3.0) - b.normalise(30.0)) < EPS_LOOSE);
}

// ═════════════════════════════════════════════════════════════════════════════
// Normalise value above mean gives positive z; below gives negative z
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_sign_of_z_score() {
    CoordinateNormalizer norm{5};
    for (double v : {10.0, 20.0, 30.0, 40.0, 50.0}) norm.update(v); // mean=30

    STREAM_CHECK(norm.normalise(31.0) > 0.0);   // above mean
    STREAM_CHECK(norm.normalise(29.0) < 0.0);   // below mean
    STREAM_CHECK_NEAR(norm.normalise(30.0), 0.0, EPS_LOOSE); // at mean
}

// ═════════════════════════════════════════════════════════════════════════════
// Window accessor returns correct configured size
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_window_accessor() {
    CoordinateNormalizer n2{2};
    CoordinateNormalizer n20{20};
    CoordinateNormalizer n100{100};
    STREAM_CHECK(n2.window()   == 2u);
    STREAM_CHECK(n20.window()  == 20u);
    STREAM_CHECK(n100.window() == 100u);
}

// ═════════════════════════════════════════════════════════════════════════════
// sigma^2 = population variance (Bessel-uncorrected)
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_sigma_squared_is_population_variance() {
    CoordinateNormalizer norm{4};
    // Values: 1, 2, 3, 4 → mean = 2.5, pop variance = 1.25
    for (double v : {1.0, 2.0, 3.0, 4.0}) norm.update(v);

    double s = norm.sigma();
    STREAM_CHECK_NEAR(s * s, 1.25, EPS_LOOSE);
}

// ═════════════════════════════════════════════════════════════════════════════
// Normalise at mean+2*sigma = 2.0 (z = 2)
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_z_score_of_2_sigma() {
    CoordinateNormalizer norm{5};
    for (double v : {10.0, 20.0, 30.0, 40.0, 50.0}) norm.update(v);

    double sigma = norm.sigma();
    double mean  = norm.mean();
    STREAM_CHECK_NEAR(norm.normalise(mean + 2.0 * sigma), 2.0, EPS_LOOSE);
    STREAM_CHECK_NEAR(norm.normalise(mean - 2.0 * sigma), -2.0, EPS_LOOSE);
}

// ═════════════════════════════════════════════════════════════════════════════
// After partial fill, count() returns actual number, not window size
// ═════════════════════════════════════════════════════════════════════════════

static void test_normalizer_count_before_window_full() {
    CoordinateNormalizer norm{10};
    for (int i = 1; i <= 7; ++i) {
        norm.update(static_cast<double>(i));
        STREAM_CHECK(norm.count() == static_cast<std::size_t>(i));
    }
    STREAM_CHECK(!norm.warmed_up());
}

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("SRFM Stream — CoordinateNormalizer Tests\n");
    std::printf("==========================================\n");

    STREAM_SUITE("pre-warmup returns zero",        test_normalizer_pre_warmup_returns_zero);
    STREAM_SUITE("constant series returns zero",   test_normalizer_constant_series_returns_zero);
    STREAM_SUITE("known mean",                     test_normalizer_known_mean);
    STREAM_SUITE("known sigma",                    test_normalizer_known_sigma);
    STREAM_SUITE("z-score of mean is 0",           test_normalizer_z_score_in_range);
    STREAM_SUITE("sliding window evicts old",      test_normalizer_sliding_window_evicts_old);
    STREAM_SUITE("large window (200) stability",   test_normalizer_large_window);
    STREAM_SUITE("reset reverts state",            test_normalizer_reset);
    STREAM_SUITE("warmed_up() semantics",          test_normalizer_warm_up_flag);
    STREAM_SUITE("symmetric series",               test_normalizer_symmetric_series);
    STREAM_SUITE("monotonic series",               test_normalizer_monotonic_series);
    STREAM_SUITE("count() accessor",               test_normalizer_count_accessor);
    STREAM_SUITE("two instances independent",      test_normalizer_two_instances_are_independent);
    STREAM_SUITE("sign of z-score",                test_normalizer_sign_of_z_score);
    STREAM_SUITE("window() accessor",              test_normalizer_window_accessor);
    STREAM_SUITE("sigma^2 = population variance",  test_normalizer_sigma_squared_is_population_variance);
    STREAM_SUITE("z-score at ±2-sigma",            test_normalizer_z_score_of_2_sigma);
    STREAM_SUITE("count before window full",       test_normalizer_count_before_window_full);

    return stream_test::report();
}
