/**
 * @file  test_tick_validation.cpp
 * @brief Unit tests for tick.hpp — OHLCVTick and tick_is_valid().
 *
 * Test structure:
 *   test_valid_tick_accepted              — canonical valid tick
 *   test_nan_fields_rejected              — NaN in each price field
 *   test_inf_fields_rejected              — +/-Inf in each price field
 *   test_non_positive_price_rejected      — zero and negative prices
 *   test_negative_volume_rejected         — negative volume
 *   test_zero_volume_accepted             — zero volume is allowed
 *   test_invalid_timestamp_rejected       — zero and negative timestamp
 *   test_ohlc_inconsistency_rejected      — high < low, etc.
 *   test_boundary_values_accepted         — epsilon-above-zero prices
 *   test_regime_to_str                    — regime_to_str() coverage
 */

#include "srfm_stream_test.hpp"
#include "../../include/srfm/stream/tick.hpp"
#include "../../include/srfm/stream/stream_signal.hpp"

#include <cstring>
#include <limits>

using namespace srfm::stream;

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
static constexpr double Inf = std::numeric_limits<double>::infinity();

// ── Helper: construct a fully valid tick ─────────────────────────────────────

static OHLCVTick valid_tick() noexcept {
    OHLCVTick t;
    t.open  = 100.0;
    t.high  = 105.0;
    t.low   =  98.0;
    t.close = 102.0;
    t.volume = 10000.0;
    t.timestamp_ns = 1700000000000000000LL;
    return t;
}

// ═════════════════════════════════════════════════════════════════════════════
// Valid tick accepted
// ═════════════════════════════════════════════════════════════════════════════

static void test_valid_tick_accepted() {
    STREAM_CHECK(tick_is_valid(valid_tick()));

    // Tick with open == low == close == high (doji candle) is valid.
    OHLCVTick doji = valid_tick();
    doji.open = doji.high = doji.low = doji.close = 50.0;
    STREAM_CHECK(tick_is_valid(doji));

    // Very small positive price.
    OHLCVTick small = valid_tick();
    small.open = small.high = small.low = small.close = 1e-6;
    STREAM_CHECK(tick_is_valid(small));

    // Very large price.
    OHLCVTick big = valid_tick();
    big.open = big.low = 1e12;
    big.high  = 1e12 + 1.0;
    big.close = 1e12;
    STREAM_CHECK(tick_is_valid(big));

    // Zero volume is valid.
    OHLCVTick novol = valid_tick();
    novol.volume = 0.0;
    STREAM_CHECK(tick_is_valid(novol));
}

// ═════════════════════════════════════════════════════════════════════════════
// NaN in each price field
// ═════════════════════════════════════════════════════════════════════════════

static void test_nan_fields_rejected() {
    {
        auto t = valid_tick(); t.open = NaN;
        STREAM_CHECK(!tick_is_valid(t));
    }
    {
        auto t = valid_tick(); t.high = NaN;
        STREAM_CHECK(!tick_is_valid(t));
    }
    {
        auto t = valid_tick(); t.low = NaN;
        STREAM_CHECK(!tick_is_valid(t));
    }
    {
        auto t = valid_tick(); t.close = NaN;
        STREAM_CHECK(!tick_is_valid(t));
    }
    {
        auto t = valid_tick(); t.volume = NaN;
        STREAM_CHECK(!tick_is_valid(t));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Inf in each price field
// ═════════════════════════════════════════════════════════════════════════════

static void test_inf_fields_rejected() {
    {
        auto t = valid_tick(); t.open = Inf;
        STREAM_CHECK(!tick_is_valid(t));
    }
    {
        auto t = valid_tick(); t.high = Inf;
        STREAM_CHECK(!tick_is_valid(t));
    }
    {
        auto t = valid_tick(); t.low = -Inf;
        STREAM_CHECK(!tick_is_valid(t));
    }
    {
        auto t = valid_tick(); t.close = -Inf;
        STREAM_CHECK(!tick_is_valid(t));
    }
    {
        auto t = valid_tick(); t.volume = Inf;
        STREAM_CHECK(!tick_is_valid(t));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Non-positive price rejected
// ═════════════════════════════════════════════════════════════════════════════

static void test_non_positive_price_rejected() {
    // Zero price.
    {
        auto t = valid_tick(); t.open = 0.0;
        STREAM_CHECK(!tick_is_valid(t));
    }
    {
        auto t = valid_tick(); t.close = 0.0;
        STREAM_CHECK(!tick_is_valid(t));
    }
    // Negative price.
    {
        auto t = valid_tick(); t.high = -1.0;
        STREAM_CHECK(!tick_is_valid(t));
    }
    {
        auto t = valid_tick(); t.low = -0.001;
        STREAM_CHECK(!tick_is_valid(t));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Negative volume rejected; zero accepted
// ═════════════════════════════════════════════════════════════════════════════

static void test_negative_volume_rejected() {
    auto t = valid_tick(); t.volume = -1.0;
    STREAM_CHECK(!tick_is_valid(t));

    auto t2 = valid_tick(); t2.volume = -0.0001;
    STREAM_CHECK(!tick_is_valid(t2));
}

static void test_zero_volume_accepted() {
    auto t = valid_tick(); t.volume = 0.0;
    STREAM_CHECK(tick_is_valid(t));
}

// ═════════════════════════════════════════════════════════════════════════════
// Invalid timestamp
// ═════════════════════════════════════════════════════════════════════════════

static void test_invalid_timestamp_rejected() {
    {
        auto t = valid_tick(); t.timestamp_ns = 0;
        STREAM_CHECK(!tick_is_valid(t));
    }
    {
        auto t = valid_tick(); t.timestamp_ns = -1;
        STREAM_CHECK(!tick_is_valid(t));
    }
    {
        auto t = valid_tick(); t.timestamp_ns = -1000000000LL;
        STREAM_CHECK(!tick_is_valid(t));
    }
    // Positive timestamp accepted.
    {
        auto t = valid_tick(); t.timestamp_ns = 1;
        STREAM_CHECK(tick_is_valid(t));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// OHLC inconsistency rejected
// ═════════════════════════════════════════════════════════════════════════════

static void test_ohlc_inconsistency_rejected() {
    // high < low (inverted candle).
    {
        auto t = valid_tick();
        t.high = 90.0; t.low = 100.0;
        STREAM_CHECK(!tick_is_valid(t));
    }
    // high < open.
    {
        auto t = valid_tick();
        t.open = 110.0; t.high = 108.0; t.low = 100.0; t.close = 105.0;
        STREAM_CHECK(!tick_is_valid(t));
    }
    // high < close.
    {
        auto t = valid_tick();
        t.close = 110.0; t.high = 108.0;
        STREAM_CHECK(!tick_is_valid(t));
    }
    // low > open.
    {
        auto t = valid_tick();
        t.low = 102.0; t.open = 100.0; t.high = 105.0; t.close = 103.0;
        STREAM_CHECK(!tick_is_valid(t));
    }
    // low > close.
    {
        auto t = valid_tick();
        t.low = 104.0; t.close = 102.0;
        STREAM_CHECK(!tick_is_valid(t));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Boundary values accepted
// ═════════════════════════════════════════════════════════════════════════════

static void test_boundary_values_accepted() {
    // Just above zero.
    double eps = std::numeric_limits<double>::min(); // smallest positive normal
    OHLCVTick t;
    t.open = t.high = t.low = t.close = eps;
    t.volume = 0.0;
    t.timestamp_ns = 1;
    STREAM_CHECK(tick_is_valid(t));
}

// ═════════════════════════════════════════════════════════════════════════════
// regime_to_str
// ═════════════════════════════════════════════════════════════════════════════

static void test_regime_to_str() {
    STREAM_CHECK(std::strcmp(regime_to_str(Regime::TIMELIKE),  "TIMELIKE")  == 0);
    STREAM_CHECK(std::strcmp(regime_to_str(Regime::LIGHTLIKE), "LIGHTLIKE") == 0);
    STREAM_CHECK(std::strcmp(regime_to_str(Regime::SPACELIKE), "SPACELIKE") == 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("SRFM Stream — Tick Validation Tests\n");
    std::printf("====================================\n");

    STREAM_SUITE("valid tick accepted",          test_valid_tick_accepted);
    STREAM_SUITE("NaN fields rejected",          test_nan_fields_rejected);
    STREAM_SUITE("Inf fields rejected",          test_inf_fields_rejected);
    STREAM_SUITE("non-positive price rejected",  test_non_positive_price_rejected);
    STREAM_SUITE("negative volume rejected",     test_negative_volume_rejected);
    STREAM_SUITE("zero volume accepted",         test_zero_volume_accepted);
    STREAM_SUITE("invalid timestamp rejected",   test_invalid_timestamp_rejected);
    STREAM_SUITE("OHLC inconsistency rejected",  test_ohlc_inconsistency_rejected);
    STREAM_SUITE("boundary values accepted",     test_boundary_values_accepted);
    STREAM_SUITE("regime_to_str",                test_regime_to_str);

    return stream_test::report();
}
