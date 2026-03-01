#pragma once
/**
 * @file  srfm_stream_test.hpp
 * @brief Minimal, dependency-free unit-test runner for SRFM streaming C++20 modules.
 *
 * Extended from tests/momentum/srfm_test.hpp with additional helpers suited to
 * the streaming pipeline (range checks, signal assertions, counter checks).
 *
 * Usage:
 *   #include "srfm_stream_test.hpp"
 *   void test_foo() { STREAM_CHECK(1 + 1 == 2); }
 *   int main() {
 *       STREAM_SUITE("Foo", test_foo);
 *       return stream_test::report();
 *   }
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Pull in the tick definition for helper factories.
#include "../../include/srfm/stream/tick.hpp"
#include "../../include/srfm/stream/stream_signal.hpp"

namespace stream_test {

struct Result {
    bool        passed;
    std::string expr;
    std::string location;
};

inline std::vector<Result>& all_results() {
    static std::vector<Result> r;
    return r;
}

inline void record(bool cond, const char* expr,
                   const char* file, int line) noexcept {
    all_results().push_back({cond, expr,
        std::string(file) + ":" + std::to_string(line)});
}

inline int report() noexcept {
    int pass = 0, fail = 0;
    for (const auto& r : all_results()) {
        if (r.passed) { ++pass; }
        else {
            ++fail;
            std::fprintf(stderr, "  FAIL  %s  @ %s\n",
                         r.expr.c_str(), r.location.c_str());
        }
    }
    std::printf("\n  Results: %d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}

} // namespace stream_test

// ── Core assertion macros ──────────────────────────────────────────────────────

#define STREAM_CHECK(cond) \
    stream_test::record(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

#define STREAM_CHECK_NEAR(a, b, eps) \
    stream_test::record(std::abs((a) - (b)) < (eps), \
                        #a " ~= " #b, __FILE__, __LINE__)

#define STREAM_CHECK_GT(a, b) \
    stream_test::record((a) > (b), #a " > " #b, __FILE__, __LINE__)

#define STREAM_CHECK_GE(a, b) \
    stream_test::record((a) >= (b), #a " >= " #b, __FILE__, __LINE__)

#define STREAM_CHECK_LT(a, b) \
    stream_test::record((a) < (b), #a " < " #b, __FILE__, __LINE__)

#define STREAM_CHECK_LE(a, b) \
    stream_test::record((a) <= (b), #a " <= " #b, __FILE__, __LINE__)

#define STREAM_CHECK_FINITE(v) \
    stream_test::record(std::isfinite(v), #v " is finite", __FILE__, __LINE__)

#define STREAM_HAS_VALUE(opt) \
    stream_test::record((opt).has_value(), #opt ".has_value()", __FILE__, __LINE__)

#define STREAM_NO_VALUE(opt) \
    stream_test::record(!(opt).has_value(), "!" #opt ".has_value()", __FILE__, __LINE__)

#define STREAM_SUITE(name, fn) \
    do { \
        std::printf("[suite] %s\n", (name)); \
        (fn)(); \
    } while (false)

// ── Tick factory helpers ───────────────────────────────────────────────────────

namespace stream_test {

/**
 * @brief Construct a valid OHLCVTick with a given close price.
 *
 * Sets open = low = close - 0.01 (floored at 0.01) and high = close + 0.01
 * to satisfy OHLC consistency.
 */
inline srfm::stream::OHLCVTick make_tick(double close,
                                          double volume = 1000.0,
                                          std::int64_t ts = 1000000000LL) noexcept {
    srfm::stream::OHLCVTick t;
    t.close        = close;
    t.open         = close - 0.01 > 0.0 ? close - 0.01 : close;
    t.high         = close + 0.01;
    t.low          = t.open;
    t.volume       = volume;
    t.timestamp_ns = ts;
    return t;
}

/**
 * @brief Construct a sequence of N ticks with linearly increasing close prices.
 *
 * @param start  First close price.
 * @param step   Price increment per tick.
 * @param n      Number of ticks.
 */
inline std::vector<srfm::stream::OHLCVTick>
make_tick_sequence(double start, double step, std::size_t n) noexcept {
    std::vector<srfm::stream::OHLCVTick> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        double close = start + step * static_cast<double>(i);
        v.push_back(make_tick(close, 1000.0,
                              static_cast<std::int64_t>(1000000000LL + i)));
    }
    return v;
}

} // namespace stream_test
