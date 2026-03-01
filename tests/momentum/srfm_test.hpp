#pragma once
/**
 * @file  srfm_test.hpp
 * @brief Minimal, dependency-free unit-test runner for SRFM C++20 modules.
 *
 * Usage
 * -----
 *   #include "srfm_test.hpp"
 *   void test_foo() { SRFM_CHECK(1 + 1 == 2); }
 *   int main() {
 *       SRFM_SUITE("Foo", test_foo);
 *       return srfm_test::report();
 *   }
 */

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace srfm_test {

struct Result {
    bool        passed;
    std::string expr;
    std::string location;
};

inline std::vector<Result>& all_results() {
    static std::vector<Result> r;
    return r;
}

inline void record(bool cond,
                   const char* expr,
                   const char* file,
                   int line) noexcept {
    all_results().push_back({
        cond,
        expr,
        std::string(file) + ":" + std::to_string(line)
    });
}

/// Print summary and return 0 (pass) or 1 (fail).
inline int report() noexcept {
    int pass = 0;
    int fail = 0;
    for (const auto& r : all_results()) {
        if (r.passed) {
            ++pass;
        } else {
            ++fail;
            std::fprintf(stderr, "  FAIL  %s  @ %s\n",
                         r.expr.c_str(), r.location.c_str());
        }
    }
    std::printf("\n  Results: %d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}

} // namespace srfm_test

// ── Assertion macros ──────────────────────────────────────────────────────────

/// Assert that cond is true.
#define SRFM_CHECK(cond) \
    srfm_test::record(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

/// Assert that |a - b| < eps.
#define SRFM_CHECK_NEAR(a, b, eps) \
    srfm_test::record(std::abs((a) - (b)) < (eps), \
                      #a " ~= " #b, __FILE__, __LINE__)

/// Assert that optional has a value.
#define SRFM_HAS_VALUE(opt) \
    srfm_test::record((opt).has_value(), #opt ".has_value()", __FILE__, __LINE__)

/// Assert that optional is empty.
#define SRFM_NO_VALUE(opt) \
    srfm_test::record(!(opt).has_value(), "!" #opt ".has_value()", __FILE__, __LINE__)

/// Run a test function and label it.
#define SRFM_SUITE(name, fn) \
    do { \
        std::printf("[suite] %s\n", (name)); \
        (fn)(); \
    } while (false)
