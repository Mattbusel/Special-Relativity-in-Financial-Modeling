#pragma once
/**
 * @file  srfm_test_n_asset.hpp
 * @brief Test infrastructure for Stage 4 N-Asset Manifold tests.
 *
 * Re-exports all macros from srfm_test.hpp and adds a variadic SRFM_SUITE
 * overload that accepts multiple test functions:
 *
 *   SRFM_SUITE("My Suite", test_a, test_b, test_c);
 *
 * All original SRFM_CHECK / SRFM_HAS_VALUE / SRFM_NO_VALUE / SRFM_CHECK_NEAR
 * macros remain available.
 */

#include "../../tests/momentum/srfm_test.hpp"

#include <cstdio>

// Variadic SRFM_SUITE: accepts one or more void() functions.
// Uses a C++17 fold-expression to call each function in order.
namespace srfm_test_n_asset_detail {
    template<typename... Fns>
    inline void run_suite(const char* name, Fns... fns) {
        std::printf("[suite] %s\n", name);
        (fns(), ...);
    }
} // namespace srfm_test_n_asset_detail

// Override SRFM_SUITE to accept variadic function arguments.
#undef SRFM_SUITE
#define SRFM_SUITE(name, ...) \
    srfm_test_n_asset_detail::run_suite((name), __VA_ARGS__)
