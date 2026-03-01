/**
 * @file  test_beta_calculator.cpp
 * @brief Unit tests for BetaCalculator<N> — online market-velocity β estimator.
 *
 * Test structure:
 *   test_beta_pre_warmup_returns_zero       — β = 0 until N returns accumulated
 *   test_beta_warmup_exact_threshold        — precise warmup tick count
 *   test_beta_zero_returns_zero_beta        — flat price → zero log-returns → β=0
 *   test_beta_rising_price_positive_beta    — rising price → positive β
 *   test_beta_falling_price_negative_beta   — falling price → negative β
 *   test_beta_clamped_below_max_safe        — |β| < BETA_MAX_SAFE always
 *   test_beta_invalid_close_ignored         — NaN/negative close ignored
 *   test_beta_reset_reverts_state           — reset() clears counters
 *   test_beta_fix3_uses_last_3_returns      — sliding average property
 *   test_beta_c_market_scaling              — different c_market values
 *   test_beta_fix1_mode                     — BetaCalculator<1> single return
 *   test_beta_warmed_up_flag                — warmed_up() semantics
 *   test_beta_close_count_accessor          — close_count() tracks total ingested
 */

#include "srfm_stream_test.hpp"
#include "../../include/srfm/stream/beta_calculator.hpp"

#include <cmath>

using namespace srfm::stream;

static constexpr double EPS = 1e-9;

// ═════════════════════════════════════════════════════════════════════════════
// Pre-warmup returns zero
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_pre_warmup_returns_zero() {
    BetaCalculatorFix3 calc;

    // No updates → β = 0.
    STREAM_CHECK_NEAR(calc.beta(), 0.0, EPS);

    // 1 update → only 0 log-returns (need prev_close), still β = 0.
    calc.update(100.0);
    STREAM_CHECK_NEAR(calc.beta(), 0.0, EPS);

    // 2 updates → 1 log-return (need 3 for FIX-3), β = 0.
    calc.update(101.0);
    STREAM_CHECK_NEAR(calc.beta(), 0.0, EPS);

    // 3 updates → 2 log-returns, still below N=3.
    calc.update(102.0);
    STREAM_CHECK_NEAR(calc.beta(), 0.0, EPS);

    STREAM_CHECK(!calc.warmed_up());
}

// ═════════════════════════════════════════════════════════════════════════════
// Exact warmup threshold
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_warmup_exact_threshold() {
    BetaCalculatorFix3 calc;

    // Need N+1 = 4 close prices to accumulate N=3 log-returns.
    for (int i = 0; i < 3; ++i) {
        calc.update(100.0 + i);
        STREAM_CHECK(!calc.warmed_up());
    }
    calc.update(103.0); // 4th price → 3rd log-return → warmed up.
    STREAM_CHECK(calc.warmed_up());
}

// ═════════════════════════════════════════════════════════════════════════════
// Flat price → zero log-returns → β = 0
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_zero_returns_zero_beta() {
    BetaCalculatorFix3 calc;

    for (int i = 0; i < 10; ++i) calc.update(50.0);

    STREAM_CHECK(calc.warmed_up());
    STREAM_CHECK_NEAR(calc.beta(), 0.0, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// Rising price → positive β
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_rising_price_positive_beta() {
    BetaCalculatorFix3 calc;

    // Each tick: price increases by 1% → log-return ≈ 0.01.
    double price = 100.0;
    for (int i = 0; i < 10; ++i) {
        price *= 1.01;
        calc.update(price);
    }

    STREAM_CHECK(calc.warmed_up());
    STREAM_CHECK(calc.beta() > 0.0);
    STREAM_CHECK(calc.beta() < BETA_MAX_SAFE);
    STREAM_CHECK_FINITE(calc.beta());
}

// ═════════════════════════════════════════════════════════════════════════════
// Falling price → negative β
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_falling_price_negative_beta() {
    BetaCalculatorFix3 calc;

    double price = 100.0;
    for (int i = 0; i < 10; ++i) {
        price *= 0.99;
        calc.update(price);
    }

    STREAM_CHECK(calc.warmed_up());
    STREAM_CHECK(calc.beta() < 0.0);
    STREAM_CHECK(calc.beta() > -BETA_MAX_SAFE);
    STREAM_CHECK_FINITE(calc.beta());
}

// ═════════════════════════════════════════════════════════════════════════════
// |β| always < BETA_MAX_SAFE
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_clamped_below_max_safe() {
    // Extreme: price doubles every tick (huge log-return).
    BetaCalculatorFix3 calc;
    double price = 1.0;
    for (int i = 0; i < 20; ++i) {
        price *= 2.0;
        calc.update(price);
        if (calc.warmed_up()) {
            STREAM_CHECK(calc.beta() < BETA_MAX_SAFE);
            STREAM_CHECK(calc.beta() > -BETA_MAX_SAFE);
        }
    }

    // Extreme: price halves every tick.
    BetaCalculatorFix3 calc2;
    price = 1e15;
    for (int i = 0; i < 20; ++i) {
        price *= 0.5;
        calc2.update(price);
        if (calc2.warmed_up()) {
            STREAM_CHECK(calc2.beta() < BETA_MAX_SAFE);
            STREAM_CHECK(calc2.beta() > -BETA_MAX_SAFE);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Invalid close values are ignored
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_invalid_close_ignored() {
    BetaCalculatorFix3 calc;

    calc.update(100.0);

    const std::size_t count_before = calc.close_count();

    // NaN ignored.
    calc.update(std::numeric_limits<double>::quiet_NaN());
    STREAM_CHECK(calc.close_count() == count_before);

    // Zero ignored.
    calc.update(0.0);
    STREAM_CHECK(calc.close_count() == count_before);

    // Negative ignored.
    calc.update(-50.0);
    STREAM_CHECK(calc.close_count() == count_before);

    // Infinity ignored.
    calc.update(std::numeric_limits<double>::infinity());
    STREAM_CHECK(calc.close_count() == count_before);

    // A valid value is still accepted.
    calc.update(101.0);
    STREAM_CHECK(calc.close_count() == count_before + 1u);
}

// ═════════════════════════════════════════════════════════════════════════════
// reset() reverts all state
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_reset_reverts_state() {
    BetaCalculatorFix3 calc;

    for (int i = 0; i < 20; ++i) calc.update(100.0 + i);
    STREAM_CHECK(calc.warmed_up());

    calc.reset();

    STREAM_CHECK(!calc.warmed_up());
    STREAM_CHECK(calc.ret_count()   == 0u);
    STREAM_CHECK(calc.close_count() == 0u);
    STREAM_CHECK_NEAR(calc.beta(),    0.0, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// FIX-3 sliding average: verifies that β reflects last 3 returns
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_fix3_uses_last_3_returns() {
    // c_market=1 so β = mean log-return directly.
    BetaCalculatorFix3 calc{1.0};

    // Push 4 constant values (0 log-returns) to warm up.
    for (int i = 0; i < 4; ++i) calc.update(100.0);
    STREAM_CHECK(calc.warmed_up());
    STREAM_CHECK_NEAR(calc.beta(), 0.0, 1e-12);

    // Now push a price that gives log-return = ln(110/100) ≈ 0.0953.
    calc.update(110.0);
    // Last 3 returns: 0, 0, 0.0953 → mean ≈ 0.0318.
    double expected = std::log(110.0 / 100.0) / 3.0;
    STREAM_CHECK_NEAR(calc.beta(), expected, 1e-6);

    // Push 110 again (log-return = 0) twice.
    calc.update(110.0);
    calc.update(110.0);
    // Last 3 returns: 0.0953, 0, 0 → mean ≈ 0.0318 still.
    STREAM_CHECK_NEAR(calc.beta(), expected, 1e-6);

    // One more push: return slides out of window.
    calc.update(110.0);
    // Last 3 returns: 0, 0, 0 → β = 0.
    STREAM_CHECK_NEAR(calc.beta(), 0.0, 1e-10);
}

// ═════════════════════════════════════════════════════════════════════════════
// c_market scaling
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_c_market_scaling() {
    // c_market = 0.01 → β saturates more quickly.
    BetaCalculatorFix3 fast{0.01};
    // c_market = 0.10 → β takes larger moves to saturate.
    BetaCalculatorFix3 slow{0.10};

    double price = 100.0;
    for (int i = 0; i < 10; ++i) {
        price *= 1.005; // +0.5% per tick
        fast.update(price);
        slow.update(price);
    }

    // Both should be positive (rising price).
    STREAM_CHECK(fast.beta() > 0.0);
    STREAM_CHECK(slow.beta() > 0.0);

    // fast.c_market is smaller → normalised β is larger for the same raw velocity.
    STREAM_CHECK(fast.beta() > slow.beta());

    // β stays in bounds.
    STREAM_CHECK(fast.beta() < BETA_MAX_SAFE);
    STREAM_CHECK(slow.beta() < BETA_MAX_SAFE);
}

// ═════════════════════════════════════════════════════════════════════════════
// BetaCalculator<1> — single-return mode
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_fix1_mode() {
    BetaCalculator<1> calc{1.0}; // c_market = 1

    calc.update(100.0);  // prev_close set
    calc.update(110.0);  // first return = ln(1.1)

    STREAM_CHECK(calc.warmed_up());
    STREAM_CHECK_NEAR(calc.beta(), std::log(1.1), 1e-9);
}

// ═════════════════════════════════════════════════════════════════════════════
// warmed_up() semantics
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_warmed_up_flag() {
    BetaCalculatorFix3 calc;

    STREAM_CHECK(!calc.warmed_up());
    calc.update(100.0);
    STREAM_CHECK(!calc.warmed_up());
    calc.update(101.0);
    STREAM_CHECK(!calc.warmed_up());
    calc.update(102.0);
    STREAM_CHECK(!calc.warmed_up());
    calc.update(103.0);
    STREAM_CHECK(calc.warmed_up());
}

// ═════════════════════════════════════════════════════════════════════════════
// close_count() accessor tracks total ingested
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_close_count_accessor() {
    BetaCalculatorFix3 calc;

    for (int i = 0; i < 50; ++i) {
        STREAM_CHECK(calc.close_count() == static_cast<std::size_t>(i));
        calc.update(100.0 + i);
    }
    STREAM_CHECK(calc.close_count() == 50u);
}

// ═════════════════════════════════════════════════════════════════════════════
// Two independent instances track different price series
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_two_instances_independent() {
    BetaCalculatorFix3 rising;
    BetaCalculatorFix3 falling;

    double pr = 100.0, pf = 100.0;
    for (int i = 0; i < 10; ++i) {
        pr *= 1.02;  // rising: β > 0
        pf *= 0.98;  // falling: β < 0
        rising.update(pr);
        falling.update(pf);
    }

    STREAM_CHECK(rising.warmed_up());
    STREAM_CHECK(falling.warmed_up());
    STREAM_CHECK(rising.beta() > 0.0);
    STREAM_CHECK(falling.beta() < 0.0);
}

// ═════════════════════════════════════════════════════════════════════════════
// c_market() accessor returns configured value
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_c_market_accessor() {
    BetaCalculatorFix3 a{0.01};
    BetaCalculatorFix3 b{0.05};
    BetaCalculatorFix3 c;

    STREAM_CHECK_NEAR(a.c_market(), 0.01,            EPS);
    STREAM_CHECK_NEAR(b.c_market(), 0.05,            EPS);
    STREAM_CHECK_NEAR(c.c_market(), DEFAULT_C_MARKET, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// Default c_market guards against invalid construction
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_invalid_c_market_falls_back_to_default() {
    // Zero c_market → should fall back to DEFAULT_C_MARKET.
    BetaCalculatorFix3 calc_zero{0.0};
    STREAM_CHECK_NEAR(calc_zero.c_market(), DEFAULT_C_MARKET, EPS);

    // Negative c_market → fall back.
    BetaCalculatorFix3 calc_neg{-1.0};
    STREAM_CHECK_NEAR(calc_neg.c_market(), DEFAULT_C_MARKET, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// After reset, re-warmup from scratch works correctly
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_re_warmup_after_reset() {
    BetaCalculatorFix3 calc{1.0}; // c_market=1 for exact comparison

    // First run: steady +1% price.
    for (int i = 0; i < 10; ++i) {
        calc.update(100.0 * std::pow(1.01, i));
    }
    double beta_run1 = calc.beta();
    STREAM_CHECK(beta_run1 > 0.0);

    // Reset and run the same sequence again.
    calc.reset();
    for (int i = 0; i < 10; ++i) {
        calc.update(100.0 * std::pow(1.01, i));
    }
    double beta_run2 = calc.beta();

    // Deterministic: same input → same output.
    STREAM_CHECK_NEAR(beta_run1, beta_run2, 1e-12);
}

// ═════════════════════════════════════════════════════════════════════════════
// Ret count saturates at N after N+ updates
// ═════════════════════════════════════════════════════════════════════════════

static void test_beta_ret_count_saturates() {
    BetaCalculatorFix3 calc;

    for (int i = 0; i < 20; ++i) {
        calc.update(100.0 + i);
    }

    // ret_count should be exactly N=3 (the window size).
    STREAM_CHECK(calc.ret_count() == 3u);
}

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("SRFM Stream — BetaCalculator Tests\n");
    std::printf("====================================\n");

    STREAM_SUITE("pre-warmup returns zero",         test_beta_pre_warmup_returns_zero);
    STREAM_SUITE("exact warmup threshold",          test_beta_warmup_exact_threshold);
    STREAM_SUITE("flat price → β = 0",              test_beta_zero_returns_zero_beta);
    STREAM_SUITE("rising price → β > 0",            test_beta_rising_price_positive_beta);
    STREAM_SUITE("falling price → β < 0",           test_beta_falling_price_negative_beta);
    STREAM_SUITE("|β| < BETA_MAX_SAFE always",      test_beta_clamped_below_max_safe);
    STREAM_SUITE("invalid close ignored",           test_beta_invalid_close_ignored);
    STREAM_SUITE("reset() reverts state",           test_beta_reset_reverts_state);
    STREAM_SUITE("FIX-3 sliding average",           test_beta_fix3_uses_last_3_returns);
    STREAM_SUITE("c_market scaling",                test_beta_c_market_scaling);
    STREAM_SUITE("FIX-1 single-return mode",        test_beta_fix1_mode);
    STREAM_SUITE("warmed_up() semantics",           test_beta_warmed_up_flag);
    STREAM_SUITE("close_count() accessor",          test_beta_close_count_accessor);
    STREAM_SUITE("two instances independent",       test_beta_two_instances_independent);
    STREAM_SUITE("c_market() accessor",             test_beta_c_market_accessor);
    STREAM_SUITE("invalid c_market → default",      test_beta_invalid_c_market_falls_back_to_default);
    STREAM_SUITE("re-warmup after reset",           test_beta_re_warmup_after_reset);
    STREAM_SUITE("ret_count saturates at N",        test_beta_ret_count_saturates);

    return stream_test::report();
}
