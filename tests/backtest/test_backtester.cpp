#include <gtest/gtest.h>
#include "srfm/backtest.hpp"
#include "srfm/constants.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace srfm;
using namespace srfm::backtest;
using namespace srfm::constants;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build a bar series with constant β and alternating +/- signals.
static std::vector<BarData> make_bars(std::size_t n,
                                       double beta_val,
                                       double signal_magnitude = 1.0,
                                       double benchmark       = 0.001) {
    std::vector<BarData> bars(n);
    for (std::size_t i = 0; i < n; ++i) {
        bars[i] = {
            .raw_signal = (i % 2 == 0 ? signal_magnitude : -signal_magnitude),
            .beta       = BetaVelocity{beta_val},
            .benchmark  = benchmark,
        };
    }
    return bars;
}

/// Asset returns that follow the sign of the alternating signal pattern above.
/// sign(signal_t) × asset_ret = predictable P&L
static std::vector<double> make_aligned_returns(std::size_t n,
                                                  double per_bar_ret = 0.005) {
    std::vector<double> r(n);
    for (std::size_t i = 0; i < n; ++i) {
        // Signal at bar i is positive when i is even, so return also positive
        r[i] = (i % 2 == 0) ? per_bar_ret : -per_bar_ret;
    }
    // Strategy return = sign(signal) × asset_return = always positive
    return r;
}

// ─── Backtester::run — length mismatches ─────────────────────────────────────

TEST(Backtester_Run, EmptyBars_Nullopt) {
    Backtester bt;
    std::vector<BarData> bars;
    std::vector<double>  rets;
    EXPECT_FALSE(bt.run(bars, rets).has_value());
}

TEST(Backtester_Run, MismatchedLengths_Nullopt) {
    Backtester bt;
    auto bars = make_bars(10, 0.3);
    std::vector<double> rets(5, 0.001);
    EXPECT_FALSE(bt.run(bars, rets).has_value());
}

TEST(Backtester_Run, TooFewBars_Nullopt) {
    Backtester bt;
    auto bars = make_bars(1, 0.3);
    std::vector<double> rets = {0.001};
    EXPECT_FALSE(bt.run(bars, rets).has_value());
}

// ─── Backtester::run — structural correctness ────────────────────────────────

TEST(Backtester_Run, NewtonianRegime_SameAsRaw) {
    // β = 0 everywhere → γ = 1 → relativistic signals = raw signals
    // → relativistic metrics must equal raw metrics
    BacktestConfig cfg;
    cfg.effective_mass = 1.0;
    Backtester bt(cfg);

    // We need a non-trivial return series with variance
    const std::size_t N = 200;
    auto bars = make_bars(N, 0.0, 1.0);
    // Predictable positive strategy returns
    std::vector<double> asset_ret = make_aligned_returns(N, 0.003);

    auto result = bt.run(bars, asset_ret);
    ASSERT_TRUE(result.has_value());

    // In Newtonian limit (β=0, γ=1, m_eff=1), adjusted signal = raw signal
    // → strategy returns are identical → metrics must be equal
    EXPECT_NEAR(result->raw.sharpe_ratio,
                result->relativistic.sharpe_ratio, 1e-8);
    EXPECT_NEAR(result->raw.sortino_ratio,
                result->relativistic.sortino_ratio, 1e-8);
    EXPECT_NEAR(result->raw.max_drawdown,
                result->relativistic.max_drawdown, 1e-8);
}

TEST(Backtester_Run, RelativisticRegime_MetricsExist) {
    // β = 0.6 → γ = 1.25 → relativistic signals amplified
    // Run with a long series; metrics should be non-null
    const std::size_t N = 300;
    auto bars      = make_bars(N, 0.6, 1.0, 0.001);
    auto asset_ret = make_aligned_returns(N, 0.003);

    Backtester bt;
    auto result = bt.run(bars, asset_ret);
    ASSERT_TRUE(result.has_value());

    // Both sides should have finite metrics
    EXPECT_TRUE(std::isfinite(result->raw.sharpe_ratio));
    EXPECT_TRUE(std::isfinite(result->relativistic.sharpe_ratio));
    EXPECT_TRUE(std::isfinite(result->raw.max_drawdown));
    EXPECT_TRUE(std::isfinite(result->relativistic.max_drawdown));
}

TEST(Backtester_Run, MaxDrawdownNonNegative) {
    // MDD is always in [0, 1]
    const std::size_t N = 100;
    auto bars      = make_bars(N, 0.5, 1.0, 0.001);
    auto asset_ret = make_aligned_returns(N, 0.002);

    Backtester bt;
    auto result = bt.run(bars, asset_ret);
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result->raw.max_drawdown,        0.0);
    EXPECT_GE(result->relativistic.max_drawdown, 0.0);
    EXPECT_LE(result->raw.max_drawdown,        1.0);
    EXPECT_LE(result->relativistic.max_drawdown, 1.0);
}

// ─── Backtester::apply_corrections ───────────────────────────────────────────

TEST(Backtester_Corrections, EmptyBars_Nullopt) {
    Backtester bt;
    EXPECT_FALSE(bt.apply_corrections(std::span<const BarData>{}).has_value());
}

TEST(Backtester_Corrections, OutputLengthMatchesInput) {
    const std::size_t N = 50;
    Backtester bt;
    auto bars   = make_bars(N, 0.4, 1.0);
    auto result = bt.apply_corrections(bars);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->gamma_factors.size(),    N);
    EXPECT_EQ(result->adjusted_signals.size(), N);
}

TEST(Backtester_Corrections, GammaIsMonotonicallyIncreasingWithBeta) {
    // Higher β → higher γ: test with ascending β values
    std::vector<BarData> bars = {
        {1.0, BetaVelocity{0.1}, 0.0},
        {1.0, BetaVelocity{0.3}, 0.0},
        {1.0, BetaVelocity{0.6}, 0.0},
        {1.0, BetaVelocity{0.9}, 0.0},
    };
    Backtester bt;
    auto result = bt.apply_corrections(bars);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->gamma_factors.size(), 4u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_LT(result->gamma_factors[i], result->gamma_factors[i + 1])
            << "γ should increase with β at index " << i;
    }
}

TEST(Backtester_Corrections, KnownGamma_Beta06) {
    std::vector<BarData> bars = {{1.0, BetaVelocity{0.6}, 0.0}};
    Backtester bt;
    auto result = bt.apply_corrections(bars);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->gamma_factors[0], 1.25, 1e-8);
    EXPECT_NEAR(result->adjusted_signals[0], 1.25, 1e-8);  // γ × m_eff(1) × signal(1)
}

// ─── BacktestComparison::to_string ───────────────────────────────────────────

TEST(BacktestComparison_ToString, NonEmpty) {
    BacktestComparison cmp;
    cmp.raw          = {0.5,  0.6,  0.15, 0.3};
    cmp.relativistic = {0.65, 0.78, 0.12, 0.42};
    auto s = cmp.to_string();
    EXPECT_FALSE(s.empty());
    // Should contain relevant label strings
    EXPECT_NE(s.find("Sharpe"), std::string::npos);
    EXPECT_NE(s.find("Sortino"), std::string::npos);
    EXPECT_NE(s.find("Drawdown"), std::string::npos);
}

// ─── PerformanceMetrics::to_string ───────────────────────────────────────────

TEST(PerformanceMetrics_ToString, NonEmpty) {
    PerformanceMetrics m{1.2, 1.5, 0.08, 0.9};
    auto s = m.to_string();
    EXPECT_FALSE(s.empty());
}

// ─── End-to-end: relativistic lift on strongly trending signal ───────────────

TEST(Backtester_EndToEnd, RelativisticLiftOnTrendingSignal) {
    // Construct a scenario where γ > 1 consistently amplifies a winning signal.
    // β is constant at 0.7 → γ ≈ 1.4003.
    // A winning signal means strategy returns (sign(signal) × asset_return)
    // are consistently positive.  With m_eff = 1, γ scales the signal but
    // not the realised asset return — so the strategy P&L is the same per bar.
    // The signal just changes direction more decisively.
    // (The lift on Sharpe comes from the benchmark-normalised IR.)
    const std::size_t N = 500;
    auto bars      = make_bars(N, 0.7, 1.0, 0.001);
    auto asset_ret = make_aligned_returns(N, 0.004);

    Backtester bt;
    auto result = bt.run(bars, asset_ret);
    ASSERT_TRUE(result.has_value());

    // Both strategies should have valid (finite) metrics
    EXPECT_TRUE(std::isfinite(result->raw.sharpe_ratio));
    EXPECT_TRUE(std::isfinite(result->relativistic.sharpe_ratio));

    // Gamma-weighted IR should be amplified for relativistic strategy
    EXPECT_GE(result->relativistic.gamma_weighted_ir,
              result->raw.gamma_weighted_ir - 1e-6);
}

// ─── Mathematical identity: γ(β=0) = 1 preserved end-to-end ─────────────────

TEST(Backtester_MathIdentity, NewtonianLimit_GammaOne) {
    std::vector<BarData> bars(100);
    for (auto& b : bars) b = {1.0, BetaVelocity{0.0}, 0.001};

    Backtester bt;
    auto corr = bt.apply_corrections(bars);
    ASSERT_TRUE(corr.has_value());
    for (double g : corr->gamma_factors) {
        EXPECT_NEAR(g, 1.0, FLOAT_EPSILON);
    }
}

// ─── Property: adjusted_signal preserves sign of raw signal ──────────────────

TEST(Backtester_Property, AdjustedSignalPreservesSign) {
    // γ > 0 and m_eff > 0, so sign(adjusted) = sign(raw)
    std::vector<BarData> bars = {
        { 3.0, BetaVelocity{0.5}, 0.0},
        {-2.0, BetaVelocity{0.5}, 0.0},
        { 0.0, BetaVelocity{0.5}, 0.0},   // zero stays zero
    };
    Backtester bt;
    auto corr = bt.apply_corrections(bars);
    ASSERT_TRUE(corr.has_value());
    EXPECT_GT(corr->adjusted_signals[0], 0.0);
    EXPECT_LT(corr->adjusted_signals[1], 0.0);
    EXPECT_NEAR(corr->adjusted_signals[2], 0.0, FLOAT_EPSILON);
}
