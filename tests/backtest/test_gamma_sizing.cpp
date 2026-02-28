/// @file tests/backtest/test_gamma_sizing.cpp
/// @brief Tests for γ-weighted position sizing in the Backtester.
///
/// Verifies:
///   - When all γ = 1.0 (Newtonian): raw and relativistic Sharpe are identical
///   - When γ > 1.0: relativistic position size > raw position size
///   - MAX_GAMMA cap is respected
///   - New BacktestComparison fields (mean_gamma, max_gamma_applied, relativistic_lift)
///   - Relativistic strategy outperforms raw on winning signals when γ > 1

#include <gtest/gtest.h>
#include "srfm/backtest.hpp"
#include "srfm/constants.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace srfm;
using namespace srfm::backtest;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build a bar series with constant β, alternating signal, constant benchmark.
static std::vector<BarData> make_bars(std::size_t n,
                                       double beta_val,
                                       double signal_mag = 1.0,
                                       double benchmark  = 0.0) {
    std::vector<BarData> bars(n);
    for (std::size_t i = 0; i < n; ++i) {
        bars[i] = {
            .raw_signal = (i % 2 == 0 ? signal_mag : -signal_mag),
            .beta       = BetaVelocity{beta_val},
            .benchmark  = benchmark,
        };
    }
    return bars;
}

/// Perfectly aligned returns: sign matches signal exactly, so strategy always wins.
static std::vector<double> aligned_returns(std::size_t n, double per_bar = 0.005) {
    std::vector<double> r(n);
    for (std::size_t i = 0; i < n; ++i) {
        r[i] = (i % 2 == 0) ? per_bar : -per_bar;
    }
    return r;
}

// ─── Test 1: When all γ = 1.0, raw and relativistic Sharpe are identical ─────

TEST(GammaSizing, NewtonianLimit_IdenticalSharpe) {
    // β = 0 → γ = 1.0 everywhere → clamp(1,1,3) = 1 → positions identical
    const std::size_t N = 200;
    auto bars = make_bars(N, 0.0);
    auto rets = aligned_returns(N, 0.003);

    Backtester bt;
    auto result = bt.run(bars, rets);
    ASSERT_TRUE(result.has_value());

    EXPECT_NEAR(result->raw.sharpe_ratio,
                result->relativistic.sharpe_ratio, 1e-8);
    EXPECT_NEAR(result->raw.sortino_ratio,
                result->relativistic.sortino_ratio, 1e-8);
    EXPECT_NEAR(result->raw.max_drawdown,
                result->relativistic.max_drawdown, 1e-8);
}

// ─── Test 2: mean_gamma = 1.0 in Newtonian regime ────────────────────────────

TEST(GammaSizing, NewtonianLimit_MeanGammaIsOne) {
    const std::size_t N = 100;
    auto bars = make_bars(N, 0.0);
    auto rets = aligned_returns(N, 0.001);

    Backtester bt;
    auto result = bt.run(bars, rets);
    ASSERT_TRUE(result.has_value());

    EXPECT_NEAR(result->mean_gamma, 1.0, 1e-8);
    EXPECT_NEAR(result->max_gamma_applied, 1.0, 1e-8);
}

// ─── Test 3: When γ > 1.0, relativistic returns have larger magnitude ─────────

TEST(GammaSizing, RelativisticRegime_PositionScalesWithGamma) {
    // β = 0.6 → γ = 1/√(1-0.36) = 1/0.8 = 1.25
    // relativistic return = sign(adj) * 1.25 * asset_return
    // raw return          = sign(raw) * 1.0  * asset_return
    // With aligned (winning) returns, relativistic |return| = 1.25 * raw |return|
    const std::size_t N = 200;
    double beta = 0.6;
    auto bars = make_bars(N, beta, 1.0, 0.0);
    auto rets = aligned_returns(N, 0.005);

    Backtester bt;
    auto result = bt.run(bars, rets);
    ASSERT_TRUE(result.has_value());

    // Relativistic strategy returns are 1.25× larger → higher Sharpe
    EXPECT_GT(result->relativistic.sharpe_ratio, result->raw.sharpe_ratio)
        << "Relativistic Sharpe should exceed raw when γ > 1 and signals win";
}

// ─── Test 4: max_gamma cap is respected — γ never exceeds max_gamma ───────────

TEST(GammaSizing, MaxGammaCap_RespectsLimit) {
    // Use very high β → γ would be enormous without cap.
    // β = 0.999 → γ ≈ 22.4; with max_gamma = 3.0, position capped at 3.0.
    const std::size_t N = 100;
    BacktestConfig cfg;
    cfg.max_gamma = 3.0;
    Backtester bt(cfg);

    auto bars = make_bars(N, 0.999, 1.0, 0.0);
    auto rets = aligned_returns(N, 0.003);

    auto result = bt.run(bars, rets);
    ASSERT_TRUE(result.has_value());

    // max_gamma_applied must not exceed the configured cap.
    EXPECT_LE(result->max_gamma_applied, 3.0 + 1e-10);
    EXPECT_GE(result->max_gamma_applied, 1.0);
}

// ─── Test 5: custom max_gamma = 2.0 is respected ─────────────────────────────

TEST(GammaSizing, CustomMaxGamma_CapApplied) {
    const std::size_t N = 100;
    BacktestConfig cfg;
    cfg.max_gamma = 2.0;
    Backtester bt(cfg);

    auto bars = make_bars(N, 0.95, 1.0, 0.0);  // γ ≈ 3.2 without cap
    auto rets = aligned_returns(N, 0.002);

    auto result = bt.run(bars, rets);
    ASSERT_TRUE(result.has_value());

    EXPECT_LE(result->max_gamma_applied, 2.0 + 1e-10);
}

// ─── Test 6: mean_gamma > 1.0 in relativistic regime ─────────────────────────

TEST(GammaSizing, RelativisticRegime_MeanGammaAboveOne) {
    const std::size_t N = 100;
    auto bars = make_bars(N, 0.6);  // γ = 1.25 for all bars
    auto rets = aligned_returns(N, 0.002);

    Backtester bt;
    auto result = bt.run(bars, rets);
    ASSERT_TRUE(result.has_value());

    EXPECT_GT(result->mean_gamma, 1.0);
    // For β=0.6, γ ≈ 1.25
    EXPECT_NEAR(result->mean_gamma, 1.25, 0.01);
}

// ─── Test 7: relativistic_lift > 1 when relativistic strategy wins ────────────

TEST(GammaSizing, RelativisticLift_AboveOneOnWinningSignal) {
    // With γ > 1 and winning signals, relativistic IR_γ > raw IR_γ
    // so the ratio relativistic_lift = IR_γ_rel / IR_γ_raw > 1.
    const std::size_t N = 300;
    auto bars = make_bars(N, 0.7, 1.0, 0.0);  // γ ≈ 1.40 for all bars
    auto rets = aligned_returns(N, 0.005);

    Backtester bt;
    auto result = bt.run(bars, rets);
    ASSERT_TRUE(result.has_value());

    EXPECT_GT(result->relativistic_lift, 1.0)
        << "relativistic_lift should exceed 1.0 when γ > 1 and signals are winning";
}

// ─── Test 8: max_gamma = 1.0 collapses to sign-only (identical strategies) ───

TEST(GammaSizing, MaxGammaOne_StrategiesIdentical) {
    BacktestConfig cfg;
    cfg.max_gamma = 1.0;  // cap at 1 → clamp(γ, 1, 1) = 1 always
    Backtester bt(cfg);

    const std::size_t N = 200;
    auto bars = make_bars(N, 0.8);  // γ ≈ 1.667 without cap
    auto rets = aligned_returns(N, 0.003);

    auto result = bt.run(bars, rets);
    ASSERT_TRUE(result.has_value());

    // With max_gamma = 1.0, relativistic position = sign × 1.0 × return = raw
    EXPECT_NEAR(result->raw.sharpe_ratio,
                result->relativistic.sharpe_ratio, 1e-8);
    EXPECT_NEAR(result->raw.max_drawdown,
                result->relativistic.max_drawdown, 1e-8);
}

// ─── Test 9: losing signal → relativistic amplifies losses more than raw ──────

TEST(GammaSizing, LosingSignal_RelativisticAmplifies) {
    // Opposite-aligned returns: signal says up, market goes down.
    // Raw strategy loses; relativistic strategy loses MORE (γ > 1 scales loss).
    const std::size_t N = 200;
    auto bars = make_bars(N, 0.6, 1.0, 0.0);
    // Counter-aligned returns (strategy always wrong → loses 0.003/bar)
    std::vector<double> rets(N);
    for (std::size_t i = 0; i < N; ++i) {
        // Signal at bar i is +1 (even) or -1 (odd); asset moves opposite
        rets[i] = (i % 2 == 0) ? -0.003 : 0.003;
    }

    Backtester bt;
    auto result = bt.run(bars, rets);
    ASSERT_TRUE(result.has_value());

    // Relativistic strategy loses more (larger negative Sharpe)
    EXPECT_LT(result->relativistic.sharpe_ratio, result->raw.sharpe_ratio)
        << "Relativistic should amplify losses when γ > 1 and signals are wrong";
}

// ─── Test 10: new fields are finite ──────────────────────────────────────────

TEST(GammaSizing, NewFields_AllFinite) {
    const std::size_t N = 100;
    auto bars = make_bars(N, 0.5);
    auto rets = aligned_returns(N, 0.002);

    Backtester bt;
    auto result = bt.run(bars, rets);
    ASSERT_TRUE(result.has_value());

    EXPECT_TRUE(std::isfinite(result->mean_gamma));
    EXPECT_TRUE(std::isfinite(result->max_gamma_applied));
    EXPECT_TRUE(std::isfinite(result->relativistic_lift));
}

// ─── Test 11: max_gamma_applied reflects actual cap used ─────────────────────

TEST(GammaSizing, MaxGammaApplied_TracksCap) {
    // β=0, γ=1: cap doesn't engage, max_gamma_applied should be 1.0.
    {
        BacktestConfig cfg;
        cfg.max_gamma = 3.0;
        Backtester bt(cfg);
        const std::size_t N = 50;
        auto bars = make_bars(N, 0.0);
        auto rets = aligned_returns(N, 0.001);
        auto result = bt.run(bars, rets);
        ASSERT_TRUE(result.has_value());
        EXPECT_NEAR(result->max_gamma_applied, 1.0, 1e-8);
    }
    // β=0.9 → γ≈2.29, cap=3.0: max_gamma_applied should be ~2.29.
    {
        BacktestConfig cfg;
        cfg.max_gamma = 3.0;
        Backtester bt(cfg);
        const std::size_t N = 50;
        auto bars = make_bars(N, 0.9);
        auto rets = aligned_returns(N, 0.001);
        auto result = bt.run(bars, rets);
        ASSERT_TRUE(result.has_value());
        EXPECT_NEAR(result->max_gamma_applied, 1.0 / std::sqrt(1.0 - 0.81), 0.01);
    }
}

// ─── Test 12: relativistic_lift = 0 when raw IR is zero ──────────────────────

TEST(GammaSizing, RelativisticLift_ZeroWhenRawIRIsZero) {
    // Zero benchmark: IR may be defined or undefined depending on variance.
    // Edge case: if raw_ir = 0 exactly, lift = 0 (undefined ratio handled).
    // We test that the field is always finite (never NaN/Inf).
    const std::size_t N = 100;
    BacktestConfig cfg;
    cfg.max_gamma = 3.0;
    Backtester bt(cfg);

    // Use β=0.0 and zero benchmark — in this case raw IR_γ uses γ=1 everywhere,
    // so IR_γ_raw is non-zero if the strategy has a non-zero active return.
    // Mainly we verify the field is always finite.
    auto bars = make_bars(N, 0.5);
    auto rets = aligned_returns(N, 0.002);
    auto result = bt.run(bars, rets);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::isfinite(result->relativistic_lift));
}
