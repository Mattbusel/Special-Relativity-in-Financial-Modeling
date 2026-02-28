#include <gtest/gtest.h>
#include "srfm/backtest.hpp"
#include "srfm/constants.hpp"

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

using namespace srfm;
using namespace srfm::backtest;
using namespace srfm::constants;

// ─── Helpers ─────────────────────────────────────────────────────────────────

static std::vector<double> make_constant_returns(std::size_t n, double val) {
    return std::vector<double>(n, val);
}

static std::vector<double> make_returns_with_mean_stddev(
        double target_mean, double target_stddev, std::size_t n) {
    // Alternating series: half at (mean + stddev), half at (mean - stddev)
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = (i % 2 == 0) ? target_mean + target_stddev
                              : target_mean - target_stddev;
    }
    return v;
}

// ─── Sharpe: Basic Correctness ────────────────────────────────────────────────

TEST(PerformanceCalculator_Sharpe, ZeroRiskFree_ZeroMean_ReturnsZero) {
    // All zero returns → mean = 0, Sharpe = 0
    auto returns = make_constant_returns(100, 0.0);
    auto result  = PerformanceCalculator::sharpe(returns, 0.0, 1.0);
    // σ = 0 → nullopt (degenerate), OR mean/σ undefined
    // Here all equal returns → σ = 0 → degenerate
    EXPECT_FALSE(result.has_value())
        << "Constant-zero returns have zero variance — Sharpe undefined";
}

TEST(PerformanceCalculator_Sharpe, ConstantPositiveReturns_ZeroVariance_Nullopt) {
    // Constant positive returns: σ = 0 → Sharpe undefined
    auto returns = make_constant_returns(50, 0.01);
    auto result  = PerformanceCalculator::sharpe(returns, 0.0, 1.0);
    EXPECT_FALSE(result.has_value());
}

TEST(PerformanceCalculator_Sharpe, KnownValues_CorrectAnnualised) {
    // Daily returns with known mean and std.
    // Daily mean = 0.001, daily std = 0.01
    // Annualised Sharpe (252 days) = (0.001 / 0.01) * sqrt(252) ≈ 1.5874
    auto returns = make_returns_with_mean_stddev(0.001, 0.01, 500);
    auto result  = PerformanceCalculator::sharpe(returns, 0.0, ANNUALISATION_FACTOR);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.001 / 0.01 * std::sqrt(252.0), 0.05);
}

TEST(PerformanceCalculator_Sharpe, RiskFreeSubtracted) {
    // mean = 0.001, r_f = 0.0005, std = 0.01
    // Expected: (0.001 - 0.0005) / 0.01 * sqrt(252) ≈ 0.7937
    auto returns = make_returns_with_mean_stddev(0.001, 0.01, 500);
    auto result  = PerformanceCalculator::sharpe(returns, 0.0005, ANNUALISATION_FACTOR);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.0005 / 0.01 * std::sqrt(252.0), 0.05);
}

TEST(PerformanceCalculator_Sharpe, TooFewReturns_Nullopt) {
    // Need at least MIN_RETURN_SERIES_LENGTH = 2
    auto result = PerformanceCalculator::sharpe(std::span<const double>{}, 0.0, 1.0);
    EXPECT_FALSE(result.has_value());

    std::vector<double> one = {0.01};
    EXPECT_FALSE(PerformanceCalculator::sharpe(one, 0.0, 1.0).has_value());
}

TEST(PerformanceCalculator_Sharpe, NaNInput_Nullopt) {
    std::vector<double> returns = {0.01, std::numeric_limits<double>::quiet_NaN(), 0.02};
    EXPECT_FALSE(PerformanceCalculator::sharpe(returns, 0.0, 1.0).has_value());
}

TEST(PerformanceCalculator_Sharpe, InfInput_Nullopt) {
    std::vector<double> returns = {0.01, std::numeric_limits<double>::infinity(), 0.02};
    EXPECT_FALSE(PerformanceCalculator::sharpe(returns, 0.0, 1.0).has_value());
}

TEST(PerformanceCalculator_Sharpe, NegativeMean_NegativeSharpe) {
    // Daily losses → negative Sharpe
    auto returns = make_returns_with_mean_stddev(-0.001, 0.01, 500);
    auto result  = PerformanceCalculator::sharpe(returns, 0.0, ANNUALISATION_FACTOR);
    ASSERT_TRUE(result.has_value());
    EXPECT_LT(*result, 0.0);
}

// ─── Sortino ─────────────────────────────────────────────────────────────────

TEST(PerformanceCalculator_Sortino, NoDownsideReturns_Nullopt) {
    // All returns above threshold → σ_down = 0 → Sortino undefined
    auto returns = make_constant_returns(100, 0.01);
    // r_f = 0, all returns > 0 → no downside
    auto result = PerformanceCalculator::sortino(returns, 0.0, 1.0);
    EXPECT_FALSE(result.has_value());
}

TEST(PerformanceCalculator_Sortino, MixedReturns_HigherThanSharpe) {
    // Asymmetric returns: mostly positive, a few small negatives
    // Sortino should be >= Sharpe for positively-skewed distributions
    std::vector<double> returns;
    for (int i = 0; i < 100; ++i) {
        returns.push_back(i % 10 == 0 ? -0.005 : 0.002);
    }
    auto sh  = PerformanceCalculator::sharpe(returns, 0.0, ANNUALISATION_FACTOR);
    auto so  = PerformanceCalculator::sortino(returns, 0.0, ANNUALISATION_FACTOR);
    ASSERT_TRUE(sh.has_value());
    ASSERT_TRUE(so.has_value());
    // Sortino uses only downside vol (< total vol), so |Sortino| >= |Sharpe|
    EXPECT_GE(std::abs(*so), std::abs(*sh) * 0.95);  // allow 5% tolerance
}

TEST(PerformanceCalculator_Sortino, TooFewReturns_Nullopt) {
    std::vector<double> one = {-0.01};
    EXPECT_FALSE(PerformanceCalculator::sortino(one, 0.0, 1.0).has_value());
}

TEST(PerformanceCalculator_Sortino, NaNInput_Nullopt) {
    std::vector<double> r = {0.01, std::numeric_limits<double>::quiet_NaN(), -0.01};
    EXPECT_FALSE(PerformanceCalculator::sortino(r, 0.0, 1.0).has_value());
}

// ─── Max Drawdown ─────────────────────────────────────────────────────────────

TEST(PerformanceCalculator_MaxDrawdown, EmptyInput_Nullopt) {
    EXPECT_FALSE(PerformanceCalculator::max_drawdown(
        std::span<const double>{}).has_value());
}

TEST(PerformanceCalculator_MaxDrawdown, MonotonicallyRising_ZeroDrawdown) {
    // All positive returns → equity curve always at new highs → MDD = 0
    auto returns = make_constant_returns(100, 0.01);
    auto result  = PerformanceCalculator::max_drawdown(returns);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.0, FLOAT_EPSILON);
}

TEST(PerformanceCalculator_MaxDrawdown, SingleDropThenRecover_KnownMDD) {
    // Equity: 1 → 1.1 → 0.9 → 1.1
    // Drawdown at trough: (1.1 − 0.9) / 1.1 ≈ 0.1818
    // Returns: +0.1, −0.1818..., +0.2222...
    std::vector<double> returns = {
        0.10,                       // 1.0 → 1.10
        (0.90 - 1.10) / 1.10,       // 1.10 → 0.90  (approx -0.1818)
        (1.10 - 0.90) / 0.90        // 0.90 → 1.10  (approx +0.2222)
    };
    auto result = PerformanceCalculator::max_drawdown(returns);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, (1.10 - 0.90) / 1.10, 1e-6);
}

TEST(PerformanceCalculator_MaxDrawdown, AllNegativeReturns_LargeDrawdown) {
    // Equity falls every bar → MDD measured from start to end
    std::vector<double> returns(20, -0.01);
    auto result = PerformanceCalculator::max_drawdown(returns);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.0);
    EXPECT_LE(*result, 1.0);  // drawdown is in [0,1]
}

TEST(PerformanceCalculator_MaxDrawdown, SingleReturn_ZeroDrawdown) {
    std::vector<double> r = {0.05};
    auto result = PerformanceCalculator::max_drawdown(r);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.0, FLOAT_EPSILON);
}

TEST(PerformanceCalculator_MaxDrawdown, MaxDrawdownInRange) {
    // MDD must always be in [0, 1]
    std::vector<double> r;
    for (int i = 0; i < 200; ++i) {
        r.push_back((i % 3 == 0) ? -0.05 : 0.02);
    }
    auto result = PerformanceCalculator::max_drawdown(r);
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(*result, 0.0);
    EXPECT_LE(*result, 1.0);
}

// ─── Gamma-Weighted IR ────────────────────────────────────────────────────────

TEST(PerformanceCalculator_GammaIR, EmptyInput_Nullopt) {
    EXPECT_FALSE(PerformanceCalculator::gamma_weighted_ir(
        std::span<const double>{},
        std::span<const double>{},
        std::span<const double>{}).has_value());
}

TEST(PerformanceCalculator_GammaIR, MismatchedLengths_Nullopt) {
    std::vector<double> a = {0.01, 0.02, 0.03};
    std::vector<double> b = {0.005, 0.01};
    std::vector<double> g = {1.0, 1.1, 1.2};
    EXPECT_FALSE(PerformanceCalculator::gamma_weighted_ir(a, b, g).has_value());
}

TEST(PerformanceCalculator_GammaIR, UnitGammas_EquivalentToClassicIR) {
    // γ = 1 everywhere → IR_γ = mean(active) / σ(active) = classic IR
    std::vector<double> strat, bench, gammas;
    for (int i = 0; i < 100; ++i) {
        strat.push_back(0.001 + (i % 2 == 0 ? 0.005 : -0.005));
        bench.push_back(0.0005);
        gammas.push_back(1.0);
    }
    auto ir_gamma  = PerformanceCalculator::gamma_weighted_ir(strat, bench, gammas);
    // Classic IR
    std::vector<double> active(strat.size());
    for (std::size_t i = 0; i < strat.size(); ++i) active[i] = strat[i] - bench[i];
    auto classic_sharpe = PerformanceCalculator::sharpe(active, 0.0, 1.0);

    ASSERT_TRUE(ir_gamma.has_value());
    ASSERT_TRUE(classic_sharpe.has_value());
    // With γ=1, IR_γ = classic IR (non-annualised)
    EXPECT_NEAR(*ir_gamma, *classic_sharpe, 0.01);
}

TEST(PerformanceCalculator_GammaIR, HighGammaAmplifies_PositiveActive) {
    // High γ should amplify the IR when active returns are positive
    std::vector<double> strat(50, 0.002), bench(50, 0.001);
    std::vector<double> g_high(50, 2.0), g_low(50, 1.0);
    // Make active return non-constant so IR is well-defined
    for (std::size_t i = 0; i < 50; ++i) {
        strat[i] = 0.002 + (i % 2 == 0 ? 0.001 : -0.001);
    }

    auto ir_high = PerformanceCalculator::gamma_weighted_ir(strat, bench, g_high);
    auto ir_low  = PerformanceCalculator::gamma_weighted_ir(strat, bench, g_low);

    ASSERT_TRUE(ir_high.has_value());
    ASSERT_TRUE(ir_low.has_value());
    // IR_γ with γ=2 should be 2x that with γ=1 (same denominator, 2x numerator)
    EXPECT_NEAR(*ir_high, 2.0 * (*ir_low), 0.01);
}

TEST(PerformanceCalculator_GammaIR, ZeroActiveReturn_NulloptOrZero) {
    // Identical strategy and benchmark → all-zero active returns → σ = 0
    std::vector<double> same(20, 0.01), gammas(20, 1.2);
    auto result = PerformanceCalculator::gamma_weighted_ir(same, same, gammas);
    // σ(active)=0 → undefined → nullopt
    EXPECT_FALSE(result.has_value());
}

// ─── LorentzSignalAdjuster ────────────────────────────────────────────────────

TEST(LorentzSignalAdjuster, EmptyBars_Nullopt) {
    LorentzSignalAdjuster adj(1.0);
    EXPECT_FALSE(adj.adjust(std::span<const BarData>{}).has_value());
}

TEST(LorentzSignalAdjuster, ZeroMass_ConstructionAllowed_AdjustNullopt) {
    // Zero mass is invalid for adjustment
    LorentzSignalAdjuster adj(0.0);
    std::vector<BarData> bars = {{1.0, BetaVelocity{0.0}, 0.0}};
    EXPECT_FALSE(adj.adjust(bars).has_value());
}

TEST(LorentzSignalAdjuster, NewtonianBeta_GammaIsOne) {
    // β = 0 → γ = 1 → adjusted = 1 × m_eff × raw = raw (with m_eff=1)
    LorentzSignalAdjuster adj(1.0);
    std::vector<BarData> bars = {{2.5, BetaVelocity{0.0}, 0.0}};
    auto result = adj.adjust(bars);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->gamma_factors[0], 1.0, FLOAT_EPSILON);
    EXPECT_NEAR(result->adjusted_signals[0], 2.5, FLOAT_EPSILON);
}

TEST(LorentzSignalAdjuster, HighBeta_AmplifiedSignal) {
    // β = 0.6 → γ = 1.25 → adjusted = 1.25 × 1.0 × 1.0 = 1.25
    LorentzSignalAdjuster adj(1.0);
    std::vector<BarData> bars = {{1.0, BetaVelocity{0.6}, 0.0}};
    auto result = adj.adjust(bars);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->gamma_factors[0], 1.25, 1e-8);
    EXPECT_NEAR(result->adjusted_signals[0], 1.25, 1e-8);
}

TEST(LorentzSignalAdjuster, InvalidBeta_FallsBackToRaw) {
    // β >= BETA_MAX_SAFE → invalid → fallback to raw signal (γ = 1)
    LorentzSignalAdjuster adj(1.0);
    std::vector<BarData> bars = {{3.0, BetaVelocity{1.5}, 0.0}};
    auto result = adj.adjust(bars);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->gamma_factors[0], 1.0, FLOAT_EPSILON);
    EXPECT_NEAR(result->adjusted_signals[0], 3.0, FLOAT_EPSILON);
}

TEST(LorentzSignalAdjuster, GammaAlwaysAtLeastOne) {
    // γ ≥ 1 for all valid β
    LorentzSignalAdjuster adj(1.0);
    std::vector<BarData> bars = {
        {1.0, BetaVelocity{0.0},  0.0},
        {1.0, BetaVelocity{0.3},  0.0},
        {1.0, BetaVelocity{0.6},  0.0},
        {1.0, BetaVelocity{0.9},  0.0},
        {1.0, BetaVelocity{0.999},0.0},
    };
    auto result = adj.adjust(bars);
    ASSERT_TRUE(result.has_value());
    for (double g : result->gamma_factors) {
        EXPECT_GE(g, 1.0) << "γ must be >= 1 for valid β";
    }
}

TEST(LorentzSignalAdjuster, NegativeSignal_Preserved) {
    // Negative raw signal should remain negative after γ scaling (γ > 0)
    LorentzSignalAdjuster adj(1.0);
    std::vector<BarData> bars = {{-2.0, BetaVelocity{0.6}, 0.0}};
    auto result = adj.adjust(bars);
    ASSERT_TRUE(result.has_value());
    // γ = 1.25, m_eff = 1.0 → adjusted = -2.5
    EXPECT_NEAR(result->adjusted_signals[0], -2.5, 1e-8);
}

TEST(LorentzSignalAdjuster, EffectiveMassScales) {
    // m_eff = 2.0, β = 0.6 → γ = 1.25 → adjusted = 1.25 × 2.0 × 1.0 = 2.5
    LorentzSignalAdjuster adj(2.0);
    std::vector<BarData> bars = {{1.0, BetaVelocity{0.6}, 0.0}};
    auto result = adj.adjust(bars);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->adjusted_signals[0], 2.5, 1e-8);
}

TEST(LorentzSignalAdjuster, SingleGamma_Beta06) {
    // Standalone lorentz_gamma helper
    auto g = LorentzSignalAdjuster::lorentz_gamma(BetaVelocity{0.6});
    ASSERT_TRUE(g.has_value());
    EXPECT_NEAR(*g, 1.25, 1e-10);
}

TEST(LorentzSignalAdjuster, SingleGamma_InvalidBeta_Nullopt) {
    auto g = LorentzSignalAdjuster::lorentz_gamma(BetaVelocity{1.0});
    EXPECT_FALSE(g.has_value());
}

// ─── BacktestComparison helpers ───────────────────────────────────────────────

TEST(BacktestComparison, Lifts_CorrectArithmetic) {
    BacktestComparison cmp;
    cmp.raw          = {1.0, 1.2, 0.10, 0.5};
    cmp.relativistic = {1.3, 1.5, 0.08, 0.7};

    EXPECT_NEAR(cmp.sharpe_lift(),    0.3, 1e-12);
    EXPECT_NEAR(cmp.sortino_lift(),   0.3, 1e-12);
    EXPECT_NEAR(cmp.drawdown_delta(), 0.02, 1e-12);  // raw.mdd − rel.mdd
    EXPECT_NEAR(cmp.ir_lift(),        0.2, 1e-12);
}
