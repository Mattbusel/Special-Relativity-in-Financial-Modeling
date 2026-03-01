#include <gtest/gtest.h>
#include "srfm/backtest.hpp"
#include "srfm/constants.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace srfm;
using namespace srfm::backtest;
using namespace srfm::constants;

// ─── Precision & boundary tests for PerformanceCalculator ────────────────────

// Sharpe: sign invariance — negating all returns negates Sharpe
TEST(PerformanceCalc_Precision, Sharpe_SignInversion) {
    std::vector<double> pos = {0.01, -0.005, 0.008, -0.003, 0.012,
                                -0.002, 0.009, -0.004, 0.011, -0.001};
    std::vector<double> neg(pos.size());
    for (std::size_t i = 0; i < pos.size(); ++i) neg[i] = -pos[i];

    auto sh_pos = PerformanceCalculator::sharpe(pos, 0.0, 1.0);
    auto sh_neg = PerformanceCalculator::sharpe(neg, 0.0, 1.0);
    ASSERT_TRUE(sh_pos.has_value());
    ASSERT_TRUE(sh_neg.has_value());
    // Negating returns negates the mean while σ stays the same → Sharpe negates
    EXPECT_NEAR(*sh_pos, -*sh_neg, 1e-10);
}

// Sharpe: scaling invariance — multiplying all returns by k scales Sharpe by sign(k)
// (μ → kμ, σ → |k|σ → Sharpe → k/|k| × Sharpe = sign(k) × Sharpe)
TEST(PerformanceCalc_Precision, Sharpe_ScalingInvariance_PositiveK) {
    std::vector<double> base = {0.01, -0.005, 0.008, -0.002, 0.007,
                                 -0.003, 0.006, -0.001, 0.009, -0.004};
    constexpr double k = 3.0;
    std::vector<double> scaled(base.size());
    for (std::size_t i = 0; i < base.size(); ++i) scaled[i] = base[i] * k;

    auto sh_base   = PerformanceCalculator::sharpe(base,   0.0, 1.0);
    auto sh_scaled = PerformanceCalculator::sharpe(scaled, 0.0, 1.0);
    ASSERT_TRUE(sh_base.has_value());
    ASSERT_TRUE(sh_scaled.has_value());
    // Positive k → sign(k)=1 → Sharpe unchanged
    EXPECT_NEAR(*sh_base, *sh_scaled, 1e-9);
}

// Max drawdown: MDD of constant gain = 0 regardless of magnitude
TEST(PerformanceCalc_Precision, MaxDrawdown_ConstantGain_Zero) {
    for (double gain : {0.001, 0.01, 0.05, 0.1}) {
        std::vector<double> r(50, gain);
        auto mdd = PerformanceCalculator::max_drawdown(r);
        ASSERT_TRUE(mdd.has_value()) << "gain=" << gain;
        EXPECT_NEAR(*mdd, 0.0, FLOAT_EPSILON) << "gain=" << gain;
    }
}

// Max drawdown: MDD is bounded by [0, 1]
TEST(PerformanceCalc_Precision, MaxDrawdown_Bounded) {
    // Worst possible: lose everything each bar
    std::vector<double> catastrophic(30, -0.99);
    auto mdd = PerformanceCalculator::max_drawdown(catastrophic);
    ASSERT_TRUE(mdd.has_value());
    EXPECT_GE(*mdd, 0.0);
    EXPECT_LE(*mdd, 1.0);
}

// Sortino must equal Sharpe when all returns are below r_f (all downside)
TEST(PerformanceCalc_Precision, Sortino_AllBelowRiskFree_Defined) {
    // All returns are negative → all below r_f=0 → σ_down = σ → Sortino ≈ Sharpe
    std::vector<double> neg = {-0.01, -0.02, -0.015, -0.005, -0.025,
                                -0.008, -0.012, -0.003, -0.018, -0.022};
    auto sh = PerformanceCalculator::sharpe(neg,  0.0, 1.0);
    auto so = PerformanceCalculator::sortino(neg, 0.0, 1.0);
    ASSERT_TRUE(sh.has_value());
    ASSERT_TRUE(so.has_value());
    // When all returns < threshold, σ_down uses n-1 denom and equals σ
    // within small numerical error
    EXPECT_NEAR(*sh, *so, 0.1);  // coarse tolerance: Bessel corrections differ
}

// Gamma IR: linearity in mean active return
TEST(PerformanceCalc_Precision, GammaIR_Linearity_InMeanActive) {
    // Double all active returns (keep benchmark at 0, double strategy)
    // IR_γ = mean(active) × mean(γ) / σ(active)
    // Doubling active: mean doubles, σ doubles → ratio unchanged
    std::vector<double> strat, bench, g;
    for (int i = 0; i < 60; ++i) {
        double s = 0.002 + (i % 2 == 0 ? 0.003 : -0.001);
        strat.push_back(s);
        bench.push_back(0.0);
        g.push_back(1.25);
    }
    std::vector<double> strat2(strat.size());
    for (std::size_t i = 0; i < strat.size(); ++i) strat2[i] = strat[i] * 2.0;

    auto ir1 = PerformanceCalculator::gamma_weighted_ir(strat,  bench, g);
    auto ir2 = PerformanceCalculator::gamma_weighted_ir(strat2, bench, g);
    ASSERT_TRUE(ir1.has_value());
    ASSERT_TRUE(ir2.has_value());
    // Scaling both mean and σ by 2 → ratio is identical
    EXPECT_NEAR(*ir1, *ir2, 1e-8);
}

// ─── LorentzSignalAdjuster — symmetry & monotonicity ─────────────────────────

// Adjusted signal magnitude always ≥ raw magnitude (γ ≥ 1, m_eff = 1)
TEST(LorentzAdjuster_Precision, AdjustedMagnitudeAtLeastRaw) {
    LorentzSignalAdjuster adj(1.0);
    std::vector<BarData> bars = {
        {2.0, BetaVelocity{0.1}, 0.0},
        {2.0, BetaVelocity{0.5}, 0.0},
        {2.0, BetaVelocity{0.9}, 0.0},
    };
    auto result = adj.adjust(bars);
    ASSERT_TRUE(result.has_value());
    for (std::size_t i = 0; i < bars.size(); ++i) {
        EXPECT_GE(std::abs(result->adjusted_signals[i]),
                  std::abs(bars[i].raw_signal))
            << "adjusted must be >= raw at bar " << i;
    }
}

// γ must be monotonically increasing with |β|
TEST(LorentzAdjuster_Precision, GammaMonotonicallyIncreasesWithBeta) {
    LorentzSignalAdjuster adj(1.0);
    std::vector<BetaVelocity> betas = {
        BetaVelocity{0.0},
        BetaVelocity{0.2},
        BetaVelocity{0.4},
        BetaVelocity{0.6},
        BetaVelocity{0.8},
        BetaVelocity{0.99},
    };
    double prev = 0.0;
    for (const auto& b : betas) {
        auto g = LorentzSignalAdjuster::lorentz_gamma(b);
        ASSERT_TRUE(g.has_value()) << "β=" << b.value;
        EXPECT_GE(*g, prev) << "γ not monotone at β=" << b.value;
        prev = *g;
    }
}

// β = 0 always gives γ = 1 exactly
TEST(LorentzAdjuster_Precision, BetaZero_GammaExactlyOne) {
    auto g = LorentzSignalAdjuster::lorentz_gamma(BetaVelocity{0.0});
    ASSERT_TRUE(g.has_value());
    EXPECT_DOUBLE_EQ(*g, 1.0);
}

// NaN β → nullopt
TEST(LorentzAdjuster_Precision, NanBeta_Nullopt) {
    auto g = LorentzSignalAdjuster::lorentz_gamma(
        BetaVelocity{std::numeric_limits<double>::quiet_NaN()});
    EXPECT_FALSE(g.has_value());
}

// Negative β: γ(−β) = γ(β)  (symmetry)
TEST(LorentzAdjuster_Precision, NegativeBeta_SameGammaAsPositive) {
    for (double b : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        auto gp = LorentzSignalAdjuster::lorentz_gamma(BetaVelocity{ b});
        auto gn = LorentzSignalAdjuster::lorentz_gamma(BetaVelocity{-b});
        ASSERT_TRUE(gp.has_value()) << "β=+" << b;
        ASSERT_TRUE(gn.has_value()) << "β=-" << b;
        EXPECT_NEAR(*gp, *gn, FLOAT_EPSILON)
            << "γ not symmetric at β=" << b;
    }
}

// ─── Backtester config propagation ───────────────────────────────────────────

// Custom risk-free rate shifts Sharpe
TEST(Backtester_Config, RiskFreeRate_ShiftsSharpe) {
    BacktestConfig cfg0, cfg_rf;
    cfg0.risk_free_rate  = 0.0;
    cfg_rf.risk_free_rate = 0.001;
    cfg0.annualisation   = 1.0;
    cfg_rf.annualisation  = 1.0;

    const std::size_t N = 200;
    std::vector<BarData> bars(N);
    std::vector<double>  rets(N);
    for (std::size_t i = 0; i < N; ++i) {
        bars[i] = {(i % 2 == 0 ? 1.0 : -1.0), BetaVelocity{0.0}, 0.0};
        rets[i] = (i % 2 == 0 ? 0.005 : -0.005);
    }

    Backtester bt0(cfg0), bt_rf(cfg_rf);
    auto r0  = bt0.run(bars, rets);
    auto rrf = bt_rf.run(bars, rets);
    ASSERT_TRUE(r0.has_value());
    ASSERT_TRUE(rrf.has_value());
    // Higher r_f → lower excess return → lower (or equal) Sharpe
    EXPECT_LE(rrf->raw.sharpe_ratio, r0->raw.sharpe_ratio + 1e-6);
}
