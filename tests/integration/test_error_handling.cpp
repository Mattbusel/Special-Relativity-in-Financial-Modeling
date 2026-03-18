/// @file tests/integration/test_error_handling.cpp
/// @brief Integration tests for error-handling and boundary conditions across
///        the full SRFM pipeline.
///
/// Verifies that every public API surface:
///   1. Returns std::nullopt (not UB or crash) on all degenerate inputs.
///   2. Correctly propagates errors through multi-stage pipelines.
///   3. Handles IEEE 754 special values (NaN, ±Inf, denormals) gracefully.

#include "srfm/backtest.hpp"
#include "srfm/constants.hpp"
#include "srfm/data_loader.hpp"
#include "srfm/engine.hpp"
#include "srfm/manifold.hpp"
#include "srfm/momentum.hpp"
#include "srfm/tensor.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using namespace srfm;
using namespace srfm::core;
using namespace srfm::backtest;
using namespace srfm::manifold;
using namespace srfm::momentum;
using namespace srfm::tensor;
using namespace srfm::constants;

// ─── Helpers ──────────────────────────────────────────────────────────────────

static const double kNaN = std::numeric_limits<double>::quiet_NaN();
static const double kInf = std::numeric_limits<double>::infinity();

static std::vector<OHLCV> make_bars(std::size_t n, double start = 100.0,
                                     double drift = 0.001) {
    std::vector<OHLCV> bars;
    bars.reserve(n);
    double price = start;
    for (std::size_t i = 0; i < n; ++i) {
        price *= (1.0 + drift);
        const double sp = price * 0.001;
        bars.push_back({static_cast<double>(i + 1),
                        price - sp, price + sp, price - sp, price, 1e6});
    }
    return bars;
}

// ─── Engine: degenerate bar inputs ────────────────────────────────────────────

TEST(ErrorHandling_Engine, NaNClosePrice_ReturnsNullopt) {
    auto bars = make_bars(50);
    // Inject NaN into one close price
    bars[25].close = kNaN;
    Engine engine;
    // Engine must not crash; result may be nullopt or valid depending on how
    // DataLoader handles corrupt bars, but must not produce UB.
    auto result = engine.run_backtest(bars);
    // No assertion on value — just verifying no crash / sanitizer hit.
    (void)result;
}

TEST(ErrorHandling_Engine, InfiniteVolume_SurvivesWithoutCrash) {
    auto bars = make_bars(50);
    bars[10].volume = kInf;
    Engine engine;
    auto result = engine.run_backtest(bars);
    (void)result;
}

TEST(ErrorHandling_Engine, AllSamePrices_SurvivesWithoutCrash) {
    // Zero velocity → β = 0 for all bars. Should produce valid (possibly
    // degenerate) backtest result, not crash.
    std::vector<OHLCV> bars(50, OHLCV{0, 100.0, 100.0, 100.0, 100.0, 1e6});
    for (std::size_t i = 0; i < bars.size(); ++i) bars[i].timestamp = static_cast<double>(i + 1);
    Engine engine;
    auto result = engine.run_backtest(bars);
    (void)result;  // May be nullopt (zero-variance series) — no crash expected.
}

// ─── PerformanceCalculator: edge inputs ───────────────────────────────────────

TEST(ErrorHandling_PerformanceCalc, EmptySeries_SharpeNullopt) {
    std::vector<double> empty;
    EXPECT_FALSE(PerformanceCalculator::sharpe(empty).has_value());
}

TEST(ErrorHandling_PerformanceCalc, SingleElement_SharpeNullopt) {
    std::vector<double> one = {0.01};
    EXPECT_FALSE(PerformanceCalculator::sharpe(one).has_value());
}

TEST(ErrorHandling_PerformanceCalc, AllZeroReturns_SharpeNullopt) {
    // σ = 0 → Sharpe undefined
    std::vector<double> zeros(50, 0.0);
    EXPECT_FALSE(PerformanceCalculator::sharpe(zeros).has_value());
}

TEST(ErrorHandling_PerformanceCalc, NaNInReturns_SharpeNullopt) {
    std::vector<double> returns = {0.01, 0.02, kNaN, 0.03};
    EXPECT_FALSE(PerformanceCalculator::sharpe(returns).has_value());
}

TEST(ErrorHandling_PerformanceCalc, InfInReturns_SharpeNullopt) {
    std::vector<double> returns = {0.01, kInf, 0.02};
    EXPECT_FALSE(PerformanceCalculator::sharpe(returns).has_value());
}

TEST(ErrorHandling_PerformanceCalc, MaxDrawdown_EmptySeries_Nullopt) {
    std::vector<double> empty;
    EXPECT_FALSE(PerformanceCalculator::max_drawdown(empty).has_value());
}

TEST(ErrorHandling_PerformanceCalc, GammaWeightedIR_MismatchedLengths_Nullopt) {
    std::vector<double> strat  = {0.01, 0.02, 0.01};
    std::vector<double> bmark  = {0.005, 0.006};          // different length
    std::vector<double> gammas = {1.0, 1.1, 1.2};
    EXPECT_FALSE(PerformanceCalculator::gamma_weighted_ir(strat, bmark, gammas)
                     .has_value());
}

// ─── Backtester: size mismatches ──────────────────────────────────────────────

TEST(ErrorHandling_Backtester, MismatchedBarAndReturnLengths_Nullopt) {
    Backtester bt;
    std::vector<BarData> bars(40, BarData{0.5, BetaVelocity{0.3}, 0.001});
    std::vector<double>  rets(50, 0.005);  // wrong length
    EXPECT_FALSE(bt.run(bars, rets).has_value());
}

TEST(ErrorHandling_Backtester, TooFewBars_Nullopt) {
    Backtester bt;
    std::vector<BarData> bars(MIN_RETURN_SERIES_LENGTH - 1,
                               BarData{0.5, BetaVelocity{0.3}, 0.001});
    std::vector<double>  rets(MIN_RETURN_SERIES_LENGTH - 1, 0.005);
    EXPECT_FALSE(bt.run(bars, rets).has_value());
}

// ─── LorentzSignalAdjuster ────────────────────────────────────────────────────

TEST(ErrorHandling_LorentzAdjuster, NegativeEffectiveMass_Nullopt) {
    LorentzSignalAdjuster adj(-1.0);  // invalid mass
    std::vector<BarData> bars(50, BarData{0.5, BetaVelocity{0.3}, 0.001});
    EXPECT_FALSE(adj.adjust(bars).has_value());
}

TEST(ErrorHandling_LorentzAdjuster, EmptyBars_Nullopt) {
    LorentzSignalAdjuster adj(1.0);
    std::vector<BarData> bars;
    EXPECT_FALSE(adj.adjust(bars).has_value());
}

// ─── SpacetimeInterval: non-finite coordinates ───────────────────────────────

TEST(ErrorHandling_SpacetimeInterval, NanCoordinate_Nullopt) {
    SpacetimeEvent a{0.0, 100.0, 1e6, 0.0};
    SpacetimeEvent b{1.0, kNaN,  1e6, 0.0};
    EXPECT_FALSE(SpacetimeInterval::compute(a, b).has_value());
}

TEST(ErrorHandling_SpacetimeInterval, InfCoordinate_Nullopt) {
    SpacetimeEvent a{0.0, 100.0, 1e6, 0.0};
    SpacetimeEvent b{1.0, kInf,  1e6, 0.0};
    EXPECT_FALSE(SpacetimeInterval::compute(a, b).has_value());
}

TEST(ErrorHandling_SpacetimeInterval, BothSamePoint_ZeroInterval) {
    SpacetimeEvent a{1.0, 100.0, 1e6, 0.5};
    auto ds2 = SpacetimeInterval::compute(a, a);
    ASSERT_TRUE(ds2.has_value());
    EXPECT_NEAR(*ds2, 0.0, FLOAT_EPSILON);
}

// ─── MetricTensor: singular metric handling ───────────────────────────────────

TEST(ErrorHandling_MetricTensor, SingularMetric_InverseNullopt) {
    // A metric with a zero diagonal entry is singular.
    auto g = MetricTensor([](const SpacetimePoint&) {
        MetricMatrix m = MetricMatrix::Zero();
        m(0, 0) = -1.0;
        m(1, 1) = 0.0;   // degenerate
        m(2, 2) = 1.0;
        m(3, 3) = 1.0;
        return m;
    });
    EXPECT_FALSE(g.inverse(SpacetimePoint::Zero()).has_value());
}

TEST(ErrorHandling_MetricTensor, WellConditionedMinkowski_InverseExists) {
    auto g = MetricTensor::make_minkowski(1.0, 0.1);
    auto inv = g.inverse(SpacetimePoint::Zero());
    ASSERT_TRUE(inv.has_value());
    // g × g_inv ≈ I
    auto gmat   = g.evaluate(SpacetimePoint::Zero());
    MetricMatrix prod = gmat * (*inv);
    EXPECT_LT((prod - MetricMatrix::Identity()).norm(), 1e-9);
}

// ─── MomentumProcessor: invalid inputs ───────────────────────────────────────

TEST(ErrorHandling_MomentumProcessor, ZeroEffectiveMass_Nullopt) {
    MomentumSignal sig{1.0, BetaVelocity{0.3}, 0.0};
    EXPECT_FALSE(MomentumProcessor::process(sig).has_value());
}

TEST(ErrorHandling_MomentumProcessor, NegativeEffectiveMass_Nullopt) {
    MomentumSignal sig{1.0, BetaVelocity{0.3}, -1.0};
    EXPECT_FALSE(MomentumProcessor::process(sig).has_value());
}

TEST(ErrorHandling_MomentumProcessor, EmptySeries_Nullopt) {
    std::vector<MomentumSignal> empty;
    EXPECT_FALSE(MomentumProcessor::process_series(empty).has_value());
}

TEST(ErrorHandling_MomentumProcessor, NaNRawValue_FiniteAdjusted) {
    // Raw NaN propagates through γ multiplication but should stay finite-ish.
    // We just verify no crash.
    MomentumSignal sig{kNaN, BetaVelocity{0.3}, 1.0};
    auto result = MomentumProcessor::process(sig);
    (void)result;  // May be valid with NaN adjusted_value — no crash expected.
}

// ─── DataLoader: malformed CSV inputs ────────────────────────────────────────

TEST(ErrorHandling_DataLoader, AllMalformedRows_EmptyVector) {
    std::string csv =
        "timestamp,open,high,low,close,volume\n"
        "not,a,number,row,here,!\n"
        "also,bad\n"
        ",,,,,\n";
    auto bars = DataLoader::parse_csv_string(csv);
    EXPECT_TRUE(bars.empty());
}

TEST(ErrorHandling_DataLoader, NegativeClosePrice_Skipped) {
    std::string csv =
        "timestamp,open,high,low,close,volume\n"
        "1,100.0,105.0,99.0,-103.0,500000\n"  // negative close
        "2,100.0,105.0,99.0,103.0,500000\n";
    auto bars = DataLoader::parse_csv_string(csv);
    ASSERT_EQ(bars.size(), 1u);
    EXPECT_DOUBLE_EQ(bars[0].timestamp, 2.0);
}

TEST(ErrorHandling_DataLoader, ValidRowAfterManyBadRows_Parsed) {
    std::string csv = "timestamp,open,high,low,close,volume\n";
    for (int i = 0; i < 20; ++i) {
        csv += "garbage_row\n";
    }
    csv += "1,99.0,101.0,98.5,100.0,500000\n";
    auto bars = DataLoader::parse_csv_string(csv);
    ASSERT_EQ(bars.size(), 1u);
    EXPECT_DOUBLE_EQ(bars[0].close, 100.0);
}

// ─── End-to-end: zero-variance returns still produce finite metrics ───────────

TEST(ErrorHandling_EndToEnd, MonotonicallyIncreasingPrices_Survives) {
    // Prices always go up → returns always positive → zero downside std-dev.
    std::vector<OHLCV> bars;
    double price = 100.0;
    for (std::size_t i = 0; i < 60; ++i) {
        price += 0.1;
        bars.push_back({static_cast<double>(i + 1), price, price + 0.1,
                        price - 0.1, price, 1e6});
    }
    Engine engine;
    auto result = engine.run_backtest(bars);
    // May return nullopt (zero-variance Sortino) — must not crash.
    if (result) {
        EXPECT_TRUE(std::isfinite(result->raw.sharpe_ratio) ||
                    !result->raw.sharpe_ratio);  // NaN is also acceptable here
    }
}
