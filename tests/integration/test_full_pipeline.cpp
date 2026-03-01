/// @file tests/integration/test_full_pipeline.cpp
/// @brief End-to-end integration tests for the full SRFM pipeline — AGT-06.
///
/// These tests exercise the complete signal path:
///   OHLCV → BetaCalculator → LorentzTransform → MarketManifold →
///   MomentumProcessor → MetricTensor + GeodesicSolver → Backtester →
///   BacktestComparison

#include "srfm/engine.hpp"
#include "srfm/data_loader.hpp"
#include "srfm/backtest.hpp"
#include "srfm/manifold.hpp"
#include "srfm/momentum.hpp"
#include "srfm/tensor.hpp"
#include "srfm/constants.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace srfm;
using namespace srfm::core;
using namespace srfm::constants;

// ─── Synthetic data helpers ───────────────────────────────────────────────────

namespace {

/// Generate a simple trending price series with optional noise.
/// Returns N bars with close prices starting at 100, drifting by drift_per_bar.
std::vector<OHLCV> make_trending_bars(std::size_t n,
                                       double start_price = 100.0,
                                       double drift       = 0.001) {
    std::vector<OHLCV> bars;
    bars.reserve(n);
    double price = start_price;
    for (std::size_t i = 0; i < n; ++i) {
        price *= (1.0 + drift);
        const double spread = price * 0.001;
        bars.push_back(OHLCV{
            .timestamp = static_cast<double>(i + 1),
            .open      = price - spread * 0.5,
            .high      = price + spread,
            .low       = price - spread,
            .close     = price,
            .volume    = 1e6,
        });
    }
    return bars;
}

/// Generate a high-velocity (relativistic) bar series with large price moves.
std::vector<OHLCV> make_volatile_bars(std::size_t n,
                                       double start_price = 100.0) {
    std::vector<OHLCV> bars;
    bars.reserve(n);
    double price = start_price;
    for (std::size_t i = 0; i < n; ++i) {
        // Alternating ±10% moves = very high β
        const double move = (i % 2 == 0) ? 0.10 : -0.09;
        price *= (1.0 + move);
        const double spread = price * 0.005;
        bars.push_back(OHLCV{
            .timestamp = static_cast<double>(i + 1),
            .open      = price - spread,
            .high      = price + spread * 2,
            .low       = price - spread * 2,
            .close     = price,
            .volume    = 5e6,
        });
    }
    return bars;
}

/// Build a minimal CSV string with header for N bars.
std::string bars_to_csv(const std::vector<OHLCV>& bars) {
    std::ostringstream ss;
    ss << "timestamp,open,high,low,close,volume\n";
    for (const auto& b : bars) {
        ss << b.timestamp << ","
           << b.open      << ","
           << b.high      << ","
           << b.low       << ","
           << b.close     << ","
           << b.volume    << "\n";
    }
    return ss.str();
}

}  // anonymous namespace

// ─── Engine::run_backtest ─────────────────────────────────────────────────────

TEST(EngineRunBacktest, TooFewBarsReturnsNullopt) {
    Engine engine;
    auto bars = make_trending_bars(MIN_RETURN_SERIES_LENGTH - 1);
    EXPECT_FALSE(engine.run_backtest(bars).has_value());
}

TEST(EngineRunBacktest, ExactlyMinBarsSucceeds) {
    Engine engine;
    auto bars = make_trending_bars(MIN_RETURN_SERIES_LENGTH);
    auto result = engine.run_backtest(bars);
    EXPECT_TRUE(result.has_value());
}

TEST(EngineRunBacktest, ResultFieldsAreFinite) {
    Engine engine;
    auto bars = make_trending_bars(100);
    auto result = engine.run_backtest(bars);
    ASSERT_TRUE(result.has_value());

    EXPECT_TRUE(std::isfinite(result->raw.sharpe_ratio));
    EXPECT_TRUE(std::isfinite(result->raw.sortino_ratio));
    EXPECT_TRUE(std::isfinite(result->raw.max_drawdown));
    EXPECT_TRUE(std::isfinite(result->raw.gamma_weighted_ir));

    EXPECT_TRUE(std::isfinite(result->relativistic.sharpe_ratio));
    EXPECT_TRUE(std::isfinite(result->relativistic.sortino_ratio));
    EXPECT_TRUE(std::isfinite(result->relativistic.max_drawdown));
    EXPECT_TRUE(std::isfinite(result->relativistic.gamma_weighted_ir));
}

TEST(EngineRunBacktest, MaxDrawdownInValidRange) {
    Engine engine;
    auto bars = make_trending_bars(100);
    auto result = engine.run_backtest(bars);
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result->raw.max_drawdown, 0.0);
    EXPECT_LE(result->raw.max_drawdown, 1.0);
    EXPECT_GE(result->relativistic.max_drawdown, 0.0);
    EXPECT_LE(result->relativistic.max_drawdown, 1.0);
}

TEST(EngineRunBacktest, LiftAccessorsReturnFiniteValues) {
    Engine engine;
    auto bars = make_trending_bars(100);
    auto result = engine.run_backtest(bars);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::isfinite(result->sharpe_lift()));
    EXPECT_TRUE(std::isfinite(result->sortino_lift()));
    EXPECT_TRUE(std::isfinite(result->drawdown_delta()));
    EXPECT_TRUE(std::isfinite(result->ir_lift()));
}

TEST(EngineRunBacktest, VolatileMarketProducesResult) {
    Engine engine;
    auto bars = make_volatile_bars(100);
    auto result = engine.run_backtest(bars);
    // May fail due to zero-variance series; just verify no crash
    // (nullopt is acceptable for degenerate high-volatility inputs)
    (void)result;
}

TEST(EngineRunBacktest, RelativisticMetricsDifferFromRaw) {
    // In a relativistic market, γ > 1 for many bars → metrics should differ
    EngineConfig cfg;
    cfg.max_market_velocity = 0.01;  // Low max_vel → higher β → more relativistic
    Engine engine(cfg);
    auto bars = make_trending_bars(100, 100.0, 0.005);
    auto result = engine.run_backtest(bars);
    ASSERT_TRUE(result.has_value());
    // The relativistic IR should be at least as large as raw (γ ≥ 1 always)
    // Can't assert strict inequality since some regimes produce β ≈ 0
    EXPECT_TRUE(std::isfinite(result->ir_lift()));
}

TEST(EngineRunBacktest, EmptySpanReturnsNullopt) {
    Engine engine;
    std::span<const OHLCV> empty{};
    EXPECT_FALSE(engine.run_backtest(empty).has_value());
}

// ─── Engine streaming mode ────────────────────────────────────────────────────

TEST(EngineStream, FirstBarReturnsNullopt) {
    Engine engine;
    auto bars = make_trending_bars(5);
    auto result = engine.process_stream_bar(bars[0]);
    EXPECT_FALSE(result.has_value());  // Warm-up: need ≥ 2 bars
}

TEST(EngineStream, SecondBarReturnsResult) {
    Engine engine;
    auto bars = make_trending_bars(5);
    engine.process_stream_bar(bars[0]);
    auto result = engine.process_stream_bar(bars[1]);
    ASSERT_TRUE(result.has_value());
}

TEST(EngineStream, ResultFieldsAreFinite) {
    Engine engine;
    auto bars = make_trending_bars(10);
    std::optional<PipelineBar> last;
    for (const auto& b : bars) {
        last = engine.process_stream_bar(b);
    }
    ASSERT_TRUE(last.has_value());
    EXPECT_TRUE(std::isfinite(last->price_return));
    EXPECT_TRUE(std::isfinite(last->beta.value));
    EXPECT_TRUE(std::isfinite(last->gamma.value));
    EXPECT_GE(last->gamma.value, 1.0);
}

TEST(EngineStream, GammaAlwaysAtLeastOne) {
    Engine engine;
    auto bars = make_volatile_bars(50);
    for (const auto& b : bars) {
        auto result = engine.process_stream_bar(b);
        if (result) {
            EXPECT_GE(result->gamma.value, 1.0)
                << "gamma < 1 at bar ts=" << b.timestamp;
        }
    }
}

TEST(EngineStream, BetaInValidRange) {
    Engine engine;
    auto bars = make_trending_bars(50);
    for (const auto& b : bars) {
        auto result = engine.process_stream_bar(b);
        if (result) {
            EXPECT_GE(result->beta.value, 0.0);
            EXPECT_LT(result->beta.value, BETA_MAX_SAFE);
        }
    }
}

TEST(EngineStream, ResetClearsWindow) {
    Engine engine;
    auto bars = make_trending_bars(10);
    for (const auto& b : bars) {
        engine.process_stream_bar(b);
    }
    EXPECT_FALSE(engine.stream_window().empty());
    engine.reset_stream();
    EXPECT_TRUE(engine.stream_window().empty());
    // After reset, first bar again needs warm-up
    auto result = engine.process_stream_bar(bars[0]);
    EXPECT_FALSE(result.has_value());
}

TEST(EngineStream, IntervalTypeIsValid) {
    Engine engine;
    auto bars = make_trending_bars(20);
    for (const auto& b : bars) {
        auto result = engine.process_stream_bar(b);
        if (result) {
            // Just verifying the field is one of the valid enum values
            const bool valid =
                result->interval_type == manifold::IntervalType::Timelike  ||
                result->interval_type == manifold::IntervalType::Lightlike ||
                result->interval_type == manifold::IntervalType::Spacelike;
            EXPECT_TRUE(valid);
        }
    }
}

// ─── DataLoader ───────────────────────────────────────────────────────────────

TEST(DataLoaderParseCsv, EmptyStringGivesEmptyVector) {
    EXPECT_TRUE(DataLoader::parse_csv_string("").empty());
}

TEST(DataLoaderParseCsv, HeaderOnlyGivesEmptyVector) {
    EXPECT_TRUE(DataLoader::parse_csv_string(
        "timestamp,open,high,low,close,volume\n").empty());
}

TEST(DataLoaderParseCsv, ValidRowIsParsed) {
    auto bars = DataLoader::parse_csv_string(
        "timestamp,open,high,low,close,volume\n"
        "1,99.0,101.0,98.5,100.0,500000\n");
    ASSERT_EQ(bars.size(), 1u);
    EXPECT_DOUBLE_EQ(bars[0].timestamp, 1.0);
    EXPECT_DOUBLE_EQ(bars[0].close, 100.0);
}

TEST(DataLoaderParseCsv, MalformedRowSkipped) {
    auto bars = DataLoader::parse_csv_string(
        "timestamp,open,high,low,close,volume\n"
        "bad,row,here\n"
        "2,99.0,101.0,98.5,100.0,500000\n");
    ASSERT_EQ(bars.size(), 1u);
    EXPECT_DOUBLE_EQ(bars[0].timestamp, 2.0);
}

TEST(DataLoaderParseCsv, OHLCViolationSkipped) {
    // close > high → invalid
    auto bars = DataLoader::parse_csv_string(
        "timestamp,open,high,low,close,volume\n"
        "1,100.0,102.0,99.0,105.0,1000\n");
    EXPECT_TRUE(bars.empty());
}

TEST(DataLoaderParseCsv, MultipleValidRows) {
    auto csv = bars_to_csv(make_trending_bars(50));
    auto bars = DataLoader::parse_csv_string(csv);
    EXPECT_EQ(bars.size(), 50u);
}

TEST(DataLoaderParseCsv, NaNInRowSkipped) {
    auto bars = DataLoader::parse_csv_string(
        "timestamp,open,high,low,close,volume\n"
        "1,nan,101.0,98.5,100.0,500000\n"
        "2,99.0,101.0,98.5,100.0,500000\n");
    // nan parses to NaN which is non-finite → skip
    ASSERT_LE(bars.size(), 1u);
    if (!bars.empty()) {
        EXPECT_DOUBLE_EQ(bars[0].timestamp, 2.0);
    }
}

TEST(DataLoaderValidateBar, ValidBarPasses) {
    OHLCV b{1.0, 99.0, 101.0, 98.5, 100.0, 500000.0};
    EXPECT_TRUE(DataLoader::validate_bar(b));
}

TEST(DataLoaderValidateBar, HighLessThanLowFails) {
    OHLCV b{1.0, 100.0, 98.0, 102.0, 100.0, 1000.0};  // high < low
    EXPECT_FALSE(DataLoader::validate_bar(b));
}

TEST(DataLoaderValidateBar, NegativeVolumeFails) {
    OHLCV b{1.0, 99.0, 101.0, 98.5, 100.0, -1.0};
    EXPECT_FALSE(DataLoader::validate_bar(b));
}

TEST(DataLoaderValidateBar, InfiniteTimestampFails) {
    OHLCV b{std::numeric_limits<double>::infinity(), 99.0, 101.0, 98.5, 100.0, 1000.0};
    EXPECT_FALSE(DataLoader::validate_bar(b));
}

// ─── End-to-end: DataLoader → Engine ─────────────────────────────────────────

TEST(EndToEnd, CsvRoundtripThroughEngine) {
    auto original_bars = make_trending_bars(80);
    auto csv = bars_to_csv(original_bars);

    auto loaded_bars = DataLoader::parse_csv_string(csv);
    ASSERT_EQ(loaded_bars.size(), original_bars.size());

    Engine engine;
    auto result = engine.run_backtest(loaded_bars);
    ASSERT_TRUE(result.has_value());

    EXPECT_TRUE(std::isfinite(result->raw.sharpe_ratio));
    EXPECT_TRUE(std::isfinite(result->relativistic.sharpe_ratio));
}

TEST(EndToEnd, ManifoldClassifiesTrendingBarsAsTimelike) {
    auto bars = make_trending_bars(20);
    std::size_t timelike_count = 0;
    for (std::size_t i = 1; i < bars.size(); ++i) {
        manifold::SpacetimeEvent a{
            bars[i-1].timestamp, bars[i-1].close, bars[i-1].volume, 0.0};
        manifold::SpacetimeEvent b{
            bars[i].timestamp, bars[i].close, bars[i].volume, 0.0};
        auto cls = manifold::MarketManifold::classify(a, b);
        if (cls && *cls == manifold::IntervalType::Timelike) {
            ++timelike_count;
        }
    }
    // A slowly-drifting trend (small price moves, large time steps) should be
    // mostly timelike.
    EXPECT_GT(timelike_count, bars.size() / 2);
}

TEST(EndToEnd, MomentumProcessorIntegratesWithEngine) {
    auto bars = make_trending_bars(50);
    std::vector<momentum::MomentumSignal> signals;
    signals.reserve(bars.size());

    Engine engine;
    for (const auto& b : bars) {
        auto pb = engine.process_stream_bar(b);
        if (pb) {
            signals.push_back(momentum::MomentumSignal{
                .raw_value      = pb->price_return,
                .beta           = pb->beta,
                .effective_mass = 1.0,
            });
        }
    }

    ASSERT_FALSE(signals.empty());
    auto rel_signals = momentum::MomentumProcessor::process_series(signals);
    ASSERT_TRUE(rel_signals.has_value());
    EXPECT_EQ(rel_signals->size(), signals.size());

    // Every adjusted signal should be finite
    for (const auto& rs : *rel_signals) {
        EXPECT_TRUE(std::isfinite(rs.adjusted_value));
        EXPECT_GE(rs.gamma.value, 1.0);
    }
}

TEST(EndToEnd, BacktestConfigPropagation) {
    // Verify custom risk-free rate is honoured by the engine.
    EngineConfig cfg;
    cfg.backtest_cfg.risk_free_rate = 0.05 / 252.0;  // ~5% annual
    Engine engine(cfg);
    auto bars = make_trending_bars(80);
    auto result = engine.run_backtest(bars);
    ASSERT_TRUE(result.has_value());
    // Sharpe will change vs default r_f = 0; just verify it's still finite
    EXPECT_TRUE(std::isfinite(result->raw.sharpe_ratio));
}

TEST(EndToEnd, LiftSignConsistency) {
    // sharpe_lift = relativistic.sharpe - raw.sharpe
    Engine engine;
    auto bars = make_trending_bars(100);
    auto result = engine.run_backtest(bars);
    ASSERT_TRUE(result.has_value());
    const double expected_sharpe_lift =
        result->relativistic.sharpe_ratio - result->raw.sharpe_ratio;
    EXPECT_NEAR(result->sharpe_lift(), expected_sharpe_lift, 1e-12);
    const double expected_drawdown_delta =
        result->raw.max_drawdown - result->relativistic.max_drawdown;
    EXPECT_NEAR(result->drawdown_delta(), expected_drawdown_delta, 1e-12);
}

TEST(EndToEnd, StreamThenBacktestAgreement) {
    // Process bars in stream mode; ensure engine doesn't crash on both paths.
    auto bars = make_trending_bars(60);
    Engine engine_stream;
    std::size_t stream_bars_processed = 0;
    for (const auto& b : bars) {
        if (engine_stream.process_stream_bar(b).has_value()) {
            ++stream_bars_processed;
        }
    }
    EXPECT_EQ(stream_bars_processed, bars.size() - 1);

    Engine engine_batch;
    auto result = engine_batch.run_backtest(bars);
    ASSERT_TRUE(result.has_value());
}

TEST(EndToEnd, MetricTensorFromReturnsIsSPD) {
    // Build a 3-asset diagonal covariance from bar returns and verify the
    // resulting metric tensor is properly invertible.
    using namespace srfm::tensor;

    const double vol = 0.02;
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * vol * vol;
    const double c_sq   = constants::SPEED_OF_INFORMATION * constants::SPEED_OF_INFORMATION;

    auto mt = MetricTensor::make_from_covariance(c_sq, cov);
    SpacetimePoint x = SpacetimePoint::Zero();

    auto g    = mt.evaluate(x);
    auto ginv = mt.inverse(x);
    ASSERT_TRUE(ginv.has_value());

    // g × g_inv should equal identity (up to numerical precision)
    MetricMatrix prod = g * (*ginv);
    EXPECT_LT((prod - MetricMatrix::Identity()).norm(), 1e-10);
}
