/// @file tests/integration/test_e2e_pipeline.cpp
/// @brief End-to-end pipeline integration tests using the fixture CSV file.
///
/// These tests verify the complete path:
///   DataLoader::load_csv → Engine::run_backtest →
///   BacktestComparison → PerformanceCalculator assertions
///
/// The fixture file at tests/fixtures/sample_ohlcv.csv contains 100 rows of
/// deterministic synthetic OHLCV data in the 150–218 price range.

#include "srfm/engine.hpp"
#include "srfm/data_loader.hpp"
#include "srfm/backtest.hpp"
#include "srfm/constants.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

using namespace srfm;
using namespace srfm::core;
using namespace srfm::backtest;

namespace {

/// Resolve the fixture path relative to this translation unit.
/// CMake sets SRFM_FIXTURES_DIR; fall back to a relative heuristic.
std::string fixture_path() {
#ifdef SRFM_FIXTURES_DIR
    return std::string(SRFM_FIXTURES_DIR) + "/sample_ohlcv.csv";
#else
    // Walk up from the binary's working directory until we find tests/fixtures/.
    std::filesystem::path p = std::filesystem::current_path();
    for (int depth = 0; depth < 6; ++depth) {
        auto candidate = p / "tests" / "fixtures" / "sample_ohlcv.csv";
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
        if (p.has_parent_path() && p != p.parent_path()) {
            p = p.parent_path();
        } else {
            break;
        }
    }
    return "tests/fixtures/sample_ohlcv.csv";
#endif
}

}  // anonymous namespace

// ─── Fixture loading ──────────────────────────────────────────────────────────

TEST(E2EPipeline, LoadFixtureCsvSucceeds) {
    auto bars = DataLoader::load_csv(fixture_path());
    ASSERT_TRUE(bars.has_value()) << "DataLoader::load_csv returned nullopt for "
                                  << fixture_path();
    EXPECT_GE(bars->size(), static_cast<std::size_t>(50))
        << "Expected at least 50 bars in the fixture CSV";
}

TEST(E2EPipeline, FixtureBarsAreValid) {
    auto bars = DataLoader::load_csv(fixture_path());
    ASSERT_TRUE(bars.has_value());
    for (const auto& bar : *bars) {
        EXPECT_TRUE(DataLoader::validate_bar(bar))
            << "Invalid bar at timestamp=" << bar.timestamp;
    }
}

TEST(E2EPipeline, FixturePricesInExpectedRange) {
    auto bars = DataLoader::load_csv(fixture_path());
    ASSERT_TRUE(bars.has_value());
    for (const auto& bar : *bars) {
        EXPECT_GE(bar.close, 100.0) << "Close below lower bound at ts=" << bar.timestamp;
        EXPECT_LE(bar.close, 300.0) << "Close above upper bound at ts=" << bar.timestamp;
        EXPECT_GE(bar.volume, 0.0);
    }
}

// ─── Full pipeline: DataLoader → Engine → BacktestComparison ─────────────────

TEST(E2EPipeline, RunBacktestOnFixtureSucceeds) {
    auto bars = DataLoader::load_csv(fixture_path());
    ASSERT_TRUE(bars.has_value());

    Engine engine;
    auto result = engine.run_backtest(*bars);
    ASSERT_TRUE(result.has_value())
        << "Engine::run_backtest returned nullopt on " << bars->size() << " bars";
}

TEST(E2EPipeline, SharpeRatioInReasonableRange) {
    auto bars = DataLoader::load_csv(fixture_path());
    ASSERT_TRUE(bars.has_value());

    Engine engine;
    auto result = engine.run_backtest(*bars);
    ASSERT_TRUE(result.has_value());

    // Sharpe should be finite and in [-5, 5] for any realistic dataset.
    EXPECT_TRUE(std::isfinite(result->raw.sharpe_ratio))
        << "Raw Sharpe is not finite";
    EXPECT_GE(result->raw.sharpe_ratio, -5.0)
        << "Raw Sharpe is unreasonably negative";
    EXPECT_LE(result->raw.sharpe_ratio, 5.0)
        << "Raw Sharpe is unreasonably large";

    EXPECT_TRUE(std::isfinite(result->relativistic.sharpe_ratio))
        << "Relativistic Sharpe is not finite";
    EXPECT_GE(result->relativistic.sharpe_ratio, -5.0)
        << "Relativistic Sharpe is unreasonably negative";
    EXPECT_LE(result->relativistic.sharpe_ratio, 5.0)
        << "Relativistic Sharpe is unreasonably large";
}

TEST(E2EPipeline, NoNaNOrInfInOutputs) {
    auto bars = DataLoader::load_csv(fixture_path());
    ASSERT_TRUE(bars.has_value());

    Engine engine;
    auto result = engine.run_backtest(*bars);
    ASSERT_TRUE(result.has_value());

    // Raw metrics
    EXPECT_TRUE(std::isfinite(result->raw.sharpe_ratio))        << "raw.sharpe NaN/Inf";
    EXPECT_TRUE(std::isfinite(result->raw.sortino_ratio))       << "raw.sortino NaN/Inf";
    EXPECT_TRUE(std::isfinite(result->raw.max_drawdown))        << "raw.max_drawdown NaN/Inf";
    EXPECT_TRUE(std::isfinite(result->raw.gamma_weighted_ir))   << "raw.gamma_ir NaN/Inf";

    // Relativistic metrics
    EXPECT_TRUE(std::isfinite(result->relativistic.sharpe_ratio))
        << "relativistic.sharpe NaN/Inf";
    EXPECT_TRUE(std::isfinite(result->relativistic.sortino_ratio))
        << "relativistic.sortino NaN/Inf";
    EXPECT_TRUE(std::isfinite(result->relativistic.max_drawdown))
        << "relativistic.max_drawdown NaN/Inf";
    EXPECT_TRUE(std::isfinite(result->relativistic.gamma_weighted_ir))
        << "relativistic.gamma_ir NaN/Inf";

    // Aggregate γ diagnostics
    EXPECT_TRUE(std::isfinite(result->mean_gamma))        << "mean_gamma NaN/Inf";
    EXPECT_TRUE(std::isfinite(result->max_gamma_applied)) << "max_gamma_applied NaN/Inf";
    EXPECT_TRUE(std::isfinite(result->relativistic_lift)) << "relativistic_lift NaN/Inf";
}

TEST(E2EPipeline, MaxDrawdownInUnitInterval) {
    auto bars = DataLoader::load_csv(fixture_path());
    ASSERT_TRUE(bars.has_value());

    Engine engine;
    auto result = engine.run_backtest(*bars);
    ASSERT_TRUE(result.has_value());

    EXPECT_GE(result->raw.max_drawdown, 0.0);
    EXPECT_LE(result->raw.max_drawdown, 1.0);
    EXPECT_GE(result->relativistic.max_drawdown, 0.0);
    EXPECT_LE(result->relativistic.max_drawdown, 1.0);
}

TEST(E2EPipeline, MeanGammaAtLeastOne) {
    auto bars = DataLoader::load_csv(fixture_path());
    ASSERT_TRUE(bars.has_value());

    Engine engine;
    auto result = engine.run_backtest(*bars);
    ASSERT_TRUE(result.has_value());

    // γ = 1/√(1−β²) ≥ 1 by construction; the mean should reflect that.
    EXPECT_GE(result->mean_gamma, 1.0)
        << "mean_gamma below 1.0; Lorentz factor invariant violated";
}

TEST(E2EPipeline, LiftAccessorsFiniteAndConsistent) {
    auto bars = DataLoader::load_csv(fixture_path());
    ASSERT_TRUE(bars.has_value());

    Engine engine;
    auto result = engine.run_backtest(*bars);
    ASSERT_TRUE(result.has_value());

    EXPECT_TRUE(std::isfinite(result->sharpe_lift()));
    EXPECT_TRUE(std::isfinite(result->sortino_lift()));
    EXPECT_TRUE(std::isfinite(result->drawdown_delta()));
    EXPECT_TRUE(std::isfinite(result->ir_lift()));

    // sharpe_lift must equal relativistic.sharpe - raw.sharpe.
    const double expected = result->relativistic.sharpe_ratio - result->raw.sharpe_ratio;
    EXPECT_NEAR(result->sharpe_lift(), expected, 1e-12);
}

// ─── PerformanceCalculator standalone assertions ──────────────────────────────

TEST(E2EPipeline, PerformanceCalculatorSharpeOnSyntheticData) {
    // Constant positive returns → large positive Sharpe.
    const std::vector<double> returns(60, 0.001);
    auto sharpe = PerformanceCalculator::sharpe(returns);
    ASSERT_TRUE(sharpe.has_value());
    EXPECT_GT(*sharpe, 0.0);
    EXPECT_TRUE(std::isfinite(*sharpe));
}

TEST(E2EPipeline, PerformanceCalculatorSharpeOnFixtureReturns) {
    auto bars = DataLoader::load_csv(fixture_path());
    ASSERT_TRUE(bars.has_value());
    ASSERT_GE(bars->size(), 2u);

    // Build simple close-to-close returns.
    std::vector<double> returns;
    returns.reserve(bars->size() - 1);
    for (std::size_t i = 1; i < bars->size(); ++i) {
        const double r = ((*bars)[i].close - (*bars)[i-1].close) / (*bars)[i-1].close;
        returns.push_back(r);
    }

    auto sharpe = PerformanceCalculator::sharpe(returns);
    ASSERT_TRUE(sharpe.has_value());
    EXPECT_GE(*sharpe, -5.0);
    EXPECT_LE(*sharpe, 5.0);
    EXPECT_TRUE(std::isfinite(*sharpe));
}

TEST(E2EPipeline, PerformanceCalculatorMaxDrawdownNonNegative) {
    auto bars = DataLoader::load_csv(fixture_path());
    ASSERT_TRUE(bars.has_value());
    ASSERT_GE(bars->size(), 2u);

    std::vector<double> returns;
    returns.reserve(bars->size() - 1);
    for (std::size_t i = 1; i < bars->size(); ++i) {
        const double r = ((*bars)[i].close - (*bars)[i-1].close) / (*bars)[i-1].close;
        returns.push_back(r);
    }

    auto mdd = PerformanceCalculator::max_drawdown(returns);
    ASSERT_TRUE(mdd.has_value());
    EXPECT_GE(*mdd, 0.0);
    EXPECT_LE(*mdd, 1.0);
}

// ─── Reproducibility: two runs on the same data produce identical results ─────

TEST(E2EPipeline, DeterministicResults) {
    auto bars = DataLoader::load_csv(fixture_path());
    ASSERT_TRUE(bars.has_value());

    Engine engine_a;
    Engine engine_b;

    auto result_a = engine_a.run_backtest(*bars);
    auto result_b = engine_b.run_backtest(*bars);

    ASSERT_TRUE(result_a.has_value());
    ASSERT_TRUE(result_b.has_value());

    EXPECT_DOUBLE_EQ(result_a->raw.sharpe_ratio, result_b->raw.sharpe_ratio);
    EXPECT_DOUBLE_EQ(result_a->relativistic.sharpe_ratio,
                     result_b->relativistic.sharpe_ratio);
    EXPECT_DOUBLE_EQ(result_a->mean_gamma, result_b->mean_gamma);
}
