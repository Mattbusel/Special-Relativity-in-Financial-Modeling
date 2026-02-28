#pragma once

/// @file include/srfm/engine.hpp
/// @brief Core Integration Engine — AGT-06 public API.
///
/// # Module: Integration Engine
///
/// ## Responsibility
/// Orchestrate the full SRFM signal-processing pipeline:
///   OHLCV bar data → BetaCalculator → LorentzTransform →
///   MarketManifold → MomentumProcessor →
///   MetricTensor + GeodesicSolver → Backtester → BacktestComparison
///
/// ## Usage
/// ```cpp
/// Engine engine;
/// auto bars = DataLoader::load_csv("prices.csv");
/// if (bars) {
///     auto result = engine.run_backtest(*bars);
///     if (result) fmt::print("{}\n", result->to_string());
/// }
/// ```
///
/// ## Guarantees
/// - Zero panics: all fallible paths return `std::optional` or `bool`
/// - Thread-safe reads: `run_backtest` and `run_stream_bar` are const
/// - No raw pointers

#include "srfm/backtest.hpp"
#include "srfm/manifold.hpp"
#include "srfm/types.hpp"
#include "srfm/constants.hpp"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace srfm::core {

// ─── OHLCV ────────────────────────────────────────────────────────────────────

/// A single OHLCV bar of market data.
struct OHLCV {
    double timestamp;  ///< Bar index or Unix epoch seconds
    double open;       ///< Opening price
    double high;       ///< High price
    double low;        ///< Low price
    double close;      ///< Closing price
    double volume;     ///< Traded volume
};

// ─── EngineConfig ─────────────────────────────────────────────────────────────

/// Configuration parameters for the core engine.
struct EngineConfig {
    /// Maximum market velocity used to normalise β (BetaCalculator denominator).
    double max_market_velocity = 1.0;

    /// Effective mass proxy for momentum corrections (m_eff in p_rel = γ m v).
    double effective_mass = 1.0;

    /// Minimum number of bars required to run a backtest.
    std::size_t min_bars = constants::MIN_RETURN_SERIES_LENGTH;

    /// Backtest configuration forwarded to the Backtester.
    backtest::BacktestConfig backtest_cfg{};

    /// If true, emit per-bar debug output to stderr.
    bool verbose = false;
};

// ─── PipelineBar ──────────────────────────────────────────────────────────────

/// Intermediate state for one bar after full pipeline processing.
struct PipelineBar {
    OHLCV                  raw;            ///< Original bar data
    double                 price_return;   ///< log or simple return for this bar
    BetaVelocity           beta;           ///< Normalised market velocity
    LorentzFactor          gamma;          ///< Lorentz factor at this bar
    manifold::SpacetimeEvent event;        ///< Spacetime embedding
    manifold::IntervalType interval_type;  ///< Causal character vs prior bar
};

// ─── Engine ───────────────────────────────────────────────────────────────────

/// Orchestrates the full relativistic signal-processing pipeline.
class Engine {
public:
    /// Construct with optional configuration.
    explicit Engine(EngineConfig config = EngineConfig{});

    /// Run a complete backtest on a sequence of OHLCV bars.
    ///
    /// # Pipeline
    /// 1. Compute close-to-close simple returns.
    /// 2. Compute β for each bar via BetaCalculator (rolling window = 5 bars).
    /// 3. Classify each bar's spacetime interval via MarketManifold.
    /// 4. Apply relativistic momentum corrections.
    /// 5. Feed BarData + returns into Backtester::run().
    ///
    /// # Returns
    /// `nullopt` if fewer than `config.min_bars` bars provided, or any
    /// downstream computation is numerically degenerate.
    [[nodiscard]] std::optional<backtest::BacktestComparison>
    run_backtest(std::span<const OHLCV> bars) const noexcept;

    /// Process a single bar in streaming mode, updating internal rolling state.
    ///
    /// Returns the PipelineBar result for the just-processed bar, or `nullopt`
    /// if the internal window has fewer than 2 bars (warm-up phase).
    [[nodiscard]] std::optional<PipelineBar>
    process_stream_bar(const OHLCV& bar) noexcept;

    /// Reset streaming state (clear internal bar window).
    void reset_stream() noexcept;

    /// Return a read-only view of the current streaming bar window.
    [[nodiscard]] std::span<const OHLCV> stream_window() const noexcept;

private:
    /// Compute simple returns from a price series.
    /// Returns vector of length (prices.size() − 1).
    [[nodiscard]] static std::vector<double>
    compute_returns(std::span<const double> prices) noexcept;

    /// Compute β for each bar using a rolling 5-bar window.
    /// Falls back to 0 (Newtonian) for the first few bars.
    [[nodiscard]] static std::vector<BetaVelocity>
    compute_betas(std::span<const double> prices,
                  double max_velocity) noexcept;

    /// Build a SpacetimeEvent from an OHLCV bar.
    [[nodiscard]] static manifold::SpacetimeEvent
    to_event(const OHLCV& bar, double momentum_indicator) noexcept;

    EngineConfig             config_;
    std::vector<OHLCV>       stream_window_;  ///< Rolling bar buffer for streaming
};

}  // namespace srfm::core
