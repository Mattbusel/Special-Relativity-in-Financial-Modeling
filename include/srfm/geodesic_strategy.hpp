#pragma once

/// @file include/srfm/geodesic_strategy.hpp
/// @brief Extended Backtester with Geodesic Deviation Strategy — AGT-07.
///
/// # Module: Geodesic Strategy
///
/// ## Responsibility
/// Adds a third strategy mode — GEODESIC_DEVIATION — to the existing
/// Backtester infrastructure.  The geodesic deviation strategy goes long when
/// the market deviates above the rolling 75th percentile of recent deviations
/// (mean-reversion hypothesis: pulled-away markets snap back).
///
/// ## Strategy Modes
/// - RAW              — Unit position, unmodified signal (baseline)
/// - RELATIVISTIC     — γ-scaled position (original Backtester strategy)
/// - GEODESIC_DEVIATION — Long when deviation > rolling p75, flat otherwise
///
/// ## Guarantees
/// - All fallible paths return std::optional
/// - No raw pointers
/// - Thread-safe reads (const methods)

#include "srfm/backtest.hpp"
#include "srfm/types.hpp"

#include <optional>
#include <string>
#include <vector>

namespace srfm::backtest {

// ─── Types ────────────────────────────────────────────────────────────────────

/// Extended bar data that carries the geodesic deviation signal alongside the
/// standard BarData used by the base Backtester.
struct GeodesicBarData {
    BarData base;                ///< Standard bar: raw_signal, beta, benchmark
    double  geodesic_deviation;  ///< ||x_actual − x_geodesic||₂ from GeodesicDeviationCalculator
};

/// Strategy selection for ExtendedBacktester.
enum class StrategyMode {
    RAW,                 ///< Baseline: sign(signal) × return
    RELATIVISTIC,        ///< γ-scaled: sign(γ × signal) × return
    GEODESIC_DEVIATION,  ///< Mean-reversion: long when deviation > rolling p75
};

// ─── ExtendedBacktester ───────────────────────────────────────────────────────

/// Runs all three strategy modes side by side and reports metrics.
///
/// Usage
/// -----
/// ```cpp
/// ExtendedBacktester ext;
/// auto result = ext.run_triple(bars, asset_returns, "AAPL");
/// if (result) {
///     fmt::print("RAW Sharpe:        {:.3f}\n", result->raw.sharpe_ratio);
///     fmt::print("Relativistic Sharpe: {:.3f}\n", result->relativistic.sharpe_ratio);
///     fmt::print("Geodesic Sharpe:   {:.3f}\n", result->geodesic.sharpe_ratio);
/// }
/// ```
class ExtendedBacktester {
public:
    /// Three-way performance comparison across all strategy modes.
    struct TripleComparison {
        PerformanceMetrics raw;          ///< RAW strategy metrics
        PerformanceMetrics relativistic;  ///< RELATIVISTIC strategy metrics
        PerformanceMetrics geodesic;      ///< GEODESIC_DEVIATION strategy metrics
        std::string        ticker;        ///< Ticker symbol (informational)

        /// Human-readable comparison table.
        [[nodiscard]] std::string to_string() const;
    };

    /// Construct with backtesting configuration and rolling window size.
    ///
    /// # Arguments
    /// * `config`         — Risk-free rate, annualisation, effective mass, etc.
    /// * `rolling_window` — Window for rolling p75 deviation (in bars).
    explicit ExtendedBacktester(
        BacktestConfig config = BacktestConfig{},
        std::size_t    rolling_window = 100) noexcept;

    /// Run all three strategies and return a three-way comparison.
    ///
    /// # Arguments
    /// * `bars`          — One GeodesicBarData per time step.
    /// * `asset_returns` — Realised asset returns aligned to bars (same length).
    /// * `ticker`        — Ticker label for the output (informational only).
    ///
    /// # Returns
    /// TripleComparison, or nullopt if:
    ///   - bars and asset_returns have different sizes
    ///   - Fewer than MIN_RETURN_SERIES_LENGTH bars
    ///   - Any metric computation is numerically degenerate
    [[nodiscard]] std::optional<TripleComparison>
    run_triple(const std::vector<GeodesicBarData>& bars,
               const std::vector<double>&           asset_returns,
               const std::string&                   ticker = "") const noexcept;

    /// Compute only the geodesic strategy return series (for inspection).
    ///
    /// # Returns
    /// Return series, or nullopt on invalid input.
    [[nodiscard]] std::optional<std::vector<double>>
    geodesic_returns(const std::vector<GeodesicBarData>& bars,
                     const std::vector<double>&           asset_returns) const noexcept;

private:
    /// Compute rolling 75th percentile of geodesic_deviation up to (and
    /// including) bar i, looking back at most rolling_window_ bars.
    [[nodiscard]] double rolling_p75(
        const std::vector<GeodesicBarData>& bars,
        std::size_t                          i) const noexcept;

    /// Build the position series for the GEODESIC_DEVIATION strategy:
    ///   position_i = 1.0 if deviation_i > rolling_p75_i, else 0.0
    [[nodiscard]] std::vector<double>
    geodesic_positions(const std::vector<GeodesicBarData>& bars) const noexcept;

    BacktestConfig config_;
    Backtester     base_backtester_;
    std::size_t    rolling_window_;
};

} // namespace srfm::backtest
