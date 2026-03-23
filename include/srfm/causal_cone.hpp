#pragma once

/// @file include/srfm/causal_cone.hpp
/// @brief Causal Cone Filter — light-cone analogue for financial time series.
///
/// # Module: Causal Cone Filter
///
/// ## Conceptual Basis
/// In special relativity, an event B is causally influenced by event A only
/// if A lies within the *past light cone* of B — i.e. the spacetime interval
/// ds²(A→B) < 0 (TIMELIKE).  Events outside the light cone (SPACELIKE) cannot
/// causally affect B.
///
/// This module applies the same logic to financial OHLCV bars:
///
///   - A historical bar A is a *causal predecessor* of the current bar B iff
///     ``ds²(A, B) < 0`` (TIMELIKE or near-LIGHTLIKE) in the embedded
///     financial spacetime.
///   - Only causal predecessors contribute to feature computation.
///   - SPACELIKE bars are excluded as "causally disconnected" noise sources.
///
/// ## Hypothesis
/// Signals computed exclusively from causally-connected bars should exhibit
/// higher predictive accuracy because they exclude market noise that arrived
/// at speeds inconsistent with the bar's causal past.
///
/// ## Core Types
/// - ``CausalHistory``   — the causal predecessors of one bar
/// - ``CausalConeFilter`` — scans a bar sequence and builds causal histories
/// - ``CausalSignal``    — a feature vector built only from causal predecessors
/// - ``CausalBacktest``  — compares CausalSignal vs all-bars signal on a dataset
///
/// ## Guarantees
/// - All methods are `noexcept` where no allocation can fail
/// - No raw pointers; ownership by value or `const&`
/// - Returns `std::optional` for all fallible operations
/// - Thread-safe reads: const methods can be called concurrently

#include "srfm/manifold.hpp"
#include "srfm/backtest.hpp"
#include "srfm/engine.hpp"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace srfm::causal {

// ─── CausalHistory ────────────────────────────────────────────────────────────

/// The set of past bars that causally "matter" to a given bar.
///
/// A bar is a causal predecessor iff the spacetime interval from that bar to
/// the current bar is TIMELIKE (ds² < 0).  SPACELIKE bars are excluded.
struct CausalHistory {
    /// The current bar (the "effect" event).
    manifold::SpacetimeEvent event;

    /// Ordered list of causal predecessor events (oldest first).
    /// Contains only TIMELIKE (ds² < 0) predecessors of ``event``.
    std::vector<manifold::SpacetimeEvent> causal_predecessors;

    /// Indices into the original bar array for each causal predecessor.
    std::vector<std::size_t> predecessor_indices;

    /// Number of bars scanned (including SPACELIKE/excluded ones).
    std::size_t total_bars_scanned{0};

    /// Return the fraction of scanned bars that are causal predecessors.
    [[nodiscard]] double causal_fraction() const noexcept {
        if (total_bars_scanned == 0) return 0.0;
        return static_cast<double>(causal_predecessors.size())
             / static_cast<double>(total_bars_scanned);
    }
};

// ─── CausalSignal ─────────────────────────────────────────────────────────────

/// A trading feature vector computed only from causally-connected bars.
///
/// Features:
///   - causal_mean_return:   Mean return of causal predecessor bars.
///   - causal_vol:           Volatility (std dev) of causal predecessor returns.
///   - causal_momentum:      Weighted sum of returns, γ-weighted by causal strength.
///   - causal_fraction:      Fraction of look-back bars that are causal.
///   - all_bars_mean_return: Same metric computed over ALL bars (for comparison).
struct CausalSignal {
    double causal_mean_return{0.0};   ///< Mean return, causal bars only.
    double causal_vol{0.0};           ///< Volatility of causal bar returns.
    double causal_momentum{0.0};      ///< Weighted momentum from causal bars.
    double causal_fraction{0.0};      ///< Fraction of bars that are causal.
    double all_bars_mean_return{0.0}; ///< Baseline: mean over ALL look-back bars.
    std::size_t n_causal{0};          ///< Number of causal predecessor bars used.
    std::size_t n_total{0};           ///< Total bars in the look-back window.
};

// ─── CausalConeFilter ─────────────────────────────────────────────────────────

/// Filters a bar sequence to identify causal predecessors for each bar.
///
/// For every bar B in the sequence, looks back up to ``look_back`` bars and
/// classifies each as TIMELIKE (causal) or SPACELIKE (excluded) using
/// ``SpacetimeInterval::classify``.
///
/// Usage:
/// ```cpp
/// CausalConeFilter filter(look_back = 20);
/// auto histories = filter.build_histories(bars);
/// for (auto& h : histories) {
///     auto sig = filter.compute_signal(h, bars, bar_index);
///     // use sig.causal_mean_return instead of all-bars mean
/// }
/// ```
class CausalConeFilter {
public:
    /// Configuration for the causal cone filter.
    struct Config {
        /// Number of past bars to scan for causal predecessors.
        std::size_t look_back{20};

        /// Speed of information (same as SPEED_OF_INFORMATION in manifold).
        double c_market{constants::SPEED_OF_INFORMATION};

        /// If true, include LIGHTLIKE (ds² ≈ 0) bars in the causal set.
        bool include_lightlike{true};

        /// If true, emit per-bar diagnostics to stderr.
        bool verbose{false};
    };

    /// Construct with an optional configuration.
    explicit CausalConeFilter(Config config = Config{}) noexcept;

    /// Build a CausalHistory for every bar in the sequence.
    ///
    /// The first ``look_back`` bars will have fewer predecessors as the
    /// rolling window fills up.
    ///
    /// @param bars     Sequence of OHLCV bars (must be pre-converted to events).
    /// @param events   Corresponding spacetime events for each bar.
    /// @returns        One CausalHistory per bar (same size as ``bars``).
    [[nodiscard]] std::vector<CausalHistory>
    build_histories(
        std::span<const core::OHLCV>            bars,
        std::span<const manifold::SpacetimeEvent> events
    ) const noexcept;

    /// Compute a CausalSignal for bar at index ``bar_idx``.
    ///
    /// @param history  The CausalHistory for this bar.
    /// @param returns  Close-to-close returns for the full bar sequence.
    /// @param bar_idx  Index of the current bar in ``returns``.
    /// @returns        CausalSignal or nullopt if history has no causal predecessors.
    [[nodiscard]] std::optional<CausalSignal>
    compute_signal(
        const CausalHistory&          history,
        std::span<const double>       returns,
        std::size_t                   bar_idx
    ) const noexcept;

    /// Return the configured look_back window.
    [[nodiscard]] std::size_t look_back() const noexcept { return config_.look_back; }

private:
    Config config_;

    /// True if the interval type should be included in the causal set.
    [[nodiscard]] bool is_causal(manifold::IntervalType t) const noexcept;
};

// ─── CausalBacktestResult ─────────────────────────────────────────────────────

/// Comparison of CausalSignal-based strategy vs all-bars baseline strategy.
struct CausalBacktestResult {
    /// Performance of the causal-only signal strategy.
    backtest::PerformanceMetrics causal_metrics;

    /// Performance of the all-bars baseline strategy.
    backtest::PerformanceMetrics baseline_metrics;

    /// Mean fraction of bars that were causal across the full dataset.
    double mean_causal_fraction{0.0};

    /// Improvement in Sharpe ratio: causal − baseline.
    [[nodiscard]] double sharpe_improvement() const noexcept {
        return causal_metrics.sharpe_ratio - baseline_metrics.sharpe_ratio;
    }

    /// Improvement in Sortino ratio: causal − baseline.
    [[nodiscard]] double sortino_improvement() const noexcept {
        return causal_metrics.sortino_ratio - baseline_metrics.sortino_ratio;
    }

    /// Format a one-line human-readable summary.
    [[nodiscard]] std::string to_string() const;
};

// ─── CausalBacktest ───────────────────────────────────────────────────────────

/// Runs a backtest comparison: CausalSignal strategy vs all-bars baseline.
///
/// Usage:
/// ```cpp
/// CausalBacktest cb;
/// auto result = cb.run(bars);
/// if (result) fmt::print("{}\n", result->to_string());
/// ```
class CausalBacktest {
public:
    /// Construct with optional filter and engine configs.
    explicit CausalBacktest(
        CausalConeFilter::Config filter_cfg = CausalConeFilter::Config{},
        core::EngineConfig       engine_cfg = core::EngineConfig{}
    ) noexcept;

    /// Run the full comparison backtest on a sequence of OHLCV bars.
    ///
    /// Pipeline:
    /// 1. Convert bars to spacetime events via the Engine.
    /// 2. Build CausalHistory for every bar.
    /// 3. Compute CausalSignal for every bar.
    /// 4. Evaluate both strategies via the Backtester.
    ///
    /// @param bars  Input OHLCV sequence (minimum look_back + 30 bars).
    /// @returns     CausalBacktestResult or nullopt on insufficient data.
    [[nodiscard]] std::optional<CausalBacktestResult>
    run(std::span<const core::OHLCV> bars) const noexcept;

private:
    CausalConeFilter filter_;
    core::Engine     engine_;
};

}  // namespace srfm::causal
