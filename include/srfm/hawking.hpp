#pragma once

/// @file include/srfm/hawking.hpp
/// @brief Hawking Radiation Analogy — event-horizon detection for price series.
///
/// # Module: Hawking Radiation Signal
///
/// ## Conceptual Basis
/// Stephen Hawking showed that black holes emit thermal radiation due to
/// quantum effects near the event horizon.  As a black hole loses mass, its
/// temperature rises — an accelerating approach to its own end state.
///
/// In financial markets, an analogous "event horizon" occurs when a price
/// series approaches an extreme: a 3σ Bollinger Band exceedance.  Near this
/// horizon:
///
///   - **Information leaks** from the system: the price can no longer sustain
///     the extreme trend and must reverse or consolidate.
///   - **Hawking Temperature** T_H measures how close the price is to the
///     event horizon, scaled by the rate of approach.
///   - High T_H → high informational entropy → trend exhaustion / reversal.
///   - Low T_H  → low entropy → trend continuation likely.
///
/// ## Hawking Temperature Formula
///
///     T_H(t) = |P(t) − μ(t)| / σ(t) × (dP/dt) / σ(t)
///
/// where μ and σ are the rolling Bollinger Band mean and standard deviation.
/// The first factor measures how far into the band we are (proximity to
/// horizon) and the second measures the velocity of approach.
///
/// Equivalently, using z-score:
///
///     T_H(t) = z(t) × Δz(t)
///
/// where z = (P − μ) / σ and Δz is the bar-to-bar change in z-score.
///
/// ## Trading Signal
/// - T_H > HAWKING_HOT_THRESHOLD  → expect reversal (trade against trend)
/// - T_H < HAWKING_COLD_THRESHOLD → expect continuation (trade with trend)
/// - Between thresholds           → neutral / no position
///
/// ## Guarantees
/// - All public methods are noexcept (return std::optional on failure)
/// - No dynamic dispatch; all types are concrete
/// - Thread-safe reads: const methods safe for concurrent access

#include "srfm/backtest.hpp"
#include "srfm/engine.hpp"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace srfm::hawking {

// ─── Constants ────────────────────────────────────────────────────────────────

/// Default Bollinger Band window (bars).
inline constexpr std::size_t DEFAULT_BB_WINDOW = 20;

/// Default Bollinger Band multiplier for outer band (σ-multiple).
inline constexpr double DEFAULT_BB_SIGMA = 3.0;

/// T_H above this → "hot" → reversal signal.
inline constexpr double HAWKING_HOT_THRESHOLD  =  2.0;

/// T_H below this → "cold" → continuation signal.
inline constexpr double HAWKING_COLD_THRESHOLD = -2.0;

// ─── HawkingTemperature ───────────────────────────────────────────────────────

/// Per-bar Hawking temperature measurement.
///
/// Attributes:
///   temperature    Dimensionless T_H = z × Δz.
///   z_score        Current Bollinger Band z-score: (P − μ) / σ.
///   delta_z        Bar-to-bar change in z-score (velocity of approach).
///   bollinger_mean Rolling mean μ over the BB window.
///   bollinger_std  Rolling standard deviation σ over the BB window.
///   near_horizon   True iff |z_score| >= bb_sigma (outside the outer band).
struct HawkingTemperature {
    double temperature{0.0};      ///< T_H = z × Δz
    double z_score{0.0};          ///< (P − μ) / σ
    double delta_z{0.0};          ///< Change in z-score from previous bar
    double bollinger_mean{0.0};   ///< Rolling μ
    double bollinger_std{0.0};    ///< Rolling σ
    bool   near_horizon{false};   ///< |z_score| ≥ bb_sigma

    /// Format a compact one-liner for logging.
    [[nodiscard]] std::string to_string() const;
};

// ─── HawkingSignal ────────────────────────────────────────────────────────────

/// Trading signal derived from Hawking temperature.
enum class HawkingDirection {
    Reversal,      ///< T_H high → trend exhaustion → fade the move
    Continuation,  ///< T_H low  → healthy trend → follow the trend
    Neutral,       ///< Temperature in the normal range → no signal
};

/// Return a human-readable string for a HawkingDirection value.
[[nodiscard]] const char* to_string(HawkingDirection d) noexcept;

/// A single bar's Hawking-derived trading signal.
struct HawkingSignal {
    HawkingTemperature temperature;    ///< Full temperature measurement.
    HawkingDirection   direction;      ///< Trading implication.
    double             strength;       ///< |T_H| normalized to [0, 1].
    int                action;         ///< +1 (buy), -1 (sell), 0 (neutral).

    /// Format a compact one-liner for logging.
    [[nodiscard]] std::string to_string() const;
};

// ─── BollingerState ───────────────────────────────────────────────────────────

/// Internal rolling state for Bollinger Band computation.
struct BollingerState {
    std::vector<double> window;    ///< Rolling price window (oldest first).
    double sum{0.0};               ///< Running sum for fast mean.
    double sum_sq{0.0};            ///< Running sum of squares for fast σ.
    std::size_t max_size{DEFAULT_BB_WINDOW};

    /// Push a new price and pop the oldest if window is full.
    void push(double price) noexcept;

    /// Return the rolling mean (NaN-safe).
    [[nodiscard]] double mean() const noexcept;

    /// Return the rolling standard deviation (population σ).
    [[nodiscard]] double stddev() const noexcept;

    /// True iff window has reached its maximum size.
    [[nodiscard]] bool full() const noexcept { return window.size() == max_size; }
};

// ─── PriceEventHorizon ────────────────────────────────────────────────────────

/// Detects when a price series approaches its "event horizon" —
/// the Bollinger Band boundary at which the current trend cannot continue.
///
/// Usage:
/// ```cpp
/// PriceEventHorizon peh(bb_window=20, bb_sigma=3.0);
/// for (auto& bar : bars) {
///     auto temp = peh.update(bar.close);
///     if (temp) fmt::print("{}\n", temp->to_string());
/// }
/// ```
class PriceEventHorizon {
public:
    /// Configuration for the Bollinger Band event horizon detector.
    struct Config {
        /// Rolling window size for Bollinger Band mean and std dev.
        std::size_t bb_window{DEFAULT_BB_WINDOW};

        /// Band multiplier: outer band = μ ± bb_sigma × σ.
        double bb_sigma{DEFAULT_BB_SIGMA};

        /// T_H threshold above which "near horizon" is signalled.
        double hot_threshold{HAWKING_HOT_THRESHOLD};

        /// T_H threshold below which "cold" (continuation) is signalled.
        double cold_threshold{HAWKING_COLD_THRESHOLD};
    };

    explicit PriceEventHorizon(Config config = Config{}) noexcept;

    /// Update state with a new closing price and compute T_H.
    ///
    /// @param close  Closing price of the current bar.
    /// @returns      HawkingTemperature or nullopt during warm-up
    ///               (until ``bb_window`` bars have been seen).
    [[nodiscard]] std::optional<HawkingTemperature>
    update(double close) noexcept;

    /// Reset the internal rolling state (e.g. at session boundaries).
    void reset() noexcept;

    /// Return current window fill level (0 → full).
    [[nodiscard]] std::size_t window_fill() const noexcept {
        return state_.window.size();
    }

private:
    Config        config_;
    BollingerState state_;
    double        prev_z_{0.0};   ///< Previous bar's z-score (for Δz)
    bool          has_prev_z_{false};
};

// ─── HawkingSignalGenerator ───────────────────────────────────────────────────

/// Wraps PriceEventHorizon and converts temperatures to actionable signals.
///
/// Usage:
/// ```cpp
/// HawkingSignalGenerator gen;
/// for (auto& bar : bars) {
///     auto sig = gen.update(bar.close);
///     if (sig) {
///         // sig->action is +1, -1, or 0
///     }
/// }
/// ```
class HawkingSignalGenerator {
public:
    explicit HawkingSignalGenerator(
        PriceEventHorizon::Config config = PriceEventHorizon::Config{}
    ) noexcept;

    /// Update with a new closing price and generate a HawkingSignal.
    ///
    /// @param close  Closing price of the current bar.
    /// @returns      HawkingSignal or nullopt during warm-up.
    [[nodiscard]] std::optional<HawkingSignal>
    update(double close) noexcept;

    /// Reset internal state.
    void reset() noexcept;

private:
    PriceEventHorizon horizon_;
    const PriceEventHorizon::Config config_;

    /// Map a temperature to a direction.
    [[nodiscard]] HawkingDirection classify(double temperature) const noexcept;

    /// Map a temperature to a normalised strength in [0, 1].
    [[nodiscard]] static double strength(double temperature) noexcept;

    /// Map direction to a discrete action.
    [[nodiscard]] static int direction_to_action(HawkingDirection d) noexcept;
};

// ─── HawkingBacktestResult ────────────────────────────────────────────────────

/// Comparison of Hawking-signal strategy vs the existing TIMELIKE classifier.
struct HawkingBacktestResult {
    /// Performance of the Hawking temperature signal strategy.
    backtest::PerformanceMetrics hawking_metrics;

    /// Performance of the TIMELIKE (standard relativistic) strategy.
    backtest::PerformanceMetrics timelike_metrics;

    /// Mean |T_H| across all bars with a non-neutral signal.
    double mean_temperature{0.0};

    /// Fraction of bars where the Hawking signal was non-neutral.
    double signal_coverage{0.0};

    /// Improvement in Sharpe: hawking − timelike.
    [[nodiscard]] double sharpe_improvement() const noexcept {
        return hawking_metrics.sharpe_ratio - timelike_metrics.sharpe_ratio;
    }

    /// Format a one-line human-readable summary.
    [[nodiscard]] std::string to_string() const;
};

// ─── HawkingBacktest ──────────────────────────────────────────────────────────

/// Runs a backtest comparing the Hawking signal against the TIMELIKE classifier.
///
/// Usage:
/// ```cpp
/// HawkingBacktest hb;
/// auto result = hb.run(bars);
/// if (result) fmt::print("{}\n", result->to_string());
/// ```
class HawkingBacktest {
public:
    explicit HawkingBacktest(
        PriceEventHorizon::Config hawking_cfg = PriceEventHorizon::Config{},
        core::EngineConfig        engine_cfg  = core::EngineConfig{}
    ) noexcept;

    /// Run the comparison backtest on a sequence of OHLCV bars.
    ///
    /// Pipeline:
    /// 1. Compute Hawking signals for every bar via HawkingSignalGenerator.
    /// 2. Compute TIMELIKE signals via the standard Engine pipeline.
    /// 3. Evaluate both strategies using backtest::Backtester.
    ///
    /// @param bars  Input OHLCV sequence (minimum bb_window + 30 bars).
    /// @returns     HawkingBacktestResult or nullopt on insufficient data.
    [[nodiscard]] std::optional<HawkingBacktestResult>
    run(std::span<const core::OHLCV> bars) const noexcept;

private:
    HawkingSignalGenerator signal_gen_;
    core::Engine           engine_;
    PriceEventHorizon::Config hawking_cfg_;
};

}  // namespace srfm::hawking
