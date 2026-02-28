#pragma once

/// @file include/srfm/backtest.hpp
/// @brief Relativistic Backtester — AGT-05 public API.
///
/// # Module: Relativistic Backtester
///
/// ## Responsibility
/// Feed every strategy signal through Lorentz corrections (γ-weighted) before
/// evaluation, and measure the performance lift — or cost — of relativistic
/// adjustment versus classical raw-signal strategies.
///
/// ## The Core Idea
/// In high-velocity market regimes (high β), conventional strategy signals
/// underweight information that is arriving "fast" relative to the market
/// observer frame.  Applying the Lorentz factor γ = 1/√(1−β²) re-weights each
/// signal proportional to the "market speed" at the time it was generated:
///
///     adjusted_signal_t = γ(β_t) · raw_signal_t
///
/// A strategy evaluated on adjusted signals implicitly up-weights signals from
/// fast-moving markets and down-weights signals from quiet, near-Newtonian
/// regimes (β ≈ 0, γ ≈ 1).
///
/// ## Performance Metrics
/// Four metrics are reported for both raw and relativistic strategies:
///   - Sharpe ratio:               (mean − r_f) / σ         (annualised)
///   - Sortino ratio:              (mean − r_f) / σ_down    (annualised)
///   - Maximum drawdown:           max peak-to-trough loss
///   - γ-weighted information ratio:
///       IR_γ = (mean(rel_ret − bm_ret) · mean(γ)) / σ(rel_ret − bm_ret)
///
/// ## Guarantees
/// - Zero UB: all fallible operations return `std::optional` or `bool`
/// - No raw pointers: ownership by value or const-reference
/// - Thread-safe reads: const member functions are safe concurrently
/// - Returns must be finite; NaN/Inf inputs produce `std::nullopt`
///
/// ## NOT Responsible For
/// - Signal generation (see src/momentum/)
/// - Covariance estimation (see src/tensor/)
/// - Data loading (see src/core/)

#include "srfm/types.hpp"
#include "srfm/constants.hpp"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace srfm::backtest {

// ─── Types ────────────────────────────────────────────────────────────────────

/// A single time-bar of backtester input.
struct BarData {
    double raw_signal;   ///< Strategy signal before relativistic correction
    BetaVelocity beta;   ///< Market velocity β at this bar
    double benchmark;    ///< Benchmark return for information-ratio computation
};

/// Strategy return series derived by applying a sign-following rule:
///   return_t = sign(signal_t) × asset_return_t
/// The caller is responsible for supplying the return series directly; the
/// backtester does not compute asset returns from prices.
using ReturnSeries = std::vector<double>;

/// A complete set of relativistic corrections for one return series.
struct LorentzCorrectedSeries {
    std::vector<double> gamma_factors;    ///< γ(β_t) for every bar
    std::vector<double> adjusted_signals; ///< γ_t × raw_signal_t
};

/// Performance metrics for a single strategy evaluation.
struct PerformanceMetrics {
    double sharpe_ratio;           ///< (mean_ret − r_f) / σ,  annualised
    double sortino_ratio;          ///< (mean_ret − r_f) / σ_down, annualised
    double max_drawdown;           ///< Peak-to-trough fractional loss (≥ 0)
    double gamma_weighted_ir;      ///< γ-weighted information ratio vs benchmark

    /// Human-readable summary line.
    [[nodiscard]] std::string to_string() const;
};

/// Side-by-side comparison of raw vs relativistic strategy metrics.
struct BacktestComparison {
    PerformanceMetrics raw;          ///< Metrics from unmodified signals
    PerformanceMetrics relativistic; ///< Metrics from γ-corrected signals

    double sharpe_lift()    const noexcept; ///< rel.sharpe − raw.sharpe
    double sortino_lift()   const noexcept; ///< rel.sortino − raw.sortino
    double drawdown_delta() const noexcept; ///< raw.mdd − rel.mdd  (positive = improvement)
    double ir_lift()        const noexcept; ///< rel.ir − raw.ir

    /// Formatted comparison table.
    [[nodiscard]] std::string to_string() const;
};

/// Configuration for a backtest run.
struct BacktestConfig {
    double risk_free_rate     = constants::DEFAULT_RISK_FREE_RATE;
    double annualisation      = constants::ANNUALISATION_FACTOR;
    double effective_mass     = 1.0;  ///< m_eff in p_rel = γ m_eff signal
    bool   verbose            = false;
};

// ─── PerformanceCalculator ────────────────────────────────────────────────────

/// Stateless utility for computing financial performance metrics.
///
/// All methods are static and operate on `std::span<const double>` for
/// zero-copy access to any contiguous container.
class PerformanceCalculator {
public:
    /// Compute annualised Sharpe ratio.
    ///
    /// # Formula
    ///   Sharpe = (mean(R) − r_f) / σ(R) × √ann
    ///
    /// # Returns
    /// `nullopt` if series has fewer than 2 elements, σ = 0, or any NaN/Inf.
    [[nodiscard]] static std::optional<double>
    sharpe(std::span<const double> returns,
           double risk_free_rate  = constants::DEFAULT_RISK_FREE_RATE,
           double annualisation   = constants::ANNUALISATION_FACTOR) noexcept;

    /// Compute annualised Sortino ratio (downside-deviation denominator).
    ///
    /// # Formula
    ///   Sortino = (mean(R) − r_f) / σ_down(R) × √ann
    ///
    /// where σ_down is the standard deviation of returns below `r_f`.
    ///
    /// # Returns
    /// `nullopt` if series is too short, downside-vol is zero, or any NaN/Inf.
    [[nodiscard]] static std::optional<double>
    sortino(std::span<const double> returns,
            double risk_free_rate  = constants::DEFAULT_RISK_FREE_RATE,
            double annualisation   = constants::ANNUALISATION_FACTOR) noexcept;

    /// Compute maximum drawdown of an equity curve.
    ///
    /// # Formula
    ///   MDD = max over t of { (peak_t − trough_t) / peak_t }
    ///   where peak_t = max_{s ≤ t} equity_curve[s]
    ///
    /// The equity curve is constructed by cumulative-summing the return series.
    ///
    /// # Returns
    /// Maximum drawdown in [0, 1].  Returns `nullopt` on empty input.
    [[nodiscard]] static std::optional<double>
    max_drawdown(std::span<const double> returns) noexcept;

    /// Compute γ-weighted information ratio.
    ///
    /// # Formula
    ///   IR_γ = (mean(active_ret) × mean(γ)) / σ(active_ret)
    ///   where active_ret_t = strategy_ret_t − benchmark_ret_t
    ///
    /// The γ factor up-weights the mean active return when signals were
    /// generated in high-velocity (high-γ) market regimes.
    ///
    /// # Returns
    /// `nullopt` if inputs are mismatched in length, too short, or numerically
    /// degenerate.
    [[nodiscard]] static std::optional<double>
    gamma_weighted_ir(std::span<const double> strategy_returns,
                      std::span<const double> benchmark_returns,
                      std::span<const double> gamma_factors) noexcept;

private:
    /// Mean of a span.  Unchecked — caller must ensure non-empty, finite.
    static double mean(std::span<const double> v) noexcept;
    /// Sample std-dev of a span.  Unchecked — caller ensures length ≥ 2.
    static double stddev(std::span<const double> v, double mean_val) noexcept;
    /// Downside std-dev relative to `threshold`.
    static double downside_stddev(std::span<const double> v,
                                  double threshold) noexcept;
};

// ─── LorentzSignalAdjuster ────────────────────────────────────────────────────

/// Applies relativistic Lorentz corrections to a raw signal series.
///
/// For each bar t:
///   γ_t = 1 / √(1 − β_t²)               (Lorentz factor)
///   adjusted_t = γ_t × m_eff × raw_t     (relativistic momentum analog)
///
/// When β_t is invalid (|β| ≥ BETA_MAX_SAFE or non-finite), the corrected bar
/// falls back to the raw signal (effectively γ = 1).
class LorentzSignalAdjuster {
public:
    /// Construct with effective mass parameter m_eff.
    ///
    /// # Arguments
    /// * `effective_mass` — Liquidity proxy, must be > 0.  Defaults to 1.0.
    explicit LorentzSignalAdjuster(double effective_mass = 1.0);

    /// Apply Lorentz corrections to a bar series.
    ///
    /// # Arguments
    /// * `bars` — Input bars containing raw_signal and beta for each time step.
    ///
    /// # Returns
    /// `LorentzCorrectedSeries` with per-bar γ values and adjusted signals.
    /// Returns `nullopt` if `bars` is empty or `effective_mass <= 0`.
    [[nodiscard]] std::optional<LorentzCorrectedSeries>
    adjust(std::span<const BarData> bars) const noexcept;

    /// Compute the Lorentz factor γ for a single β value.
    ///
    /// # Returns
    /// γ ≥ 1.0, or `nullopt` for invalid β.
    [[nodiscard]] static std::optional<double>
    lorentz_gamma(BetaVelocity beta) noexcept;

private:
    double effective_mass_;
};

// ─── Backtester ───────────────────────────────────────────────────────────────

/// Runs raw and relativistic strategies side by side and reports metrics.
///
/// Usage pattern:
/// ```cpp
/// BacktestConfig cfg;
/// cfg.risk_free_rate = 0.02 / 252.0;   // daily r_f
/// cfg.annualisation  = 252.0;
///
/// Backtester bt(cfg);
/// auto cmp = bt.run(bars, returns);
/// if (cmp) fmt::print("{}\n", cmp->to_string());
/// ```
class Backtester {
public:
    /// Construct with configuration.
    explicit Backtester(BacktestConfig config = BacktestConfig{});

    /// Run a full side-by-side backtest.
    ///
    /// # Arguments
    /// * `bars`    — One entry per time step: raw signal, β, benchmark return.
    /// * `returns` — Realised asset returns aligned to `bars` (same length).
    ///              Signalling rule: strategy return = sign(signal) × asset ret.
    ///
    /// # Returns
    /// `BacktestComparison` containing both metric sets, or `nullopt` if:
    ///   - Input lengths mismatch
    ///   - Fewer than MIN_RETURN_SERIES_LENGTH bars provided
    ///   - Any metric calculation is numerically degenerate
    [[nodiscard]] std::optional<BacktestComparison>
    run(std::span<const BarData> bars,
        std::span<const double>  asset_returns) const noexcept;

    /// Compute only the Lorentz-corrected signal series (no strategy eval).
    /// Useful for inspection / visualisation.
    [[nodiscard]] std::optional<LorentzCorrectedSeries>
    apply_corrections(std::span<const BarData> bars) const noexcept;

private:
    /// Compute PerformanceMetrics for a given return series + γ + benchmark.
    std::optional<PerformanceMetrics>
    compute_metrics(std::span<const double> returns,
                    std::span<const double> benchmark_returns,
                    std::span<const double> gamma_factors) const noexcept;

    BacktestConfig        config_;
    LorentzSignalAdjuster adjuster_;
};

}  // namespace srfm::backtest
