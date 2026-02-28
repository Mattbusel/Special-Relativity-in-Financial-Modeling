/// @file src/backtest/backtester.cpp
/// @brief Implementation of the Backtester class.
///
/// The Backtester orchestrates:
///   1. Lorentz signal correction via LorentzSignalAdjuster
///   2. Strategy return construction (sign-following rule)
///   3. Performance metric evaluation for both raw and relativistic strategies
///   4. Side-by-side comparison via BacktestComparison

#include "srfm/backtest.hpp"
#include "srfm/constants.hpp"

#include <cmath>
#include <span>
#include <vector>

namespace srfm::backtest {

// ─── Backtester ───────────────────────────────────────────────────────────────

Backtester::Backtester(BacktestConfig config)
    : config_(config)
    , adjuster_(config.effective_mass) {}

std::optional<LorentzCorrectedSeries>
Backtester::apply_corrections(std::span<const BarData> bars) const noexcept {
    return adjuster_.adjust(bars);
}

std::optional<PerformanceMetrics>
Backtester::compute_metrics(std::span<const double> returns,
                             std::span<const double> benchmark_returns,
                             std::span<const double> gamma_factors) const noexcept {
    auto sh = PerformanceCalculator::sharpe(
        returns, config_.risk_free_rate, config_.annualisation);
    if (!sh.has_value()) return std::nullopt;

    auto so = PerformanceCalculator::sortino(
        returns, config_.risk_free_rate, config_.annualisation);
    if (!so.has_value()) return std::nullopt;

    auto mdd = PerformanceCalculator::max_drawdown(returns);
    if (!mdd.has_value()) return std::nullopt;

    auto ir = PerformanceCalculator::gamma_weighted_ir(
        returns, benchmark_returns, gamma_factors);
    if (!ir.has_value()) return std::nullopt;

    return PerformanceMetrics{
        .sharpe_ratio      = *sh,
        .sortino_ratio     = *so,
        .max_drawdown      = *mdd,
        .gamma_weighted_ir = *ir,
    };
}

std::optional<BacktestComparison>
Backtester::run(std::span<const BarData> bars,
                std::span<const double>  asset_returns) const noexcept {
    // ── Input validation ────────────────────────────────────────────────────
    if (bars.size() < constants::MIN_RETURN_SERIES_LENGTH) return std::nullopt;
    if (asset_returns.size() != bars.size())               return std::nullopt;

    const std::size_t n = bars.size();

    // ── Step 1: Compute Lorentz-corrected signals ───────────────────────────
    auto corrected = adjuster_.adjust(bars);
    if (!corrected.has_value()) return std::nullopt;

    // ── Step 2: Extract benchmark returns and gamma factors ─────────────────
    std::vector<double> benchmark(n);
    for (std::size_t i = 0; i < n; ++i) {
        benchmark[i] = bars[i].benchmark;
    }

    const std::vector<double>& gammas = corrected->gamma_factors;

    // ── Step 3: Construct strategy returns via sign-following rule ──────────
    // raw_strategy_return_t  = sign(raw_signal_t)  × asset_return_t
    // adj_strategy_return_t  = sign(adj_signal_t)  × asset_return_t
    //
    // Because γ > 0 and m_eff > 0, sign(adj_signal) = sign(raw_signal).
    // The direction of the trade is identical; γ alters the signal magnitude
    // which in a pure sign-following strategy does not change the P&L per bar.
    // The relativistic lift in this model comes through the γ-weighted IR
    // (the benchmark-normalised performance metric), which explicitly weights
    // the information content by the market velocity at each bar.
    std::vector<double> raw_returns(n);
    std::vector<double> adj_returns(n);

    for (std::size_t i = 0; i < n; ++i) {
        const double raw_sign = (bars[i].raw_signal >= 0.0) ? 1.0 : -1.0;
        const double adj_sign = (corrected->adjusted_signals[i] >= 0.0) ? 1.0 : -1.0;
        raw_returns[i] = raw_sign * asset_returns[i];
        adj_returns[i] = adj_sign * asset_returns[i];
    }

    // ── Step 4: Compute metrics for both strategies ──────────────────────────
    // For γ-weighted IR: raw strategy uses uniform γ=1; adjusted uses actual γ.
    std::vector<double> unit_gammas(n, 1.0);

    auto raw_metrics = compute_metrics(raw_returns, benchmark, unit_gammas);
    if (!raw_metrics.has_value()) return std::nullopt;

    auto adj_metrics = compute_metrics(adj_returns, benchmark, gammas);
    if (!adj_metrics.has_value()) return std::nullopt;

    return BacktestComparison{
        .raw          = *raw_metrics,
        .relativistic = *adj_metrics,
    };
}

}  // namespace srfm::backtest
