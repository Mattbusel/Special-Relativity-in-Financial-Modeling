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

    // ── Step 3: Construct strategy returns ──────────────────────────────────
    //
    // Raw strategy (unit position):
    //   return_t = sign(raw_signal_t) × asset_return_t
    //
    // Relativistic strategy (γ-magnitude-weighted position):
    //   position_t = sign(adj_signal_t) × clamp(γ_t, 1.0, max_gamma)
    //   return_t   = position_t × asset_return_t
    //
    // In high-velocity markets (γ > 1) the relativistic strategy scales up
    // its position, up to max_gamma times the base unit size.  The raw
    // strategy always uses unit position size regardless of market velocity.
    // This produces measurably different Sharpe/Sortino/MDD between the two.
    std::vector<double> raw_returns(n);
    std::vector<double> adj_returns(n);

    double gamma_sum        = 0.0;
    double max_gamma_seen   = 1.0;
    const double max_gamma_cap = config_.max_gamma;

    for (std::size_t i = 0; i < n; ++i) {
        const double raw_sign   = (bars[i].raw_signal >= 0.0) ? 1.0 : -1.0;
        const double adj_sign   = (corrected->adjusted_signals[i] >= 0.0) ? 1.0 : -1.0;

        // γ multiplier: clamp to [1.0, max_gamma_cap] so position is always
        // at least unit size and never more than max_gamma_cap times unit.
        const double gamma_i    = gammas[i];
        const double gamma_mult = std::max(1.0, std::min(gamma_i, max_gamma_cap));

        raw_returns[i] = raw_sign * asset_returns[i];
        adj_returns[i] = adj_sign * gamma_mult * asset_returns[i];

        gamma_sum     += gamma_i;
        if (gamma_mult > max_gamma_seen) {
            max_gamma_seen = gamma_mult;
        }
    }

    const double mean_gamma_val = gamma_sum / static_cast<double>(n);

    // ── Step 4: Compute metrics for both strategies ──────────────────────────
    // For γ-weighted IR: raw strategy uses uniform γ=1; adjusted uses actual γ.
    std::vector<double> unit_gammas(n, 1.0);

    auto raw_metrics = compute_metrics(raw_returns, benchmark, unit_gammas);
    if (!raw_metrics.has_value()) return std::nullopt;

    auto adj_metrics = compute_metrics(adj_returns, benchmark, gammas);
    if (!adj_metrics.has_value()) return std::nullopt;

    // relativistic_lift = IR_γ_rel / IR_γ_raw  (ratio, not difference)
    const double raw_ir = raw_metrics->gamma_weighted_ir;
    const double rel_ir = adj_metrics->gamma_weighted_ir;
    const double lift   = (std::abs(raw_ir) > 1e-12) ? (rel_ir / raw_ir) : 0.0;

    return BacktestComparison{
        .raw               = *raw_metrics,
        .relativistic      = *adj_metrics,
        .mean_gamma        = mean_gamma_val,
        .max_gamma_applied = max_gamma_seen,
        .relativistic_lift = lift,
    };
}

}  // namespace srfm::backtest
