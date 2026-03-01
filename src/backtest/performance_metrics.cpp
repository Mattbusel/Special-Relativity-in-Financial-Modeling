/// @file src/backtest/performance_metrics.cpp
/// @brief Implementation of PerformanceCalculator and LorentzSignalAdjuster.
///
/// All math is documented inline. Fallible paths return std::nullopt; no
/// function ever calls abort(), assert(), or throws an exception.

#include "srfm/backtest.hpp"
#include "srfm/constants.hpp"

#include <algorithm>
#include <cmath>
#include <fmt/format.h>
#include <numeric>
#include <span>

namespace srfm::backtest {

// ─── Internal helpers (file-local) ────────────────────────────────────────────

namespace {

/// Return false if any element of `v` is NaN or ±Inf.
[[nodiscard]] bool all_finite(std::span<const double> v) noexcept {
    for (double x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

}  // namespace

// ─── PerformanceCalculator — private statics ──────────────────────────────────

double PerformanceCalculator::mean(std::span<const double> v) noexcept {
    // Unchecked: caller guarantees non-empty, finite.
    const double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / static_cast<double>(v.size());
}

double PerformanceCalculator::stddev(std::span<const double> v,
                                      double mean_val) noexcept {
    // Sample std-dev (Bessel-corrected, n−1 denominator).
    // Caller guarantees v.size() >= 2.
    double sq_sum = 0.0;
    for (double x : v) {
        const double d = x - mean_val;
        sq_sum += d * d;
    }
    return std::sqrt(sq_sum / static_cast<double>(v.size() - 1));
}

double PerformanceCalculator::downside_stddev(std::span<const double> v,
                                               double threshold) noexcept {
    // Downside deviation: RMS of returns below `threshold` (Bessel-corrected).
    // If fewer than 2 negative deviations exist, returns 0.0.
    double sq_sum = 0.0;
    std::size_t count = 0;
    for (double x : v) {
        if (x < threshold) {
            const double d = x - threshold;
            sq_sum += d * d;
            ++count;
        }
    }
    if (count < 2) return 0.0;
    return std::sqrt(sq_sum / static_cast<double>(count - 1));
}

// ─── PerformanceCalculator — Sharpe ──────────────────────────────────────────

std::optional<double>
PerformanceCalculator::sharpe(std::span<const double> returns,
                               double risk_free_rate,
                               double annualisation) noexcept {
    if (returns.size() < constants::MIN_RETURN_SERIES_LENGTH) return std::nullopt;
    if (!all_finite(returns))                                  return std::nullopt;
    if (!std::isfinite(risk_free_rate))                        return std::nullopt;
    if (annualisation <= 0.0)                                  return std::nullopt;

    const double mu = mean(returns);
    const double sd = stddev(returns, mu);

    if (sd <= 0.0) return std::nullopt;  // zero variance — ratio undefined

    // Annualised Sharpe: (μ − r_f) / σ × √ann
    return (mu - risk_free_rate) / sd * std::sqrt(annualisation);
}

// ─── PerformanceCalculator — Sortino ─────────────────────────────────────────

std::optional<double>
PerformanceCalculator::sortino(std::span<const double> returns,
                                double risk_free_rate,
                                double annualisation) noexcept {
    if (returns.size() < constants::MIN_RETURN_SERIES_LENGTH) return std::nullopt;
    if (!all_finite(returns))                                  return std::nullopt;
    if (!std::isfinite(risk_free_rate))                        return std::nullopt;
    if (annualisation <= 0.0)                                  return std::nullopt;

    const double mu     = mean(returns);
    const double sd_dn  = downside_stddev(returns, risk_free_rate);

    // Zero downside deviation → all returns above threshold → Sortino undefined
    if (sd_dn <= 0.0) return std::nullopt;

    // Annualised Sortino: (μ − r_f) / σ_down × √ann
    return (mu - risk_free_rate) / sd_dn * std::sqrt(annualisation);
}

// ─── PerformanceCalculator — MaxDrawdown ──────────────────────────────────────

std::optional<double>
PerformanceCalculator::max_drawdown(std::span<const double> returns) noexcept {
    if (returns.empty())         return std::nullopt;
    if (!all_finite(returns))    return std::nullopt;

    // Build equity curve from cumulative sum, starting at 1.0.
    double equity  = 1.0;
    double peak    = 1.0;
    double max_dd  = 0.0;

    for (double r : returns) {
        equity *= (1.0 + r);
        if (equity > peak) {
            peak = equity;
        } else {
            const double dd = (peak - equity) / peak;
            if (dd > max_dd) max_dd = dd;
        }
    }
    return max_dd;
}

// ─── PerformanceCalculator — γ-Weighted IR ───────────────────────────────────

std::optional<double>
PerformanceCalculator::gamma_weighted_ir(
        std::span<const double> strategy_returns,
        std::span<const double> benchmark_returns,
        std::span<const double> gamma_factors) noexcept {

    const std::size_t n = strategy_returns.size();
    if (n < constants::MIN_RETURN_SERIES_LENGTH)  return std::nullopt;
    if (benchmark_returns.size() != n)             return std::nullopt;
    if (gamma_factors.size()     != n)             return std::nullopt;
    if (!all_finite(strategy_returns))             return std::nullopt;
    if (!all_finite(benchmark_returns))            return std::nullopt;
    if (!all_finite(gamma_factors))                return std::nullopt;

    // Active returns: strategy − benchmark
    std::vector<double> active(n);
    for (std::size_t i = 0; i < n; ++i) {
        active[i] = strategy_returns[i] - benchmark_returns[i];
    }

    const double mu_active = mean(active);
    const double sd_active = stddev(active, mu_active);
    if (sd_active <= 0.0) return std::nullopt;  // degenerate

    // Mean Lorentz factor over the window
    const double mu_gamma = mean(gamma_factors);

    // IR_γ = mean(active) × mean(γ) / σ(active)
    return (mu_active * mu_gamma) / sd_active;
}

// ─── LorentzSignalAdjuster ────────────────────────────────────────────────────

LorentzSignalAdjuster::LorentzSignalAdjuster(double effective_mass)
    : effective_mass_(effective_mass) {}

std::optional<double>
LorentzSignalAdjuster::lorentz_gamma(BetaVelocity beta) noexcept {
    // γ = 1 / √(1 − β²).  Invalid for |β| ≥ BETA_MAX_SAFE or non-finite.
    if (!std::isfinite(beta.value))                        return std::nullopt;
    if (std::abs(beta.value) >= constants::BETA_MAX_SAFE)  return std::nullopt;

    const double beta2 = beta.value * beta.value;
    const double denom = std::sqrt(1.0 - beta2);
    if (denom <= 0.0) return std::nullopt;  // guard (BETA_MAX_SAFE ensures this)

    return 1.0 / denom;
}

std::optional<LorentzCorrectedSeries>
LorentzSignalAdjuster::adjust(std::span<const BarData> bars) const noexcept {
    if (bars.empty())              return std::nullopt;
    if (effective_mass_ <= 0.0)    return std::nullopt;

    const std::size_t n = bars.size();
    LorentzCorrectedSeries out;
    out.gamma_factors.reserve(n);
    out.adjusted_signals.reserve(n);

    for (const auto& bar : bars) {
        const auto g_opt = lorentz_gamma(bar.beta);
        // If β is invalid, fall back to γ = 1 (Newtonian — no correction)
        const double g = g_opt.value_or(1.0);
        out.gamma_factors.push_back(g);
        // Relativistic momentum analog: p = γ × m_eff × raw_signal
        out.adjusted_signals.push_back(g * effective_mass_ * bar.raw_signal);
    }

    return out;
}

// ─── PerformanceMetrics ───────────────────────────────────────────────────────

std::string PerformanceMetrics::to_string() const {
    return fmt::format(
        "Sharpe={:.4f}  Sortino={:.4f}  MaxDrawdown={:.4f}  GammaIR={:.4f}",
        sharpe_ratio, sortino_ratio, max_drawdown, gamma_weighted_ir);
}

// ─── BacktestComparison ───────────────────────────────────────────────────────

double BacktestComparison::sharpe_lift()    const noexcept {
    return relativistic.sharpe_ratio - raw.sharpe_ratio;
}
double BacktestComparison::sortino_lift()   const noexcept {
    return relativistic.sortino_ratio - raw.sortino_ratio;
}
double BacktestComparison::drawdown_delta() const noexcept {
    return raw.max_drawdown - relativistic.max_drawdown;
}
double BacktestComparison::ir_lift() const noexcept {
    return relativistic.gamma_weighted_ir - raw.gamma_weighted_ir;
}

std::string BacktestComparison::to_string() const {
    return fmt::format(
        "┌─────────────────────────────────────────────────────────┐\n"
        "│         Relativistic Backtester — Side-by-Side          │\n"
        "├──────────────────┬──────────────┬──────────────┬────────┤\n"
        "│ Metric           │     Raw      │ Relativistic │  Lift  │\n"
        "├──────────────────┼──────────────┼──────────────┼────────┤\n"
        "│ Sharpe Ratio     │   {:9.4f}  │   {:9.4f}  │ {:+.4f}│\n"
        "│ Sortino Ratio    │   {:9.4f}  │   {:9.4f}  │ {:+.4f}│\n"
        "│ Max Drawdown     │   {:9.4f}  │   {:9.4f}  │ {:+.4f}│\n"
        "│ γ-Weighted IR    │   {:9.4f}  │   {:9.4f}  │ {:+.4f}│\n"
        "├──────────────────┴──────────────┴──────────────┴────────┤\n"
        "│ Mean γ: {:.4f}   Max γ applied: {:.4f}   IR lift: {:.4f}x │\n"
        "└──────────────────────────────────────────────────────────┘\n",
        raw.sharpe_ratio,  relativistic.sharpe_ratio,  sharpe_lift(),
        raw.sortino_ratio, relativistic.sortino_ratio, sortino_lift(),
        raw.max_drawdown,  relativistic.max_drawdown,  drawdown_delta(),
        raw.gamma_weighted_ir, relativistic.gamma_weighted_ir, ir_lift(),
        mean_gamma, max_gamma_applied, relativistic_lift);
}

}  // namespace srfm::backtest
