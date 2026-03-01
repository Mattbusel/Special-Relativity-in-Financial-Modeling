/// @file src/backtest/geodesic_strategy.cpp
/// @brief Implementation of ExtendedBacktester with GEODESIC_DEVIATION strategy.
///
/// Extends the Backtester to support three strategy modes:
///   RAW              — classical baseline (unit position)
///   RELATIVISTIC     — Lorentz-corrected γ-scaled position (original Backtester)
///   GEODESIC_DEVIATION — long when deviation > rolling p75 (mean-reversion)
///
/// The geodesic deviation strategy encodes the hypothesis:
///   "When the market has been pulled far from its natural geodesic path,
///    expect mean reversion back toward the geodesic."
/// Concretely: position = 1 when deviation_i > p75 of the last W bars;
/// otherwise position = 0 (flat, no short).

#include "srfm/geodesic_strategy.hpp"
#include "srfm/constants.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace srfm::backtest {

// ─── TripleComparison::to_string ─────────────────────────────────────────────

std::string ExtendedBacktester::TripleComparison::to_string() const {
    std::ostringstream os;
    os << "══════════════════════════════════════════════════════════════\n";
    os << "  SRFM Strategy Comparison — Ticker: " << ticker << "\n";
    os << "══════════════════════════════════════════════════════════════\n";
    os << "  Strategy            Sharpe   Sortino      MDD    γ-IR\n";
    os << "──────────────────────────────────────────────────────────────\n";

    auto fmt_row = [&](const std::string& name, const PerformanceMetrics& m) {
        os << "  " << name;
        for (int i = static_cast<int>(name.size()); i < 20; ++i) os << ' ';
        os << "  " << std::fixed;
        os.precision(4);
        os << m.sharpe_ratio << "  "
           << m.sortino_ratio << "  "
           << m.max_drawdown << "  "
           << m.gamma_weighted_ir << "\n";
    };

    fmt_row("RAW", raw);
    fmt_row("RELATIVISTIC", relativistic);
    fmt_row("GEODESIC_DEV", geodesic);
    os << "══════════════════════════════════════════════════════════════\n";

    // Deltas
    auto delta = [](double a, double b) -> std::string {
        double d = a - b;
        std::ostringstream s;
        s << std::fixed;
        s.precision(4);
        if (d >= 0.0) s << '+';
        s << d;
        return s.str();
    };

    os << "  Relativistic Sharpe lift:    " << delta(relativistic.sharpe_ratio,  raw.sharpe_ratio)  << "\n";
    os << "  Geodesic     Sharpe alpha:   " << delta(geodesic.sharpe_ratio,       raw.sharpe_ratio)  << "\n";
    os << "  Relativistic MDD delta:      " << delta(raw.max_drawdown,            relativistic.max_drawdown) << "\n";
    os << "  Geodesic     MDD delta:      " << delta(raw.max_drawdown,            geodesic.max_drawdown) << "\n";
    os << "══════════════════════════════════════════════════════════════\n";
    return os.str();
}

// ─── Construction ─────────────────────────────────────────────────────────────

ExtendedBacktester::ExtendedBacktester(
    BacktestConfig config,
    std::size_t    rolling_window) noexcept
    : config_(config)
    , base_backtester_(config)
    , rolling_window_(rolling_window)
{}

// ─── Private: rolling_p75 ────────────────────────────────────────────────────

double ExtendedBacktester::rolling_p75(
    const std::vector<GeodesicBarData>& bars,
    std::size_t i) const noexcept
{
    // Window: [max(0, i - rolling_window_ + 1), i]
    const std::size_t start =
        (i + 1 > rolling_window_) ? (i + 1 - rolling_window_) : 0;

    std::vector<double> window;
    window.reserve(i - start + 1);
    for (std::size_t j = start; j <= i; ++j) {
        double dev = bars[j].geodesic_deviation;
        if (std::isfinite(dev)) {
            window.push_back(dev);
        }
    }

    if (window.empty()) {
        return 0.0;
    }

    // Partial sort to find p75
    std::size_t p75_idx = static_cast<std::size_t>(
        std::ceil(0.75 * static_cast<double>(window.size())) - 1.0);
    p75_idx = std::min(p75_idx, window.size() - 1);

    std::nth_element(window.begin(),
                     window.begin() + static_cast<std::ptrdiff_t>(p75_idx),
                     window.end());
    return window[p75_idx];
}

// ─── Private: geodesic_positions ─────────────────────────────────────────────

std::vector<double> ExtendedBacktester::geodesic_positions(
    const std::vector<GeodesicBarData>& bars) const noexcept
{
    const std::size_t n = bars.size();
    std::vector<double> positions;
    positions.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        double dev = bars[i].geodesic_deviation;
        double p75 = rolling_p75(bars, i);
        // Long (position = 1) when deviation exceeds rolling p75
        positions.push_back((std::isfinite(dev) && dev > p75) ? 1.0 : 0.0);
    }

    return positions;
}

// ─── geodesic_returns ────────────────────────────────────────────────────────

std::optional<std::vector<double>> ExtendedBacktester::geodesic_returns(
    const std::vector<GeodesicBarData>& bars,
    const std::vector<double>&           asset_returns) const noexcept
{
    if (bars.size() != asset_returns.size()) return std::nullopt;
    if (bars.size() < constants::MIN_RETURN_SERIES_LENGTH) return std::nullopt;

    const std::size_t n = bars.size();
    std::vector<double> positions = geodesic_positions(bars);

    std::vector<double> returns;
    returns.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        double ret = positions[i] * asset_returns[i];
        returns.push_back(std::isfinite(ret) ? ret : 0.0);
    }
    return returns;
}

// ─── run_triple ──────────────────────────────────────────────────────────────

std::optional<ExtendedBacktester::TripleComparison>
ExtendedBacktester::run_triple(
    const std::vector<GeodesicBarData>& bars,
    const std::vector<double>&           asset_returns,
    const std::string&                   ticker) const noexcept
{
    // ── Input validation ────────────────────────────────────────────────────
    if (bars.size() != asset_returns.size()) return std::nullopt;
    if (bars.size() < constants::MIN_RETURN_SERIES_LENGTH) return std::nullopt;

    const std::size_t n = bars.size();

    // ── Extract base BarData for the delegate Backtester ────────────────────
    std::vector<BarData> base_bars;
    base_bars.reserve(n);
    for (const auto& b : bars) {
        base_bars.push_back(b.base);
    }

    // ── RAW & RELATIVISTIC via base Backtester ───────────────────────────────
    auto comparison = base_backtester_.run(base_bars, asset_returns);
    if (!comparison.has_value()) return std::nullopt;

    // ── GEODESIC_DEVIATION ───────────────────────────────────────────────────
    auto geo_rets = geodesic_returns(bars, asset_returns);
    if (!geo_rets.has_value()) return std::nullopt;

    // Build γ factors (reuse from base comparison — same beta inputs)
    // For the γ-weighted IR we use a unit gamma vector (geodesic strategy
    // does not apply relativistic γ scaling).
    std::vector<double> unit_gamma(n, 1.0);
    std::vector<double> benchmark(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        benchmark[i] = base_bars[i].benchmark;
    }

    auto sh  = PerformanceCalculator::sharpe(*geo_rets,
                                              config_.risk_free_rate,
                                              config_.annualisation);
    auto so  = PerformanceCalculator::sortino(*geo_rets,
                                               config_.risk_free_rate,
                                               config_.annualisation);
    auto mdd = PerformanceCalculator::max_drawdown(*geo_rets);
    auto ir  = PerformanceCalculator::gamma_weighted_ir(*geo_rets,
                                                         benchmark,
                                                         unit_gamma);

    if (!sh || !so || !mdd || !ir) return std::nullopt;

    PerformanceMetrics geo_metrics{
        .sharpe_ratio      = *sh,
        .sortino_ratio     = *so,
        .max_drawdown      = *mdd,
        .gamma_weighted_ir = *ir,
    };

    return TripleComparison{
        .raw          = comparison->raw,
        .relativistic = comparison->relativistic,
        .geodesic     = geo_metrics,
        .ticker       = ticker,
    };
}

} // namespace srfm::backtest
