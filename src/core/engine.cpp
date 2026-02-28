/// @file src/core/engine.cpp
/// @brief Core Integration Engine — AGT-06.

#include "srfm/engine.hpp"
#include "srfm/backtest.hpp"
#include "srfm/manifold.hpp"
#include "srfm/momentum.hpp"

#include "../lorentz/lorentz_transform.hpp"
#include "../lorentz/beta_calculator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace srfm::core {

// ─── Engine constructor ───────────────────────────────────────────────────────

Engine::Engine(EngineConfig config)
    : config_(std::move(config))
{}

// ─── Engine::run_backtest ─────────────────────────────────────────────────────

std::optional<backtest::BacktestComparison>
Engine::run_backtest(std::span<const OHLCV> bars) const noexcept {
    if (bars.size() < config_.min_bars) {
        return std::nullopt;
    }

    // ── Step 1: Extract close prices ──────────────────────────────────────────
    std::vector<double> closes;
    closes.reserve(bars.size());
    for (const auto& b : bars) {
        closes.push_back(b.close);
    }

    // ── Step 2: Compute simple close-to-close returns ─────────────────────────
    auto returns = compute_returns(closes);
    if (returns.empty()) {
        return std::nullopt;
    }

    // ── Step 3: Compute β per bar ─────────────────────────────────────────────
    auto betas = compute_betas(closes, config_.max_market_velocity);

    // returns has length (N-1); trim betas to match
    // betas[i] corresponds to bar i; strip the last bar
    if (betas.size() > returns.size()) {
        betas.resize(returns.size());
    }

    // ── Step 4: Build BarData for backtester ─────────────────────────────────
    // We use momentum-corrected signal = 1.0 (trivial trend-following: always long)
    // adjusted by γ.  Real alpha signals would come from a strategy layer.
    std::vector<backtest::BarData> bar_data;
    bar_data.reserve(returns.size());

    for (std::size_t i = 0; i < returns.size(); ++i) {
        const BetaVelocity beta = (i < betas.size()) ? betas[i] : BetaVelocity{0.0};

        // Raw signal = +1 (constant long signal — demonstrates relativistic lift)
        const double raw_signal = 1.0;

        // Benchmark return = 0 (no benchmark; IR computed vs zero)
        const double benchmark = 0.0;

        bar_data.push_back(backtest::BarData{
            .raw_signal = raw_signal,
            .beta       = beta,
            .benchmark  = benchmark,
        });
    }

    // ── Step 5: Run Backtester ────────────────────────────────────────────────
    backtest::Backtester bt(config_.backtest_cfg);
    return bt.run(bar_data, returns);
}

// ─── Engine::process_stream_bar ───────────────────────────────────────────────

std::optional<PipelineBar>
Engine::process_stream_bar(const OHLCV& bar) noexcept {
    stream_window_.push_back(bar);

    // Need at least 2 bars to compute a return and β.
    if (stream_window_.size() < 2) {
        return std::nullopt;
    }

    const OHLCV& prev = stream_window_[stream_window_.size() - 2];
    const OHLCV& curr = stream_window_.back();

    // Compute simple return for this bar.
    double price_return = 0.0;
    if (std::isfinite(prev.close) && prev.close > 0.0 && std::isfinite(curr.close)) {
        price_return = (curr.close - prev.close) / prev.close;
    }

    // Compute β from the rolling window of closes.
    std::vector<double> window_closes;
    const std::size_t window_size = std::min(stream_window_.size(), std::size_t{5});
    const std::size_t start = stream_window_.size() - window_size;
    for (std::size_t i = start; i < stream_window_.size(); ++i) {
        window_closes.push_back(stream_window_[i].close);
    }

    BetaVelocity beta{0.0};
    auto betas = compute_betas(window_closes, config_.max_market_velocity);
    if (!betas.empty()) {
        beta = betas.back();
    }

    // Compute γ.
    LorentzFactor gamma{1.0};
    auto g = lorentz::LorentzTransform::gamma(beta);
    if (g) {
        gamma = *g;
    }

    // Build spacetime event (momentum indicator = price return × volume).
    const double momentum_indicator = price_return * curr.volume;
    const auto event = to_event(curr, momentum_indicator);

    // Classify interval vs previous bar.
    const auto prev_event = to_event(prev, 0.0);
    manifold::IntervalType interval_type = manifold::IntervalType::Timelike;
    auto cls = manifold::MarketManifold::classify(prev_event, event);
    if (cls) {
        interval_type = *cls;
    }

    return PipelineBar{
        .raw            = curr,
        .price_return   = price_return,
        .beta           = beta,
        .gamma          = gamma,
        .event          = event,
        .interval_type  = interval_type,
    };
}

// ─── Engine::reset_stream ─────────────────────────────────────────────────────

void Engine::reset_stream() noexcept {
    stream_window_.clear();
}

// ─── Engine::stream_window ────────────────────────────────────────────────────

std::span<const OHLCV> Engine::stream_window() const noexcept {
    return stream_window_;
}

// ─── Engine::compute_returns ─────────────────────────────────────────────────

std::vector<double>
Engine::compute_returns(std::span<const double> prices) noexcept {
    if (prices.size() < 2) {
        return {};
    }

    std::vector<double> rets;
    rets.reserve(prices.size() - 1);

    for (std::size_t i = 1; i < prices.size(); ++i) {
        const double prev = prices[i - 1];
        const double curr = prices[i];

        if (!std::isfinite(prev) || !std::isfinite(curr) || prev <= 0.0) {
            rets.push_back(0.0);
        } else {
            rets.push_back((curr - prev) / prev);
        }
    }

    return rets;
}

// ─── Engine::compute_betas ───────────────────────────────────────────────────

std::vector<BetaVelocity>
Engine::compute_betas(std::span<const double> prices,
                       double max_velocity) noexcept {
    if (prices.empty() || max_velocity <= 0.0) {
        return {};
    }

    std::vector<BetaVelocity> betas;
    betas.reserve(prices.size());

    constexpr std::size_t WINDOW = 5;

    for (std::size_t i = 0; i < prices.size(); ++i) {
        // Use available prices up to WINDOW in length.
        const std::size_t win_size = std::min(i + 1, WINDOW);
        const std::size_t start    = i + 1 - win_size;
        auto window = prices.subspan(start, win_size);

        BetaVelocity beta{0.0};
        if (win_size >= 2) {
            auto b = lorentz::BetaCalculator::fromRollingWindow(
                window, win_size, max_velocity, /*time_delta=*/1.0);
            if (b) {
                beta = *b;
            }
        }
        betas.push_back(beta);
    }

    return betas;
}

// ─── Engine::to_event ────────────────────────────────────────────────────────

manifold::SpacetimeEvent
Engine::to_event(const OHLCV& bar, double momentum_indicator) noexcept {
    return manifold::SpacetimeEvent{
        .time     = bar.timestamp,
        .price    = bar.close,
        .volume   = bar.volume,
        .momentum = momentum_indicator,
    };
}

}  // namespace srfm::core
