/// @file src/hawking.cpp
/// @brief Implementation of the Hawking Radiation analogy for price series.
///
/// See include/srfm/hawking.hpp for full documentation.

#include "srfm/hawking.hpp"
#include "srfm/constants.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmt/core.h>
#include <numeric>
#include <sstream>

namespace srfm::hawking {

// ─── HawkingTemperature ───────────────────────────────────────────────────────

std::string HawkingTemperature::to_string() const {
    return fmt::format(
        "HawkingTemp T_H={:+.4f} z={:+.4f} Δz={:+.4f} "
        "μ={:.2f} σ={:.4f} horizon={}",
        temperature, z_score, delta_z,
        bollinger_mean, bollinger_std,
        near_horizon ? "YES" : "no"
    );
}

// ─── HawkingSignal ────────────────────────────────────────────────────────────

const char* to_string(HawkingDirection d) noexcept {
    switch (d) {
        case HawkingDirection::Reversal:     return "Reversal";
        case HawkingDirection::Continuation: return "Continuation";
        case HawkingDirection::Neutral:      return "Neutral";
    }
    return "Unknown";
}

std::string HawkingSignal::to_string() const {
    return fmt::format(
        "HawkingSignal dir={} strength={:.4f} action={:+d} | {}",
        hawking::to_string(direction),
        strength,
        action,
        temperature.to_string()
    );
}

// ─── BollingerState ───────────────────────────────────────────────────────────

void BollingerState::push(double price) noexcept {
    if (window.size() == max_size) {
        // Remove oldest.
        const double oldest = window.front();
        sum    -= oldest;
        sum_sq -= oldest * oldest;
        window.erase(window.begin());
    }
    window.push_back(price);
    sum    += price;
    sum_sq += price * price;
}

double BollingerState::mean() const noexcept {
    if (window.empty()) return 0.0;
    return sum / static_cast<double>(window.size());
}

double BollingerState::stddev() const noexcept {
    const std::size_t n = window.size();
    if (n < 2) return 0.0;
    const double n_d   = static_cast<double>(n);
    const double m     = sum / n_d;
    const double var   = (sum_sq / n_d) - (m * m);
    return (var > 0.0) ? std::sqrt(var) : 0.0;
}

// ─── PriceEventHorizon ────────────────────────────────────────────────────────

PriceEventHorizon::PriceEventHorizon(Config config) noexcept
    : config_(config)
{
    state_.max_size = config.bb_window;
}

void PriceEventHorizon::reset() noexcept {
    state_ = BollingerState{};
    state_.max_size = config_.bb_window;
    prev_z_     = 0.0;
    has_prev_z_ = false;
}

std::optional<HawkingTemperature>
PriceEventHorizon::update(double close) noexcept {
    if (!std::isfinite(close)) return std::nullopt;

    state_.push(close);

    // Need a full window before emitting.
    if (!state_.full()) return std::nullopt;

    const double mu    = state_.mean();
    const double sigma = state_.stddev();

    // Avoid division by zero for flat price series.
    if (sigma < 1e-12) return std::nullopt;

    const double z = (close - mu) / sigma;
    const double delta_z = has_prev_z_ ? (z - prev_z_) : 0.0;

    // Hawking temperature: T_H = z × Δz
    // Interpretation:
    //   - z > 0 and Δz > 0: price is above mean AND accelerating → hot (reversal)
    //   - z < 0 and Δz < 0: price is below mean AND falling faster → hot (reversal)
    //   - z > 0 and Δz < 0: price above mean but decelerating → cold (continuation)
    const double temperature = z * delta_z;

    HawkingTemperature ht;
    ht.temperature    = temperature;
    ht.z_score        = z;
    ht.delta_z        = delta_z;
    ht.bollinger_mean = mu;
    ht.bollinger_std  = sigma;
    ht.near_horizon   = std::fabs(z) >= config_.bb_sigma;

    prev_z_     = z;
    has_prev_z_ = true;

    return ht;
}

// ─── HawkingSignalGenerator ───────────────────────────────────────────────────

HawkingSignalGenerator::HawkingSignalGenerator(
    PriceEventHorizon::Config config
) noexcept
    : horizon_(config)
    , config_(config)
{}

void HawkingSignalGenerator::reset() noexcept {
    horizon_.reset();
}

HawkingDirection HawkingSignalGenerator::classify(double temperature) const noexcept {
    if (temperature > config_.hot_threshold)  return HawkingDirection::Reversal;
    if (temperature < config_.cold_threshold) return HawkingDirection::Continuation;
    return HawkingDirection::Neutral;
}

double HawkingSignalGenerator::strength(double temperature) noexcept {
    // Map |T_H| to [0, 1] via a sigmoid-like normalisation.
    // |T_H| > 5 → strength ≈ 1.
    const double abs_t = std::fabs(temperature);
    return abs_t / (abs_t + 1.0);   // monotone, bounded in [0, 1)
}

int HawkingSignalGenerator::direction_to_action(HawkingDirection d) noexcept {
    switch (d) {
        case HawkingDirection::Reversal:
            // Trade against the current trend (fade the extreme move).
            // The sign of z_score tells us which direction to fade.
            // We return -1 as a generic "fade" signal; callers should invert
            // based on z_score sign if needed.
            return -1;
        case HawkingDirection::Continuation:
            return +1;
        case HawkingDirection::Neutral:
            return 0;
    }
    return 0;
}

std::optional<HawkingSignal>
HawkingSignalGenerator::update(double close) noexcept {
    auto temp = horizon_.update(close);
    if (!temp.has_value()) return std::nullopt;

    const HawkingDirection dir    = classify(temp->temperature);
    const double           str    = strength(temp->temperature);
    const int              action = direction_to_action(dir);

    // For REVERSAL, the action direction depends on z_score sign:
    // if z > 0 (price above mean), we expect it to fall → sell (-1)
    // if z < 0 (price below mean), we expect it to rise → buy (+1)
    int final_action = action;
    if (dir == HawkingDirection::Reversal) {
        final_action = (temp->z_score > 0.0) ? -1 : +1;
    } else if (dir == HawkingDirection::Continuation) {
        // Continue in the direction of the current z_score trend.
        final_action = (temp->z_score > 0.0) ? +1 : -1;
    }

    return HawkingSignal{
        .temperature = *temp,
        .direction   = dir,
        .strength    = str,
        .action      = final_action,
    };
}

// ─── HawkingBacktestResult ────────────────────────────────────────────────────

std::string HawkingBacktestResult::to_string() const {
    return fmt::format(
        "HawkingBacktest | "
        "HawkingSharpe={:.4f} TimelikeSharpe={:.4f} "
        "SharpeImprovement={:+.4f} | "
        "MeanTemp={:.4f} Coverage={:.1%}",
        hawking_metrics.sharpe_ratio,
        timelike_metrics.sharpe_ratio,
        sharpe_improvement(),
        mean_temperature,
        signal_coverage
    );
}

// ─── HawkingBacktest ──────────────────────────────────────────────────────────

HawkingBacktest::HawkingBacktest(
    PriceEventHorizon::Config hawking_cfg,
    core::EngineConfig        engine_cfg
) noexcept
    : signal_gen_(hawking_cfg)
    , engine_(engine_cfg)
    , hawking_cfg_(hawking_cfg)
{}

std::optional<HawkingBacktestResult>
HawkingBacktest::run(std::span<const core::OHLCV> bars) const noexcept {
    const std::size_t N       = bars.size();
    const std::size_t min_req = hawking_cfg_.bb_window + 30;
    if (N < min_req) return std::nullopt;

    // Build pipeline bars via Engine streaming.
    std::vector<core::PipelineBar> pipeline_bars;
    pipeline_bars.reserve(N);
    {
        core::Engine streaming_engine;
        for (const auto& bar : bars) {
            auto pb = streaming_engine.process_stream_bar(bar);
            if (pb) pipeline_bars.push_back(*pb);
        }
    }

    if (pipeline_bars.size() < min_req) return std::nullopt;
    const std::size_t M = pipeline_bars.size();

    // Compute returns.
    std::vector<double> returns;
    returns.reserve(M);
    for (const auto& pb : pipeline_bars) returns.push_back(pb.price_return);

    // Generate Hawking signals and TIMELIKE-based signals.
    // Use a mutable local generator (not shared with member to keep run() const).
    HawkingSignalGenerator local_gen(hawking_cfg_);

    std::vector<backtest::BarData> hawking_data;
    std::vector<backtest::BarData> timelike_data;
    hawking_data.reserve(M);
    timelike_data.reserve(M);

    double total_temperature = 0.0;
    std::size_t non_neutral  = 0;
    const std::size_t offset = bars.size() - M;

    for (std::size_t i = 0; i < M; ++i) {
        const auto& pb = pipeline_bars[i];
        const double close = bars[offset + i].close;

        // Hawking signal.
        auto sig = local_gen.update(close);
        double h_signal = sig ? static_cast<double>(sig->action) : 0.0;
        if (sig && sig->direction != HawkingDirection::Neutral) {
            total_temperature += std::fabs(sig->temperature.temperature);
            ++non_neutral;
        }

        // TIMELIKE baseline: +1 if bar is TIMELIKE, else 0.
        double t_signal = (pb.interval_type == manifold::IntervalType::Timelike) ? 1.0 : 0.0;

        hawking_data.push_back({
            .raw_signal = h_signal,
            .beta       = pb.beta,
            .benchmark  = pb.price_return,
        });
        timelike_data.push_back({
            .raw_signal = t_signal,
            .beta       = pb.beta,
            .benchmark  = pb.price_return,
        });
    }

    // Run backtests.
    backtest::Backtester bt;
    auto hawking_result  = bt.run(hawking_data,  returns);
    auto timelike_result = bt.run(timelike_data, returns);

    if (!hawking_result || !timelike_result) return std::nullopt;

    HawkingBacktestResult result;
    result.hawking_metrics  = hawking_result->relativistic;
    result.timelike_metrics = timelike_result->relativistic;
    result.mean_temperature = (non_neutral > 0)
                            ? (total_temperature / static_cast<double>(non_neutral))
                            : 0.0;
    result.signal_coverage  = static_cast<double>(non_neutral)
                            / static_cast<double>(M);
    return result;
}

}  // namespace srfm::hawking
