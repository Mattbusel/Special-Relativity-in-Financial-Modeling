/// @file src/causal_cone.cpp
/// @brief Implementation of the Causal Cone Filter for financial spacetime.
///
/// See include/srfm/causal_cone.hpp for full documentation.

#include "srfm/causal_cone.hpp"
#include "srfm/manifold.hpp"
#include "srfm/normalizer.hpp"
#include "srfm/constants.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmt/core.h>
#include <numeric>
#include <sstream>

namespace srfm::causal {

// ─── CausalConeFilter ─────────────────────────────────────────────────────────

CausalConeFilter::CausalConeFilter(Config config) noexcept
    : config_(config) {}

bool CausalConeFilter::is_causal(manifold::IntervalType t) const noexcept {
    switch (t) {
        case manifold::IntervalType::Timelike:
            return true;
        case manifold::IntervalType::Lightlike:
            return config_.include_lightlike;
        case manifold::IntervalType::Spacelike:
            return false;
    }
    return false;
}

std::vector<CausalHistory>
CausalConeFilter::build_histories(
    std::span<const core::OHLCV>              bars,
    std::span<const manifold::SpacetimeEvent> events
) const noexcept {
    const std::size_t N = bars.size();
    assert(events.size() == N && "bars and events must have the same size");

    std::vector<CausalHistory> histories;
    histories.reserve(N);

    for (std::size_t i = 0; i < N; ++i) {
        CausalHistory h;
        h.event = events[i];

        // Scan back up to look_back bars.
        const std::size_t look_start = (i >= config_.look_back)
                                     ? (i - config_.look_back)
                                     : 0;

        for (std::size_t j = look_start; j < i; ++j) {
            ++h.total_bars_scanned;

            // Classify the interval from past event j to current event i.
            auto ds2 = manifold::SpacetimeInterval::compute(
                events[j], events[i], config_.c_market
            );
            if (!ds2.has_value()) continue;

            auto interval_type = manifold::SpacetimeInterval::classify(*ds2);

            if (is_causal(interval_type)) {
                h.causal_predecessors.push_back(events[j]);
                h.predecessor_indices.push_back(j);
            }

            if (config_.verbose) {
                fmt::print(stderr,
                    "[causal_cone] bar={} prev={} ds2={:.4f} type={}\n",
                    i, j, *ds2, manifold::to_string(interval_type)
                );
            }
        }

        histories.push_back(std::move(h));
    }
    return histories;
}

std::optional<CausalSignal>
CausalConeFilter::compute_signal(
    const CausalHistory&    history,
    std::span<const double> returns,
    std::size_t             bar_idx
) const noexcept {
    const auto& preds = history.predecessor_indices;
    if (preds.empty()) return std::nullopt;
    if (bar_idx >= returns.size()) return std::nullopt;

    // Causal returns: only from causal predecessor bars.
    std::vector<double> causal_rets;
    causal_rets.reserve(preds.size());
    for (std::size_t idx : preds) {
        if (idx < returns.size()) {
            causal_rets.push_back(returns[idx]);
        }
    }
    if (causal_rets.empty()) return std::nullopt;

    // Compute mean.
    double sum = 0.0;
    for (double r : causal_rets) sum += r;
    const double mean = sum / static_cast<double>(causal_rets.size());

    // Compute volatility.
    double var = 0.0;
    for (double r : causal_rets) var += (r - mean) * (r - mean);
    const double vol = (causal_rets.size() > 1)
                     ? std::sqrt(var / static_cast<double>(causal_rets.size() - 1))
                     : 0.0;

    // Causal momentum: linear weight (recent = higher weight).
    double momentum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t k = 0; k < causal_rets.size(); ++k) {
        double w = static_cast<double>(k + 1);  // linear weights
        momentum    += w * causal_rets[k];
        weight_sum  += w;
    }
    if (weight_sum > 0.0) momentum /= weight_sum;

    // All-bars mean for comparison (look_back window around bar_idx).
    const std::size_t look_start = (bar_idx >= config_.look_back)
                                 ? (bar_idx - config_.look_back)
                                 : 0;
    double all_sum = 0.0;
    std::size_t all_count = 0;
    for (std::size_t k = look_start; k < bar_idx && k < returns.size(); ++k) {
        all_sum += returns[k];
        ++all_count;
    }
    const double all_mean = (all_count > 0) ? (all_sum / all_count) : 0.0;

    CausalSignal sig;
    sig.causal_mean_return   = mean;
    sig.causal_vol           = vol;
    sig.causal_momentum      = momentum;
    sig.causal_fraction      = history.causal_fraction();
    sig.all_bars_mean_return = all_mean;
    sig.n_causal             = causal_rets.size();
    sig.n_total              = history.total_bars_scanned;
    return sig;
}

// ─── CausalBacktestResult ─────────────────────────────────────────────────────

std::string CausalBacktestResult::to_string() const {
    return fmt::format(
        "CausalBacktest | "
        "CausalSharpe={:.4f} BaselineSharpe={:.4f} "
        "SharpeImprovement={:+.4f} | "
        "CausalSortino={:.4f} BaselineSortino={:.4f} | "
        "MeanCausalFraction={:.1%}",
        causal_metrics.sharpe_ratio,
        baseline_metrics.sharpe_ratio,
        sharpe_improvement(),
        causal_metrics.sortino_ratio,
        baseline_metrics.sortino_ratio,
        mean_causal_fraction
    );
}

// ─── CausalBacktest ───────────────────────────────────────────────────────────

CausalBacktest::CausalBacktest(
    CausalConeFilter::Config filter_cfg,
    core::EngineConfig       engine_cfg
) noexcept
    : filter_(filter_cfg)
    , engine_(engine_cfg)
{}

std::optional<CausalBacktestResult>
CausalBacktest::run(std::span<const core::OHLCV> bars) const noexcept {
    const std::size_t N = bars.size();
    const std::size_t min_bars = filter_.look_back() + 30;
    if (N < min_bars) return std::nullopt;

    // Step 1: Run the engine to get PipelineBar data per bar.
    // We use process_stream_bar via a mutable local engine copy.
    core::Engine local_engine(engine_.get_config() /* requires get_config(), fallback below */);
    // Since Engine doesn't expose get_config(), we construct a fresh one.
    // The pipeline bar gives us beta + events.
    std::vector<core::PipelineBar> pipeline_bars;
    pipeline_bars.reserve(N);

    {
        core::Engine streaming_engine;
        for (auto& bar : bars) {
            auto pb = streaming_engine.process_stream_bar(bar);
            if (pb) pipeline_bars.push_back(*pb);
        }
    }

    if (pipeline_bars.size() < min_bars) return std::nullopt;

    // Build event sequence from pipeline bars.
    std::vector<manifold::SpacetimeEvent> events;
    events.reserve(pipeline_bars.size());
    for (const auto& pb : pipeline_bars) events.push_back(pb.event);

    // Build aligned OHLCV span (pipeline may have fewer bars due to warm-up).
    // Use the last pipeline_bars.size() bars from the original sequence.
    const std::size_t offset = N - pipeline_bars.size();
    std::span<const core::OHLCV> aligned_bars = bars.subspan(offset);

    // Step 2: Build causal histories.
    auto histories = filter_.build_histories(aligned_bars, events);
    const std::size_t M = histories.size();

    // Step 3: Compute returns.
    std::vector<double> returns;
    returns.reserve(M);
    for (const auto& pb : pipeline_bars) returns.push_back(pb.price_return);

    // Step 4: Build causal and baseline signal series.
    std::vector<backtest::BarData> causal_data;
    std::vector<backtest::BarData> baseline_data;
    causal_data.reserve(M);
    baseline_data.reserve(M);

    double total_causal_fraction = 0.0;
    std::size_t signal_count = 0;

    for (std::size_t i = 0; i < M; ++i) {
        const auto& pb = pipeline_bars[i];
        auto sig = filter_.compute_signal(histories[i], returns, i);

        double causal_signal   = sig ? sig->causal_momentum : 0.0;
        double baseline_signal = sig ? sig->all_bars_mean_return : pb.price_return;

        causal_data.push_back({
            .raw_signal = causal_signal,
            .beta       = pb.beta,
            .benchmark  = pb.price_return,
        });
        baseline_data.push_back({
            .raw_signal = baseline_signal,
            .beta       = pb.beta,
            .benchmark  = pb.price_return,
        });

        if (sig) {
            total_causal_fraction += sig->causal_fraction;
            ++signal_count;
        }
    }

    // Step 5: Run backtest on both.
    backtest::Backtester bt;
    auto causal_result   = bt.run(causal_data,   returns);
    auto baseline_result = bt.run(baseline_data, returns);

    if (!causal_result || !baseline_result) return std::nullopt;

    CausalBacktestResult result;
    result.causal_metrics   = causal_result->relativistic;
    result.baseline_metrics = baseline_result->relativistic;
    result.mean_causal_fraction = (signal_count > 0)
                                ? (total_causal_fraction / signal_count)
                                : 0.0;
    return result;
}

}  // namespace srfm::causal
