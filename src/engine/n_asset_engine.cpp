/**
 * @file  n_asset_engine.cpp
 * @brief Implementation of NAssetEngine.
 *
 * See include/srfm/engine/n_asset_engine.hpp for the public API contract.
 *
 * ## Pipeline summary
 *   1. ingest(): push bars into deque ring buffer.
 *   2. estimate_covariance(): compute N×N empirical covariance from log-returns.
 *   3. process(): build manifold, per-asset momenta, portfolio interval.
 */

#include "../../include/srfm/engine/n_asset_engine.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace srfm::engine {

// ── Constructor ───────────────────────────────────────────────────────────────

NAssetEngine::NAssetEngine(AssetUniverse universe,
                            EngineConfig  cfg) noexcept
    : universe_(std::move(universe))
    , cfg_(cfg)
{}

// ── Ingestion ─────────────────────────────────────────────────────────────────

std::optional<std::monostate>
NAssetEngine::ingest(std::span<const OHLCVBar> bars) noexcept {
    // Must provide exactly one bar per asset.
    if (static_cast<int>(bars.size()) != universe_.n()) {
        return std::nullopt;
    }

    // Append to ring buffer.
    history_.push_back(std::vector<OHLCVBar>(bars.begin(), bars.end()));

    // Trim ring buffer to avoid unbounded growth (keep 2× lookback).
    const int max_history = cfg_.lookback_bars * 2;
    while (static_cast<int>(history_.size()) > max_history) {
        history_.pop_front();
    }

    return std::monostate{};
}

// ── Covariance estimation ─────────────────────────────────────────────────────

std::optional<Eigen::MatrixXd>
NAssetEngine::estimate_covariance() const noexcept {
    const int N = universe_.n();
    const int T = static_cast<int>(history_.size());

    // Need at least 2 bars to compute one log-return, and T-1 >= lookback_bars - 1.
    if (T < 2) { return std::nullopt; }

    const int n_returns = T - 1;

    // Compute log-returns matrix: rows = time, cols = asset.
    Eigen::MatrixXd returns(n_returns, N);

    for (int t = 0; t < n_returns; ++t) {
        const auto& prev = history_[static_cast<std::size_t>(t)];
        const auto& curr = history_[static_cast<std::size_t>(t + 1)];

        for (int i = 0; i < N; ++i) {
            double prev_close = prev[static_cast<std::size_t>(i)].close;
            double curr_close = curr[static_cast<std::size_t>(i)].close;

            // Guard against non-positive prices.
            if (prev_close <= 0.0 || curr_close <= 0.0) {
                return std::nullopt;
            }

            returns(t, i) = std::log(curr_close / prev_close);
        }
    }

    // Demean columns.
    Eigen::VectorXd mean = returns.colwise().mean();
    Eigen::MatrixXd demeaned = returns.rowwise() - mean.transpose();

    // Sample covariance: Cov = (1/(T-1)) * demeaned^T * demeaned.
    if (n_returns < 2) {
        // With only one return we cannot compute sample covariance.
        // Use population covariance (divide by 1) — or return nullopt for safety.
        // Return an identity-scaled matrix as a fallback.
        Eigen::MatrixXd cov = demeaned.transpose() * demeaned;
        // Add small regularisation to ensure positive definiteness.
        cov += 1e-8 * Eigen::MatrixXd::Identity(N, N);
        return cov;
    }

    Eigen::MatrixXd cov = (demeaned.transpose() * demeaned) /
                           static_cast<double>(n_returns - 1);

    // Add small regularisation to ensure positive definiteness even when
    // assets are nearly perfectly correlated.
    cov += 1e-8 * Eigen::MatrixXd::Identity(N, N);

    return cov;
}

// ── Beta computation ──────────────────────────────────────────────────────────

std::optional<double>
NAssetEngine::compute_beta(double prev_close, double curr_close) const noexcept {
    if (prev_close <= 0.0) { return std::nullopt; }

    double delta = curr_close - prev_close;
    double raw_beta = std::abs(delta / (cfg_.c_market * prev_close));

    // Clamp to [0, ENGINE_BETA_MAX_SAFE - 1e-9].
    const double beta_max = ENGINE_BETA_MAX_SAFE - 1e-9;
    if (raw_beta >= beta_max) {
        raw_beta = beta_max;
    }

    return raw_beta;
}

// ── Portfolio interval ────────────────────────────────────────────────────────

std::optional<std::pair<double, IntervalType>>
NAssetEngine::compute_portfolio_interval(
        const NAssetManifold& manifold) const noexcept {
    const int T = static_cast<int>(history_.size());
    if (T < 2) { return std::nullopt; }

    const int N = universe_.n();

    // Build events from the last two bar snapshots.
    const auto& prev_bars = history_[static_cast<std::size_t>(T - 2)];
    const auto& curr_bars = history_[static_cast<std::size_t>(T - 1)];

    Eigen::VectorXd prev_prices(N), curr_prices(N);
    for (int i = 0; i < N; ++i) {
        prev_prices(i) = prev_bars[static_cast<std::size_t>(i)].close;
        curr_prices(i) = curr_bars[static_cast<std::size_t>(i)].close;
    }

    double t_prev = prev_bars[0].timestamp;
    double t_curr = curr_bars[0].timestamp;

    auto ev_a = NAssetEvent::make(t_prev, prev_prices);
    auto ev_b = NAssetEvent::make(t_curr, curr_prices);
    if (!ev_a || !ev_b) { return std::nullopt; }

    NAssetInterval interval_calc;
    auto result = interval_calc.compute(*ev_a, *ev_b, manifold);
    if (!result) { return std::nullopt; }

    return std::make_pair(result->ds_sq, result->type);
}

// ── Processing ────────────────────────────────────────────────────────────────

std::optional<EngineOutput> NAssetEngine::process() const noexcept {
    if (!ready()) { return std::nullopt; }

    const int N = universe_.n();

    // 1. Estimate covariance.
    auto cov_opt = estimate_covariance();
    if (!cov_opt) { return std::nullopt; }

    // 2. Build NAssetManifold.
    auto manifold_opt = NAssetManifold::make(N, *cov_opt, cfg_.c_market);
    if (!manifold_opt) { return std::nullopt; }
    const NAssetManifold& manifold = *manifold_opt;

    // 3. Retrieve latest two bar snapshots.
    const int T = static_cast<int>(history_.size());
    const auto& prev_bars = history_[static_cast<std::size_t>(T - 2)];
    const auto& curr_bars = history_[static_cast<std::size_t>(T - 1)];

    // 4. Compute per-asset results.
    EngineOutput output;
    output.timestamp = curr_bars[0].timestamp;
    output.assets.reserve(static_cast<std::size_t>(N));

    for (int i = 0; i < N; ++i) {
        const OHLCVBar& prev = prev_bars[static_cast<std::size_t>(i)];
        const OHLCVBar& curr = curr_bars[static_cast<std::size_t>(i)];

        // Beta.
        auto beta_opt = compute_beta(prev.close, curr.close);
        if (!beta_opt) { return std::nullopt; }
        double beta = *beta_opt;

        // Gamma.
        double gamma = 1.0 / std::sqrt(1.0 - beta * beta);

        // Effective mass.
        double m_eff = curr.volume / cfg_.adv_baseline;

        // Relativistic momentum.
        double p_rel = gamma * m_eff * curr.close;

        // Per-asset regime: classify the price displacement as spacetime interval.
        // Build single-asset events.
        Eigen::VectorXd prev_p(1), curr_p(1);
        prev_p(0) = prev.close;
        curr_p(0) = curr.close;

        // Use the full N-asset metric for interval classification.
        // We project to the i-th spatial axis only.
        // Simpler: classify via beta — β < 1 always, regime from interval.
        // Use the single-asset sub-manifold for classification.
        Eigen::MatrixXd cov_1x1(1, 1);
        cov_1x1(0, 0) = (*cov_opt)(i, i);

        IntervalType asset_regime = IntervalType::SPACELIKE;
        auto sub_manifold_opt = NAssetManifold::make(1, cov_1x1, cfg_.c_market);
        if (sub_manifold_opt) {
            auto ea = NAssetEvent::make(prev.timestamp, prev_p);
            auto eb = NAssetEvent::make(curr.timestamp, curr_p);
            if (ea && eb) {
                NAssetInterval calc;
                auto ir = calc.compute(*ea, *eb, *sub_manifold_opt);
                if (ir) {
                    asset_regime = ir->type;
                }
            }
        }

        AssetMomentumResult res;
        res.asset_name            = universe_.names[static_cast<std::size_t>(i)];
        res.beta                  = beta;
        res.gamma                 = gamma;
        res.m_eff                 = m_eff;
        res.relativistic_momentum = p_rel;
        res.regime                = asset_regime;

        output.assets.push_back(std::move(res));
    }

    // 5. Portfolio interval.
    auto portfolio_opt = compute_portfolio_interval(manifold);
    if (!portfolio_opt) {
        output.portfolio_interval_sq = 0.0;
        output.portfolio_regime      = IntervalType::LIGHTLIKE;
    } else {
        output.portfolio_interval_sq = portfolio_opt->first;
        output.portfolio_regime      = portfolio_opt->second;
    }

    return output;
}

std::optional<EngineOutput>
NAssetEngine::ingest_and_process(std::span<const OHLCVBar> bars) noexcept {
    auto r = ingest(bars);
    if (!r) { return std::nullopt; }
    return process();
}

} // namespace srfm::engine
