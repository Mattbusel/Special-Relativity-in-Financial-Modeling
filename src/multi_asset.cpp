/// @file src/multi_asset.cpp
/// @brief Implementation of MultiAssetInterval, CorrelationMetric,
///        MultiAssetLorentz, and PortfolioGeodesic.

#include "srfm/multi_asset.hpp"
#include "srfm/constants.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace srfm::multi_asset {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// @brief Clamp a double to [lo, hi].
constexpr double clamp_d(double v, double lo, double hi) noexcept {
    return v < lo ? lo : (v > hi ? hi : v);
}

/// @brief Compute per-asset log-returns from consecutive price vectors.
Eigen::VectorXd log_returns(const Eigen::VectorXd& prev,
                            const Eigen::VectorXd& curr) noexcept {
    assert(prev.size() == curr.size());
    Eigen::VectorXd ret(prev.size());
    for (Eigen::Index i = 0; i < prev.size(); ++i) {
        if (prev[i] > 0.0 && curr[i] > 0.0) {
            ret[i] = std::log(curr[i] / prev[i]);
        } else {
            ret[i] = 0.0;
        }
    }
    return ret;
}

/// @brief Compute scalar Lorentz factor γ = 1/sqrt(1 − β²).
double lorentz_gamma(double beta) noexcept {
    const double b2 = beta * beta;
    if (b2 >= 1.0) return 1.0 / std::sqrt(1.0 - constants::BETA_MAX_SAFE *
                                                  constants::BETA_MAX_SAFE);
    return 1.0 / std::sqrt(1.0 - b2);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// MultiAssetInterval
// ---------------------------------------------------------------------------

std::optional<double>
MultiAssetInterval::compute(const MultiAssetEvent& a,
                             const MultiAssetEvent& b,
                             const Eigen::MatrixXd& metric) noexcept
{
    if (!a.is_valid() || !b.is_valid()) return std::nullopt;
    if (a.n_assets() != b.n_assets()) return std::nullopt;

    const std::size_t N = a.n_assets();
    const Eigen::Index dim = static_cast<Eigen::Index>(N + 1);
    if (metric.rows() != dim || metric.cols() != dim) return std::nullopt;

    // Build the displacement 4-vector: Δx^μ = (Δt, ΔP_1, ..., ΔP_N).
    Eigen::VectorXd dx(dim);
    const double dt = static_cast<double>(b.timestamp - a.timestamp) / 1000.0; // ms → s
    dx[0] = dt;
    for (std::size_t i = 0; i < N; ++i) {
        dx[static_cast<Eigen::Index>(i + 1)] = b.prices[i] - a.prices[i];
    }

    // ds² = g_μν Δx^μ Δx^ν
    const double ds2 = (dx.transpose() * metric * dx).value();
    return ds2;
}

MultiAssetInterval::IntervalType
MultiAssetInterval::classify(double ds2) noexcept {
    if (std::abs(ds2) <= kLightlikeEps) return IntervalType::Lightlike;
    if (ds2 < 0.0)                      return IntervalType::Timelike;
    return IntervalType::Spacelike;
}

std::optional<std::pair<double, MultiAssetInterval::IntervalType>>
MultiAssetInterval::compute_and_classify(const MultiAssetEvent& a,
                                          const MultiAssetEvent& b,
                                          const Eigen::MatrixXd& metric) noexcept
{
    auto ds2 = compute(a, b, metric);
    if (!ds2) return std::nullopt;
    return std::make_pair(*ds2, classify(*ds2));
}

// ---------------------------------------------------------------------------
// CorrelationMetric
// ---------------------------------------------------------------------------

CorrelationMetric::CorrelationMetric(std::size_t n_assets, Config cfg)
    : n_assets_(n_assets)
    , cfg_(cfg)
{
    assert(n_assets_ > 0);
    log_returns_window_.resize(cfg_.window_size);
    for (auto& v : log_returns_window_) {
        v.setZero(static_cast<Eigen::Index>(n_assets_));
    }
    prev_prices_.setZero(static_cast<Eigen::Index>(n_assets_));
}

std::optional<Eigen::MatrixXd>
CorrelationMetric::update(const std::vector<double>& prices) {
    if (prices.size() != n_assets_) return std::nullopt;

    // Convert to Eigen vector.
    Eigen::VectorXd p(static_cast<Eigen::Index>(n_assets_));
    for (std::size_t i = 0; i < n_assets_; ++i)
        p[static_cast<Eigen::Index>(i)] = prices[i];

    if (!has_prev_) {
        prev_prices_ = p;
        has_prev_ = true;
        obs_count_++;
        return std::nullopt; // Need at least 2 observations.
    }

    // Compute log-return and push into circular buffer.
    Eigen::VectorXd ret = log_returns(prev_prices_, p);
    log_returns_window_[window_head_] = ret;
    window_head_ = (window_head_ + 1) % cfg_.window_size;
    obs_count_++;
    prev_prices_ = p;

    if (obs_count_ < 2) return std::nullopt;

    metric_dirty_ = true;
    recompute_metric();
    return current_metric_;
}

std::optional<Eigen::MatrixXd>
CorrelationMetric::current_metric() const noexcept {
    return current_metric_;
}

std::optional<Eigen::MatrixXd>
CorrelationMetric::correlation_matrix() const noexcept {
    if (!current_metric_) return std::nullopt;
    const Eigen::Index N = static_cast<Eigen::Index>(n_assets_);
    // Extract spatial block and normalise by diagonal (variances).
    Eigen::MatrixXd spatial = current_metric_->block(1, 1, N, N);
    Eigen::VectorXd variances = spatial.diagonal();
    Eigen::MatrixXd corr(N, N);
    for (Eigen::Index i = 0; i < N; ++i) {
        for (Eigen::Index j = 0; j < N; ++j) {
            const double denom = std::sqrt(variances[i] * variances[j]);
            corr(i, j) = (denom > 1e-12) ? spatial(i, j) / denom : (i == j ? 1.0 : 0.0);
        }
    }
    return corr;
}

void CorrelationMetric::reset() noexcept {
    obs_count_ = 0;
    window_head_ = 0;
    has_prev_ = false;
    metric_dirty_ = true;
    current_metric_ = std::nullopt;
    for (auto& v : log_returns_window_)
        v.setZero(static_cast<Eigen::Index>(n_assets_));
}

Eigen::MatrixXd CorrelationMetric::compute_covariance() const {
    const Eigen::Index N = static_cast<Eigen::Index>(n_assets_);
    // Collect valid observations (up to window_size or obs_count-1).
    const std::size_t n_obs = std::min(obs_count_ - 1, cfg_.window_size);
    if (n_obs < 2) return Eigen::MatrixXd::Identity(N, N) * 1e-4;

    // Compute sample mean.
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(N);
    for (std::size_t k = 0; k < n_obs; ++k) {
        // Walk backward from (window_head_ - 1) to access most recent n_obs.
        const std::size_t idx = (window_head_ + cfg_.window_size - 1 - k) % cfg_.window_size;
        mean += log_returns_window_[idx];
    }
    mean /= static_cast<double>(n_obs);

    // Compute sample covariance.
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(N, N);
    for (std::size_t k = 0; k < n_obs; ++k) {
        const std::size_t idx = (window_head_ + cfg_.window_size - 1 - k) % cfg_.window_size;
        const Eigen::VectorXd z = log_returns_window_[idx] - mean;
        cov += z * z.transpose();
    }
    cov /= static_cast<double>(n_obs - 1);

    return cov;
}

void CorrelationMetric::recompute_metric() {
    const Eigen::Index N = static_cast<Eigen::Index>(n_assets_);
    const Eigen::Index dim = N + 1;

    Eigen::MatrixXd cov = compute_covariance();

    // Regularise if not positive definite.
    Eigen::LLT<Eigen::MatrixXd> llt(cov);
    if (llt.info() != Eigen::Success) {
        cov += Eigen::MatrixXd::Identity(N, N) * cfg_.regularisation_eps;
    }

    // Build (N+1)×(N+1) Lorentzian metric.
    Eigen::MatrixXd g = Eigen::MatrixXd::Zero(dim, dim);
    g(0, 0) = -(cfg_.c_market * cfg_.c_market);
    g.block(1, 1, N, N) = cov;

    current_metric_ = g;
    metric_dirty_ = false;
}

// ---------------------------------------------------------------------------
// MultiAssetLorentz
// ---------------------------------------------------------------------------

std::optional<MultiAssetLorentz::TransformResult>
MultiAssetLorentz::transform(const MultiAssetEvent& a,
                              const MultiAssetEvent& b,
                              const Eigen::MatrixXd& metric) noexcept
{
    if (!a.is_valid() || !b.is_valid()) return std::nullopt;
    if (a.n_assets() != b.n_assets()) return std::nullopt;

    const std::size_t N = a.n_assets();
    const Eigen::Index dim = static_cast<Eigen::Index>(N + 1);
    if (metric.rows() != dim || metric.cols() != dim) return std::nullopt;

    const double dt = static_cast<double>(b.timestamp - a.timestamp) / 1000.0;
    if (std::abs(dt) < 1e-12) return std::nullopt;

    // Compute per-asset β_i = ΔP_i / (c · Δt).
    const double c = std::sqrt(std::abs(metric(0, 0)));
    std::vector<double> betas(N);
    std::vector<double> gammas(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double dp = b.prices[i] - a.prices[i];
        const double beta = clamp_d(dp / (c * dt),
                                    -constants::BETA_MAX_SAFE,
                                     constants::BETA_MAX_SAFE);
        betas[i]  = beta;
        gammas[i] = lorentz_gamma(beta);
    }

    // Portfolio β (metric-weighted).
    const double beta_p = portfolio_beta(betas, metric);
    const double gamma_p = lorentz_gamma(beta_p);

    // Apply γ_portfolio to each asset's price adjustment.
    std::vector<double> adjusted(N);
    for (std::size_t i = 0; i < N; ++i) {
        adjusted[i] = gammas[i] * (b.prices[i] - a.prices[i]);
    }

    return TransformResult{
        adjusted, beta_p, LorentzFactor{gamma_p}, betas, gammas
    };
}

double
MultiAssetLorentz::portfolio_beta(const std::vector<double>& betas,
                                   const Eigen::MatrixXd&     metric) noexcept
{
    const std::size_t N = betas.size();
    if (N == 0) return 0.0;
    const Eigen::Index dim = metric.rows();
    if (dim < static_cast<Eigen::Index>(N + 1)) return 0.0;

    // Build β vector (spatial components only).
    Eigen::VectorXd b(static_cast<Eigen::Index>(N));
    for (std::size_t i = 0; i < N; ++i)
        b[static_cast<Eigen::Index>(i)] = betas[i];

    // Extract spatial block of metric.
    const Eigen::MatrixXd g_spatial = metric.block(1, 1,
        static_cast<Eigen::Index>(N),
        static_cast<Eigen::Index>(N));

    // β_portfolio² = g_ij β_i β_j
    const double b2 = (b.transpose() * g_spatial * b).value();
    if (b2 <= 0.0) return 0.0;

    return clamp_d(std::sqrt(b2), 0.0, constants::BETA_MAX_SAFE);
}

// ---------------------------------------------------------------------------
// PortfolioGeodesic
// ---------------------------------------------------------------------------

std::vector<PortfolioGeodesic::GeodesicStep>
PortfolioGeodesic::integrate(const MultiAssetEvent& initial,
                              const Eigen::VectorXd& four_velocity,
                              const Eigen::MatrixXd& metric,
                              std::size_t            n_steps,
                              double                 dt) noexcept
{
    if (!initial.is_valid() || n_steps == 0 || std::abs(dt) < 1e-15)
        return {};

    const std::size_t N = initial.n_assets();
    const Eigen::Index dim = static_cast<Eigen::Index>(N + 1);
    if (four_velocity.size() != dim) return {};
    if (metric.rows() != dim || metric.cols() != dim) return {};

    std::vector<GeodesicStep> path;
    path.reserve(n_steps);

    // Current position in spacetime.
    Eigen::VectorXd x(dim);
    x[0] = static_cast<double>(initial.timestamp) / 1000.0; // ms → s
    for (std::size_t i = 0; i < N; ++i)
        x[static_cast<Eigen::Index>(i + 1)] = initial.prices[i];

    // For a flat (constant metric) manifold the geodesic is a straight line:
    // x^μ(τ) = x^μ(0) + u^μ · τ
    // Christoffel symbols vanish so there is no acceleration term.
    double tau = 0.0;
    double arc_length = 0.0;

    for (std::size_t step = 0; step < n_steps; ++step) {
        // Euler step.
        x += four_velocity * dt;
        tau += dt;

        // Compute proper arc length element: ds² = g_μν u^μ u^ν · dτ²
        const double ds2_dot = (four_velocity.transpose() * metric * four_velocity).value();
        const double ds = std::sqrt(std::abs(ds2_dot)) * std::abs(dt);
        arc_length += ds;

        // Build MultiAssetEvent from current x.
        MultiAssetEvent ev;
        ev.timestamp = static_cast<int64_t>(x[0] * 1000.0); // s → ms
        ev.prices.resize(N);
        for (std::size_t i = 0; i < N; ++i)
            ev.prices[i] = x[static_cast<Eigen::Index>(i + 1)];
        if (!initial.symbols.empty()) ev.symbols = initial.symbols;
        if (!initial.volumes.empty())  ev.volumes = initial.volumes;

        path.push_back({ev, tau, arc_length, 0.0 /* filled by deviation_series */});
    }
    return path;
}

std::vector<double>
PortfolioGeodesic::deviation_series(
    const std::vector<GeodesicStep>&   predicted,
    const std::vector<MultiAssetEvent>& actual) noexcept
{
    if (predicted.size() != actual.size()) return {};
    std::vector<double> devs;
    devs.reserve(predicted.size());

    for (std::size_t i = 0; i < predicted.size(); ++i) {
        const auto& pred_prices   = predicted[i].event.prices;
        const auto& actual_prices = actual[i].prices;
        if (pred_prices.size() != actual_prices.size()) {
            devs.push_back(0.0);
            continue;
        }
        double sq_sum = 0.0;
        for (std::size_t j = 0; j < pred_prices.size(); ++j) {
            const double d = pred_prices[j] - actual_prices[j];
            sq_sum += d * d;
        }
        devs.push_back(std::sqrt(sq_sum));
    }
    return devs;
}

std::vector<Eigen::VectorXd>
PortfolioGeodesic::portfolio_weights(const std::vector<GeodesicStep>& steps,
                                      const Eigen::VectorXd&            four_velocity,
                                      double                            gross_exposure) noexcept
{
    if (steps.empty() || four_velocity.size() < 2) return {};
    const Eigen::Index N = four_velocity.size() - 1; // spatial dimensions

    // Extract spatial components of the four-velocity.
    const Eigen::VectorXd u_spatial = four_velocity.tail(N);
    const double norm = u_spatial.lpNorm<1>(); // L1 norm for weight normalisation

    std::vector<Eigen::VectorXd> weights;
    weights.reserve(steps.size());

    for (std::size_t i = 0; i < steps.size(); ++i) {
        Eigen::VectorXd w(N);
        if (norm < 1e-12) {
            w.setZero();
        } else {
            w = u_spatial * (gross_exposure / norm);
        }
        weights.push_back(w);
    }
    return weights;
}

} // namespace srfm::multi_asset
