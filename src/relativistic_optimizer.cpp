/// @file src/relativistic_optimizer.cpp
/// @brief Implementation of RelativisticPortfolio optimizer.
///
/// See include/relativistic_optimizer.hpp for the public API contract.
///
/// ## Algorithm
/// The optimization problem is:
///   minimise    risk_tolerance · w^T Σ_st w
///   subject to  w^T μ_rel ≥ r_target
///               Σ_i w_i = 1,  w_i ≥ 0
///
/// Solved via projected gradient descent:
///   w_{k+1} = Π_simplex(w_k − α · ∇_w L(w_k))
/// where
///   L(w) = risk_tolerance · w^T Σ_st w − λ · (w^T μ_rel − r_target)
///   ∇_w L(w) = 2 · risk_tolerance · Σ_st · w − λ · μ_rel
///
/// λ is adaptively set so the return constraint is enforced at each step.

#include "relativistic_optimizer.hpp"

#include "srfm/constants.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace srfm::portfolio {

// ─── Construction ─────────────────────────────────────────────────────────────

RelativisticPortfolio::RelativisticPortfolio(OptimizerConfig config) noexcept
    : config_(config) {}

// ─── Asset management ─────────────────────────────────────────────────────────

void RelativisticPortfolio::add_asset(AssetEvent event,
                                       double expected_return) {
    events_.push_back(std::move(event));
    expected_returns_.push_back(expected_return);
}

std::size_t RelativisticPortfolio::n_assets() const noexcept {
    return events_.size();
}

void RelativisticPortfolio::clear() noexcept {
    events_.clear();
    expected_returns_.clear();
}

// ─── Lorentz factor ───────────────────────────────────────────────────────────

double RelativisticPortfolio::compute_beta(const AssetEvent& e) const noexcept {
    // Approximate β from the event's price coordinate relative to origin.
    // β = |P| / (c_market · |t|) when |t| > 0, else 0.
    const double c = config_.c_market;
    if (std::abs(e.t) < constants::FLOAT_EPSILON) {
        return 0.0;
    }
    const double beta = std::abs(e.P) / (c * std::abs(e.t));
    // Clamp to BETA_MAX_SAFE to prevent γ divergence.
    return std::min(beta, constants::BETA_MAX_SAFE);
}

double RelativisticPortfolio::gamma_factor(double beta) noexcept {
    // γ(β) = 1 / √(1 − β²), clamped for stability.
    const double beta_clamped = std::min(std::abs(beta), constants::BETA_MAX_SAFE);
    const double denom = std::sqrt(1.0 - beta_clamped * beta_clamped);
    // denom should be > 0 by BETA_MAX_SAFE clamping, but guard anyway.
    if (denom < constants::FLOAT_EPSILON) {
        return 1.0 / std::sqrt(1.0 - constants::BETA_MAX_SAFE *
                                         constants::BETA_MAX_SAFE);
    }
    return 1.0 / denom;
}

// ─── Relativistic returns ─────────────────────────────────────────────────────

std::optional<Eigen::VectorXd>
RelativisticPortfolio::relativistic_returns() const noexcept {
    const std::size_t N = events_.size();
    if (N == 0) {
        return std::nullopt;
    }

    Eigen::VectorXd mu_rel(static_cast<int>(N));
    for (std::size_t i = 0; i < N; ++i) {
        const double beta  = compute_beta(events_[i]);
        const double gamma = gamma_factor(beta);
        mu_rel(static_cast<int>(i)) = gamma * expected_returns_[i];
    }
    return mu_rel;
}

// ─── Spacetime covariance ─────────────────────────────────────────────────────

std::optional<Eigen::MatrixXd>
RelativisticPortfolio::spacetime_covariance() const noexcept {
    if (events_.size() < 2) {
        return std::nullopt;
    }
    MinkowskiCovariance mc(config_.c_market);
    for (const auto& ev : events_) {
        mc.add_asset(ev);
    }
    return mc.compute_spacetime_covariance();
}

// ─── Simplex projection ───────────────────────────────────────────────────────

Eigen::VectorXd
RelativisticPortfolio::project_simplex(const Eigen::VectorXd& v) noexcept {
    // Algorithm: Duchi et al. (2008), "Efficient Projections onto the l1-Ball".
    // Projects onto  Δ_N = { w ∈ R^N : Σw_i = 1, w_i >= 0 }.
    const int n = static_cast<int>(v.size());

    // Sort descending.
    Eigen::VectorXd u = v;
    std::sort(u.data(), u.data() + n, std::greater<double>());

    // Compute rho: largest index where u_j - (cumsum_j - 1)/j > 0.
    double cumsum = 0.0;
    int rho = 0;
    for (int j = 0; j < n; ++j) {
        cumsum += u(j);
        if (u(j) - (cumsum - 1.0) / static_cast<double>(j + 1) > 0.0) {
            rho = j;
        }
    }

    // Compute threshold theta.
    double cumsum_rho = 0.0;
    for (int j = 0; j <= rho; ++j) {
        cumsum_rho += u(j);
    }
    const double theta = (cumsum_rho - 1.0) / static_cast<double>(rho + 1);

    // Project.
    Eigen::VectorXd w(n);
    for (int i = 0; i < n; ++i) {
        w(i) = std::max(v(i) - theta, 0.0);
    }
    return w;
}

// ─── Geodesic gradient ────────────────────────────────────────────────────────

Eigen::VectorXd
RelativisticPortfolio::geodesic_gradient(const Eigen::MatrixXd& sigma_st,
                                          const Eigen::VectorXd& w) noexcept {
    // Gradient of  d²_geo(w) = w^T Σ_st w  is  2 Σ_st w.
    return 2.0 * sigma_st * w;
}

// ─── optimize_weights ─────────────────────────────────────────────────────────

std::optional<OptimizationResult>
RelativisticPortfolio::optimize_weights(double target_return,
                                         double risk_tolerance) const noexcept {
    const std::size_t N = events_.size();
    if (N < 2) {
        return std::nullopt;
    }

    // Build spacetime covariance matrix (risk metric).
    auto sigma_opt = spacetime_covariance();
    if (!sigma_opt) {
        return std::nullopt;
    }
    Eigen::MatrixXd sigma_st = *sigma_opt;

    // Apply Tikhonov regularisation: Σ_reg = Σ_st + λI.
    // This prevents singular or ill-conditioned geodesic metrics.
    sigma_st += config_.regularisation *
                Eigen::MatrixXd::Identity(static_cast<int>(N),
                                          static_cast<int>(N));

    // Relativistic returns μ_rel.
    auto mu_opt = relativistic_returns();
    if (!mu_opt) {
        return std::nullopt;
    }
    const Eigen::VectorXd mu_rel = *mu_opt;

    // Initialise weights to uniform (maximum entropy starting point).
    Eigen::VectorXd w = Eigen::VectorXd::Constant(
        static_cast<int>(N), 1.0 / static_cast<double>(N));

    const double alpha     = config_.step_size;
    const double tol       = config_.convergence_tol;
    const int    max_iter  = config_.max_iterations;

    int  iter      = 0;
    bool converged = false;

    for (iter = 0; iter < max_iter; ++iter) {
        // Geodesic risk gradient: ∇_w (risk_tolerance · w^T Σ_st w)
        Eigen::VectorXd grad = risk_tolerance * geodesic_gradient(sigma_st, w);

        // Return constraint penalty gradient.
        // L_aug = risk * w^T Σ w − penalty * max(0, w^T μ − r_target)
        // Penalty: when return falls short, pull weights toward high-μ assets.
        const double current_return = w.dot(mu_rel);
        if (current_return < target_return) {
            // Gradient ascent on return: subtract mu_rel to reduce the deficit.
            const double penalty = risk_tolerance;
            grad -= penalty * mu_rel;
        }

        // Projected gradient step.
        Eigen::VectorXd w_new = project_simplex(w - alpha * grad);

        // Check convergence.
        const double step_norm = (w_new - w).norm();
        w = w_new;

        if (step_norm < tol) {
            converged = true;
            ++iter;
            break;
        }
    }

    // Compute final risk and return.
    const double geo_risk = risk_tolerance * w.dot(sigma_st * w);
    const double exp_ret  = w.dot(mu_rel);

    return OptimizationResult{
        .weights         = w,
        .geodesic_risk   = geo_risk,
        .expected_return = exp_ret,
        .iterations      = iter,
        .converged       = converged,
    };
}

} // namespace srfm::portfolio
