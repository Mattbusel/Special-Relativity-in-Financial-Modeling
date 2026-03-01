/**
 * @file  christoffel_n.cpp
 * @brief Implementation of ChristoffelN.
 *
 * See include/srfm/tensor/christoffel_n.hpp for the public API contract.
 *
 * ## Algorithm
 * Christoffel symbols of the second kind:
 *
 *   Γ^λ_μν = ½ g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
 *
 * Metric derivatives are evaluated by central finite differences:
 *
 *   ∂_α g_μν ≈ [g_μν(x + h·e_α) - g_μν(x - h·e_α)] / (2h)
 *
 * For a constant metric (NAssetManifold base case) all derivatives vanish,
 * so all Christoffel symbols are identically zero.
 */

#include "../../include/srfm/tensor/christoffel_n.hpp"

#include <cmath>

namespace srfm::tensor {

// ── Constructor ───────────────────────────────────────────────────────────────

ChristoffelN::ChristoffelN(const NAssetManifold& manifold) noexcept
    : manifold_(manifold)
{}

// ── Internal helpers ──────────────────────────────────────────────────────────

std::optional<double>
ChristoffelN::metric_deriv(int alpha, int mu, int nu,
                           const Eigen::VectorXd& x) const noexcept {
    const int D = manifold_.dim();

    if (alpha < 0 || alpha >= D) { return std::nullopt; }
    if (mu    < 0 || mu    >= D) { return std::nullopt; }
    if (nu    < 0 || nu    >= D) { return std::nullopt; }
    if (x.size() != D)           { return std::nullopt; }

    // Forward perturb.
    Eigen::VectorXd xp = x;
    xp(alpha) += FD_STEP;
    auto gp = manifold_.metric_at(xp);
    if (!gp) { return std::nullopt; }

    // Backward perturb.
    Eigen::VectorXd xm = x;
    xm(alpha) -= FD_STEP;
    auto gm = manifold_.metric_at(xm);
    if (!gm) { return std::nullopt; }

    return ((*gp)(mu, nu) - (*gm)(mu, nu)) / (2.0 * FD_STEP);
}

std::optional<Eigen::MatrixXd>
ChristoffelN::inv_metric_at(const Eigen::VectorXd& x) const noexcept {
    return manifold_.inverse_metric_at(x);
}

// ── Public methods ────────────────────────────────────────────────────────────

std::optional<double>
ChristoffelN::symbol(int lambda, int mu, int nu,
                     const Eigen::VectorXd& x) const noexcept {
    const int D = manifold_.dim();

    // Validate indices.
    if (lambda < 0 || lambda >= D) { return std::nullopt; }
    if (mu     < 0 || mu     >= D) { return std::nullopt; }
    if (nu     < 0 || nu     >= D) { return std::nullopt; }
    if (x.size() != D)             { return std::nullopt; }

    // Obtain inverse metric.
    auto g_inv_opt = inv_metric_at(x);
    if (!g_inv_opt) { return std::nullopt; }
    const Eigen::MatrixXd& g_inv = *g_inv_opt;

    // Γ^λ_μν = ½ Σ_σ g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
    double result = 0.0;
    for (int sigma = 0; sigma < D; ++sigma) {
        auto d_mu_g_nu_sigma  = metric_deriv(mu,    nu, sigma, x);
        auto d_nu_g_mu_sigma  = metric_deriv(nu,    mu, sigma, x);
        auto d_sigma_g_mu_nu  = metric_deriv(sigma, mu, nu,    x);

        if (!d_mu_g_nu_sigma || !d_nu_g_mu_sigma || !d_sigma_g_mu_nu) {
            return std::nullopt;
        }

        double bracket = *d_mu_g_nu_sigma + *d_nu_g_mu_sigma - *d_sigma_g_mu_nu;
        result += g_inv(lambda, sigma) * bracket;
    }

    return 0.5 * result;
}

std::optional<std::vector<std::vector<std::vector<double>>>>
ChristoffelN::all_symbols(const Eigen::VectorXd& x) const noexcept {
    const int D = manifold_.dim();
    if (x.size() != D) { return std::nullopt; }

    // Obtain inverse metric once.
    auto g_inv_opt = inv_metric_at(x);
    if (!g_inv_opt) { return std::nullopt; }
    const Eigen::MatrixXd& g_inv = *g_inv_opt;

    // Pre-compute all metric derivatives: deriv[alpha][mu][nu] = ∂_alpha g_mu_nu.
    // Use a flat 3D array stored as vector-of-vector-of-vector.
    std::vector<std::vector<std::vector<double>>> dg(
        D, std::vector<std::vector<double>>(D, std::vector<double>(D, 0.0)));

    for (int alpha = 0; alpha < D; ++alpha) {
        for (int mu = 0; mu < D; ++mu) {
            for (int nu = 0; nu < D; ++nu) {
                auto val = metric_deriv(alpha, mu, nu, x);
                if (!val) { return std::nullopt; }
                dg[alpha][mu][nu] = *val;
            }
        }
    }

    // Compute full Christoffel tensor.
    std::vector<std::vector<std::vector<double>>> gamma(
        D, std::vector<std::vector<double>>(D, std::vector<double>(D, 0.0)));

    for (int lambda = 0; lambda < D; ++lambda) {
        for (int mu = 0; mu < D; ++mu) {
            for (int nu = 0; nu < D; ++nu) {
                double val = 0.0;
                for (int sigma = 0; sigma < D; ++sigma) {
                    double bracket = dg[mu][nu][sigma]
                                   + dg[nu][mu][sigma]
                                   - dg[sigma][mu][nu];
                    val += g_inv(lambda, sigma) * bracket;
                }
                gamma[lambda][mu][nu] = 0.5 * val;
            }
        }
    }

    return gamma;
}

bool ChristoffelN::verify_symmetry(const Eigen::VectorXd& x,
                                   double tol) const noexcept {
    const int D = manifold_.dim();
    if (x.size() != D) { return false; }

    for (int lambda = 0; lambda < D; ++lambda) {
        for (int mu = 0; mu < D; ++mu) {
            for (int nu = mu + 1; nu < D; ++nu) {
                auto s_mn = symbol(lambda, mu, nu, x);
                auto s_nm = symbol(lambda, nu, mu, x);
                if (!s_mn || !s_nm) { return false; }
                if (std::abs(*s_mn - *s_nm) > tol) { return false; }
            }
        }
    }
    return true;
}

} // namespace srfm::tensor
