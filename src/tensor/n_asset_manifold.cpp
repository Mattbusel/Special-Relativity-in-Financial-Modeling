/**
 * @file  n_asset_manifold.cpp
 * @brief Implementation of NAssetManifold.
 *
 * See include/srfm/tensor/n_asset_manifold.hpp for the public API contract.
 */

#include "../../include/srfm/tensor/n_asset_manifold.hpp"

#include <algorithm>
#include <cmath>

namespace srfm::tensor {

// ── Constructor ───────────────────────────────────────────────────────────────

NAssetManifold::NAssetManifold(int n_assets,
                               Eigen::MatrixXd covariance,
                               double c_market) noexcept
    : n_assets_(n_assets)
    , c_market_(c_market)
    , covariance_(std::move(covariance))
    , inv_valid_(false)
{
    build_metric();
    build_inverse_metric();
}

// ── Factory ───────────────────────────────────────────────────────────────────

std::optional<NAssetManifold>
NAssetManifold::make(int n, Eigen::MatrixXd cov, double c_market) noexcept {
    // Validate n_assets >= 1.
    if (n < 1) {
        return std::nullopt;
    }

    // Validate c_market > 0.
    if (c_market <= 0.0) {
        return std::nullopt;
    }

    // Validate covariance dimensions.
    if (cov.rows() != n || cov.cols() != n) {
        return std::nullopt;
    }

    // Validate symmetry (allow small numerical noise).
    const double sym_tol = 1e-10;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (std::abs(cov(i, j) - cov(j, i)) > sym_tol) {
                return std::nullopt;
            }
        }
    }

    // Symmetrise numerically.
    Eigen::MatrixXd cov_sym = 0.5 * (cov + cov.transpose());

    // Validate positive-definiteness: all eigenvalues must be > 0.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov_sym,
                                                        Eigen::EigenvaluesOnly);
    if (eig.info() != Eigen::Success) {
        return std::nullopt;
    }
    if (eig.eigenvalues().minCoeff() <= 0.0) {
        return std::nullopt;
    }

    return NAssetManifold(n, std::move(cov_sym), c_market);
}

// ── Metric construction ───────────────────────────────────────────────────────

void NAssetManifold::build_metric() noexcept {
    const int D = n_assets_ + 1;
    metric_ = Eigen::MatrixXd::Zero(D, D);

    // Time-time component.
    metric_(0, 0) = -(c_market_ * c_market_);

    // Spatial block: copy covariance matrix into rows/cols 1..N.
    if (covariance_.rows() == n_assets_ && covariance_.cols() == n_assets_) {
        metric_.block(1, 1, n_assets_, n_assets_) = covariance_;
    }
    // Time-space cross terms remain zero.
}

void NAssetManifold::build_inverse_metric() noexcept {
    const int D = n_assets_ + 1;

    // Use full-pivot LU for numerical stability.
    Eigen::FullPivLU<Eigen::MatrixXd> lu(metric_);
    if (!lu.isInvertible()) {
        inv_metric_ = Eigen::MatrixXd::Zero(D, D);
        inv_valid_  = false;
        return;
    }

    inv_metric_ = lu.inverse();
    inv_valid_  = true;
}

// ── Metric tensor accessors ───────────────────────────────────────────────────

std::optional<Eigen::MatrixXd>
NAssetManifold::metric_at(const Eigen::VectorXd& x) const noexcept {
    if (x.size() != dim()) {
        return std::nullopt;
    }
    // Constant metric — x is unused.
    return metric_;
}

std::optional<Eigen::MatrixXd>
NAssetManifold::inverse_metric_at(const Eigen::VectorXd& x) const noexcept {
    if (x.size() != dim()) {
        return std::nullopt;
    }
    if (!inv_valid_) {
        return std::nullopt;
    }
    return inv_metric_;
}

// ── Line element ──────────────────────────────────────────────────────────────

std::optional<double>
NAssetManifold::line_element_sq(const Eigen::VectorXd& x,
                                const Eigen::VectorXd& dx) const noexcept {
    if (x.size() != dim() || dx.size() != dim()) {
        return std::nullopt;
    }
    // ds² = dx^T · g · dx
    return dx.dot(metric_ * dx);
}

// ── Other public methods ──────────────────────────────────────────────────────

std::optional<Eigen::MatrixXd> NAssetManifold::covariance() const noexcept {
    return covariance_;
}

bool NAssetManifold::reduces_to_4d(const NAssetManifold& other) const noexcept {
    // Compatible with 4D means other has n_assets == 3.
    if (other.n_assets() != 3) {
        return false;
    }
    // This manifold must have at least 3 spatial dimensions.
    return n_assets_ >= 3;
}

} // namespace srfm::tensor
