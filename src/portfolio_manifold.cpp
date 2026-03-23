/// @file src/portfolio_manifold.cpp
/// @brief Implementation of MinkowskiCovariance and SpacetimeCausalGraph.
///
/// See include/portfolio_manifold.hpp for the public API contract.

#include "portfolio_manifold.hpp"

#include "srfm/constants.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace srfm::portfolio {

// ─── AssetEvent helpers ───────────────────────────────────────────────────────

Eigen::Vector4d AssetEvent::to_four_vector() const noexcept {
    Eigen::Vector4d v;
    v << t, P, V, M;
    return v;
}

manifold::NAssetEvent AssetEvent::to_nasset_event() const noexcept {
    // Map the 4-coord event (t, P, V, M) to an NAssetEvent with prices = [P, V, M].
    Eigen::VectorXd prices(3);
    prices << P, V, M;
    // NAssetEvent::make validates non-empty prices; always succeeds here.
    auto opt = manifold::NAssetEvent::make(t, prices);
    // Fallback: construct manually if make fails (should never happen for valid data).
    if (opt) {
        return *opt;
    }
    // Return a default-constructed event with single zero price to satisfy noexcept.
    Eigen::VectorXd fallback(1);
    fallback << 0.0;
    return manifold::NAssetEvent{t, fallback};
}

// ─── MinkowskiCovariance ──────────────────────────────────────────────────────

MinkowskiCovariance::MinkowskiCovariance(double c_market) noexcept
    : c_market_(c_market) {}

void MinkowskiCovariance::add_asset(AssetEvent event) {
    events_.push_back(std::move(event));
}

std::size_t MinkowskiCovariance::size() const noexcept {
    return events_.size();
}

const std::vector<AssetEvent>& MinkowskiCovariance::events() const noexcept {
    return events_;
}

void MinkowskiCovariance::clear() noexcept {
    events_.clear();
}

double MinkowskiCovariance::raw_interval(std::size_t i,
                                          std::size_t j) const noexcept {
    // ds² = -c²·Δt² + ΔP² + ΔV² + ΔM²
    const AssetEvent& a = events_[i];
    const AssetEvent& b = events_[j];

    const double dt = b.t - a.t;
    const double dP = b.P - a.P;
    const double dV = b.V - a.V;
    const double dM = b.M - a.M;

    return -(c_market_ * c_market_) * (dt * dt)
           + (dP * dP)
           + (dV * dV)
           + (dM * dM);
}

std::optional<double>
MinkowskiCovariance::interval_correlation(std::size_t i,
                                           std::size_t j) const noexcept {
    if (i >= events_.size() || j >= events_.size()) {
        return std::nullopt;
    }
    return raw_interval(i, j);
}

std::optional<manifold::IntervalType>
MinkowskiCovariance::classify_pair(std::size_t i, std::size_t j) const noexcept {
    if (i >= events_.size() || j >= events_.size()) {
        return std::nullopt;
    }
    const double ds2 = raw_interval(i, j);
    if (std::abs(ds2) < manifold::LIGHTLIKE_THRESHOLD) {
        return manifold::IntervalType::LIGHTLIKE;
    }
    return ds2 < 0.0 ? manifold::IntervalType::TIMELIKE
                      : manifold::IntervalType::SPACELIKE;
}

std::optional<Eigen::MatrixXd>
MinkowskiCovariance::compute_spacetime_covariance() const noexcept {
    const std::size_t N = events_.size();
    if (N < 2) {
        return std::nullopt;
    }

    Eigen::MatrixXd cov(static_cast<int>(N), static_cast<int>(N));

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            if (i == j) {
                // Diagonal: perfect self-correlation.
                cov(static_cast<int>(i), static_cast<int>(j)) = 1.0;
            } else {
                // Off-diagonal: Gaussian kernel over the spacetime interval.
                // C(i,j) = exp(−|ds²(i,j)|)
                // - TIMELIKE  (ds² << 0): |ds²| large → C → 0
                // - LIGHTLIKE (ds² ≈ 0):  |ds²| ≈ 0  → C → 1
                // - SPACELIKE (ds² >> 0): |ds²| large → C → 0
                const double ds2 = raw_interval(i, j);
                cov(static_cast<int>(i), static_cast<int>(j)) =
                    std::exp(-std::abs(ds2));
            }
        }
    }

    return cov;
}

// ─── SpacetimeCausalGraph ─────────────────────────────────────────────────────

SpacetimeCausalGraph::SpacetimeCausalGraph(Eigen::MatrixXi adjacency,
                                            std::vector<std::string> ids) noexcept
    : adj_(std::move(adjacency))
    , asset_ids_(std::move(ids)) {}

std::optional<SpacetimeCausalGraph>
SpacetimeCausalGraph::build(const MinkowskiCovariance& mc) noexcept {
    const std::size_t N = mc.size();
    if (N < 2) {
        return std::nullopt;
    }

    const int n = static_cast<int>(N);
    Eigen::MatrixXi adj = Eigen::MatrixXi::Zero(n, n);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            if (i == j) {
                // No self-loops.
                continue;
            }
            // Edge (i→j) exists iff asset_i is TIMELIKE-separated from asset_j.
            // TIMELIKE means ds²(i,j) < −LIGHTLIKE_THRESHOLD.
            auto ds2_opt = mc.interval_correlation(i, j);
            if (!ds2_opt) {
                continue;
            }
            const double ds2 = *ds2_opt;
            if (ds2 < -manifold::LIGHTLIKE_THRESHOLD) {
                // TIMELIKE separation: causal influence i → j.
                adj(static_cast<int>(i), static_cast<int>(j)) = 1;
            }
        }
    }

    // Collect asset IDs.
    std::vector<std::string> ids;
    ids.reserve(N);
    for (const auto& ev : mc.events()) {
        ids.push_back(ev.asset_id);
    }

    return SpacetimeCausalGraph(std::move(adj), std::move(ids));
}

const Eigen::MatrixXi& SpacetimeCausalGraph::adjacency_matrix() const noexcept {
    return adj_;
}

bool SpacetimeCausalGraph::has_edge(std::size_t i, std::size_t j) const noexcept {
    if (i >= static_cast<std::size_t>(adj_.rows()) ||
        j >= static_cast<std::size_t>(adj_.cols())) {
        return false;
    }
    return adj_(static_cast<int>(i), static_cast<int>(j)) != 0;
}

std::size_t SpacetimeCausalGraph::n_assets() const noexcept {
    return static_cast<std::size_t>(adj_.rows());
}

int SpacetimeCausalGraph::out_degree(std::size_t i) const noexcept {
    if (i >= static_cast<std::size_t>(adj_.rows())) {
        return 0;
    }
    return adj_.row(static_cast<int>(i)).sum();
}

int SpacetimeCausalGraph::in_degree(std::size_t j) const noexcept {
    if (j >= static_cast<std::size_t>(adj_.cols())) {
        return 0;
    }
    return adj_.col(static_cast<int>(j)).sum();
}

const std::vector<std::string>& SpacetimeCausalGraph::asset_ids() const noexcept {
    return asset_ids_;
}

} // namespace srfm::portfolio
