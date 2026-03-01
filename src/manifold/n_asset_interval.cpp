/**
 * @file  n_asset_interval.cpp
 * @brief Implementation of NAssetEvent and NAssetInterval.
 *
 * See include/srfm/manifold/n_asset_interval.hpp for the public API contract.
 */

#include "../../include/srfm/manifold/n_asset_interval.hpp"

#include <cmath>

namespace srfm::manifold {

// ── NAssetEvent ───────────────────────────────────────────────────────────────

Eigen::VectorXd NAssetEvent::to_coords() const noexcept {
    const int N = static_cast<int>(prices.size());
    Eigen::VectorXd coords(N + 1);
    coords(0) = t;
    for (int i = 0; i < N; ++i) {
        coords(i + 1) = prices(i);
    }
    return coords;
}

std::optional<NAssetEvent>
NAssetEvent::make(double t, Eigen::VectorXd prices) noexcept {
    if (prices.size() == 0) {
        return std::nullopt;
    }
    NAssetEvent ev;
    ev.t      = t;
    ev.prices = std::move(prices);
    return ev;
}

// ── NAssetInterval ────────────────────────────────────────────────────────────

std::optional<IntervalResult>
NAssetInterval::compute(const NAssetEvent&    a,
                        const NAssetEvent&    b,
                        const NAssetManifold& manifold) const noexcept {
    // Validate dimensions.
    if (static_cast<int>(a.prices.size()) != manifold.n_assets()) {
        return std::nullopt;
    }
    if (static_cast<int>(b.prices.size()) != manifold.n_assets()) {
        return std::nullopt;
    }

    // Compute displacement Δx = b - a in (N+1)-dimensional coordinates.
    Eigen::VectorXd xa = a.to_coords();
    Eigen::VectorXd xb = b.to_coords();
    Eigen::VectorXd dx = xb - xa;

    // Evaluate metric at the midpoint (constant metric → same everywhere).
    Eigen::VectorXd x_mid = 0.5 * (xa + xb);
    auto g_opt = manifold.metric_at(x_mid);
    if (!g_opt) { return std::nullopt; }

    // ds² = dx^T · g · dx.
    double ds_sq = dx.dot((*g_opt) * dx);

    IntervalResult result;
    result.ds_sq     = ds_sq;
    result.type      = classify(ds_sq);
    result.magnitude = std::sqrt(std::abs(ds_sq));

    return result;
}

std::optional<std::vector<IntervalResult>>
NAssetInterval::batch_from_reference(
        const NAssetEvent&           ref,
        std::span<const NAssetEvent> events,
        const NAssetManifold&        manifold) const noexcept {
    std::vector<IntervalResult> results;
    results.reserve(events.size());

    for (const auto& ev : events) {
        auto r = compute(ref, ev, manifold);
        if (!r) { return std::nullopt; }
        results.push_back(std::move(*r));
    }

    return results;
}

} // namespace srfm::manifold
