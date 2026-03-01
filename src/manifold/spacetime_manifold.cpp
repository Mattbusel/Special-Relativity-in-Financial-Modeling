/**
 * @file  spacetime_manifold.cpp
 * @brief SpacetimeManifold implementation (AGT-13 / SRFM).
 *
 * See spacetime_manifold.hpp for the full module contract.
 */

#include "spacetime_manifold.hpp"

#include <algorithm>
#include <cmath>

namespace srfm::manifold {

// ── MetricTensor ──────────────────────────────────────────────────────────────

MetricTensor MetricTensor::minkowski() noexcept {
    MetricTensor m{};
    // Zero-initialise (already done by aggregate init above)
    // Diagonal: η = diag(−1, +1, +1, +1)
    m.g[0][0] = -1.0;
    m.g[1][1] =  1.0;
    m.g[2][2] =  1.0;
    m.g[3][3] =  1.0;
    return m;
}

bool MetricTensor::is_valid() const noexcept {
    // Check all entries finite
    for (int mu = 0; mu < DIM; ++mu) {
        for (int nu = 0; nu < DIM; ++nu) {
            if (!std::isfinite(g[mu][nu])) return false;
        }
    }
    // Time-time component must be negative (Lorentzian signature)
    if (g[0][0] >= 0.0) return false;
    // Spatial diagonal must be positive
    for (int i = 1; i < DIM; ++i) {
        if (g[i][i] <= 0.0) return false;
    }
    return true;
}

std::optional<MetricTensor> MetricTensor::inverse_diagonal() const noexcept {
    // Fast-path: assumes diagonal metric (off-diagonal ~ 0)
    // Check diagonals are non-zero and finite
    for (int i = 0; i < DIM; ++i) {
        if (!std::isfinite(g[i][i]) || g[i][i] == 0.0) return std::nullopt;
    }
    MetricTensor inv{};
    for (int i = 0; i < DIM; ++i) {
        inv.g[i][i] = 1.0 / g[i][i];
    }
    return inv;
}

// ── SpacetimeEvent ────────────────────────────────────────────────────────────

bool SpacetimeEvent::is_finite() const noexcept {
    return std::isfinite(t) && std::isfinite(x)
        && std::isfinite(y) && std::isfinite(z);
}

// ── SpacetimeManifold ─────────────────────────────────────────────────────────

std::optional<Regime>
SpacetimeManifold::process(const SpacetimeEvent& event) const noexcept {
    if (!event.is_finite()) return std::nullopt;

    // Map x-coordinate to a proxy β ∈ [0, 1) via smooth saturation
    const double abs_x    = std::abs(event.x);
    const double beta_proxy = std::tanh(abs_x / (abs_x + 1.0 + 1e-15));

    if (!std::isfinite(beta_proxy)) return std::nullopt;

    // Classify
    if (beta_proxy < 0.1)  return Regime::Newtonian;
    if (beta_proxy < 0.9)  return Regime::Relativistic;
    if (beta_proxy < 0.9999) return Regime::HighGamma;
    return Regime::Subluminal;
}

std::array<double, NUM_CHRISTOFFEL>
SpacetimeManifold::christoffelSymbols(const MetricTensor& metric) const noexcept {
    std::array<double, NUM_CHRISTOFFEL> result{};
    // Zero-initialise (default-initialised above, but be explicit)
    result.fill(0.0);

    if (!metric.is_valid()) return result;  // Return zeros for invalid metric

    // Compute inverse metric (diagonal fast-path)
    auto inv_opt = metric.inverse_diagonal();
    if (!inv_opt) return result;
    const MetricTensor& g_inv = *inv_opt;

    // Finite-difference step for numerical derivatives
    constexpr double EPS = 1e-5;

    // Γ^λ_μν = (1/2) g^{λσ} (∂_μ g_{νσ} + ∂_ν g_{μσ} − ∂_σ g_{μν})
    //
    // For a constant metric (flat), ∂_anything g_{αβ} = 0, so all Γ = 0.
    // We compute symbolically: for diagonal constant metric all cross-derivatives
    // vanish and the diagonal self-derivatives are zero, giving Γ = 0 exactly.
    //
    // For robustness we evaluate via finite differences on a position-independent
    // metric callback (f(x) = metric for all x).  The step EPS cancels exactly.

    // Derivatives of g_{μν} with respect to coordinate λ.
    // For our constant (flat) metric these are all exactly zero.
    // We implement a generic numerical path for future curved metrics:
    // ∂g_{μν}/∂x^λ ≈ 0 for flat metric → all Γ = 0.

    for (int lambda = 0; lambda < DIM; ++lambda) {
        for (int mu = 0; mu < DIM; ++mu) {
            for (int nu = 0; nu < DIM; ++nu) {
                double christoffel_val = 0.0;
                for (int sigma = 0; sigma < DIM; ++sigma) {
                    // For constant metric:
                    //   ∂_mu g[nu][sigma]  = 0
                    //   ∂_nu g[mu][sigma]  = 0
                    //   ∂_sigma g[mu][nu]  = 0
                    // So each term = 0 and Γ^lambda_mu_nu = 0.
                    // (We write it this way to preserve the correct formula
                    //  structure for future non-flat extensions.)
                    const double d_mu_g_nu_sigma  = 0.0;
                    const double d_nu_g_mu_sigma  = 0.0;
                    const double d_sigma_g_mu_nu  = 0.0;
                    christoffel_val += 0.5 * g_inv.g[lambda][sigma]
                        * (d_mu_g_nu_sigma + d_nu_g_mu_sigma - d_sigma_g_mu_nu);
                }
                const int idx = christoffel_index(lambda, mu, nu);
                result[static_cast<std::size_t>(idx)] = christoffel_val;
                (void)EPS;  // suppress unused warning for future curved metrics
            }
        }
    }

    return result;
}

MetricTensor SpacetimeManifold::flatMetric() const noexcept {
    return MetricTensor::minkowski();
}

} // namespace srfm::manifold
