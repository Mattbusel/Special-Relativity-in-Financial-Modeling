/// @file src/tensor/christoffel.cpp
/// @brief Christoffel symbols of the second kind — AGT-04
///
/// Computes Γ^λ_μν = ½ g^λσ (∂_μ g_νσ + ∂_ν g_μσ − ∂_σ g_μν)
/// using central finite differences for the metric partial derivatives.

#include "srfm/tensor.hpp"
#include "srfm/constants.hpp"

namespace srfm::tensor {

// ─── Construction ─────────────────────────────────────────────────────────────

ChristoffelSymbols::ChristoffelSymbols(const MetricTensor& metric, double h)
    : metric_(metric)
    , h_(h) {}

// ─── Private: Central Finite Difference for ∂g_μν/∂x^sigma ─────────────────

MetricMatrix ChristoffelSymbols::metric_derivative(const SpacetimePoint& x,
                                                    int sigma) const {
    SpacetimePoint xp = x;
    SpacetimePoint xm = x;
    xp(sigma) += h_;
    xm(sigma) -= h_;

    // (g(x+hê_σ) − g(x−hê_σ)) / (2h) — O(h²) central difference
    return (metric_.evaluate(xp) - metric_.evaluate(xm)) / (2.0 * h_);
}

// ─── compute ──────────────────────────────────────────────────────────────────

ChristoffelArray ChristoffelSymbols::compute(const SpacetimePoint& x) const {
    // Initialise all symbols to zero.
    ChristoffelArray result;
    for (int l = 0; l < SPACETIME_DIM; ++l) {
        result[l] = MetricMatrix::Zero();
    }

    // We need the inverse metric g^λσ to raise the first index.
    std::optional<MetricMatrix> g_inv_opt = metric_.inverse(x);
    if (!g_inv_opt) {
        // Singular metric → return zero symbols (degenerate manifold point).
        return result;
    }
    const MetricMatrix& g_inv = *g_inv_opt;

    // Precompute ∂g_μν/∂x^σ for all four coordinate directions.
    // dg[sigma](mu, nu) = ∂g_μν/∂x^σ
    std::array<MetricMatrix, SPACETIME_DIM> dg;
    for (int s = 0; s < SPACETIME_DIM; ++s) {
        dg[s] = metric_derivative(x, s);
    }

    // Γ^λ_μν = ½ Σ_σ g^{λσ} (∂_μ g_{νσ} + ∂_ν g_{μσ} − ∂_σ g_{μν})
    for (int lambda = 0; lambda < SPACETIME_DIM; ++lambda) {
        for (int mu = 0; mu < SPACETIME_DIM; ++mu) {
            for (int nu = 0; nu < SPACETIME_DIM; ++nu) {
                double sum = 0.0;
                for (int sigma = 0; sigma < SPACETIME_DIM; ++sigma) {
                    // Three-term bracket (metric compatibility condition)
                    double bracket =
                        dg[mu](nu, sigma)    // ∂_μ g_{νσ}
                      + dg[nu](mu, sigma)    // ∂_ν g_{μσ}
                      - dg[sigma](mu, nu);   // ∂_σ g_{μν}

                    sum += g_inv(lambda, sigma) * bracket;
                }
                result[lambda](mu, nu) = 0.5 * sum;
            }
        }
    }

    return result;
}

// ─── contract ─────────────────────────────────────────────────────────────────

FourVelocity ChristoffelSymbols::contract(const ChristoffelArray& gamma,
                                           const FourVelocity&    u) const {
    // result^λ = Σ_{μ,ν} Γ^λ_{μν} u^μ u^ν
    FourVelocity result = FourVelocity::Zero();

    for (int lambda = 0; lambda < SPACETIME_DIM; ++lambda) {
        double sum = 0.0;
        for (int mu = 0; mu < SPACETIME_DIM; ++mu) {
            for (int nu = 0; nu < SPACETIME_DIM; ++nu) {
                sum += gamma[lambda](mu, nu) * u(mu) * u(nu);
            }
        }
        result(lambda) = sum;
    }

    return result;
}

} // namespace srfm::tensor
