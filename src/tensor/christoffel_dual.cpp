/// @file src/tensor/christoffel_dual.cpp
/// @brief Christoffel symbols via dual-number automatic differentiation.
///
/// Replaces the O(h²) central finite-difference approximation in
/// christoffel.cpp with exact forward-mode autodiff using DualNumber.
///
/// ## Method
/// For each coordinate direction σ ∈ {0,1,2,3}:
///   1. Seed the input point: xd[k] = {x(k), k==σ ? 1.0 : 0.0}
///   2. Evaluate the dual-number metric function at xd.
///   3. The .deriv component of each matrix entry is ∂g_μν/∂x^σ — exactly,
///      with no finite-difference truncation error.
///
/// The assembled Christoffel symbols then use these exact derivatives:
///   Γ^λ_μν = ½ g^λσ (∂_μ g_{νσ} + ∂_ν g_{μσ} − ∂_σ g_{μν})
///
/// ## Comparison with ChristoffelSymbols (finite differences)
/// - Central FD:   8 metric evaluations, error O(h²), step-size sensitive
/// - Dual-number:  4 metric evaluations, error 0 (machine epsilon only)

#include "srfm/tensor.hpp"
#include "srfm/constants.hpp"

namespace srfm::tensor {

// ─── Construction ─────────────────────────────────────────────────────────────

ChristoffelSymbolsDual::ChristoffelSymbolsDual(const MetricTensor& metric,
                                                DualMetricFunction  dual_fn)
    : metric_(metric)
    , dual_fn_(std::move(dual_fn)) {}

// ─── Private: exact derivative via dual evaluation ────────────────────────────

MetricMatrix ChristoffelSymbolsDual::dual_metric_derivative(
    const SpacetimePoint& x, int sigma) const
{
    // Build the dual-number point:
    //   xd[k] = x(k) + (k==sigma ? 1 : 0)·ε
    // So the ε-component of dual_fn(xd)(μ,ν) equals ∂g_μν/∂x^sigma exactly.
    DualSpacetimePoint xd;
    for (int k = 0; k < SPACETIME_DIM; ++k) {
        xd(k) = DualNumber{x(k), k == sigma ? 1.0 : 0.0};
    }

    // Evaluate the dual metric.
    DualMetricMatrix gd = dual_fn_(xd);

    // Extract the derivative component into a plain MetricMatrix.
    MetricMatrix dg;
    for (int mu = 0; mu < SPACETIME_DIM; ++mu) {
        for (int nu = 0; nu < SPACETIME_DIM; ++nu) {
            dg(mu, nu) = gd(mu, nu).deriv;
        }
    }
    return dg;
}

// ─── compute ──────────────────────────────────────────────────────────────────

ChristoffelArray ChristoffelSymbolsDual::compute(const SpacetimePoint& x) const {
    // Initialise all symbols to zero.
    ChristoffelArray result;
    for (int l = 0; l < SPACETIME_DIM; ++l) {
        result[l] = MetricMatrix::Zero();
    }

    // Obtain the inverse metric g^λσ via the base MetricTensor (which now
    // applies Tikhonov regularization on near-singular metrics — Task 4).
    std::optional<MetricMatrix> g_inv_opt = metric_.inverse(x);
    if (!g_inv_opt) {
        // Still singular even after regularization — return zero Christoffels.
        return result;
    }
    const MetricMatrix& g_inv = *g_inv_opt;

    // Compute exact metric partial derivatives via dual-number autodiff.
    // dg[sigma](mu, nu) = ∂g_μν/∂x^σ  (no truncation error)
    std::array<MetricMatrix, SPACETIME_DIM> dg;
    for (int s = 0; s < SPACETIME_DIM; ++s) {
        dg[s] = dual_metric_derivative(x, s);
    }

    // Γ^λ_μν = ½ Σ_σ g^{λσ} (∂_μ g_{νσ} + ∂_ν g_{μσ} − ∂_σ g_{μν})
    for (int lambda = 0; lambda < SPACETIME_DIM; ++lambda) {
        for (int mu = 0; mu < SPACETIME_DIM; ++mu) {
            for (int nu = 0; nu < SPACETIME_DIM; ++nu) {
                double sum = 0.0;
                for (int sigma = 0; sigma < SPACETIME_DIM; ++sigma) {
                    const double bracket =
                        dg[mu](nu, sigma)   // ∂_μ g_{νσ}
                      + dg[nu](mu, sigma)   // ∂_ν g_{μσ}
                      - dg[sigma](mu, nu);  // ∂_σ g_{μν}
                    sum += g_inv(lambda, sigma) * bracket;
                }
                result[lambda](mu, nu) = 0.5 * sum;
            }
        }
    }

    return result;
}

// ─── contract ─────────────────────────────────────────────────────────────────

FourVelocity ChristoffelSymbolsDual::contract(const ChristoffelArray& gamma,
                                               const FourVelocity&    u) const {
    // Identical to ChristoffelSymbols::contract:
    //   result^λ = Σ_{μ,ν} Γ^λ_{μν} u^μ u^ν
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
