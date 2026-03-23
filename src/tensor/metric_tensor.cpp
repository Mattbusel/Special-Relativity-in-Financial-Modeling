/// @file src/tensor/metric_tensor.cpp
/// @brief Implementation of MetricTensor — AGT-04

#include "srfm/tensor.hpp"
#include "srfm/constants.hpp"

#include <Eigen/Eigenvalues>
#include <cstdio>   // std::fprintf for regularization warning

namespace srfm::tensor {

// ─── Tikhonov regularization constant ────────────────────────────────────────
// When the metric matrix is near-singular, we add λI to stabilise the
// LU decomposition before inverting.  The value λ = 1e-10 is small enough
// to preserve the geometry to full double precision for well-conditioned
// metrics while rescuing degenerate (zero-volatility) configurations.
static constexpr double TIKHONOV_LAMBDA = 1e-10;

// ─── Construction ─────────────────────────────────────────────────────────────

MetricTensor::MetricTensor(MetricFunction metric_fn)
    : metric_fn_(std::move(metric_fn)) {}

// ─── Core Operations ──────────────────────────────────────────────────────────

MetricMatrix MetricTensor::evaluate(const SpacetimePoint& x) const {
    return metric_fn_(x);
}

std::optional<MetricMatrix> MetricTensor::inverse(const SpacetimePoint& x) const {
    MetricMatrix g = evaluate(x);

    // Use full-pivoting LU decomposition for numerical robustness.
    // Full-pivoting is slower than partial but detects near-singularity reliably.
    Eigen::FullPivLU<MetricMatrix> lu(g);

    if (!lu.isInvertible()) {
        // FIX (Task 4): Replace hard nullopt on singular metric with Tikhonov
        // regularization.  Adding λI to a singular matrix shifts all eigenvalues
        // by λ, making the matrix invertible while introducing only O(λ) error
        // in the geometry.  This prevents geodesic integrations from silently
        // returning zero Christoffel symbols when the covariance matrix has a
        // zero-volatility direction (e.g. a perfectly correlated asset pair).
        //
        // λ = TIKHONOV_LAMBDA = 1e-10 is chosen to be:
        //   - Large enough to regularise numerically zero eigenvalues
        //   - Small enough to be negligible for well-conditioned markets (σ ~ 0.01)
        std::fprintf(stderr,
            "[srfm::tensor::MetricTensor::inverse] WARNING: singular metric "
            "detected at x = (%.4g, %.4g, %.4g, %.4g); applying Tikhonov "
            "regularization λ = %.2e\n",
            x(0), x(1), x(2), x(3), TIKHONOV_LAMBDA);

        MetricMatrix g_reg = g;
        g_reg += TIKHONOV_LAMBDA * MetricMatrix::Identity();

        Eigen::FullPivLU<MetricMatrix> lu_reg(g_reg);
        if (!lu_reg.isInvertible()) {
            // Even after regularization the matrix is degenerate — return nullopt.
            return std::nullopt;
        }
        return lu_reg.inverse();
    }

    return lu.inverse();
}

bool MetricTensor::is_lorentzian(const SpacetimePoint& x) const {
    MetricMatrix g = evaluate(x);

    // Count eigenvalue signs.  SelfAdjointEigenSolver assumes symmetric input,
    // which the metric always is.
    Eigen::SelfAdjointEigenSolver<MetricMatrix> solver(g,
        Eigen::EigenvaluesOnly);

    int neg = 0;
    int pos = 0;
    const auto& ev = solver.eigenvalues();

    for (int i = 0; i < SPACETIME_DIM; ++i) {
        if (ev(i) < -constants::METRIC_SINGULARITY_EPSILON) {
            ++neg;
        } else if (ev(i) > constants::METRIC_SINGULARITY_EPSILON) {
            ++pos;
        }
    }

    // Lorentzian signature: exactly one negative eigenvalue, three positive.
    return (neg == 1) && (pos == 3);
}

double MetricTensor::spacetime_interval(const SpacetimePoint& x,
                                         const FourVelocity&   dx) const {
    MetricMatrix g = evaluate(x);
    // ds² = g_μν dx^μ dx^ν  (bilinear form)
    return dx.dot(g * dx);
}

// ─── Factories ────────────────────────────────────────────────────────────────

MetricTensor MetricTensor::make_minkowski(double time_scale,
                                           double spatial_scale) {
    return MetricTensor([time_scale, spatial_scale](const SpacetimePoint& /*x*/) {
        MetricMatrix g = MetricMatrix::Zero();
        g(0, 0) = -(time_scale * time_scale);
        g(1, 1) =  (spatial_scale * spatial_scale);
        g(2, 2) =  (spatial_scale * spatial_scale);
        g(3, 3) =  (spatial_scale * spatial_scale);
        return g;
    });
}

MetricTensor MetricTensor::make_diagonal(double time_scale,
                                          const std::array<double, 3>& vol) {
    return MetricTensor([time_scale, vol](const SpacetimePoint& /*x*/) {
        MetricMatrix g = MetricMatrix::Zero();
        g(0, 0) = -(time_scale * time_scale);
        for (int i = 0; i < 3; ++i) {
            g(i + 1, i + 1) = vol[i] * vol[i];
        }
        return g;
    });
}

MetricTensor MetricTensor::make_from_covariance(double time_scale,
                                                  const Eigen::Matrix3d& cov) {
    return MetricTensor([time_scale, cov](const SpacetimePoint& /*x*/) {
        MetricMatrix g = MetricMatrix::Zero();
        g(0, 0) = -(time_scale * time_scale);
        g.block<3, 3>(1, 1) = cov;
        return g;
    });
}

} // namespace srfm::tensor
