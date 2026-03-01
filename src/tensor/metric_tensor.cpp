/// @file src/tensor/metric_tensor.cpp
/// @brief Implementation of MetricTensor — AGT-04

#include "srfm/tensor.hpp"
#include "srfm/constants.hpp"

#include <Eigen/Eigenvalues>

namespace srfm::tensor {

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
        return std::nullopt;
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
