/// @file src/tensor/geodesic.cpp
/// @brief Geodesic equation integrator using 4th-order Runge-Kutta — AGT-04
///
/// Integrates the geodesic ODE:
///   dx^λ/dτ = u^λ
///   du^λ/dτ = −Γ^λ_{μν} u^μ u^ν
///
/// Natural price paths in curved financial spacetime are geodesics — the
/// trajectories that extremise proper time given the covariance geometry.

#include "srfm/tensor.hpp"
#include "srfm/constants.hpp"

namespace srfm::tensor {

// ─── Construction ─────────────────────────────────────────────────────────────

GeodesicSolver::GeodesicSolver(const MetricTensor& metric,
                                double step_size,
                                double christoffel_h)
    : metric_(metric)
    , christoffel_(metric, christoffel_h)
    , step_size_(step_size) {}

// ─── Private: RK4 Step ────────────────────────────────────────────────────────

GeodesicState GeodesicSolver::rk4_step(const GeodesicState& state) const {
    // ODE derivative: given state (x, u), return (dx/dτ, du/dτ).
    auto derivative = [this](const GeodesicState& s) -> GeodesicState {
        const SpacetimePoint& x = s.position;
        const FourVelocity&   u = s.velocity;

        // du^λ/dτ = −Γ^λ_{μν} u^μ u^ν
        auto gamma = christoffel_.compute(x);
        FourVelocity accel = -1.0 * christoffel_.contract(gamma, u);

        // dx^λ/dτ = u^λ
        return GeodesicState{u, accel};
    };

    const double h = step_size_;

    GeodesicState k1 = derivative(state);
    GeodesicState k2 = derivative(state + (h / 2.0) * k1);
    GeodesicState k3 = derivative(state + (h / 2.0) * k2);
    GeodesicState k4 = derivative(state + h * k3);

    // Classic RK4 combination: weighted average of four slope estimates
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// ─── integrate ────────────────────────────────────────────────────────────────

std::vector<GeodesicState> GeodesicSolver::integrate(
    const SpacetimePoint& x0,
    const FourVelocity&   u0,
    int steps) const
{
    std::vector<GeodesicState> trajectory;
    trajectory.reserve(static_cast<size_t>(steps) + 1);

    GeodesicState current{x0, u0};
    trajectory.push_back(current);

    for (int i = 0; i < steps; ++i) {
        current = rk4_step(current);
        trajectory.push_back(current);
    }

    return trajectory;
}

// ─── norm_squared ─────────────────────────────────────────────────────────────

double GeodesicSolver::norm_squared(const SpacetimePoint& x,
                                     const FourVelocity&   u) const {
    return metric_.spacetime_interval(x, u);
}

} // namespace srfm::tensor
