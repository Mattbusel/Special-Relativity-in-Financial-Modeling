/**
 * @file  geodesic_solver.cpp
 * @brief RK4 geodesic integrator implementation (AGT-13 / SRFM).
 *
 * See geodesic_solver.hpp for the full module contract.
 */

#include "geodesic_solver.hpp"

#include <algorithm>
#include <cmath>

namespace srfm::geodesic {

// ── GeodesicState ─────────────────────────────────────────────────────────────

bool GeodesicState::is_finite() const noexcept {
    for (int i = 0; i < DIM; ++i) {
        if (!std::isfinite(x[i])) return false;
        if (!std::isfinite(u[i])) return false;
    }
    return true;
}

// ── Internal RK4 helpers ──────────────────────────────────────────────────────

namespace {

/// Compute Christoffel acceleration: a^λ = −Γ^λ_μν u^μ u^ν
std::array<double, DIM>
geodesic_acceleration(const std::array<double, NUM_CHRISTOFFEL>& gamma,
                      const std::array<double, DIM>&             u) noexcept {
    std::array<double, DIM> accel{};
    accel.fill(0.0);
    for (int lambda = 0; lambda < DIM; ++lambda) {
        double sum = 0.0;
        for (int mu = 0; mu < DIM; ++mu) {
            for (int nu = 0; nu < DIM; ++nu) {
                const int idx = christoffel_index(lambda, mu, nu);
                sum += gamma[static_cast<std::size_t>(idx)] * u[mu] * u[nu];
            }
        }
        accel[static_cast<std::size_t>(lambda)] = -sum;
    }
    return accel;
}

/// Single RK4 step.  Returns nullopt if any component becomes non-finite.
std::optional<GeodesicState>
rk4_step(const GeodesicState&                    s,
         const std::array<double, NUM_CHRISTOFFEL>& christoffel,
         double                                  dt) noexcept {
    // k1
    const auto a1 = geodesic_acceleration(christoffel, s.u);
    GeodesicState s2{};
    for (int i = 0; i < DIM; ++i) {
        s2.x[static_cast<std::size_t>(i)] = s.x[static_cast<std::size_t>(i)] + 0.5 * dt * s.u[static_cast<std::size_t>(i)];
        s2.u[static_cast<std::size_t>(i)] = s.u[static_cast<std::size_t>(i)] + 0.5 * dt * a1[static_cast<std::size_t>(i)];
    }

    // k2
    const auto a2 = geodesic_acceleration(christoffel, s2.u);
    GeodesicState s3{};
    for (int i = 0; i < DIM; ++i) {
        s3.x[static_cast<std::size_t>(i)] = s.x[static_cast<std::size_t>(i)] + 0.5 * dt * s2.u[static_cast<std::size_t>(i)];
        s3.u[static_cast<std::size_t>(i)] = s.u[static_cast<std::size_t>(i)] + 0.5 * dt * a2[static_cast<std::size_t>(i)];
    }

    // k3
    const auto a3 = geodesic_acceleration(christoffel, s3.u);
    GeodesicState s4{};
    for (int i = 0; i < DIM; ++i) {
        s4.x[static_cast<std::size_t>(i)] = s.x[static_cast<std::size_t>(i)] + dt * s3.u[static_cast<std::size_t>(i)];
        s4.u[static_cast<std::size_t>(i)] = s.u[static_cast<std::size_t>(i)] + dt * a3[static_cast<std::size_t>(i)];
    }

    // k4
    const auto a4 = geodesic_acceleration(christoffel, s4.u);

    // Combine
    GeodesicState out{};
    for (std::size_t i = 0; i < static_cast<std::size_t>(DIM); ++i) {
        out.x[i] = s.x[i] + (dt / 6.0) * (s.u[i] + 2.0 * s2.u[i] + 2.0 * s3.u[i] + s4.u[i]);
        out.u[i] = s.u[i] + (dt / 6.0) * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]);

        if (!std::isfinite(out.x[i]) || !std::isfinite(out.u[i])) {
            return std::nullopt;
        }
    }
    return out;
}

} // namespace

// ── GeodesicSolver::solve ─────────────────────────────────────────────────────

std::optional<GeodesicState>
GeodesicSolver::solve(const GeodesicState& initial,
                      const MetricTensor&  metric,
                      int                  steps,
                      double               dt) const noexcept {
    // Validate initial state
    if (!initial.is_finite()) return std::nullopt;

    // Clamp integration parameters to safe ranges
    const int    clamped_steps = std::clamp(steps, 1, 100'000);
    const double clamped_dt    = std::clamp(dt, 1e-8, 1.0);

    // Validate metric
    if (!metric.is_valid()) return std::nullopt;

    // Compute Christoffel symbols (constant for flat metric)
    manifold::SpacetimeManifold mfld;
    const auto christoffel = mfld.christoffelSymbols(metric);

    GeodesicState state = initial;
    for (int step = 0; step < clamped_steps; ++step) {
        auto next = rk4_step(state, christoffel, clamped_dt);
        if (!next) return std::nullopt;
        state = *next;
    }

    return state;
}

} // namespace srfm::geodesic
