/**
 * @file  geodesic_n.cpp
 * @brief Implementation of GeodesicSolverN.
 *
 * See include/srfm/tensor/geodesic_n.hpp for the public API contract.
 *
 * ## RK4 scheme
 * State s = (x, u). Derivative f(s) = (u, acc) where
 *
 *   acc^λ = -Σ_{μν} Γ^λ_μν u^μ u^ν
 *
 * Standard RK4:
 *   k1 = f(s)
 *   k2 = f(s + h/2 · k1)
 *   k3 = f(s + h/2 · k2)
 *   k4 = f(s + h   · k3)
 *   s_new = s + (h/6)(k1 + 2k2 + 2k3 + k4)
 */

#include "../../include/srfm/tensor/geodesic_n.hpp"

#include <cmath>

namespace srfm::tensor {

// ── Constructor ───────────────────────────────────────────────────────────────

GeodesicSolverN::GeodesicSolverN(const NAssetManifold& manifold,
                                  const ChristoffelN&   christoffel) noexcept
    : manifold_(manifold)
    , christoffel_(christoffel)
{}

// ── Internal helpers ──────────────────────────────────────────────────────────

std::optional<std::pair<Eigen::VectorXd, Eigen::VectorXd>>
GeodesicSolverN::rhs(const GeodesicState& s) const noexcept {
    const int D = manifold_.dim();

    // dx/dτ = u.
    Eigen::VectorXd dx = s.u;

    // du^λ/dτ = -Σ_{μν} Γ^λ_μν u^μ u^ν.
    Eigen::VectorXd du = Eigen::VectorXd::Zero(D);

    for (int lambda = 0; lambda < D; ++lambda) {
        double acc = 0.0;
        for (int mu = 0; mu < D; ++mu) {
            for (int nu = 0; nu < D; ++nu) {
                auto gamma_opt = christoffel_.symbol(lambda, mu, nu, s.x);
                if (!gamma_opt) { return std::nullopt; }
                acc += (*gamma_opt) * s.u(mu) * s.u(nu);
            }
        }
        du(lambda) = -acc;
    }

    return std::make_pair(std::move(dx), std::move(du));
}

GeodesicState
GeodesicSolverN::advance(const GeodesicState&   s,
                          const Eigen::VectorXd& dx,
                          const Eigen::VectorXd& du,
                          double                 scale) noexcept {
    GeodesicState result;
    result.x = s.x + scale * dx;
    result.u = s.u + scale * du;
    return result;
}

// ── Public integration ────────────────────────────────────────────────────────

std::optional<GeodesicState>
GeodesicSolverN::step(const GeodesicState& s, double dtau) const noexcept {
    // k1.
    auto f1 = rhs(s);
    if (!f1) { return std::nullopt; }
    const auto& [dx1, du1] = *f1;

    // k2.
    GeodesicState s2 = advance(s, dx1, du1, dtau * 0.5);
    auto f2 = rhs(s2);
    if (!f2) { return std::nullopt; }
    const auto& [dx2, du2] = *f2;

    // k3.
    GeodesicState s3 = advance(s, dx2, du2, dtau * 0.5);
    auto f3 = rhs(s3);
    if (!f3) { return std::nullopt; }
    const auto& [dx3, du3] = *f3;

    // k4.
    GeodesicState s4 = advance(s, dx3, du3, dtau);
    auto f4 = rhs(s4);
    if (!f4) { return std::nullopt; }
    const auto& [dx4, du4] = *f4;

    // Combine: s_new = s + (h/6)(k1 + 2k2 + 2k3 + k4).
    GeodesicState result;
    result.x = s.x + (dtau / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4);
    result.u = s.u + (dtau / 6.0) * (du1 + 2.0 * du2 + 2.0 * du3 + du4);

    return result;
}

std::optional<std::vector<GeodesicState>>
GeodesicSolverN::integrate(GeodesicState initial,
                            double        dtau,
                            int           n_steps) const noexcept {
    if (n_steps <= 0) { return std::vector<GeodesicState>{}; }

    std::vector<GeodesicState> traj;
    traj.reserve(static_cast<std::size_t>(n_steps));

    GeodesicState current = std::move(initial);
    for (int i = 0; i < n_steps; ++i) {
        auto next = step(current, dtau);
        if (!next) { return std::nullopt; }
        current = std::move(*next);
        traj.push_back(current);
    }

    return traj;
}

std::optional<double>
GeodesicSolverN::geodesic_deviation(
        const std::vector<GeodesicState>& traj) const noexcept {
    if (traj.size() < 2) { return std::nullopt; }

    const Eigen::VectorXd& x0 = traj.front().x;
    const Eigen::VectorXd& x1 = traj.back().x;

    const int N = static_cast<int>(traj.size());
    double max_dev = 0.0;

    for (int i = 1; i < N - 1; ++i) {
        // Parameter t ∈ (0,1) for this point along the trajectory.
        double t = static_cast<double>(i) / static_cast<double>(N - 1);

        // Point on the straight line from x0 to x1.
        Eigen::VectorXd x_line = x0 + t * (x1 - x0);

        // Distance in full coordinate space.
        double dev = (traj[static_cast<std::size_t>(i)].x - x_line).norm();
        if (dev > max_dev) {
            max_dev = dev;
        }
    }

    return max_dev;
}

} // namespace srfm::tensor
