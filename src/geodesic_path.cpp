/**
 * @file  src/geodesic_path.cpp
 * @brief Geodesic Portfolio Path implementation — Round 5.
 *
 * See include/srfm/geodesic_path.hpp for the full module contract.
 *
 * The geodesic equation d^2w_i/dt^2 = -2*lambda*w_i is a simple harmonic
 * oscillator with omega = sqrt(2*lambda).  The analytical solution is:
 *
 *   w_i(t) = A_i * cos(omega * t) + B_i * sin(omega * t)
 *
 * Boundary conditions:
 *   w_i(0) = start.weights[i]   →  A_i = start.weights[i]
 *   w_i(1) = end.weights[i]     →  B_i = (end.weights[i] - A_i*cos(omega))
 *                                        / sin(omega)
 *
 * Degenerate cases:
 *   - lambda == 0 (omega == 0): straight-line interpolation.
 *   - sin(omega) ≈ 0 (omega near pi, 2pi, ...): linear interpolation fallback.
 */

#include "srfm/geodesic_path.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace srfm::portfolio {

// ─── PortfolioState ───────────────────────────────────────────────────────────

bool PortfolioState::is_valid() const noexcept {
    if (weights.empty()) return false;
    for (double w : weights) {
        if (!std::isfinite(w)) return false;
    }
    return true;
}

double PortfolioState::sum_weights() const noexcept {
    double s = 0.0;
    for (double w : weights) s += w;
    return s;
}

// ─── GeodesicSolver ───────────────────────────────────────────────────────────

// static
double GeodesicSolver::trajectory(
    double w_start,
    double w_end,
    double t_norm,
    double omega
) noexcept {
    // t_norm ∈ [0, 1]
    if (omega < 1e-12) {
        // Flat spacetime (lambda ≈ 0): straight-line interpolation
        return w_start + (w_end - w_start) * t_norm;
    }

    const double sin_omega = std::sin(omega);
    if (std::fabs(sin_omega) < 1e-12) {
        // sin(omega) ≈ 0: oscillator period divides the interval exactly.
        // Fall back to linear interpolation to avoid division by zero.
        return w_start + (w_end - w_start) * t_norm;
    }

    // Harmonic oscillator coefficients from boundary conditions
    const double A = w_start;
    const double B = (w_end - A * std::cos(omega)) / sin_omega;

    return A * std::cos(omega * t_norm) + B * std::sin(omega * t_norm);
}

// static
Geodesic GeodesicSolver::solve(
    const PortfolioState& start,
    const PortfolioState& end,
    int n_steps,
    double lambda
) {
    if (n_steps < 1) {
        throw std::invalid_argument(
            "GeodesicSolver::solve — n_steps must be >= 1, got " +
            std::to_string(n_steps)
        );
    }
    if (start.dim() != end.dim()) {
        throw std::invalid_argument(
            "GeodesicSolver::solve — start and end must have the same dimension ("
            + std::to_string(start.dim()) + " vs " + std::to_string(end.dim()) + ")"
        );
    }
    if (start.weights.empty()) {
        throw std::invalid_argument(
            "GeodesicSolver::solve — portfolio dimension must be >= 1"
        );
    }
    if (lambda < 0.0) {
        throw std::invalid_argument(
            "GeodesicSolver::solve — lambda must be >= 0, got " +
            std::to_string(lambda)
        );
    }

    const int dim = start.dim();
    const double omega = std::sqrt(2.0 * lambda);

    // Total number of waypoints = n_steps + 1 (inclusive of both endpoints)
    const int n_points = n_steps + 1;

    // Linearly interpolate timestamps
    const int64_t t0 = start.timestamp_ms;
    const int64_t t1 = end.timestamp_ms;

    Geodesic result;
    result.states.reserve(static_cast<std::size_t>(n_points));

    for (int step = 0; step < n_points; ++step) {
        const double t_norm = (n_steps == 0)
            ? 0.0
            : static_cast<double>(step) / static_cast<double>(n_steps);

        PortfolioState state;
        state.weights.resize(static_cast<std::size_t>(dim));
        state.timestamp_ms = t0 + static_cast<int64_t>(
            std::round(static_cast<double>(t1 - t0) * t_norm)
        );

        for (int i = 0; i < dim; ++i) {
            state.weights[static_cast<std::size_t>(i)] = trajectory(
                start.weights[static_cast<std::size_t>(i)],
                end.weights[static_cast<std::size_t>(i)],
                t_norm,
                omega
            );
        }

        result.states.push_back(std::move(state));
    }

    return result;
}

// ─── GeodesicLength ───────────────────────────────────────────────────────────

// static
double GeodesicLength::distance(
    const PortfolioState& a,
    const PortfolioState& b
) noexcept {
    if (a.weights.size() != b.weights.size()) return 0.0;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < a.weights.size(); ++i) {
        const double diff = a.weights[i] - b.weights[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

// static
double GeodesicLength::compute(const Geodesic& geodesic, double dt) noexcept {
    if (geodesic.states.size() < 2) return 0.0;
    const double effective_dt = (dt <= 0.0) ? 1.0 : dt;

    double length = 0.0;
    for (std::size_t i = 1; i < geodesic.states.size(); ++i) {
        // Arc-length contribution: ||dw/dt|| * dt
        // Approximate ||dw|| as Euclidean distance between consecutive states,
        // then divide by effective_dt to get velocity magnitude, then multiply.
        // Net effect: sum of Euclidean distances (independent of dt scaling).
        const double seg = distance(geodesic.states[i - 1], geodesic.states[i]);
        length += seg;
    }
    // Scale by effective_dt to convert from displacement to arc length
    // (velocity = displacement / dt → arc length = velocity * dt = displacement)
    // So arc length equals the sum of displacements directly.
    (void)effective_dt;  // dt cancels; length = sum of segment distances
    return length;
}

}  // namespace srfm::portfolio
