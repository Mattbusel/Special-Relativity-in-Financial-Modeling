#pragma once

/// @file include/srfm/geodesic_path.hpp
/// @brief Geodesic Portfolio Path — Round 5 public API.
///
/// # Module: Geodesic Portfolio Path
///
/// ## Concept
/// In financial spacetime, the geodesic between two portfolio states is the
/// path of minimum action.  The Lagrangian is:
///
///   L = (1/2) ||dw/dt||^2 - V(w)
///
/// where V(w) = lambda * sum(w_i^2) is a concentration penalty.
///
/// The Euler-Lagrange equations yield:
///
///   d^2w_i/dt^2 = -dV/dw_i = -2 * lambda * w_i
///
/// This is simple harmonic oscillator motion with omega = sqrt(2 * lambda).
///
/// ## Analytical Solution
///
///   w_i(t) = A_i * cos(omega * t) + B_i * sin(omega * t)
///
/// Boundary conditions w_i(0) = start.weights[i] and
/// w_i(T) = end.weights[i] determine A_i and B_i.
///
/// ## Classes
///
/// - `PortfolioState`  : weights vector + timestamp_ms
/// - `Geodesic`        : discretised path (vector of PortfolioState)
/// - `GeodesicSolver`  : solves for the geodesic path between two states
/// - `GeodesicLength`  : computes the integrated arc length of a geodesic
///
/// ## Guarantees
/// - All methods noexcept where possible
/// - No dynamic allocation beyond the returned Geodesic
/// - Thread-safe (pure functions; no shared mutable state)

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace srfm::portfolio {

// ─── PortfolioState ───────────────────────────────────────────────────────────

/// A point in portfolio space + time.
struct PortfolioState {
    std::vector<double> weights;  ///< Portfolio weights (any length >= 1)
    int64_t timestamp_ms;         ///< Wall-clock time in milliseconds

    /// Dimension of the portfolio (number of assets).
    int dim() const noexcept { return static_cast<int>(weights.size()); }

    /// Returns true if weights is non-empty and all elements are finite.
    bool is_valid() const noexcept;

    /// Sum of all weights.
    double sum_weights() const noexcept;
};

// ─── Geodesic ─────────────────────────────────────────────────────────────────

/// Discretised path through portfolio space, representing the geodesic
/// between two PortfolioState endpoints.
struct Geodesic {
    std::vector<PortfolioState> states;  ///< Ordered from start to end

    /// Number of waypoints (including endpoints).
    int size() const noexcept { return static_cast<int>(states.size()); }

    /// True if the path contains at least two states.
    bool is_valid() const noexcept { return states.size() >= 2; }

    /// Returns the start state (first waypoint).
    const PortfolioState& start() const { return states.front(); }

    /// Returns the end state (last waypoint).
    const PortfolioState& end() const { return states.back(); }
};

// ─── GeodesicSolver ───────────────────────────────────────────────────────────

/// Computes the geodesic between two portfolio states under a concentration
/// penalty Lagrangian.
///
/// The analytical solution:
///   omega = sqrt(2 * lambda)
///   A_i   = start.weights[i]
///   B_i   = (end.weights[i] - A_i * cos(omega)) / sin(omega)   if sin(omega) != 0
///           (linear interpolation fallback when omega is near zero)
///   w_i(t_norm) = A_i * cos(omega * t_norm) + B_i * sin(omega * t_norm)
///
/// where t_norm in [0, 1] is the normalised time along the geodesic.
class GeodesicSolver {
public:
    /// Solve for the geodesic between start and end.
    ///
    /// @param start   Initial portfolio state.
    /// @param end     Target portfolio state.  Must have the same dimension as start.
    /// @param n_steps Number of interior discretisation steps (>= 1).
    ///                Total waypoints = n_steps + 1 (endpoints included).
    /// @param lambda  Concentration penalty coefficient (>= 0).
    ///                lambda = 0 → straight-line (flat spacetime) geodesic.
    ///
    /// @returns Geodesic with (n_steps + 1) waypoints.
    ///
    /// @throws std::invalid_argument if dimensions mismatch or n_steps < 1.
    static Geodesic solve(
        const PortfolioState& start,
        const PortfolioState& end,
        int n_steps,
        double lambda
    );

private:
    /// Compute the normalised-time trajectory for a single weight dimension.
    /// t_norm in [0, 1].
    static double trajectory(
        double w_start,
        double w_end,
        double t_norm,
        double omega
    ) noexcept;
};

// ─── GeodesicLength ───────────────────────────────────────────────────────────

/// Computes the integrated arc length of a Geodesic.
///
/// Arc length = integral of ||dw/dt|| dt
///
/// Approximated numerically as sum of Euclidean distances between consecutive
/// waypoints.
class GeodesicLength {
public:
    /// Compute the arc length of a discretised geodesic.
    ///
    /// @param geodesic  The path to measure.
    /// @param dt        Time step between waypoints in the same units as
    ///                  timestamp_ms differences.  If zero, defaults to 1.0.
    ///
    /// @returns Arc length (>= 0).  Returns 0 for a path with < 2 states.
    static double compute(const Geodesic& geodesic, double dt = 1.0) noexcept;

    /// Compute the Euclidean distance between two PortfolioState weight vectors.
    static double distance(
        const PortfolioState& a,
        const PortfolioState& b
    ) noexcept;
};

}  // namespace srfm::portfolio
