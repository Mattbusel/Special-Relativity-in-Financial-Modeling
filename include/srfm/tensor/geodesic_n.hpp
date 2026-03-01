#pragma once
/**
 * @file  geodesic_n.hpp
 * @brief RK4 geodesic integrator for an (N+1)-dimensional Lorentzian manifold.
 *
 * Module:  include/srfm/tensor/
 * Stage:   4 — N-Asset Manifold
 *
 * ## Responsibility
 * Numerically integrate the geodesic equation
 *
 *   d²x^λ/dτ² = -Γ^λ_μν (dx^μ/dτ)(dx^ν/dτ)
 *
 * as the first-order system
 *
 *   dx^λ/dτ  = u^λ
 *   du^λ/dτ  = -Σ_{μν} Γ^λ_μν u^μ u^ν
 *
 * using a classic 4th-order Runge–Kutta scheme.
 *
 * ## Guarantees
 * - Thread-safe: GeodesicSolverN is stateless.
 * - All public methods are noexcept.
 * - All fallible operations return std::optional.
 *
 * ## NOT Responsible For
 * - Adaptive step-size control.
 * - Parallel transport of other tensors along the geodesic.
 */

#include "christoffel_n.hpp"
#include "n_asset_manifold.hpp"

#include <optional>
#include <vector>
#include <third_party/eigen/Eigen/Dense>

namespace srfm::tensor {

// ── State type ────────────────────────────────────────────────────────────────

/**
 * @brief Position and 4-velocity state on the manifold.
 *
 * Both vectors must have length equal to the manifold dimension (N+1).
 */
struct GeodesicState {
    Eigen::VectorXd x;  ///< Position coordinates.
    Eigen::VectorXd u;  ///< Tangent vector (4-velocity, dx/dτ).
};

// ── Solver ────────────────────────────────────────────────────────────────────

/**
 * @brief RK4 integrator for geodesics on an NAssetManifold.
 */
class GeodesicSolverN {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Construct from manifold and Christoffel symbol provider.
     *
     * Both objects must outlive this solver.
     *
     * @param manifold     The underlying NAssetManifold.
     * @param christoffel  Christoffel symbol computation object.
     */
    GeodesicSolverN(const NAssetManifold& manifold,
                    const ChristoffelN&   christoffel) noexcept;

    // ── Integration ───────────────────────────────────────────────────────────

    /**
     * @brief Advance the geodesic by one RK4 step of proper-time dtau.
     *
     * Computes k1..k4 for both position and velocity, then combines:
     *   s_new = s + (dtau/6)(k1 + 2k2 + 2k3 + k4)
     *
     * @param s     Current state (position + velocity).
     * @param dtau  Proper-time step size (may be negative for backward integration).
     * @return      Updated state, or std::nullopt if Christoffel evaluation fails.
     */
    [[nodiscard]] std::optional<GeodesicState>
    step(const GeodesicState& s, double dtau) const noexcept;

    /**
     * @brief Integrate the geodesic for n_steps steps of size dtau.
     *
     * Returns a vector of n_steps states (not including the initial state).
     *
     * @param initial  Starting state.
     * @param dtau     Proper-time step size.
     * @param n_steps  Number of steps to take.
     * @return         Vector of states, or std::nullopt if any step fails.
     */
    [[nodiscard]] std::optional<std::vector<GeodesicState>>
    integrate(GeodesicState initial,
              double dtau,
              int    n_steps) const noexcept;

    /**
     * @brief Measure the maximum deviation of a trajectory from a straight line.
     *
     * Computes the straight line from traj.front() to traj.back() and
     * returns the maximum Euclidean distance in the spatial (price) components
     * from any intermediate point to the corresponding point on that line.
     *
     * This provides a measure of how non-geodesic the trajectory is in flat space.
     *
     * @param traj  Trajectory of geodesic states.
     * @return      Max deviation, or std::nullopt if fewer than 2 states.
     */
    [[nodiscard]] std::optional<double>
    geodesic_deviation(const std::vector<GeodesicState>& traj) const noexcept;

    /**
     * @brief Return the manifold dimension.
     *
     * @return manifold_.dim().
     */
    [[nodiscard]] int dim() const noexcept;

private:
    // ── Internal helpers ──────────────────────────────────────────────────────

    /**
     * @brief Evaluate the geodesic equation RHS at state s.
     *
     * Returns (u, -Γ^λ_μν u^μ u^ν) as a pair (dx_dot, du_dot).
     *
     * @param s  Current state.
     * @return   Derivatives (dx/dτ, du/dτ), or std::nullopt on failure.
     */
    [[nodiscard]] std::optional<std::pair<Eigen::VectorXd, Eigen::VectorXd>>
    rhs(const GeodesicState& s) const noexcept;

    /**
     * @brief Advance a state by a scaled derivative vector.
     *
     * new_state.x = s.x + scale * dx
     * new_state.u = s.u + scale * du
     *
     * @param s      Base state.
     * @param dx     Position derivative.
     * @param du     Velocity derivative.
     * @param scale  Scaling factor.
     * @return       New state.
     */
    [[nodiscard]] static GeodesicState
    advance(const GeodesicState& s,
            const Eigen::VectorXd& dx,
            const Eigen::VectorXd& du,
            double scale) noexcept;

    // ── Data members ──────────────────────────────────────────────────────────

    const NAssetManifold& manifold_;      ///< Underlying manifold.
    const ChristoffelN&   christoffel_;   ///< Christoffel symbol provider.
};

// ── Inline trivial accessors ──────────────────────────────────────────────────

inline int GeodesicSolverN::dim() const noexcept {
    return manifold_.dim();
}

} // namespace srfm::tensor
