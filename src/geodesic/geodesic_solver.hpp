#pragma once
/**
 * @file  geodesic_solver.hpp
 * @brief RK4 geodesic integrator on a spacetime manifold (AGT-13 / SRFM)
 *
 * Module:  src/geodesic/
 * Owner:   AGT-13  (Adversarial hardening)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Numerically integrate the geodesic equation:
 *
 *   d²x^λ/dτ² + Γ^λ_μν (dx^μ/dτ)(dx^ν/dτ) = 0
 *
 * using a 4th-order Runge-Kutta scheme.
 *
 * Key property (tested by property suite):
 *   On the flat Minkowski metric (all Γ = 0) the solution is a straight line.
 *   Geodesic deviation after 100 RK4 steps is < 1e-8 from the linear prediction.
 *
 * Design Constraints
 * ------------------
 *   • Must always terminate (no unbounded loops; step count is bounded).
 *   • RK4 must not produce NaN even for extreme initial conditions.
 *   • All fallible operations return std::optional.
 *   • noexcept throughout.
 *
 * NOT Responsible For
 *   • Adaptive step-size control.
 *   • Parallelisation across multiple geodesics.
 *   • Non-metric coupling terms (e.g. electromagnetic).
 */

#include <array>
#include <cmath>
#include <optional>

#include "../manifold/spacetime_manifold.hpp"

namespace srfm::geodesic {

using manifold::MetricTensor;
using manifold::DIM;
using manifold::NUM_CHRISTOFFEL;
using manifold::christoffel_index;

// ── GeodesicState ─────────────────────────────────────────────────────────────

/**
 * @brief State of a particle on a geodesic: position x^μ and 4-velocity u^μ.
 *
 * The 8-dimensional phase space (x, u) is what RK4 advances each step.
 */
struct GeodesicState {
    std::array<double, DIM> x{};   ///< Position x^μ (μ = 0…3)
    std::array<double, DIM> u{};   ///< 4-velocity  u^μ = dx^μ/dτ

    /// True iff all position and velocity components are finite.
    [[nodiscard]] bool is_finite() const noexcept;
};

// ── GeodesicSolver ────────────────────────────────────────────────────────────

/**
 * @brief Integrates geodesic equations using the classical RK4 method.
 *
 * The Christoffel symbols are evaluated once per step from the supplied
 * metric (constant-metric approximation, valid for small step sizes dτ).
 *
 * @example
 * @code
 *   GeodesicSolver solver;
 *   GeodesicState init{{0,0,0,0}, {1,0,0,0}};   // at rest, proper-time flow
 *   auto metric = MetricTensor::minkowski();
 *   auto final_state = solver.solve(init, metric, 100, 0.01);
 *   // final_state->x[0] ≈ 1.0 (advanced 1 unit of proper time)
 * @endcode
 */
class GeodesicSolver {
public:
    GeodesicSolver() noexcept = default;

    /**
     * @brief Integrate the geodesic equation for `steps` RK4 steps.
     *
     * At each step, Christoffel symbols are computed from `metric` at the
     * current position.  For a constant (flat) metric this is equivalent to
     * evaluating at the origin: Γ = 0, giving straight-line trajectories.
     *
     * Safety guarantees:
     *   • Returns std::nullopt if initial state is non-finite.
     *   • Returns std::nullopt if any intermediate state becomes non-finite.
     *   • Never loops more than `steps` times.
     *   • `steps` is clamped to [1, 100'000] to prevent runaway.
     *   • `dt` (dτ) is clamped to [1e-8, 1.0] to prevent instability.
     *
     * @param initial  Initial (x, u) state.
     * @param metric   Background metric tensor (treated as constant in space).
     * @param steps    Number of RK4 integration steps.
     * @param dt       Proper-time step size dτ.
     * @return Final GeodesicState after integration, or std::nullopt on error.
     */
    [[nodiscard]] std::optional<GeodesicState>
    solve(const GeodesicState& initial,
          const MetricTensor&  metric,
          int                  steps,
          double               dt) const noexcept;
};

} // namespace srfm::geodesic
