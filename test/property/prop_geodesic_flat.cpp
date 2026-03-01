/**
 * @file  prop_geodesic_flat.cpp
 * @brief Property: ∀ flat metric, ∀ initial conditions: geodesic deviation < 1e-8
 *
 * Run with 10,000 random inputs:
 *   RC_PARAMS="max_success=10000" ./prop_geodesic_flat
 *
 * Mathematical basis:
 *   In flat Minkowski spacetime all Christoffel symbols are zero.
 *   The geodesic equation reduces to:
 *
 *     d²x^λ/dτ² = 0  →  x^λ(τ) = x₀^λ + u₀^λ · τ
 *
 *   The RK4 integrator should reproduce this straight-line solution exactly
 *   (up to floating-point rounding).  After N steps of dτ each:
 *
 *     x_expected^λ = x₀^λ + u₀^λ · (N · dτ)
 *
 *   Geodesic deviation = max_λ |x_actual^λ − x_expected^λ| < 1e-8.
 *
 * A deviation > 1e-8 would indicate:
 *   • An RK4 implementation bug (wrong k-coefficients or weights).
 *   • Non-zero Christoffel symbols being computed for the flat metric.
 *   • Numerical instability in the integration.
 */

#include <rapidcheck.h>
#include <array>
#include <cmath>
#include <algorithm>

#include "geodesic/geodesic_solver.hpp"
#include "manifold/spacetime_manifold.hpp"

using namespace srfm::geodesic;
using namespace srfm::manifold;

int main() {
    // ── Property 1: straight-line geodesic in flat spacetime ─────────────────
    rc::check(
        "geodesic_flat: RK4 deviation from straight line < 1e-8 after 100 steps",
        [](double ux, double uy, double uz) {
            // Random initial 4-velocity (time component fixed to 1 for normalisation)
            // Clamp spatial velocity to avoid extreme cases
            const double u1 = std::tanh(ux);
            const double u2 = std::tanh(uy);
            const double u3 = std::tanh(uz);

            GeodesicState init{};
            init.x = {0.0, 0.0, 0.0, 0.0};
            init.u = {1.0, u1, u2, u3};

            const MetricTensor flat = MetricTensor::minkowski();
            GeodesicSolver solver;

            constexpr int    STEPS = 100;
            constexpr double DT    = 0.001;

            auto result = solver.solve(init, flat, STEPS, DT);
            RC_ASSERT(result.has_value());
            RC_ASSERT(result->is_finite());

            // Expected straight-line position: x_expected^λ = u₀^λ * (STEPS * DT)
            const double total_tau = STEPS * DT;
            const std::array<double, DIM> x_expected = {
                init.u[0] * total_tau,
                init.u[1] * total_tau,
                init.u[2] * total_tau,
                init.u[3] * total_tau,
            };

            // Compute max deviation
            double max_dev = 0.0;
            for (std::size_t i = 0; i < static_cast<std::size_t>(DIM); ++i) {
                const double dev = std::abs(result->x[i] - x_expected[i]);
                max_dev = std::max(max_dev, dev);
            }

            RC_ASSERT(max_dev < 1e-8);
        }
    );

    // ── Property 2: 4-velocity is preserved in flat spacetime ────────────────
    rc::check(
        "geodesic_flat: 4-velocity unchanged after 100 flat geodesic steps",
        [](double ux, double uy, double uz) {
            const double u1 = std::tanh(ux);
            const double u2 = std::tanh(uy);
            const double u3 = std::tanh(uz);

            GeodesicState init{};
            init.x = {0.0, 0.0, 0.0, 0.0};
            init.u = {1.0, u1, u2, u3};

            const MetricTensor flat = MetricTensor::minkowski();
            GeodesicSolver solver;

            auto result = solver.solve(init, flat, 100, 0.001);
            RC_ASSERT(result.has_value());

            // In flat spacetime d(u^λ)/dτ = 0, so velocity is unchanged
            for (std::size_t i = 0; i < static_cast<std::size_t>(DIM); ++i) {
                const double du = std::abs(result->u[i] - init.u[i]);
                RC_ASSERT(du < 1e-12);
            }
        }
    );

    // ── Property 3: solver handles zero velocity (particle at rest) ───────────
    rc::check(
        "geodesic_flat: particle at rest advances only in time",
        []() {
            GeodesicState init{};
            init.x = {0.0, 0.0, 0.0, 0.0};
            init.u = {1.0, 0.0, 0.0, 0.0};  // at rest

            const MetricTensor flat = MetricTensor::minkowski();
            GeodesicSolver solver;

            auto result = solver.solve(init, flat, 100, 0.01);
            RC_ASSERT(result.has_value());

            // Only x[0] (time) should advance; spatial coordinates stay 0
            const double t_expected = 100 * 0.01;  // 1.0
            RC_ASSERT(std::abs(result->x[0] - t_expected) < 1e-10);
            RC_ASSERT(std::abs(result->x[1]) < 1e-14);
            RC_ASSERT(std::abs(result->x[2]) < 1e-14);
            RC_ASSERT(std::abs(result->x[3]) < 1e-14);
        }
    );

    // ── Property 4: solver returns nullopt for non-finite initial state ───────
    rc::check(
        "geodesic_flat: solver returns nullopt for NaN initial position",
        []() {
            GeodesicState bad{};
            bad.x = {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0};
            bad.u = {1.0, 0.0, 0.0, 0.0};

            const MetricTensor flat = MetricTensor::minkowski();
            GeodesicSolver solver;

            auto result = solver.solve(bad, flat, 10, 0.01);
            RC_ASSERT(!result.has_value());
        }
    );

    return 0;
}
