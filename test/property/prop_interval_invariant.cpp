/**
 * @file  prop_interval_invariant.cpp
 * @brief Property: spacetime interval ds² is Lorentz-invariant.
 *
 * Run with 10,000 random inputs:
 *   RC_PARAMS="max_success=10000" ./prop_interval_invariant
 *
 * Mathematical basis:
 *   The spacetime interval ds² = g_{μν} dx^μ dx^ν is invariant under
 *   Lorentz boosts. For the flat Minkowski metric:
 *     ds² = -c²Δt² + Δx² + Δy² + Δz²
 *
 *   Under a Lorentz boost with velocity β along the x-axis:
 *     Δt'  = γ(Δt  - β·Δx/c)
 *     Δx'  = γ(Δx  - β·c·Δt)
 *     Δy'  = Δy
 *     Δz'  = Δz
 *
 *   The interval must satisfy: ds'² = ds²  (invariance)
 *
 * Property tested:
 *   |ds'²(boosted) - ds²(original)| < |ds²| * 1e-10 + 1e-12
 */

#include <rapidcheck.h>
#include <cmath>

#include "manifold/spacetime_manifold.hpp"
#include "momentum/momentum.hpp"

using namespace srfm::manifold;
using namespace srfm::momentum;

namespace {

/// Compute the flat Minkowski interval for a 4-displacement.
double minkowski_interval(double dt, double dx, double dy, double dz) {
    // ds² = -dt² + dx² + dy² + dz² (c=1 units)
    return -dt * dt + dx * dx + dy * dy + dz * dz;
}

/// Apply a Lorentz boost along the x-axis with velocity β.
/// Returns (dt', dx', dy', dz').
std::array<double, 4> lorentz_boost_x(double dt, double dx, double dy,
                                       double dz, double gamma, double beta) {
    return {
        gamma * (dt - beta * dx),
        gamma * (dx - beta * dt),
        dy,
        dz
    };
}

} // namespace

int main() {
    // ── Property 1: interval invariant under Lorentz boost along x ───────────
    rc::check(
        "interval_invariant: ds² unchanged under Lorentz boost",
        [](double raw_beta, double dt, double dx, double dy, double dz) {
            // Map β to valid range
            const double b = std::tanh(raw_beta) * (BETA_MAX_SAFE - 1e-10);
            auto bv = BetaVelocity::make(b);
            RC_ASSERT(bv.has_value());

            auto g_opt = lorentz_gamma(*bv);
            RC_ASSERT(g_opt.has_value());

            // Clamp 4-displacement to reasonable magnitudes
            const double dt_c = std::tanh(dt);
            const double dx_c = std::tanh(dx);
            const double dy_c = std::tanh(dy);
            const double dz_c = std::tanh(dz);

            const double ds2_orig = minkowski_interval(dt_c, dx_c, dy_c, dz_c);

            // Apply Lorentz boost
            auto boosted = lorentz_boost_x(
                dt_c, dx_c, dy_c, dz_c,
                g_opt->value(), bv->value()
            );
            const double ds2_boosted = minkowski_interval(
                boosted[0], boosted[1], boosted[2], boosted[3]
            );

            // Invariance: |ds2_boosted - ds2_orig| / (|ds2_orig| + 1e-12) < 1e-10
            const double abs_scale = std::abs(ds2_orig) + 1e-12;
            const double rel_err   = std::abs(ds2_boosted - ds2_orig) / abs_scale;
            RC_ASSERT(std::isfinite(ds2_boosted));
            RC_ASSERT(rel_err < 1e-10);
        }
    );

    // ── Property 2: interval sign is preserved under boost ───────────────────
    rc::check(
        "interval_invariant: timelike interval stays timelike after boost",
        [](double raw_beta) {
            const double b = std::tanh(raw_beta) * 0.5;  // moderate boost
            auto bv = BetaVelocity::make(b);
            RC_ASSERT(bv.has_value());
            auto g_opt = lorentz_gamma(*bv);
            RC_ASSERT(g_opt.has_value());

            // Strongly timelike: dt >> dx (e.g. dt=1, dx=0.1)
            const double dt_orig = 1.0;
            const double dx_orig = 0.1;
            const double ds2_orig = minkowski_interval(dt_orig, dx_orig, 0.0, 0.0);
            RC_ASSERT(ds2_orig < 0.0);  // timelike

            auto boosted = lorentz_boost_x(
                dt_orig, dx_orig, 0.0, 0.0,
                g_opt->value(), bv->value()
            );
            const double ds2_boosted = minkowski_interval(
                boosted[0], boosted[1], boosted[2], boosted[3]
            );

            // Must remain timelike after boost
            RC_ASSERT(ds2_boosted < 1e-10);
        }
    );

    // ── Property 3: null interval remains null under boost ───────────────────
    rc::check(
        "interval_invariant: null (lightlike) interval stays null after boost",
        [](double raw_beta) {
            const double b = std::tanh(raw_beta) * 0.7;
            auto bv = BetaVelocity::make(b);
            RC_ASSERT(bv.has_value());
            auto g_opt = lorentz_gamma(*bv);
            RC_ASSERT(g_opt.has_value());

            // Null interval: dt = dx, dy = dz = 0
            const double dt_orig = 0.5;
            const double dx_orig = 0.5;  // ds² = -0.25 + 0.25 = 0
            const double ds2_orig = minkowski_interval(dt_orig, dx_orig, 0.0, 0.0);
            // ds2_orig should be very close to 0
            RC_ASSERT(std::abs(ds2_orig) < 1e-14);

            auto boosted = lorentz_boost_x(
                dt_orig, dx_orig, 0.0, 0.0,
                g_opt->value(), bv->value()
            );
            const double ds2_boosted = minkowski_interval(
                boosted[0], boosted[1], boosted[2], boosted[3]
            );

            // Null interval must stay null to < 1e-10
            RC_ASSERT(std::abs(ds2_boosted) < 1e-10);
        }
    );

    return 0;
}
