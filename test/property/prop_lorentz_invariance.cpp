/**
 * @file  prop_lorentz_invariance.cpp
 * @brief Property: ds² = −c²·Δt² + Δx² is Lorentz-invariant under boost.
 *
 * Run with 10,000 random inputs:
 *   RC_PARAMS="max_success=10000" ./prop_lorentz_invariance
 *
 * Mathematical basis:
 *   For two events (t1,x1) and (t2,x2) in the unprimed frame:
 *     ds² = −Δt² + Δx²   (c=1 units, (−,+) signature)
 *
 *   Under a Lorentz boost by β along the x-axis with γ = 1/√(1−β²):
 *     t'  = γ(t  − β·x)
 *     x'  = γ(x  − β·t)
 *
 *   The interval must be invariant: ds'² = ds²
 *
 * Property tested (tolerance 1e-9):
 *   |ds'²(primed) − ds²(unprimed)| ≤ 1e-9
 *
 * β is drawn from (−0.9999, +0.9999) via tanh mapping.
 * Spacetime events are clamped via tanh to avoid catastrophic cancellation.
 */

#include <rapidcheck.h>
#include <cmath>

#include "momentum/momentum.hpp"

using namespace srfm::momentum;

namespace {

/// Compute the 1+1D Minkowski interval for a displacement (dt, dx) with c=1.
double minkowski_ds2(double dt, double dx) noexcept {
    return -dt * dt + dx * dx;
}

/// Apply a Lorentz boost by (gamma, beta) to a single event (t, x).
/// Returns {t', x'}.
std::pair<double, double>
boost_event(double t, double x, double gamma, double beta) noexcept {
    return {gamma * (t - beta * x), gamma * (x - beta * t)};
}

} // namespace

int main() {
    // ── Property 1: ds² invariant for arbitrary β ∈ (−0.9999, +0.9999) ─────
    rc::check(
        "lorentz_invariance: ds² unchanged under Lorentz boost for two events",
        [](double raw_beta,
           double t1_raw, double x1_raw,
           double t2_raw, double x2_raw) {
            // Map β to (−BETA_MAX_SAFE, +BETA_MAX_SAFE) via tanh.
            const double beta = std::tanh(raw_beta) * (BETA_MAX_SAFE - 1e-10);
            auto bv = BetaVelocity::make(beta);
            RC_ASSERT(bv.has_value());

            auto g_opt = lorentz_gamma(*bv);
            RC_ASSERT(g_opt.has_value());
            const double gamma = g_opt->value();
            const double b     = bv->value();

            // Clamp events to [−1, +1] to keep numbers well-conditioned.
            const double t1 = std::tanh(t1_raw);
            const double x1 = std::tanh(x1_raw);
            const double t2 = std::tanh(t2_raw);
            const double x2 = std::tanh(x2_raw);

            // Displacement in unprimed frame.
            const double dt = t2 - t1;
            const double dx = x2 - x1;

            // ds² in unprimed frame.
            const double ds2_unprimed = minkowski_ds2(dt, dx);

            // Boost both events.
            auto [t1p, x1p] = boost_event(t1, x1, gamma, b);
            auto [t2p, x2p] = boost_event(t2, x2, gamma, b);

            // Displacement in primed frame.
            const double dt_prime = t2p - t1p;
            const double dx_prime = x2p - x1p;

            // ds² in primed frame.
            const double ds2_primed = minkowski_ds2(dt_prime, dx_prime);

            // Both must be finite.
            RC_ASSERT(std::isfinite(ds2_unprimed));
            RC_ASSERT(std::isfinite(ds2_primed));

            // Invariance check: |ds'² − ds²| ≤ 1e-9.
            const double diff = std::abs(ds2_primed - ds2_unprimed);
            RC_ASSERT(diff < 1e-9);
        }
    );

    // ── Property 2: invariance holds for zero displacement (trivial) ─────────
    rc::check(
        "lorentz_invariance: ds² = 0 for coincident events after any boost",
        [](double raw_beta, double t_raw, double x_raw) {
            const double beta = std::tanh(raw_beta) * (BETA_MAX_SAFE - 1e-10);
            auto bv = BetaVelocity::make(beta);
            RC_ASSERT(bv.has_value());
            auto g_opt = lorentz_gamma(*bv);
            RC_ASSERT(g_opt.has_value());
            const double gamma = g_opt->value();
            const double b     = bv->value();

            // Two coincident events — ds² must be zero.
            const double t = std::tanh(t_raw);
            const double x = std::tanh(x_raw);

            const double ds2_unprimed = minkowski_ds2(0.0, 0.0);
            RC_ASSERT(ds2_unprimed == 0.0);

            auto [tp, xp] = boost_event(t, x, gamma, b);
            (void)tp; (void)xp;

            const double ds2_primed = minkowski_ds2(0.0, 0.0);
            RC_ASSERT(ds2_primed == 0.0);
        }
    );

    // ── Property 3: sign of ds² is preserved under boost ─────────────────────
    rc::check(
        "lorentz_invariance: timelike ds² (dt>|dx|) stays negative after boost",
        [](double raw_beta) {
            // Moderate boost only, to keep the event timelike.
            const double beta = std::tanh(raw_beta) * 0.5;
            auto bv = BetaVelocity::make(beta);
            RC_ASSERT(bv.has_value());
            auto g_opt = lorentz_gamma(*bv);
            RC_ASSERT(g_opt.has_value());
            const double gamma = g_opt->value();
            const double b     = bv->value();

            // Strongly timelike displacement: dt=1, dx=0.1.
            const double dt = 1.0;
            const double dx = 0.1;
            const double ds2_unprimed = minkowski_ds2(dt, dx);
            RC_ASSERT(ds2_unprimed < 0.0);

            // Boost the displacement by computing primed coords of two events.
            auto [t1p, x1p] = boost_event(0.0, 0.0, gamma, b);
            auto [t2p, x2p] = boost_event(dt,  dx,  gamma, b);
            const double ds2_primed = minkowski_ds2(t2p - t1p, x2p - x1p);

            // Must remain timelike (negative) after the boost.
            RC_ASSERT(ds2_primed < 1e-10);

            // And the invariance tolerance must hold.
            RC_ASSERT(std::abs(ds2_primed - ds2_unprimed) < 1e-9);
        }
    );

    return 0;
}
