/**
 * @file  prop_metric_positive_definite.cpp
 * @brief Property: the spatial block of the metric tensor is positive-definite.
 *
 * Run with 10,000 random inputs:
 *   RC_PARAMS="max_success=10000" ./prop_metric_positive_definite
 *
 * Mathematical basis:
 *   For a Lorentzian (−,+,+,+) metric g_{μν}:
 *   1. g_{00} < 0  (timelike signature)
 *   2. The spatial 3×3 block g_{ij} (i,j ∈ {1,2,3}) must be positive-definite
 *      for a physically valid metric.
 *   3. All diagonal elements g_{ii} > 0 for i ≥ 1.
 *   4. det(g_{spatial}) > 0 (positive volume form).
 *
 * For the Minkowski metric η = diag(-1, +1, +1, +1):
 *   • g_{00} = -1 < 0  ✓
 *   • g_{ii} = 1 > 0   ✓  for i = 1,2,3
 *   • The spatial block is I₃ (identity), which is SPD.
 *
 * For a covariance-based metric with Σ SPD:
 *   • The spatial block equals Σ, which is SPD by construction.
 */

#include <rapidcheck.h>
#include <cmath>

#include "manifold/spacetime_manifold.hpp"

using namespace srfm::manifold;

int main() {
    // ── Property 1: Minkowski metric is_valid() always returns true ───────────
    rc::check(
        "metric_positive_definite: MetricTensor::minkowski() passes is_valid()",
        []() {
            const MetricTensor m = MetricTensor::minkowski();
            RC_ASSERT(m.is_valid());
        }
    );

    // ── Property 2: g[0][0] < 0 for Minkowski metric ─────────────────────────
    rc::check(
        "metric_positive_definite: time-time component is negative (Lorentzian sig)",
        []() {
            const MetricTensor m = MetricTensor::minkowski();
            RC_ASSERT(m.g[0][0] < 0.0);
        }
    );

    // ── Property 3: spatial diagonal > 0 for Minkowski metric ────────────────
    rc::check(
        "metric_positive_definite: spatial diagonal components are positive",
        []() {
            const MetricTensor m = MetricTensor::minkowski();
            for (int i = 1; i < DIM; ++i) {
                RC_ASSERT(m.g[i][i] > 0.0);
            }
        }
    );

    // ── Property 4: scaled Minkowski metric stays positive-definite ──────────
    rc::check(
        "metric_positive_definite: scaled diagonal metric is valid for positive scales",
        [](double s0, double s1, double s2, double s3) {
            // Map to strictly positive scales
            const double sc0 = std::exp(std::tanh(s0));  // ∈ (1/e, e)
            const double sc1 = std::exp(std::tanh(s1));
            const double sc2 = std::exp(std::tanh(s2));
            const double sc3 = std::exp(std::tanh(s3));

            MetricTensor m{};
            m.g[0][0] = -sc0 * sc0;  // must be negative
            m.g[1][1] =  sc1 * sc1;
            m.g[2][2] =  sc2 * sc2;
            m.g[3][3] =  sc3 * sc3;

            RC_ASSERT(m.is_valid());

            // Inverse should also exist for diagonal metric
            auto inv = m.inverse_diagonal();
            RC_ASSERT(inv.has_value());

            // g * g_inv = I for diagonal metric
            for (int i = 0; i < DIM; ++i) {
                const double product = m.g[i][i] * inv->g[i][i];
                RC_ASSERT(std::abs(product - 1.0) < 1e-12);
            }
        }
    );

    // ── Property 5: degenerate metric (zero diagonal) fails is_valid() ───────
    rc::check(
        "metric_positive_definite: zero spatial diagonal fails is_valid()",
        []() {
            MetricTensor bad{};
            bad.g[0][0] = -1.0;
            bad.g[1][1] =  0.0;  // degenerate!
            bad.g[2][2] =  1.0;
            bad.g[3][3] =  1.0;
            RC_ASSERT(!bad.is_valid());
        }
    );

    // ── Property 6: wrong signature (positive time-time) fails is_valid() ────
    rc::check(
        "metric_positive_definite: positive time-time component fails is_valid()",
        []() {
            MetricTensor bad{};
            bad.g[0][0] =  1.0;  // wrong sign!
            bad.g[1][1] =  1.0;
            bad.g[2][2] =  1.0;
            bad.g[3][3] =  1.0;
            RC_ASSERT(!bad.is_valid());
        }
    );

    // ── Property 7: non-finite entries fail is_valid() ───────────────────────
    rc::check(
        "metric_positive_definite: NaN entry fails is_valid()",
        []() {
            MetricTensor bad = MetricTensor::minkowski();
            bad.g[1][2] = std::numeric_limits<double>::quiet_NaN();
            RC_ASSERT(!bad.is_valid());
        }
    );

    return 0;
}
