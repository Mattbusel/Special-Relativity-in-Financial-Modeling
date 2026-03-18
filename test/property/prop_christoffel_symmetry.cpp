/**
 * @file  prop_christoffel_symmetry.cpp
 * @brief Property: Christoffel symbols are symmetric in their lower two indices.
 *
 * Run with 10,000 random inputs:
 *   RC_PARAMS="max_success=10000" ./prop_christoffel_symmetry
 *
 * Mathematical basis:
 *   The Christoffel symbols of the Levi-Civita (torsion-free) connection are:
 *     Γ^λ_μν = (1/2) g^{λσ} (∂_μ g_{νσ} + ∂_ν g_{μσ} − ∂_σ g_{μν})
 *
 *   From this formula, the lower-index symmetry is immediate:
 *     Γ^λ_μν = Γ^λ_νμ   ∀ λ, μ, ν
 *
 *   This is equivalent to saying the connection is torsion-free:
 *     T^λ_μν = Γ^λ_μν − Γ^λ_νμ = 0
 *
 *   Note: "antisymmetry" is NOT correct for the Levi-Civita connection.
 *   The correct property is symmetry (Γ^λ_μν = Γ^λ_νμ), NOT antisymmetry.
 *   This test verifies the symmetry property for the flat constant metric.
 *
 * For the flat Minkowski metric:
 *   All 64 Γ^λ_μν = 0, so symmetry holds trivially (0 = 0).
 *   The important test is that ChristoffelSymbols preserves this
 *   regardless of which pair of lower indices is queried.
 */

#include <rapidcheck.h>
#include <cmath>

#include "manifold/spacetime_manifold.hpp"

using namespace srfm::manifold;

int main() {
    // ── Property 1: Γ^λ_μν = Γ^λ_νμ for flat Minkowski metric ───────────────
    rc::check(
        "christoffel_symmetry: Gamma[l][mu][nu] == Gamma[l][nu][mu] (flat metric)",
        [](int lambda, int mu, int nu) {
            // Clamp to valid index range
            const int l = ((lambda % DIM) + DIM) % DIM;
            const int m = ((mu     % DIM) + DIM) % DIM;
            const int n = ((nu     % DIM) + DIM) % DIM;

            SpacetimeManifold manifold;
            const MetricTensor flat = MetricTensor::minkowski();
            const auto gamma = manifold.christoffelSymbols(flat);

            const auto idx_mn = static_cast<std::size_t>(christoffel_index(l, m, n));
            const auto idx_nm = static_cast<std::size_t>(christoffel_index(l, n, m));

            RC_ASSERT(std::isfinite(gamma[idx_mn]));
            RC_ASSERT(std::isfinite(gamma[idx_nm]));

            // Symmetry: Γ^λ_μν = Γ^λ_νμ
            const double diff = std::abs(gamma[idx_mn] - gamma[idx_nm]);
            RC_ASSERT(diff < 1e-12);
        }
    );

    // ── Property 2: All 64 symbols are finite for flat metric ─────────────────
    rc::check(
        "christoffel_symmetry: all 64 symbols finite for flat metric",
        []() {
            SpacetimeManifold manifold;
            const MetricTensor flat = MetricTensor::minkowski();
            const auto gamma = manifold.christoffelSymbols(flat);

            for (std::size_t i = 0; i < static_cast<std::size_t>(NUM_CHRISTOFFEL); ++i) {
                RC_ASSERT(std::isfinite(gamma[i]));
            }
        }
    );

    // ── Property 3: Γ^λ_μν = Γ^λ_νμ for any valid diagonal metric ───────────
    rc::check(
        "christoffel_symmetry: symmetry holds for scaled diagonal metric",
        [](double g11, double g22, double g33, int lambda, int mu, int nu) {
            // Construct a valid diagonal metric with random positive spatial entries
            const double s1 = std::exp(std::tanh(g11));
            const double s2 = std::exp(std::tanh(g22));
            const double s3 = std::exp(std::tanh(g33));

            MetricTensor metric{};
            metric.g[0][0] = -1.0;
            metric.g[1][1] =  s1;
            metric.g[2][2] =  s2;
            metric.g[3][3] =  s3;

            RC_ASSERT(metric.is_valid());

            SpacetimeManifold manifold;
            const auto gamma = manifold.christoffelSymbols(metric);

            const int l = ((lambda % DIM) + DIM) % DIM;
            const int m = ((mu     % DIM) + DIM) % DIM;
            const int n = ((nu     % DIM) + DIM) % DIM;

            const auto idx_mn = static_cast<std::size_t>(christoffel_index(l, m, n));
            const auto idx_nm = static_cast<std::size_t>(christoffel_index(l, n, m));

            RC_ASSERT(std::isfinite(gamma[idx_mn]));
            RC_ASSERT(std::isfinite(gamma[idx_nm]));

            // Symmetry: Γ^λ_μν = Γ^λ_νμ
            const double diff = std::abs(gamma[idx_mn] - gamma[idx_nm]);
            RC_ASSERT(diff < 1e-10);
        }
    );

    // ── Property 4: invalid metric returns all-zero array ─────────────────────
    rc::check(
        "christoffel_symmetry: invalid metric produces zero Christoffels",
        []() {
            MetricTensor bad{};
            // All zeros → not valid (g[0][0] = 0, not negative)
            SpacetimeManifold manifold;
            const auto gamma = manifold.christoffelSymbols(bad);

            for (std::size_t i = 0; i < static_cast<std::size_t>(NUM_CHRISTOFFEL); ++i) {
                RC_ASSERT(gamma[i] == 0.0);
            }
        }
    );

    return 0;
}
