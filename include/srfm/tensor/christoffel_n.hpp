#pragma once
/**
 * @file  christoffel_n.hpp
 * @brief Christoffel symbol computation for an NAssetManifold.
 *
 * Module:  include/srfm/tensor/
 * Stage:   4 — N-Asset Manifold
 *
 * ## Responsibility
 * Compute the Christoffel symbols of the second kind,
 *
 *   Γ^λ_μν = ½ g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
 *
 * for an arbitrary NAssetManifold via central finite-differences on the
 * metric tensor.
 *
 * ## Guarantees
 * - Thread-safe: ChristoffelN is stateless beyond holding a const reference
 *   to the manifold.
 * - All fallible operations return std::optional.
 * - All public methods are noexcept.
 * - For a constant metric (flat manifold) all symbols are exactly zero.
 *
 * ## NOT Responsible For
 * - Second derivatives / Riemann tensor.
 * - Geodesic integration (see geodesic_n.hpp).
 */

#include "n_asset_manifold.hpp"

#include <optional>
#include <vector>
#include <third_party/eigen/Eigen/Dense>

namespace srfm::tensor {

/**
 * @brief Computes Christoffel symbols Γ^λ_μν for an NAssetManifold.
 *
 * Metric derivatives are approximated by central finite differences with
 * step size FD_STEP. For a constant metric all derivatives vanish and every
 * Christoffel symbol is exactly zero.
 */
class ChristoffelN {
public:
    /// Central finite-difference step size used for metric derivatives.
    static constexpr double FD_STEP = 1e-5;

    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Construct from a manifold reference.
     *
     * The manifold must remain valid for the lifetime of this object.
     *
     * @param manifold  The NAssetManifold providing the metric.
     */
    explicit ChristoffelN(const NAssetManifold& manifold) noexcept;

    // ── Symbol accessors ──────────────────────────────────────────────────────

    /**
     * @brief Compute a single Christoffel symbol Γ^lambda_mu_nu at point x.
     *
     * Uses the formula:
     *   Γ^λ_μν = ½ g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
     *
     * Metric derivatives are approximated by central FD with step FD_STEP.
     *
     * @param lambda  Contravariant index, in [0, dim).
     * @param mu      First covariant index, in [0, dim).
     * @param nu      Second covariant index, in [0, dim).
     * @param x       Coordinate point, must have length dim().
     * @return        Γ^lambda_mu_nu, or std::nullopt on dimension mismatch or
     *                metric inversion failure.
     */
    [[nodiscard]] std::optional<double>
    symbol(int lambda, int mu, int nu,
           const Eigen::VectorXd& x) const noexcept;

    /**
     * @brief Compute all Christoffel symbols at point x.
     *
     * Returns a dim() × dim() × dim() tensor stored as nested std::vector:
     *   result[lambda][mu][nu] = Γ^lambda_mu_nu
     *
     * @param x  Coordinate point of length dim().
     * @return   Full tensor, or std::nullopt on failure.
     */
    [[nodiscard]] std::optional<std::vector<std::vector<std::vector<double>>>>
    all_symbols(const Eigen::VectorXd& x) const noexcept;

    /**
     * @brief Verify that Γ^λ_μν = Γ^λ_νμ for all indices at point x.
     *
     * Checks the symmetry of the lower two indices, which holds for
     * torsion-free connections.
     *
     * @param x    Coordinate point of length dim().
     * @param tol  Tolerance for the symmetry check (default 1e-10).
     * @return     true if symmetric within tolerance, false otherwise.
     *             Returns false (not nullopt) on dimension mismatch.
     */
    [[nodiscard]] bool
    verify_symmetry(const Eigen::VectorXd& x,
                    double tol = 1e-10) const noexcept;

    /**
     * @brief Return the dimension of the manifold.
     *
     * @return manifold_.dim().
     */
    [[nodiscard]] int dim() const noexcept;

private:
    // ── Internal helpers ──────────────────────────────────────────────────────

    /**
     * @brief Compute ∂_alpha g_mu_nu at x via central finite differences.
     *
     * ∂_α g_μν ≈ [g_μν(x + h·e_α) - g_μν(x - h·e_α)] / (2h)
     *
     * @param alpha  Direction index for differentiation.
     * @param mu     Row index of the metric component.
     * @param nu     Column index of the metric component.
     * @param x      Base point.
     * @return       Finite-difference approximation, or std::nullopt on failure.
     */
    [[nodiscard]] std::optional<double>
    metric_deriv(int alpha, int mu, int nu,
                 const Eigen::VectorXd& x) const noexcept;

    /**
     * @brief Return the inverse metric at x, computing it if needed.
     *
     * For a constant metric this always returns the pre-built inverse.
     *
     * @param x  Coordinate point.
     * @return   Inverse metric matrix or std::nullopt on failure.
     */
    [[nodiscard]] std::optional<Eigen::MatrixXd>
    inv_metric_at(const Eigen::VectorXd& x) const noexcept;

    // ── Data members ──────────────────────────────────────────────────────────

    const NAssetManifold& manifold_;  ///< Reference to the underlying manifold.
};

// ── Inline trivial accessors ──────────────────────────────────────────────────

inline int ChristoffelN::dim() const noexcept {
    return manifold_.dim();
}

} // namespace srfm::tensor
