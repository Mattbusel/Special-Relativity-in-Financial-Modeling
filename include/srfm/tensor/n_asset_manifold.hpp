#pragma once
/**
 * @file  n_asset_manifold.hpp
 * @brief N-Asset Lorentzian Manifold for Special Relativistic Financial Mechanics.
 *
 * Module:  include/srfm/tensor/
 * Stage:   4 — N-Asset Manifold
 *
 * ## Responsibility
 * Represent a (N+1)-dimensional pseudo-Riemannian manifold with Lorentzian
 * signature (-,+,+,...,+) where the spatial block is given by an N×N
 * asset covariance matrix and the time-time component is -c_market².
 *
 * ## Metric Structure
 * The metric tensor g_μν is a constant (N+1)×(N+1) matrix:
 *
 *   g_00       = -c_market²
 *   g_0i = g_i0 = 0              for i ∈ {1,..,N}
 *   g_ij       = Σ_ij            for i,j ∈ {1,..,N}
 *
 * where Σ is the covariance matrix (Σ_ij = ρ_ij σ_i σ_j).
 *
 * ## Guarantees
 * - Thread-safe: all methods are const and noexcept.
 * - No raw pointers in the public API.
 * - All fallible operations return std::optional.
 * - Metric is constant (flat manifold); is_flat() always returns true.
 *
 * ## NOT Responsible For
 * - Time-varying covariance (stochastic volatility models).
 * - Cross-node synchronisation.
 * - Persistence of manifold state.
 */

#include <optional>
#include <string>
#include <third_party/eigen/Eigen/Dense>

namespace srfm::tensor {

/**
 * @brief (N+1)-dimensional Lorentzian manifold for N financial assets.
 *
 * The manifold dimension is n_assets + 1 (one time axis plus N price axes).
 * The metric is assembled from a user-supplied N×N positive-definite
 * covariance matrix and a market speed-of-light parameter c_market.
 */
class NAssetManifold {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Construct an NAssetManifold directly.
     *
     * Prefer the factory @ref make() which validates inputs and returns
     * std::nullopt on failure instead of constructing an invalid object.
     *
     * @param n_assets   Number of asset price dimensions (spatial dimensions).
     * @param covariance N×N positive-definite covariance matrix Σ.
     * @param c_market   Market speed-of-light parameter (must be > 0).
     */
    NAssetManifold(int n_assets,
                   Eigen::MatrixXd covariance,
                   double c_market = 1.0) noexcept;

    /**
     * @brief Factory: validate inputs then construct.
     *
     * Validates:
     *  - n_assets >= 1
     *  - covariance is N×N
     *  - covariance is symmetric
     *  - covariance is positive-definite (all eigenvalues > 0)
     *  - c_market > 0
     *
     * @param n       Number of assets.
     * @param cov     N×N covariance matrix.
     * @param c_market Market speed of light (default 1.0).
     * @return Constructed manifold or std::nullopt if inputs are invalid.
     */
    [[nodiscard]] static std::optional<NAssetManifold>
    make(int n, Eigen::MatrixXd cov, double c_market = 1.0) noexcept;

    // ── Dimensional accessors ─────────────────────────────────────────────────

    /**
     * @brief Returns the full manifold dimension = n_assets + 1.
     *
     * @return n_assets_ + 1.
     */
    [[nodiscard]] int dim() const noexcept;

    /**
     * @brief Returns the number of asset (spatial) dimensions.
     *
     * @return n_assets_.
     */
    [[nodiscard]] int n_assets() const noexcept;

    // ── Metric tensor ─────────────────────────────────────────────────────────

    /**
     * @brief Evaluate the metric tensor at point x.
     *
     * Because this is a constant (flat) manifold the result does not depend
     * on x. The parameter is accepted for API uniformity with curved manifolds.
     *
     * Returns a (N+1)×(N+1) matrix:
     *   row/col 0 → time component  (g_00 = -c_market²)
     *   rows/cols 1..N → spatial block  (g_ij = Σ_ij)
     *
     * @param x  Coordinate vector of length dim() (unused for flat manifolds).
     * @return   The metric matrix, or std::nullopt if x has wrong dimension.
     */
    [[nodiscard]] std::optional<Eigen::MatrixXd>
    metric_at(const Eigen::VectorXd& x) const noexcept;

    /**
     * @brief Evaluate the inverse metric g^μν at point x.
     *
     * Computed via LU decomposition of the metric matrix.
     *
     * @param x  Coordinate vector of length dim().
     * @return   The inverse metric, or std::nullopt on failure.
     */
    [[nodiscard]] std::optional<Eigen::MatrixXd>
    inverse_metric_at(const Eigen::VectorXd& x) const noexcept;

    /**
     * @brief Compute the squared line element ds² = g_μν dx^μ dx^ν.
     *
     * @param x   Base point (unused for flat manifold, validated for dimension).
     * @param dx  Displacement vector of length dim().
     * @return    ds², or std::nullopt if dimensions are inconsistent.
     */
    [[nodiscard]] std::optional<double>
    line_element_sq(const Eigen::VectorXd& x,
                    const Eigen::VectorXd& dx) const noexcept;

    // ── Properties ────────────────────────────────────────────────────────────

    /**
     * @brief Returns true if the metric has no coordinate dependence.
     *
     * For NAssetManifold the metric is always constant, so this always
     * returns true.
     *
     * @return true.
     */
    [[nodiscard]] bool is_flat() const noexcept;

    /**
     * @brief Returns the covariance matrix used to build the metric.
     *
     * @return The N×N covariance matrix, or std::nullopt on internal error.
     */
    [[nodiscard]] std::optional<Eigen::MatrixXd> covariance() const noexcept;

    /**
     * @brief Check structural compatibility with a 4D (N=3) manifold.
     *
     * Returns true if `other` has n_assets == 3 and this manifold's
     * spatial block is consistent with being an extension of a 4D manifold.
     * Concretely: returns true when other.n_assets() == 3 and
     * this->n_assets() >= 3.
     *
     * @param other  Manifold to compare against.
     * @return       true if compatible.
     */
    [[nodiscard]] bool reduces_to_4d(const NAssetManifold& other) const noexcept;

    /**
     * @brief Return the market speed-of-light parameter.
     *
     * @return c_market_.
     */
    [[nodiscard]] double c_market() const noexcept;

    /**
     * @brief Direct metric accessor (no coordinate argument).
     *
     * Returns the pre-built constant metric matrix.
     *
     * @return (N+1)×(N+1) metric tensor.
     */
    [[nodiscard]] const Eigen::MatrixXd& metric() const noexcept;

    /**
     * @brief Direct inverse metric accessor.
     *
     * Returns the pre-built inverse metric matrix.
     *
     * @return (N+1)×(N+1) inverse metric tensor.
     */
    [[nodiscard]] const Eigen::MatrixXd& inverse_metric() const noexcept;

private:
    // ── Internal helpers ──────────────────────────────────────────────────────

    /**
     * @brief Build the (N+1)×(N+1) metric matrix from stored covariance.
     *
     * Called once in the constructor.
     */
    void build_metric() noexcept;

    /**
     * @brief Build the inverse metric via full-pivot LU decomposition.
     *
     * Called once in the constructor after build_metric().
     */
    void build_inverse_metric() noexcept;

    // ── Data members ──────────────────────────────────────────────────────────

    int             n_assets_;      ///< Number of asset (spatial) dimensions.
    double          c_market_;      ///< Market speed of light.
    Eigen::MatrixXd covariance_;    ///< N×N covariance matrix.
    Eigen::MatrixXd metric_;        ///< Pre-built (N+1)×(N+1) metric.
    Eigen::MatrixXd inv_metric_;    ///< Pre-built (N+1)×(N+1) inverse metric.
    bool            inv_valid_;     ///< Whether inverse metric computation succeeded.
};

// ── Inline trivial accessors ──────────────────────────────────────────────────

inline int NAssetManifold::dim() const noexcept {
    return n_assets_ + 1;
}

inline int NAssetManifold::n_assets() const noexcept {
    return n_assets_;
}

inline bool NAssetManifold::is_flat() const noexcept {
    // Metric is constant — no coordinate dependence.
    return true;
}

inline double NAssetManifold::c_market() const noexcept {
    return c_market_;
}

inline const Eigen::MatrixXd& NAssetManifold::metric() const noexcept {
    return metric_;
}

inline const Eigen::MatrixXd& NAssetManifold::inverse_metric() const noexcept {
    return inv_metric_;
}

} // namespace srfm::tensor
