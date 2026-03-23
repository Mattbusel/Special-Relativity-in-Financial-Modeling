#pragma once

/// @file include/relativistic_optimizer.hpp
/// @brief Relativistic Portfolio Optimization on the Financial Manifold.
///
/// # Module: Relativistic Portfolio Optimizer
///
/// ## Responsibility
/// Formulate the Markowitz mean-variance portfolio optimization problem as a
/// geodesic problem on the N-asset financial manifold, replacing the Euclidean
/// variance risk measure with the geodesic distance in the Lorentzian manifold.
///
/// ## Key Idea
/// Classical Markowitz minimises  w^T Σ w  subject to  w^T μ = r_target, Σw = 1.
/// Here we replace the quadratic risk measure  w^T Σ w  with the spacetime
/// geodesic distance between the current portfolio state and the target return
/// state on the financial manifold. This penalises large displacements in the
/// "causal" (TIMELIKE) direction more heavily than classical variance.
///
/// Additionally, expected returns are corrected by the Lorentz factor γ(β):
///   μ_rel_i = γ(β_i) · μ_i
/// where β_i is the normalised price velocity for asset i, computed from its
/// AssetEvent. High-velocity (noisy) assets are down-weighted; causal
/// (low-β) assets retain full expected return.
///
/// ## Optimization Problem
///   minimise    d_geo(w, w_0)²       (geodesic risk measure)
///   subject to  w^T μ_rel ≥ r_target (relativistic return constraint)
///               Σ_i w_i = 1          (full investment)
///               w_i ≥ 0              (long-only)
///
/// Solved via projected gradient descent on the simplex.
///
/// ## Guarantees
/// - Returns nullopt rather than throwing on degenerate inputs.
/// - No raw pointers; Eigen3 owns all matrix storage.
/// - All iterative algorithms are bounded (max_iterations parameter).
///
/// ## Dependencies
/// - Eigen3
/// - include/portfolio_manifold.hpp  (AssetEvent, MinkowskiCovariance)
/// - srfm/constants.hpp

#include "portfolio_manifold.hpp"

#include <Eigen/Dense>
#include <optional>
#include <vector>

namespace srfm::portfolio {

// ─── OptimizerConfig ──────────────────────────────────────────────────────────

/// Tuning parameters for the relativistic portfolio optimizer.
struct OptimizerConfig {
    /// Maximum number of projected gradient descent iterations.
    int max_iterations = 1000;

    /// Step size for gradient descent (learning rate).
    double step_size = 1e-3;

    /// Convergence tolerance: stop when ||w_{k+1} − w_k||₂ < tol.
    double convergence_tol = 1e-8;

    /// Tikhonov regularisation strength added to the geodesic metric.
    /// Prevents degenerate geodesic distances when assets are nearly collinear.
    double regularisation = 1e-6;

    /// Speed-of-information parameter for Lorentz factor computation.
    double c_market = constants::SPEED_OF_INFORMATION;
};

// ─── OptimizationResult ───────────────────────────────────────────────────────

/// Result of a single portfolio optimization run.
struct OptimizationResult {
    Eigen::VectorXd weights;          ///< Optimal asset weights (sum to 1, >= 0)
    double          geodesic_risk;    ///< Achieved geodesic risk d_geo²
    double          expected_return;  ///< w^T μ_rel (relativistic expected return)
    int             iterations;       ///< Number of gradient descent iterations
    bool            converged;        ///< True iff convergence_tol was reached
};

// ─── RelativisticPortfolio ────────────────────────────────────────────────────

/// Relativistic portfolio optimizer: Markowitz on the financial manifold.
///
/// ## Workflow
/// 1. Construct with a list of AssetEvents (one per asset in the portfolio).
/// 2. Set expected returns via set_expected_returns().
/// 3. Call optimize_weights(target_return, risk_tolerance).
///
/// The spacetime covariance matrix Σ_st is computed internally from the
/// AssetEvents using MinkowskiCovariance and used as the geodesic metric.
///
/// ## Example
/// ```cpp
/// RelativisticPortfolio rp;
/// rp.add_asset(AssetEvent{"AAPL", 1.0, 150.0, 1e8, 2.4e12}, 0.12);
/// rp.add_asset(AssetEvent{"MSFT", 1.0, 290.0, 8e7, 2.1e12}, 0.10);
/// auto result = rp.optimize_weights(0.08, 0.5);
/// if (result) {
///     std::cout << result->weights << "\n";
/// }
/// ```
class RelativisticPortfolio {
public:
    /// Construct with optional configuration.
    explicit RelativisticPortfolio(OptimizerConfig config = OptimizerConfig{}) noexcept;

    /// Add an asset with its annualised expected return.
    ///
    /// @param event           AssetEvent providing spacetime coordinates.
    /// @param expected_return Annualised expected return for this asset (e.g. 0.10 = 10%).
    void add_asset(AssetEvent event, double expected_return);

    /// Return the number of assets currently in the portfolio.
    [[nodiscard]] std::size_t n_assets() const noexcept;

    /// Compute the gamma-weighted (relativistic) expected return vector.
    ///
    /// For each asset i:
    ///   μ_rel_i = γ(β_i) · μ_i
    /// where β_i = |ΔP_i| / (c_market · |Δt_i|) approximated from the event
    /// coordinates relative to a reference event at the origin.
    ///
    /// @return Vector of length n_assets(), or nullopt if no assets added.
    [[nodiscard]] std::optional<Eigen::VectorXd>
    relativistic_returns() const noexcept;

    /// Compute the NxN spacetime covariance matrix Σ_st.
    ///
    /// Uses MinkowskiCovariance internally. Returns nullopt if fewer than 2
    /// assets are loaded.
    ///
    /// @return NxN Eigen::MatrixXd, or nullopt on failure.
    [[nodiscard]] std::optional<Eigen::MatrixXd>
    spacetime_covariance() const noexcept;

    /// Run the relativistic portfolio optimization.
    ///
    /// Formulates the problem as a geodesic minimization on the financial
    /// manifold and solves it via projected gradient descent on the simplex.
    ///
    /// @param target_return   Minimum required relativistic expected return.
    /// @param risk_tolerance  Scalar multiplier on the geodesic risk term
    ///                        (higher = more risk-averse; default 1.0).
    /// @return                OptimizationResult, or nullopt if fewer than 2
    ///                        assets are loaded or the covariance is degenerate.
    [[nodiscard]] std::optional<OptimizationResult>
    optimize_weights(double target_return,
                     double risk_tolerance = 1.0) const noexcept;

    /// Remove all assets from the portfolio.
    void clear() noexcept;

private:
    /// Compute β_i for asset i using the price/time coordinates of its event.
    [[nodiscard]] double compute_beta(const AssetEvent& e) const noexcept;

    /// Compute the Lorentz factor γ(β) clamped to BETA_MAX_SAFE.
    [[nodiscard]] static double gamma_factor(double beta) noexcept;

    /// Project a weight vector onto the probability simplex
    ///   { w : Σw_i = 1, w_i >= 0 }.
    /// Uses the O(N log N) algorithm of Duchi et al. (2008).
    [[nodiscard]] static Eigen::VectorXd
    project_simplex(const Eigen::VectorXd& v) noexcept;

    /// Compute the geodesic risk gradient:
    ///   ∇_w d²_geo(w) = 2 · Σ_st · w  (quadratic form with spacetime metric)
    [[nodiscard]] static Eigen::VectorXd
    geodesic_gradient(const Eigen::MatrixXd& sigma_st,
                      const Eigen::VectorXd& w) noexcept;

    std::vector<AssetEvent> events_;           ///< Per-asset spacetime events
    std::vector<double>     expected_returns_; ///< Per-asset expected returns
    OptimizerConfig         config_;           ///< Optimizer tuning parameters
};

} // namespace srfm::portfolio
