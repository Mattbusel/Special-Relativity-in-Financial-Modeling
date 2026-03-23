#pragma once

/// @file include/srfm/multi_asset.hpp
/// @brief Multi-asset spacetime extension for the SRFM library.
///
/// Extends the single-asset spacetime framework (SpacetimeEvent, SpacetimeInterval)
/// to handle N correlated financial assets simultaneously by:
///
///  1. Embedding a portfolio snapshot as an (N+1)-dimensional spacetime event.
///  2. Computing the N-dimensional Lorentzian interval using a rolling correlation
///     matrix as the spatial metric block.
///  3. Applying simultaneous Lorentz transforms across all correlated price series.
///  4. Computing portfolio geodesics that represent the optimal portfolio path
///     under the given metric geometry.
///
/// ## Physical Interpretation
/// In multi-asset spacetime, the metric tensor couples price dimensions through
/// the pairwise correlations ρ_ij.  A TIMELIKE interval in the multi-asset sense
/// implies that the *portfolio* as a whole moved causally — the joint price motion
/// was consistent with sub-light correlated propagation.  A SPACELIKE interval
/// implies decorrelated (stochastic) joint motion.
///
/// ## Guarantees
/// - All fallible operations return std::optional (no exceptions).
/// - All const methods are thread-safe.
/// - No raw pointers in the public API.
/// - Eigen3 is used for all linear algebra.

#include "srfm/types.hpp"
#include "srfm/constants.hpp"
#include "srfm/manifold/n_asset_interval.hpp"
#include "srfm/tensor/n_asset_manifold.hpp"

#include <Eigen/Dense>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace srfm::multi_asset {

// ---------------------------------------------------------------------------
// MultiAssetEvent
// ---------------------------------------------------------------------------

/// @brief A snapshot of N correlated financial assets at a point in time.
///
/// Represents a single row of a multi-asset OHLCV dataset embedded as a
/// spacetime event in the (N+1)-dimensional financial spacetime manifold.
struct MultiAssetEvent {
    std::vector<std::string> symbols;   ///< Asset ticker symbols (for labelling).
    std::vector<double>      prices;    ///< Mid-prices for each asset.
    std::vector<double>      volumes;   ///< Traded volumes for each asset.
    int64_t                  timestamp; ///< Unix epoch milliseconds.

    /// @brief Return the number of assets represented.
    [[nodiscard]] std::size_t n_assets() const noexcept { return prices.size(); }

    /// @brief Validate that all vectors have equal length.
    [[nodiscard]] bool is_valid() const noexcept {
        return !prices.empty() &&
               prices.size() == volumes.size() &&
               (symbols.empty() || symbols.size() == prices.size());
    }
};

// ---------------------------------------------------------------------------
// MultiAssetInterval
// ---------------------------------------------------------------------------

/// @brief N-dimensional spacetime interval between two MultiAssetEvents.
///
/// The metric tensor g_μν is:
///
///   g_00     = −c_market²         (time-time, negative signature)
///   g_0i     = 0                  (no time-space cross-terms)
///   g_ij     = Σ_ij               (price-price block: correlation-scaled covariance)
///
/// where Σ_ij = ρ_ij · σ_i · σ_j is the rolling covariance matrix.
///
/// ## Classification
///   ds² < -LIGHTLIKE_EPS  → TIMELIKE   (causal, correlated joint motion)
///   |ds²| ≤ LIGHTLIKE_EPS → LIGHTLIKE  (critical boundary)
///   ds² > +LIGHTLIKE_EPS  → SPACELIKE  (stochastic, decorrelated motion)
class MultiAssetInterval {
public:
    using IntervalType = srfm::manifold::IntervalType;

    /// @brief Threshold for LIGHTLIKE classification.
    static constexpr double kLightlikeEps = srfm::manifold::LIGHTLIKE_THRESHOLD;

    /// @brief Compute ds² between two N-asset spacetime events.
    ///
    /// @param a       First event (earlier in time).
    /// @param b       Second event (later in time).
    /// @param metric  The (N+1)×(N+1) Lorentzian metric tensor g_μν.
    /// @return ds² value, or std::nullopt if events have different asset counts
    ///         or the metric dimensions are inconsistent.
    [[nodiscard]] static std::optional<double>
    compute(const MultiAssetEvent& a,
            const MultiAssetEvent& b,
            const Eigen::MatrixXd& metric) noexcept;

    /// @brief Classify a pre-computed ds² value.
    [[nodiscard]] static IntervalType classify(double ds2) noexcept;

    /// @brief Compute and classify in one call.
    ///
    /// @return Pair (ds², IntervalType), or nullopt on failure.
    [[nodiscard]] static std::optional<std::pair<double, IntervalType>>
    compute_and_classify(const MultiAssetEvent& a,
                         const MultiAssetEvent& b,
                         const Eigen::MatrixXd& metric) noexcept;
};

// ---------------------------------------------------------------------------
// CorrelationMetric
// ---------------------------------------------------------------------------

/// @brief Builds the Lorentzian metric tensor from a rolling correlation matrix.
///
/// Maintains a sliding window of N-asset price observations and computes the
/// (N+1)×(N+1) metric tensor g_μν with:
///   - g_00 = -c_market²
///   - g_ij = rolling_correlation[i,j] × sqrt(rolling_variance[i] × rolling_variance[j])
///
/// The Cholesky decomposition of the spatial block is checked on every update;
/// if the covariance matrix is not positive-definite (ill-conditioned or
/// singular correlation), the metric is regularised by adding
/// `regularisation_eps × I` to the spatial block.
class CorrelationMetric {
public:
    /// @brief Construction parameters.
    struct Config {
        std::size_t window_size{60};           ///< Rolling window (number of bars).
        double      c_market{1.0};             ///< Market speed-of-light.
        double      regularisation_eps{1e-6};  ///< Ridge regularisation for ill-conditioned Σ.
    };

    /// @brief Construct with given configuration.
    explicit CorrelationMetric(std::size_t n_assets, Config cfg = {});

    /// @brief Add a new bar observation and update the metric.
    ///
    /// @param prices New price observations for each asset.
    /// @return Updated metric tensor, or std::nullopt if fewer than 2 observations.
    [[nodiscard]] std::optional<Eigen::MatrixXd>
    update(const std::vector<double>& prices);

    /// @brief Return the current metric tensor (last computed).
    [[nodiscard]] std::optional<Eigen::MatrixXd> current_metric() const noexcept;

    /// @brief Return the rolling correlation matrix (spatial block only).
    [[nodiscard]] std::optional<Eigen::MatrixXd> correlation_matrix() const noexcept;

    /// @brief Return the number of assets.
    [[nodiscard]] std::size_t n_assets() const noexcept { return n_assets_; }

    /// @brief Return the number of observations seen so far.
    [[nodiscard]] std::size_t observation_count() const noexcept { return obs_count_; }

    /// @brief Return the configuration.
    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

    /// @brief Reset the window (clears all observations).
    void reset() noexcept;

private:
    std::size_t             n_assets_;
    Config                  cfg_;
    std::size_t             obs_count_{0};

    // Circular buffer of log-return observations.
    std::vector<Eigen::VectorXd> log_returns_window_;
    std::size_t                  window_head_{0};  // next write position

    // Previous prices for log-return computation.
    Eigen::VectorXd prev_prices_;
    bool            has_prev_{false};

    // Cached metric tensor.
    mutable std::optional<Eigen::MatrixXd> current_metric_;
    bool                                   metric_dirty_{true};

    /// @brief Recompute the correlation matrix and metric from the current window.
    void recompute_metric();

    /// @brief Compute the sample covariance from stored log-return observations.
    [[nodiscard]] Eigen::MatrixXd compute_covariance() const;
};

// ---------------------------------------------------------------------------
// MultiAssetLorentz
// ---------------------------------------------------------------------------

/// @brief Applies simultaneous Lorentz boosts to N correlated price series.
///
/// In the multi-asset setting, each asset has its own β_i = ΔP_i / (c · Δt).
/// The off-diagonal metric terms (correlations) couple the boosts: a high-β
/// move in asset A induces a "relativistic correction" in correlated asset B.
///
/// The transform is applied column-wise in the (N+1)-dimensional manifold:
///
///   γ_portfolio = 1 / sqrt(1 − β_portfolio²)
///
/// where β_portfolio = |Σ g_ij β_i β_j| (metric-weighted portfolio velocity).
class MultiAssetLorentz {
public:
    /// @brief Result of a multi-asset Lorentz transform.
    struct TransformResult {
        std::vector<double> adjusted_prices; ///< γ-scaled price adjustments.
        double              beta_portfolio;  ///< Portfolio velocity ∈ [0, 1).
        LorentzFactor       gamma_portfolio; ///< Portfolio Lorentz factor.
        std::vector<double> per_asset_beta;  ///< Individual β_i values.
        std::vector<double> per_asset_gamma; ///< Individual γ(β_i) values.
    };

    /// @brief Apply the multi-asset Lorentz transform.
    ///
    /// @param a      Previous-bar event.
    /// @param b      Current-bar event.
    /// @param metric (N+1)×(N+1) Lorentzian metric tensor.
    /// @return TransformResult, or std::nullopt if inputs are invalid.
    [[nodiscard]] static std::optional<TransformResult>
    transform(const MultiAssetEvent& a,
              const MultiAssetEvent& b,
              const Eigen::MatrixXd& metric) noexcept;

    /// @brief Compute portfolio β from individual asset velocities and the metric.
    ///
    /// β_portfolio = sqrt(g_ij β_i β_j)  (metric-weighted speed).
    ///
    /// @param betas  Per-asset β values.
    /// @param metric (N+1)×(N+1) metric (only the spatial block is used).
    /// @return Portfolio β, clamped to [0, BETA_MAX_SAFE].
    [[nodiscard]] static double
    portfolio_beta(const std::vector<double>& betas,
                   const Eigen::MatrixXd&     metric) noexcept;
};

// ---------------------------------------------------------------------------
// PortfolioGeodesic
// ---------------------------------------------------------------------------

/// @brief Geodesic in multi-asset spacetime: the optimal portfolio path.
///
/// In the (N+1)-dimensional financial spacetime manifold, the geodesic
/// equation describes the "force-free" trajectory of a portfolio in the absence
/// of external shocks.  Deviations of actual price paths from the geodesic
/// are interpreted as trading signals.
///
/// The geodesic is computed via Euler integration of the linearised geodesic
/// equation for the flat (constant-metric) case:
///
///   d²x^μ / dτ² = 0   (flat manifold: Christoffel symbols vanish)
///
/// Under a constant metric, the geodesic is simply a straight line in
/// spacetime coordinates — the "inertial" portfolio trajectory.  Curvature
/// effects arise from the time-varying metric (handled by passing updated
/// metrics per step).
class PortfolioGeodesic {
public:
    /// @brief A single step on the geodesic path.
    struct GeodesicStep {
        MultiAssetEvent event;            ///< Predicted portfolio event.
        double          proper_time;      ///< Cumulative proper time τ.
        double          path_length;      ///< Cumulative geodesic arc length.
        double          deviation;        ///< |actual − predicted| in price space.
    };

    /// @brief Compute the geodesic from an initial event with given four-velocity.
    ///
    /// @param initial       Starting multi-asset event.
    /// @param four_velocity (N+1)-vector dx^μ/dτ at the initial point.
    /// @param metric        (N+1)×(N+1) metric tensor (assumed constant).
    /// @param n_steps       Number of integration steps.
    /// @param dt            Proper-time step size.
    /// @return Sequence of GeodesicStep, or empty if inputs are invalid.
    [[nodiscard]] static std::vector<GeodesicStep>
    integrate(const MultiAssetEvent& initial,
              const Eigen::VectorXd& four_velocity,
              const Eigen::MatrixXd& metric,
              std::size_t            n_steps,
              double                 dt) noexcept;

    /// @brief Compute the geodesic deviation between the predicted and actual path.
    ///
    /// @param predicted  Geodesic steps from integrate().
    /// @param actual     Sequence of observed MultiAssetEvents.
    /// @return Per-step deviation ||predicted_prices − actual_prices||₂,
    ///         or empty if lengths mismatch.
    [[nodiscard]] static std::vector<double>
    deviation_series(const std::vector<GeodesicStep>&  predicted,
                     const std::vector<MultiAssetEvent>& actual) noexcept;

    /// @brief Compute portfolio weights along the geodesic path.
    ///
    /// At each step the geodesic four-velocity is normalised so that the
    /// spatial components sum to the target gross exposure.  Returns one
    /// weight vector per step.
    ///
    /// @param steps         Geodesic steps from integrate().
    /// @param four_velocity Reference four-velocity (direction of travel).
    /// @param gross_exposure Target portfolio gross exposure (sum of |weights|).
    /// @return Per-step weight vectors.
    [[nodiscard]] static std::vector<Eigen::VectorXd>
    portfolio_weights(const std::vector<GeodesicStep>& steps,
                      const Eigen::VectorXd&            four_velocity,
                      double                            gross_exposure = 1.0) noexcept;
};

} // namespace srfm::multi_asset
