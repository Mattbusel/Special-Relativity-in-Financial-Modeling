#pragma once

/// @file include/srfm/tensor.hpp
/// @brief Tensor Calculus & Covariance Engine — AGT-04 public API.
///
/// # Module: Tensor Calculus & Covariance Engine
///
/// ## Responsibility
/// Implements the differential geometry machinery for the financial spacetime
/// manifold. Provides:
///   - `MetricTensor`     — 4×4 position-dependent g_μν encoding covariance
///   - `ChristoffelSymbols` — Γ^λ_μν = ½ g^λσ(∂_μg_νσ + ∂_νg_μσ − ∂_σg_μν)
///   - `GeodesicSolver`   — integrates d²x^λ/dτ² + Γ^λ_μν ẋ^μ ẋ^ν = 0
///
/// ## Physical Interpretation
/// In the financial spacetime manifold, the metric g_μν encodes the
/// covariance structure of the market: the time-time component g₀₀ scales
/// with market time; the spatial block g_ij carries the asset covariance
/// matrix. Christoffel symbols Γ^λ_μν therefore measure the *rate of change
/// of correlations* through market space. The geodesic equation describes the
/// natural, force-free price path through this curved geometry.
///
/// ## Guarantees
/// - No undefined behaviour: all fallible operations return std::optional
/// - No raw pointers: ownership is by value or const-reference
/// - Thread-safe reads: const member functions are safe to call concurrently
/// - Eigen3 is used for all matrix arithmetic (LAPACK-quality numerics)
///
/// ## NOT Responsible For
/// - Lorentz boosts (see src/lorentz/)
/// - Market manifold topology / spacetime intervals (see src/manifold/)
/// - Backtesting or portfolio construction (see src/backtest/)

#include "srfm/types.hpp"
#include "srfm/constants.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <execution>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace srfm::tensor {

// ─── Type Aliases ─────────────────────────────────────────────────────────────

/// A callable that maps a spacetime point to the metric matrix at that point.
/// Used for position-dependent (curved) metrics.
using MetricFunction = std::function<MetricMatrix(const SpacetimePoint&)>;

/// Christoffel symbols as an array of 4×4 matrices.
/// Access pattern: gamma[lambda](mu, nu) = Γ^λ_μν.
using ChristoffelArray = std::array<MetricMatrix, SPACETIME_DIM>;

// ─── MetricTensor ─────────────────────────────────────────────────────────────

/// A position-dependent 4×4 symmetric tensor g_μν encoding the geometry of
/// the financial spacetime manifold.
///
/// The metric signature is (−,+,+,+): component 0 is timelike (market time),
/// components 1–3 are spacelike (asset returns). Off-diagonal spatial entries
/// encode asset correlations; off-diagonal time-space entries encode
/// temporal momentum correlations.
///
/// # Example
/// ```cpp
/// // Flat market: uncorrelated assets, equal volatility 0.2
/// auto g = srfm::tensor::MetricTensor::make_minkowski(1.0, 0.2);
/// srfm::SpacetimePoint origin = srfm::SpacetimePoint::Zero();
/// auto gx = g.evaluate(origin);  // diag(-1, 0.04, 0.04, 0.04)
/// ```
class MetricTensor {
public:
    /// Construct from an arbitrary position-dependent metric function.
    explicit MetricTensor(MetricFunction metric_fn);

    /// Evaluate g_μν at the given spacetime point.
    ///
    /// # Arguments
    /// * `x` — Position in the 4D financial spacetime manifold
    ///
    /// # Returns
    /// The 4×4 metric matrix at x.
    MetricMatrix evaluate(const SpacetimePoint& x) const;

    /// Evaluate g_μν at x and store the result into `out` (in-place overload).
    ///
    /// Avoids an extra copy compared to `evaluate()` when the caller already
    /// holds a `MetricMatrix` to overwrite.
    ///
    /// # Arguments
    /// * `x`   — Position in the 4D financial spacetime manifold
    /// * `out` — Output matrix to overwrite with g_μν(x)
    void evaluate_into(const SpacetimePoint& x, MetricMatrix& out) const {
        out = evaluate(x);
    }

    /// Compute the inverse metric g^μν at point x.
    ///
    /// # Returns
    /// - `Some(g_inv)` if the metric is invertible at x
    /// - `None`        if the metric is singular (degenerate correlations)
    std::optional<MetricMatrix> inverse(const SpacetimePoint& x) const;

    /// Return true if the metric has Lorentzian signature (−,+,+,+) at x.
    /// A Lorentzian metric has exactly one negative eigenvalue.
    bool is_lorentzian(const SpacetimePoint& x) const;

    /// Compute the spacetime interval ds² = g_μν dx^μ dx^ν.
    ///
    /// # Returns
    /// - Negative: timelike displacement (subluminal market movement)
    /// - Zero:     null / lightlike (signal at speed of information)
    /// - Positive: spacelike displacement (acausal — outside light cone)
    double spacetime_interval(const SpacetimePoint& x,
                              const FourVelocity& dx) const;

    // ── Factories ────────────────────────────────────────────────────────────

    /// Flat Minkowski-like metric: g = diag(−time_scale², σ², σ², σ²).
    ///
    /// Equivalent to a market with uncorrelated assets of equal volatility σ.
    ///
    /// # Arguments
    /// * `time_scale`    — Scale of the time dimension (c analogue), default 1
    /// * `spatial_scale` — Common asset volatility σ, default 1
    static MetricTensor make_minkowski(double time_scale   = 1.0,
                                       double spatial_scale = 1.0);

    /// Diagonal metric from per-asset volatilities:
    /// g = diag(−time_scale², σ₁², σ₂², σ₃²).
    ///
    /// # Arguments
    /// * `time_scale` — Scale of the time dimension
    /// * `vol`        — Array of three asset volatilities {σ₁, σ₂, σ₃}
    static MetricTensor make_diagonal(double time_scale,
                                      const std::array<double, 3>& vol);

    /// Full covariance-based metric from a 3×3 asset covariance matrix.
    /// g = block-diag(−time_scale², Σ) where Σ is the asset covariance.
    ///
    /// # Arguments
    /// * `time_scale` — Scale of the time dimension
    /// * `cov`        — 3×3 asset covariance matrix (must be positive definite)
    static MetricTensor make_from_covariance(double time_scale,
                                              const Eigen::Matrix3d& cov);

private:
    MetricFunction metric_fn_;
};

// ─── CachedMetricTensor ───────────────────────────────────────────────────────

/// Cache wrapper around MetricTensor::evaluate().
///
/// MetricTensor::evaluate() is called many times per geodesic integration step
/// (once per Christoffel finite-difference probe).  When the probe points are
/// close together, the metric value changes negligibly.  This wrapper avoids
/// redundant computation by caching the most recently evaluated (point, result)
/// pair and returning the cached result whenever the requested point is within
/// `tol` (L∞ norm) of the cached point.
///
/// All other MetricTensor operations (inverse, is_lorentzian,
/// spacetime_interval) are delegated directly to the inner MetricTensor without
/// caching, as they are called far less frequently.
///
/// ## Usage
/// ```cpp
/// auto cached = CachedMetricTensor::make_minkowski(1.0, 0.2);
/// auto g = cached.evaluate(pt);  // fast on repeated nearby calls
/// cached.invalidate();            // force re-evaluation on next call
/// ```
///
/// ## Thread safety
/// NOT thread-safe — one instance per thread / per GeodesicSolver.
class CachedMetricTensor {
public:
    /// Construct wrapping an existing MetricTensor with an optional tolerance.
    ///
    /// @param metric  MetricTensor to wrap (copied by value).
    /// @param tol     L∞ distance threshold for cache hit (default 1e-8).
    explicit CachedMetricTensor(MetricTensor metric, double tol = 1e-8)
        : inner_(std::move(metric)), tol_(tol) {}

    /// Evaluate g_μν at x, returning a cached result when the point is close
    /// enough to the previously cached point (L∞ distance < tol).
    MetricMatrix evaluate(const SpacetimePoint& x) const {
        if (valid_ && (x - pt_).cwiseAbs().maxCoeff() < tol_) {
            return cached_;
        }
        cached_ = inner_.evaluate(x);
        pt_     = x;
        valid_  = true;
        return cached_;
    }

    /// Delegate: compute g^μν (inverse metric) at x.
    std::optional<MetricMatrix> inverse(const SpacetimePoint& x) const {
        return inner_.inverse(x);
    }

    /// Delegate: return true iff metric has Lorentzian signature (−,+,+,+) at x.
    bool is_lorentzian(const SpacetimePoint& x) const {
        return inner_.is_lorentzian(x);
    }

    /// Delegate: compute ds² = g_μν dx^μ dx^ν at x.
    double spacetime_interval(const SpacetimePoint& x,
                               const FourVelocity&   dx) const {
        return inner_.spacetime_interval(x, dx);
    }

    /// Invalidate the cache, forcing re-evaluation on the next call to evaluate().
    void invalidate() const noexcept { valid_ = false; }

    // ── Factories ────────────────────────────────────────────────────────────

    /// Flat Minkowski-like metric: g = diag(−time_scale², σ², σ², σ²).
    static CachedMetricTensor make_minkowski(double time_scale   = 1.0,
                                              double spatial_scale = 1.0) {
        return CachedMetricTensor(MetricTensor::make_minkowski(time_scale,
                                                               spatial_scale));
    }

    /// Diagonal metric from per-asset volatilities.
    static CachedMetricTensor make_diagonal(double time_scale,
                                             const std::array<double, 3>& vol) {
        return CachedMetricTensor(MetricTensor::make_diagonal(time_scale, vol));
    }

    /// Full covariance-based metric from a 3×3 asset covariance matrix.
    static CachedMetricTensor make_from_covariance(double time_scale,
                                                    const Eigen::Matrix3d& cov) {
        return CachedMetricTensor(
            MetricTensor::make_from_covariance(time_scale, cov));
    }

private:
    MetricTensor              inner_;
    double                    tol_;
    mutable bool              valid_{false};
    mutable SpacetimePoint    pt_{SpacetimePoint::Zero()};
    mutable MetricMatrix      cached_{};
};

// ─── ChristoffelSymbols ───────────────────────────────────────────────────────

/// Computes the Christoffel symbols of the second kind Γ^λ_μν at a spacetime
/// point by numerically differentiating the metric tensor.
///
/// The Christoffel symbols measure how the metric — and therefore the market
/// covariance structure — changes from one market state to another. They are
/// the "connection" that converts coordinate changes into physical changes in
/// the correlation geometry.
///
/// Formula (Einstein summation):
///   Γ^λ_μν = ½ g^λσ (∂_μ g_νσ + ∂_ν g_μσ − ∂_σ g_μν)
///
/// Partial derivatives are computed via central finite differences:
///   ∂g_μν/∂x^σ ≈ [g_μν(x + h·ê_σ) − g_μν(x − h·ê_σ)] / (2h)
class ChristoffelSymbols {
public:
    /// Construct from a metric tensor.
    ///
    /// # Arguments
    /// * `metric` — The position-dependent metric (held by const-reference)
    /// * `h`      — Finite-difference step for metric derivatives (default 1e-5)
    explicit ChristoffelSymbols(const MetricTensor& metric,
                                double h = constants::DEFAULT_FD_STEP);

    /// Compute all 4³ = 64 Christoffel symbols at point x.
    ///
    /// # Arguments
    /// * `x` — Spacetime point at which to evaluate Γ^λ_μν
    ///
    /// # Returns
    /// Array indexed as result[lambda](mu, nu) = Γ^λ_μν.
    /// Returns all-zero array if the metric is singular at x.
    ChristoffelArray compute(const SpacetimePoint& x) const;

    /// Contract the Christoffel symbols with a four-velocity:
    ///   result^λ = Γ^λ_μν u^μ u^ν
    ///
    /// This is the RHS of the geodesic acceleration equation (negated).
    FourVelocity contract(const ChristoffelArray& gamma,
                          const FourVelocity& u) const;

private:
    /// Compute ∂g_μν/∂x^sigma via central finite differences.
    MetricMatrix metric_derivative(const SpacetimePoint& x, int sigma) const;

    const MetricTensor& metric_;
    double              h_;
};

// ─── CachedChristoffelSymbols ────────────────────────────────────────────────

/// Thread-local cache wrapper around ChristoffelSymbols.
///
/// Caches the most recently computed ChristoffelArray so that repeated calls
/// from the RK4 geodesic solver at the same (or very nearby) spacetime point
/// do not re-evaluate all 64 finite-difference metric derivatives.
///
/// Cache invalidation: the cached result is reused when the requested point
/// is within `cache_tol` (default 1e-8) of the cached point in L∞ norm.
///
/// Thread safety: NOT thread-safe (one instance per thread / per GeodesicSolver).
class CachedChristoffelSymbols {
public:
    /// Construct from a metric tensor with an optional cache tolerance.
    explicit CachedChristoffelSymbols(const MetricTensor& metric,
                                      double h          = constants::DEFAULT_FD_STEP,
                                      double cache_tol  = 1e-8)
        : inner_{metric, h}, cache_tol_{cache_tol} {}

    /// Compute (or return cached) Christoffel symbols at point x.
    ChristoffelArray compute(const SpacetimePoint& x) const {
        if (cache_valid_ && (x - cached_point_).cwiseAbs().maxCoeff() < cache_tol_) {
            return cached_result_;
        }
        cached_result_ = inner_.compute(x);
        cached_point_  = x;
        cache_valid_   = true;
        return cached_result_;
    }

    /// Contract cached symbols with a four-velocity.
    FourVelocity contract(const ChristoffelArray& gamma,
                          const FourVelocity& u) const {
        return inner_.contract(gamma, u);
    }

    /// Invalidate the cache (e.g. after a metric parameter change).
    void invalidate() const noexcept { cache_valid_ = false; }

private:
    ChristoffelSymbols        inner_;
    double                    cache_tol_;
    mutable bool              cache_valid_{false};
    mutable SpacetimePoint    cached_point_{SpacetimePoint::Zero()};
    mutable ChristoffelArray  cached_result_{};
};

// ─── GeodesicSolver ───────────────────────────────────────────────────────────

/// Phase-space state for the geodesic ODE: position x^μ and velocity u^μ.
///
/// The geodesic equation is a second-order ODE. We reduce it to first order
/// by treating (x, u) as the state vector:
///   dx^λ/dτ = u^λ
///   du^λ/dτ = −Γ^λ_μν u^μ u^ν
struct GeodesicState {
    SpacetimePoint position; ///< x^μ: position in financial spacetime
    FourVelocity   velocity; ///< u^μ = dx^μ/dτ: four-velocity tangent vector

    /// Pointwise addition of two states (used internally by RK4).
    GeodesicState operator+(const GeodesicState& o) const noexcept {
        return {position + o.position, velocity + o.velocity};
    }

    /// Scalar multiplication (used internally by RK4).
    friend GeodesicState operator*(double s, const GeodesicState& g) noexcept {
        return {s * g.position, s * g.velocity};
    }
};

/// Integrates the geodesic equation using classical 4th-order Runge-Kutta.
///
/// The geodesic equation:
///   d²x^λ/dτ² + Γ^λ_μν (dx^μ/dτ)(dx^ν/dτ) = 0
///
/// describes the "free fall" path of an asset price in curved financial
/// spacetime — the trajectory that minimises proper time (the path of least
/// market resistance given the covariance geometry).
///
/// # Example
/// ```cpp
/// auto g   = MetricTensor::make_minkowski(1.0, 0.2);
/// auto sol = GeodesicSolver(g, 0.01);
/// srfm::SpacetimePoint x0 = srfm::SpacetimePoint::Zero();
/// srfm::FourVelocity   u0; u0 << 1.0, 0.1, 0.0, 0.0;
/// auto traj = sol.integrate(x0, u0, 100);
/// ```
class GeodesicSolver {
public:
    /// Construct with a metric, proper-time step size, and FD step for Γ.
    ///
    /// # Arguments
    /// * `metric`        — Position-dependent metric tensor
    /// * `step_size`     — Proper-time step dτ for RK4 integration
    /// * `christoffel_h` — Finite-difference step for Christoffel symbols
    GeodesicSolver(const MetricTensor& metric,
                   double step_size     = constants::DEFAULT_GEODESIC_STEP,
                   double christoffel_h = constants::DEFAULT_FD_STEP);

    /// Integrate the geodesic from (x0, u0) for `steps` proper-time steps.
    ///
    /// # Arguments
    /// * `x0`    — Initial spacetime position
    /// * `u0`    — Initial four-velocity (tangent vector)
    /// * `steps` — Number of RK4 integration steps
    ///
    /// # Returns
    /// Vector of `steps + 1` states including the initial state.
    std::vector<GeodesicState> integrate(const SpacetimePoint& x0,
                                          const FourVelocity&   u0,
                                          int steps) const;

    /// Compute g_μν u^μ u^ν to diagnose the causal character of the geodesic.
    ///
    /// # Returns
    /// - Negative: timelike (subluminal price movement — physically meaningful)
    /// - Zero:     null / lightlike (speed-of-information signal)
    /// - Positive: spacelike (acausal — indicates model error or extreme regime)
    double norm_squared(const SpacetimePoint& x,
                        const FourVelocity&   u) const;

private:
    /// Advance state by one RK4 step of size step_size_.
    GeodesicState rk4_step(const GeodesicState& state) const;

    const MetricTensor& metric_;
    ChristoffelSymbols  christoffel_;
    double              step_size_;
};

// ─── integrate_batch ──────────────────────────────────────────────────────────

/// Integrate a batch of geodesics in parallel using std::execution::par_unseq.
///
/// Each geodesic is independent (different initial conditions), so the
/// per-geodesic integrations can be run concurrently without any shared mutable
/// state.  The solver object itself is read-only; each lambda call operates on
/// its own trajectory vector.
///
/// This provides a straightforward way to amortise the overhead of
/// multiple-asset geodesic integration — one call per portfolio rebalance
/// instead of a sequential for-loop.
///
/// # Requirements
/// The C++ standard library parallel STL backend must be available.
/// - On Linux with libstdc++: link with -ltbb (Intel TBB).
/// - On MSVC: std::execution::par_unseq works out of the box.
/// - On libc++ (macOS/LLVM): may require -fexperimental-library and TBB.
///
/// # Arguments
/// * `solver`             — Configured GeodesicSolver (read-only).
/// * `initial_conditions` — Vector of (initial_position, initial_velocity) pairs.
/// * `steps`              — Number of RK4 steps per trajectory.
///
/// # Returns
/// Vector of trajectories (one per initial condition), each containing
/// `steps + 1` GeodesicState entries in proper-time order.
///
/// # Thread safety
/// The GeodesicSolver and MetricTensor are accessed as const; parallel calls
/// are safe provided the MetricFunction stored in the solver's MetricTensor is
/// itself thread-safe for concurrent const calls.
inline std::vector<std::vector<GeodesicState>>
integrate_batch(
    const GeodesicSolver& solver,
    const std::vector<std::pair<SpacetimePoint, FourVelocity>>& initial_conditions,
    int steps)
{
    std::vector<std::vector<GeodesicState>> results(initial_conditions.size());

    std::transform(
        std::execution::par_unseq,
        initial_conditions.begin(),
        initial_conditions.end(),
        results.begin(),
        [&solver, steps](const std::pair<SpacetimePoint, FourVelocity>& ic) {
            return solver.integrate(ic.first, ic.second, steps);
        });

    return results;
}

} // namespace srfm::tensor
