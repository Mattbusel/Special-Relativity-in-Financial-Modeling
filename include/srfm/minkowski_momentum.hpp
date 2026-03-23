#pragma once

/// @file include/srfm/minkowski_momentum.hpp
/// @brief Minkowski Momentum — Round 6 public API.
///
/// # Module: Minkowski Momentum
///
/// ## Concept
/// Extends classical momentum to financial spacetime by representing a
/// portfolio's exposure profile as a four-momentum vector:
///
///   p^μ = (E, p_x, p_y, p_z)
///
/// Financial interpretation:
///   - E   (energy)      = portfolio_return   (total return, the "time-like" component)
///   - p_x (x-momentum)  = equity_exposure    (e.g. equity beta × portfolio value)
///   - p_y (y-momentum)  = bond_exposure      (duration × portfolio value)
///   - p_z (z-momentum)  = commodity_exposure (commodity beta × portfolio value)
///
/// ## Key quantities
///
/// ### Invariant mass (diversification measure)
/// ```
///   m² = E² - p_x² - p_y² - p_z²
/// ```
/// A higher invariant mass indicates better diversification: the portfolio's
/// total return exceeds the sum-in-quadrature of its directional exposures.
///
/// ### Rapidity (financial velocity in equity space)
/// ```
///   y = 0.5 * ln((E + p_x) / (E - p_x))
/// ```
/// Rapidity is additive under successive equity-space boosts, making it a
/// natural measure of compounded equity momentum.
///
/// ## Guarantees
/// - All methods are `noexcept` where possible
/// - Return `std::optional` for invalid inputs
/// - No dynamic allocation; no mutable state in pure-math classes
/// - Thread-safe (stateless functions)

#include "srfm/types.hpp"
#include "srfm/constants.hpp"

#include <cmath>
#include <optional>
#include <span>
#include <vector>

namespace srfm::minkowski_momentum {

// ─── FourMomentum ─────────────────────────────────────────────────────────────

/// Financial four-momentum vector p^μ = (E, p_x, p_y, p_z).
///
/// Component semantics:
///   - energy: portfolio return (total, not annualised)
///   - px:     equity exposure
///   - py:     bond exposure
///   - pz:     commodity exposure
struct FourMomentum {
    double energy;  ///< E  — portfolio return (time-like component)
    double px;      ///< p_x — equity exposure
    double py;      ///< p_y — bond exposure
    double pz;      ///< p_z — commodity exposure
};

// ─── MinkowskiMomentum ────────────────────────────────────────────────────────

/// Stateless utility class for financial Minkowski four-momentum calculations.
class MinkowskiMomentum {
public:
    /// Compute the Minkowski interval (invariant mass squared):
    ///   m² = E² - p_x² - p_y² - p_z²
    ///
    /// A positive m² indicates a time-like four-momentum (physically
    /// realisable: the portfolio return dominates its exposures).  In the
    /// financial analogy, m² > 0 means the portfolio is well-diversified.
    ///
    /// @param p  Four-momentum vector.
    /// @return   m² = E² - p_x² - p_y² - p_z²  (may be negative).
    [[nodiscard]] static double invariant_mass_sq(const FourMomentum& p) noexcept;

    /// Compute the invariant mass (diversification measure):
    ///   m = sqrt(|m²|)  with sign preserved (negative if space-like)
    ///
    /// Returns nullopt if the four-momentum components are not all finite.
    ///
    /// @param p  Four-momentum vector.
    /// @return   Signed sqrt of |m²|, or nullopt on invalid input.
    [[nodiscard]] static std::optional<double>
    invariant_mass(const FourMomentum& p) noexcept;

    /// Compute the rapidity in equity space:
    ///   y = 0.5 * ln((E + p_x) / (E - p_x))
    ///
    /// Rapidity is finite and well-defined when |p_x| < E (i.e. equity
    /// exposure does not exceed total return).
    ///
    /// @param p  Four-momentum vector.
    /// @return   Rapidity, or nullopt if E <= |p_x| or inputs are not finite.
    [[nodiscard]] static std::optional<double>
    rapidity(const FourMomentum& p) noexcept;

    /// Compute the transverse momentum magnitude:
    ///   p_T = sqrt(p_y² + p_z²)
    ///
    /// Represents the combined non-equity exposure.
    ///
    /// @param p  Four-momentum vector.
    /// @return   p_T >= 0.
    [[nodiscard]] static double transverse_momentum(const FourMomentum& p) noexcept;

    /// Compute the spatial momentum magnitude:
    ///   |p| = sqrt(p_x² + p_y² + p_z²)
    ///
    /// @param p  Four-momentum vector.
    /// @return   |p| >= 0.
    [[nodiscard]] static double spatial_magnitude(const FourMomentum& p) noexcept;
};

// ─── FourMomentumConservation ─────────────────────────────────────────────────

/// Checks whether a set of trade four-momenta conserves total four-momentum
/// (i.e. the net change in portfolio exposure sums to the reference vector).
class FourMomentumConservation {
public:
    /// Check four-momentum conservation for a collection of trades.
    ///
    /// The trades are considered "conserving" if the sum of all four-momentum
    /// vectors equals the reference vector within *tolerance*.
    ///
    /// @param trades      Span of four-momentum changes from individual trades.
    /// @param reference   Expected net four-momentum (e.g. zeros for net-flat).
    /// @param tolerance   Per-component absolute tolerance (default 1e-9).
    /// @return            true if all four components are conserved.
    [[nodiscard]] static bool conserves(
        std::span<const FourMomentum> trades,
        const FourMomentum& reference,
        double tolerance = 1e-9) noexcept;

    /// Sum a collection of four-momenta into a single resultant.
    ///
    /// @param momenta  Span of four-momentum vectors.
    /// @return         Component-wise sum.
    [[nodiscard]] static FourMomentum sum(
        std::span<const FourMomentum> momenta) noexcept;
};

// ─── MomentumPortfolioOptimizer ───────────────────────────────────────────────

/// Finds portfolio weights that maximise the Minkowski invariant mass
/// (diversification measure) subject to the constraint that weights sum to 1.
///
/// The optimisation is performed by a simple gradient-ascent procedure:
/// at each step the weight vector is nudged in the direction of increasing m²,
/// then re-normalised to sum to 1 and clamped to [min_weight, max_weight].
class MomentumPortfolioOptimizer {
public:
    /// Configuration for the gradient-ascent optimiser.
    struct Config {
        double learning_rate  = 0.01;   ///< Step size per iteration
        int    max_iterations = 1000;   ///< Maximum number of ascent steps
        double tolerance      = 1e-8;   ///< Convergence criterion on m²
        double min_weight     = 0.0;    ///< Lower bound on each weight
        double max_weight     = 1.0;    ///< Upper bound on each weight
    };

    /// Result of a single optimisation run.
    struct Result {
        std::vector<double> weights;    ///< Optimal weight vector (sums to 1)
        double              max_inv_mass_sq; ///< m² achieved at the optimum
        int                 iterations_used; ///< Actual iterations taken
        bool                converged;  ///< true if tolerance was reached
    };

    /// Maximise invariant mass over portfolio weight combinations.
    ///
    /// @param returns     Asset return vector (one entry per asset).
    /// @param exposures   Matrix (n_assets × 3): [equity, bond, commodity]
    ///                    exposure coefficients for each asset.
    /// @param cfg         Optimiser configuration.
    /// @return            Optimisation result, or nullopt on invalid input.
    [[nodiscard]] static std::optional<Result> optimize(
        std::span<const double>              returns,
        std::span<const std::array<double,3>> exposures,
        const Config& cfg = {}) noexcept;

private:
    /// Compute the portfolio-level FourMomentum from weight vector + data.
    static FourMomentum _portfolio_momentum(
        std::span<const double>              weights,
        std::span<const double>              returns,
        std::span<const std::array<double,3>> exposures) noexcept;
};

}  // namespace srfm::minkowski_momentum
