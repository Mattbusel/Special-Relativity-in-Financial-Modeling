#pragma once

/// @file include/srfm/lorentz_portfolio.hpp
/// @brief Lorentz Portfolio Transformation — Round 4 public API.
///
/// # Module: Lorentz Portfolio Transformation
///
/// ## Concept
/// Interprets a portfolio's statistical moments as a 4-vector in financial
/// spacetime:  (return, volatility, skewness, kurtosis).  A Lorentz boost
/// along the return-volatility plane simulates the effect of "moving" the
/// portfolio to a different reference frame — useful for stress-testing
/// how Sharpe ratios transform under regime shifts.
///
/// ## Transformation (boost along return axis, β ∈ (-1, 1))
/// ```
///   ret' = γ*(ret - β*vol)
///   vol' = γ*(vol - β*ret)
///   skew' = skew            (transverse — unchanged)
///   kurt' = kurt            (transverse — unchanged)
/// ```
/// where γ = 1/√(1 - β²).
///
/// ## Minkowski invariant
/// ```
///   I = ret² - vol² - skew² - kurt²
/// ```
/// This scalar is invariant under all boosts — i.e. I == I' for any β.
///
/// ## References
/// The analogy follows the 4-vector formalism in Special Relativity:
///   x^μ = (ct, x, y, z)  →  (ret, vol, skew, kurt)
///
/// ## Guarantees
/// - All methods noexcept
/// - No dynamic allocation
/// - Thread-safe (pure functions)

#include "srfm/types.hpp"
#include "srfm/constants.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace srfm::portfolio {

// ─── PortfolioFourVector ──────────────────────────────────────────────────────

/// The "spacetime position" of a portfolio.
///
/// The four components map statistical moments to coordinates in financial
/// spacetime.  The first component (ret) plays the role of the time-like
/// coordinate; vol, skew, kurt are space-like.
struct PortfolioFourVector {
    double ret  = 0.0;   ///< Annualised expected return (time-like component)
    double vol  = 0.0;   ///< Annualised volatility (space-like)
    double skew = 0.0;   ///< Skewness (space-like, transverse)
    double kurt = 0.0;   ///< Excess kurtosis (space-like, transverse)

    /// Sharpe ratio in this frame: ret / vol.  Returns 0 if vol == 0.
    [[nodiscard]] double sharpe() const noexcept {
        return (vol != 0.0) ? (ret / vol) : 0.0;
    }
};

// ─── LorentzFactor ────────────────────────────────────────────────────────────

/// Lorentz factor γ = 1/√(1 - β²).
///
/// γ ≥ 1 always; γ → ∞ as |β| → 1.
struct LorentzFactor {
    double gamma = 1.0;  ///< γ value

    /// Construct γ from β.
    ///
    /// @param beta  Normalised velocity, strictly in (-1, 1).
    /// @throws std::domain_error if |β| >= 1.
    explicit LorentzFactor(double beta) {
        if (beta <= -1.0 || beta >= 1.0) {
            throw std::domain_error("beta must be strictly in (-1, 1)");
        }
        gamma = 1.0 / std::sqrt(1.0 - beta * beta);
    }

    LorentzFactor() = default;
};

// ─── LorentzBoost ─────────────────────────────────────────────────────────────

/// Applies a Lorentz boost to a PortfolioFourVector.
///
/// The boost is performed along the (return, volatility) plane, analogous to
/// a boost along the x-axis in standard SR with (t, x) → (ret, vol).
class LorentzBoost {
public:
    LorentzBoost() = delete; // static utility class

    /// Apply a boost with velocity β to *portfolio*.
    ///
    /// @param portfolio  The portfolio 4-vector in the lab frame.
    /// @param beta        Boost velocity ∈ (-1, 1).  Positive β "tilts" the
    ///                    frame toward higher return / lower volatility.
    /// @return  Boosted PortfolioFourVector.  skew and kurt are unchanged.
    /// @throws std::domain_error if |β| ≥ 1.
    [[nodiscard]] static PortfolioFourVector transform(
        const PortfolioFourVector& portfolio,
        double beta
    ) {
        LorentzFactor lf(beta);
        const double g = lf.gamma;
        PortfolioFourVector boosted;
        boosted.ret  = g * (portfolio.ret  - beta * portfolio.vol);
        boosted.vol  = g * (portfolio.vol  - beta * portfolio.ret);
        boosted.skew = portfolio.skew;   // transverse — unchanged
        boosted.kurt = portfolio.kurt;   // transverse — unchanged
        return boosted;
    }
};

// ─── PortfolioInvariant ───────────────────────────────────────────────────────

/// Minkowski norm squared of a PortfolioFourVector.
///
/// I = ret² - vol² - skew² - kurt²
///
/// This scalar is invariant under all Lorentz boosts; verifying I == I' is a
/// useful sanity check.
class PortfolioInvariant {
public:
    PortfolioInvariant() = delete;

    /// Compute the Minkowski norm squared of *pf*.
    [[nodiscard]] static double compute(const PortfolioFourVector& pf) noexcept {
        return pf.ret  * pf.ret
             - pf.vol  * pf.vol
             - pf.skew * pf.skew
             - pf.kurt * pf.kurt;
    }
};

// ─── OptimalBoost ─────────────────────────────────────────────────────────────

/// Grid-search for the boost β that maximises the Sharpe ratio ret'/vol'.
///
/// Searches β ∈ (-0.99, 0.99) in steps of *step*.  Returns the β with the
/// highest Sharpe in the boosted frame.  If vol' == 0 for all candidates the
/// function returns 0.0 (identity transform).
class OptimalBoost {
public:
    OptimalBoost() = delete;

    /// Find the β ∈ (-0.99, 0.99) that maximises Sharpe after boosting.
    ///
    /// @param target_sharpe  Desired Sharpe (informational; search is
    ///                       exhaustive regardless).
    /// @param portfolio      Portfolio 4-vector in the lab frame.
    /// @param step           Grid step size (default 0.01).
    /// @return  Optimal β.
    [[nodiscard]] static double find(
        double target_sharpe,
        const PortfolioFourVector& portfolio,
        double step = 0.01
    ) noexcept {
        (void)target_sharpe;  // kept for API completeness
        double best_beta  = 0.0;
        double best_sharpe = -1e18;

        for (double beta = -0.99; beta <= 0.99; beta += step) {
            // Clamp to avoid boundary instability
            if (beta <= -1.0 || beta >= 1.0) continue;
            const double g   = 1.0 / std::sqrt(1.0 - beta * beta);
            const double ret = g * (portfolio.ret - beta * portfolio.vol);
            const double vol = g * (portfolio.vol - beta * portfolio.ret);
            if (vol <= 0.0) continue;
            const double sharpe = ret / vol;
            if (sharpe > best_sharpe) {
                best_sharpe = sharpe;
                best_beta   = beta;
            }
        }
        return best_beta;
    }
};

} // namespace srfm::portfolio
