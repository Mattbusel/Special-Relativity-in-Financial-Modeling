#pragma once

/// @file include/srfm/proper_time.hpp
/// @brief Proper Time Portfolio module — Round 7 public API.
///
/// # Module: Proper Time Portfolio
///
/// ## Concept
/// In Special Relativity the *proper time* τ experienced by a moving observer
/// is less than the coordinate time t measured by a stationary observer:
///
///     τ = t / γ       where γ = 1 / √(1 − β²)
///
/// ## Financial Interpretation
/// A high-volatility ("fast-moving") portfolio is analogous to the moving
/// observer.  It takes longer to reach the same information state as a
/// low-volatility portfolio and therefore experiences "slower" proper time.
///
/// ### Mapping
/// - Coordinate time  t  → calendar time in years / trading days
/// - Velocity         v  = portfolio_vol / c   (β ∈ [0, 1))
/// - Speed of light   c  = max_vol (configurable, default 0.50 = 50 % ann.)
/// - Lorentz factor   γ  = 1 / √(1 − β²)
/// - Proper time      τ  = t / γ
///
/// ## Key Interpretation
/// - γ > 1 for any v > 0:   high-vol portfolios "age faster" in coordinate
///   time — each calendar day has a larger impact on their information state.
/// - adj_sharpe = sharpe / √(effective_age)  where effective_age = t · γ
///
/// ## Classes
/// - ProperTime              — value type (τ, γ, v)
/// - ProperTimeClock         — integrates dτ = dt / γ(v) over streaming vol obs.
/// - PortfolioAgingModel     — computes effective age and adjusted Sharpe
/// - RelativisticRebalanceTimer — fires when accumulated Δτ > threshold
///
/// ## Guarantees
/// - No dynamic allocation in the hot path
/// - All methods noexcept where indicated
/// - Thread-compatible (external locking required for concurrent mutations)

#include "srfm/constants.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace srfm::proper_time {

// ─── Constants ────────────────────────────────────────────────────────────────

/// Default maximum portfolio volatility (annualised), analogous to c.
inline constexpr double DEFAULT_MAX_VOL = 0.50;

// ─── ProperTime ───────────────────────────────────────────────────────────────

/// Value type returned by ProperTime::compute.
struct ProperTimeValue {
    double tau      = 0.0;  ///< Elapsed proper time
    double gamma    = 1.0;  ///< Lorentz factor γ = 1/√(1−β²)
    double velocity = 0.0;  ///< Normalised velocity β = vol / max_vol
};

/// Stateless computation of proper-time quantities.
struct ProperTime {
    /// Compute ProperTimeValue for a portfolio with given coordinate time and
    /// velocity.
    ///
    /// @param coordinate_time  Elapsed coordinate time (years or any unit).
    /// @param velocity         Normalised velocity β = portfolio_vol / max_vol,
    ///                         clamped to [0, BETA_MAX_SAFE].
    /// @returns                ProperTimeValue with tau = t / γ.
    [[nodiscard]] static ProperTimeValue compute(
        double coordinate_time,
        double velocity) noexcept;

    /// Compute Lorentz factor γ = 1 / √(1 − β²) for a given velocity.
    ///
    /// @param velocity  Normalised velocity β ∈ [0, 1).
    [[nodiscard]] static double gamma_factor(double velocity) noexcept;

    /// Convert a raw portfolio volatility to normalised velocity β.
    ///
    /// @param portfolio_vol  Annualised volatility (e.g. 0.20 = 20 %).
    /// @param max_vol        Speed-of-light analogue (default 0.50).
    [[nodiscard]] static double to_velocity(
        double portfolio_vol,
        double max_vol = DEFAULT_MAX_VOL) noexcept;
};

// ─── ProperTimeClock ──────────────────────────────────────────────────────────

/// Integrates proper time τ = ∑ dt / γ(v(t)) over a sequence of volatility
/// observations.
///
/// Each call to tick() advances the clock by dτ = dt_days / γ(v), where v is
/// derived from the supplied volatility.
class ProperTimeClock {
public:
    /// @param max_vol  Volatility cap analogous to the speed of light (default
    ///                 DEFAULT_MAX_VOL).
    explicit ProperTimeClock(double max_vol = DEFAULT_MAX_VOL);

    /// Advance the clock by one observation.
    ///
    /// @param new_vol   Current annualised portfolio volatility.
    /// @param dt_days   Calendar days elapsed since the last tick (default 1).
    void tick(double new_vol, double dt_days = 1.0) noexcept;

    /// Total elapsed proper time in days.
    [[nodiscard]] double elapsed_proper_time() const noexcept {
        return elapsed_tau_;
    }

    /// Total elapsed coordinate (calendar) time in days.
    [[nodiscard]] double elapsed_coordinate_time() const noexcept {
        return elapsed_t_;
    }

    /// Current Lorentz factor (from the most recent tick).
    [[nodiscard]] double current_gamma() const noexcept {
        return current_gamma_;
    }

    /// Current normalised velocity β.
    [[nodiscard]] double current_velocity() const noexcept {
        return current_beta_;
    }

    /// Maximum volatility cap.
    [[nodiscard]] double max_vol() const noexcept { return max_vol_; }

    /// Number of ticks accumulated.
    [[nodiscard]] std::size_t tick_count() const noexcept { return tick_count_; }

    /// Reset all accumulators.
    void reset() noexcept;

private:
    double max_vol_;
    double elapsed_tau_  = 0.0;
    double elapsed_t_    = 0.0;
    double current_gamma_ = 1.0;
    double current_beta_  = 0.0;
    std::size_t tick_count_ = 0;
};

// ─── PortfolioAgingModel ──────────────────────────────────────────────────────

/// Computes the "effective age" of a portfolio accounting for relativistic
/// time dilation, and produces a Lorentz-adjusted Sharpe ratio.
///
/// A high-volatility portfolio "ages faster" in coordinate time:
///
///     effective_age = coordinate_age * γ
///     adj_sharpe    = sharpe / √(effective_age)
///
/// The adjustment penalises strategies that appear attractive only because
/// they have not been exposed to enough effective information time.
class PortfolioAgingModel {
public:
    /// @param max_vol  Volatility speed-of-light cap.
    explicit PortfolioAgingModel(double max_vol = DEFAULT_MAX_VOL);

    /// Result of one aging computation.
    struct AgingResult {
        double coordinate_age;  ///< Raw age in the same units as input.
        double effective_age;   ///< coordinate_age * γ.
        double gamma;           ///< Lorentz factor.
        double adj_sharpe;      ///< sharpe / √(effective_age) or 0 if age <= 0.
    };

    /// Compute effective age and adjusted Sharpe.
    ///
    /// @param coordinate_age  Calendar age of the strategy (years or days).
    /// @param portfolio_vol   Current annualised volatility.
    /// @param sharpe          Raw (unadjusted) Sharpe ratio.
    [[nodiscard]] AgingResult compute(
        double coordinate_age,
        double portfolio_vol,
        double sharpe) const noexcept;

    [[nodiscard]] double max_vol() const noexcept { return max_vol_; }

private:
    double max_vol_;
};

// ─── RelativisticRebalanceTimer ───────────────────────────────────────────────

/// Suggests rebalancing when accumulated proper time since the last rebalance
/// exceeds a threshold.  This avoids over-trading in high-volatility regimes
/// where each calendar day has a lower informational content (slow proper time).
///
/// In high-vol regimes γ > 1 → dτ = dt / γ < dt, so the timer fires less
/// frequently in calendar time — correctly reducing turnover when it matters
/// most.
class RelativisticRebalanceTimer {
public:
    /// @param threshold_tau  Proper-time threshold in days before suggesting a
    ///                        rebalance.  Default: 5.0 days.
    /// @param max_vol        Volatility cap.
    explicit RelativisticRebalanceTimer(
        double threshold_tau = 5.0,
        double max_vol       = DEFAULT_MAX_VOL);

    /// Advance the timer by one observation.
    ///
    /// @param new_vol  Current annualised volatility.
    /// @param dt_days  Calendar days elapsed.
    /// @returns        true if a rebalance is recommended.
    bool tick(double new_vol, double dt_days = 1.0) noexcept;

    /// Accumulated proper time since the last rebalance.
    [[nodiscard]] double tau_since_rebalance() const noexcept {
        return tau_since_rebalance_;
    }

    /// Total number of rebalance events fired.
    [[nodiscard]] std::size_t rebalance_count() const noexcept {
        return rebalance_count_;
    }

    /// Threshold for rebalance trigger.
    [[nodiscard]] double threshold_tau() const noexcept { return threshold_tau_; }

    /// Reset timer (does not reset rebalance_count).
    void reset_timer() noexcept;

private:
    double threshold_tau_;
    double max_vol_;
    double tau_since_rebalance_ = 0.0;
    std::size_t rebalance_count_ = 0;
    ProperTimeClock clock_;
};

} // namespace srfm::proper_time
