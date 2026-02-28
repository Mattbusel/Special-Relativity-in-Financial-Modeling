#pragma once

/// @file src/lorentz/beta_calculator.hpp
/// @brief BetaCalculator — financial-to-physics velocity mapping (AGT-01).
///
/// # Module: Beta Calculator
///
/// ## Responsibility
/// Maps raw financial market observables (price time-series, returns, trading
/// velocity) to the normalised velocity parameter β used throughout SRFM.
///
/// This is the entry point for all financial data entering the Lorentz engine.
/// Every other SRFM module operates on β-values; this module is the only one
/// that knows about raw prices, returns, and time deltas.
///
/// ## Core Formula
/// ```
/// β_market = price_velocity / max_observed_velocity
///          = (dP/dt over window) / max_velocity
/// ```
///
/// ## Design Principle
/// The market analog of c (speed of light) is `max_velocity` — the fastest
/// sustained price movement observed or deemed possible. Setting max_velocity
/// too small risks β > 1 (superluminal, physically nonsensical); too large
/// pushes everything into the Newtonian regime where γ ≈ 1.
///
/// ## Guarantees
/// - All methods are noexcept static pure functions
/// - Returned BetaVelocity values are always within the safe range [0, BETA_MAX_SAFE)
///   or the method returns std::nullopt
/// - No dynamic allocation — no heap usage on any path
///
/// ## NOT Responsible For
/// - Applying transforms to signals (see lorentz_transform.hpp)
/// - Loading price data from disk or network (see src/core/data_loader.hpp)

#include "srfm/types.hpp"
#include "srfm/constants.hpp"

#include <optional>
#include <span>

namespace srfm::lorentz {

/// Maps financial market observables to the β velocity parameter.
///
/// All methods are static. BetaCalculator holds no state — it is a
/// transformation namespace in class form.
class BetaCalculator {
public:
    BetaCalculator() = delete;

    // ── Primary Constructors ──────────────────────────────────────────────────

    /// Compute β = |price_velocity| / max_velocity.
    ///
    /// The primary factory: given a computed price velocity (dP/dt) and the
    /// maximum reference velocity, returns the normalised β.
    ///
    /// # Arguments
    /// * `price_velocity` — dP/dt (any finite double, sign preserved for direction)
    /// * `max_velocity`   — Reference maximum velocity (must be > 0)
    ///
    /// # Returns
    /// - `Some(β)` clamped to [0, BETA_MAX_SAFE) if max_velocity > 0
    /// - `None`    if max_velocity ≤ 0 or price_velocity is non-finite
    [[nodiscard]] static std::optional<BetaVelocity>
    fromPriceVelocity(double price_velocity, double max_velocity) noexcept;

    /// Compute β from a single-period percent return and a maximum reference.
    ///
    /// β = |return| / max_return
    ///
    /// Suitable for daily/intraday return data. Negative returns become
    /// positive β (speed is always non-negative; direction is separate).
    ///
    /// # Returns
    /// - `Some(β)` ∈ [0, BETA_MAX_SAFE)
    /// - `None`    if max_return ≤ 0 or return is non-finite
    [[nodiscard]] static std::optional<BetaVelocity>
    fromReturn(double period_return, double max_return) noexcept;

    /// Compute β from a contiguous price window using central differencing.
    ///
    /// Estimates dP/dt over `window` prices with constant `time_delta` between
    /// samples, then normalises by max_velocity to get β.
    ///
    /// Uses the mean absolute velocity over the window to smooth noise.
    ///
    /// # Arguments
    /// * `prices`       — Price time series (must have at least 2 elements)
    /// * `window`       — Number of most-recent prices to include (≤ prices.size())
    /// * `max_velocity` — Reference maximum velocity (must be > 0)
    /// * `time_delta`   — Time between successive prices (must be > 0)
    ///
    /// # Returns
    /// - `Some(β)` normalised mean rolling velocity
    /// - `None`    if inputs are invalid (empty, window < 2, non-finite data)
    [[nodiscard]] static std::optional<BetaVelocity>
    fromRollingWindow(std::span<const double> prices,
                      std::size_t             window,
                      double                  max_velocity,
                      double                  time_delta) noexcept;

    // ── Velocity Estimation ───────────────────────────────────────────────────

    /// Estimate price velocity dP/dt using central finite differences.
    ///
    /// For a series p₀…pₙ₋₁ with constant step time_delta:
    ///   v_i = (p_{i+1} − p_{i-1}) / (2·time_delta)   for interior points
    ///   v_0 = (p_1 − p_0) / time_delta                for the left boundary
    ///   v_n = (pₙ − pₙ₋₁) / time_delta               for the right boundary
    ///
    /// Returns the mean absolute velocity over the series.
    ///
    /// # Returns
    /// - `Some(v)` ≥ 0
    /// - `None`    if prices has < 2 elements or time_delta ≤ 0
    [[nodiscard]] static std::optional<double>
    meanAbsVelocity(std::span<const double> prices, double time_delta) noexcept;

    // ── Classification ────────────────────────────────────────────────────────

    /// Return true if β is in the Newtonian regime (|β| < BETA_NEWTONIAN_THRESHOLD).
    ///
    /// In the Newtonian regime γ ≈ 1 + β²/2 — relativistic corrections
    /// are negligible (less than 0.5%). Classical indicators apply directly.
    [[nodiscard]] static bool isNewtonian(BetaVelocity beta) noexcept;

    /// Return true if β is in the relativistic regime (|β| ≥ BETA_NEWTONIAN_THRESHOLD).
    ///
    /// Relativistic corrections are significant; γ departs from 1 by more
    /// than ~0.5%. Lorentz-corrected indicators must be used.
    [[nodiscard]] static bool isRelativistic(BetaVelocity beta) noexcept;

    /// Return true if β is in the valid safe range (|β| < BETA_MAX_SAFE).
    [[nodiscard]] static bool isValid(BetaVelocity beta) noexcept;

    // ── Utility ───────────────────────────────────────────────────────────────

    /// Clamp an arbitrary raw_beta to the safe range (−BETA_MAX_SAFE, BETA_MAX_SAFE).
    ///
    /// Use this as a safety net when β is computed from noisy data that might
    /// occasionally exceed 1. Does not return nullopt — always produces a valid β.
    [[nodiscard]] static BetaVelocity clamp(double raw_beta) noexcept;

    /// Relativistic kinetic energy analog: E_k = (γ − 1) · m_eff · c²_market.
    ///
    /// The "excess energy" above the rest-frame baseline, representing the
    /// additional energy a market agent would need to sustain a velocity β.
    ///
    /// # Arguments
    /// * `beta`           — Market velocity
    /// * `effective_mass` — Liquidity proxy (must be > 0)
    /// * `c_market`       — Speed of information (default = SPEED_OF_INFORMATION)
    ///
    /// # Returns
    /// - `Some(E_k)` ≥ 0
    /// - `None`      if effective_mass ≤ 0 or β is invalid
    [[nodiscard]] static std::optional<double>
    kineticEnergy(BetaVelocity beta,
                  double effective_mass,
                  double c_market = constants::SPEED_OF_INFORMATION) noexcept;

    /// Relativistic Doppler factor: D = √((1 + β) / (1 − β)).
    ///
    /// Models the frequency shift of a signal emitted by a moving market.
    /// D > 1: observer sees higher frequency (market approaching — momentum).
    /// D < 1: observer sees lower frequency (market receding — mean-reversion).
    ///
    /// # Returns
    /// - `Some(D)` > 0
    /// - `None`    if β is invalid or |β| ≥ 1 (D would be undefined)
    [[nodiscard]] static std::optional<double>
    dopplerFactor(BetaVelocity beta) noexcept;
};

} // namespace srfm::lorentz
