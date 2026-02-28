#pragma once

/// @file src/lorentz/lorentz_transform.hpp
/// @brief Lorentz Transform Engine — AGT-01 public header.
///
/// # Module: Lorentz Transform Engine
///
/// ## Responsibility
/// Implements the core special-relativistic transforms applied to financial
/// signal processing. The Lorentz factor γ = 1/√(1−β²) scales indicator
/// weights in fast-moving markets; time dilation stretches signal age;
/// relativistic momentum amplifies signal magnitude for high-β regimes.
///
/// ## Financial Interpretation
/// β_market = price_velocity / max_observed_velocity
///
/// At low β (slow, Newtonian market): γ ≈ 1 — no correction applied.
/// At high β (fast, relativistic market): γ >> 1 — signals are amplified
/// and time-weighted more heavily.
///
/// ## Guarantees
/// - All functions are noexcept — no exceptions escape
/// - All fallible operations return std::optional (no UB, no silent failures)
/// - Static methods only — no mutable state
/// - Thread-safe: all operations are pure functions of their inputs
///
/// ## NOT Responsible For
/// - Computing β from raw market data (see beta_calculator.hpp)
/// - Spacetime interval geometry (see src/manifold/)
/// - Full tensor field operations (see src/tensor/)

#include "srfm/types.hpp"
#include "srfm/constants.hpp"

#include <optional>

namespace srfm::lorentz {

/// Lorentz Transform Engine.
///
/// Static utility class providing all core special-relativistic transforms
/// expressed in terms of the normalised velocity parameter β.
class LorentzTransform {
public:
    LorentzTransform() = delete; // pure static — not instantiable

    // ── Validation ────────────────────────────────────────────────────────────

    /// Return true if β is finite and strictly within the safe range.
    ///
    /// Valid range: |β| < BETA_MAX_SAFE (= 0.9999).
    /// NaN, ±infinity, and |β| ≥ 1 are all invalid.
    [[nodiscard]] static bool isValidBeta(double beta) noexcept;

    // ── Core Transforms ───────────────────────────────────────────────────────

    /// Compute the Lorentz factor γ = 1 / √(1 − β²).
    ///
    /// At β = 0: γ = 1 (Newtonian limit — no relativistic correction).
    /// At β → 1: γ → ∞ (signals infinitely amplified in the market frame).
    ///
    /// # Returns
    /// - `Some(γ)` with γ ≥ 1.0 for valid β
    /// - `None` if β is invalid (|β| ≥ 1, NaN, or ±∞)
    [[nodiscard]] static std::optional<LorentzFactor>
    gamma(BetaVelocity beta) noexcept;

    /// Apply time dilation: t_dilated = γ · τ_proper.
    ///
    /// In the financial context: a signal's effective age is stretched by γ
    /// in a fast-moving market, making it appear more recent and more relevant.
    ///
    /// # Arguments
    /// * `proper_time` — Signal age in the market's rest frame (must be ≥ 0)
    /// * `beta`        — Normalised market velocity
    ///
    /// # Returns
    /// - `Some(t)` where t ≥ proper_time (dilation never compresses time)
    /// - `None` if proper_time < 0 or β is invalid
    [[nodiscard]] static std::optional<double>
    dilateTime(double proper_time, BetaVelocity beta) noexcept;

    /// Apply relativistic momentum correction: p = γ · m_eff · raw_signal.
    ///
    /// The relativistic momentum analog amplifies signals proportionally to γ.
    /// In the Newtonian limit (β → 0) this reduces to classical momentum
    /// p = m_eff · raw_signal.
    ///
    /// # Arguments
    /// * `raw_signal`      — Unscaled signal value (any finite double)
    /// * `beta`            — Normalised market velocity
    /// * `effective_mass`  — Liquidity-proxy mass parameter (must be > 0)
    ///
    /// # Returns
    /// - `Some(signal)` with adjusted_value = γ · m_eff · raw_signal
    /// - `None` if effective_mass ≤ 0 or β is invalid
    [[nodiscard]] static std::optional<RelativisticSignal>
    applyMomentumCorrection(double raw_signal,
                             BetaVelocity beta,
                             double effective_mass) noexcept;

    /// Relativistic velocity addition: β_total = (β₁ + β₂) / (1 + β₁β₂).
    ///
    /// Composes two market velocities according to the relativistic addition
    /// law. Guarantees |β_total| < 1 when |β₁|, |β₂| < 1, preserving the
    /// sub-luminal constraint even for large individual velocities.
    ///
    /// This is not approximate: it is the exact special-relativistic formula.
    ///
    /// # Arguments
    /// * `beta1`, `beta2` — Two market velocities to compose
    ///
    /// # Returns
    /// The composed velocity. Always sub-luminal if inputs are sub-luminal.
    [[nodiscard]] static BetaVelocity
    composeVelocities(BetaVelocity beta1, BetaVelocity beta2) noexcept;

    /// Recover the proper value from a dilated value: τ = t / γ.
    ///
    /// Inverse of `dilateTime`. Useful for converting a gamma-weighted
    /// indicator back to its raw frame value.
    ///
    /// # Returns
    /// - `Some(τ)` = dilated_value / γ
    /// - `None` if β is invalid
    [[nodiscard]] static std::optional<double>
    inverseTransform(double dilated_value, BetaVelocity beta) noexcept;

    /// Apply length contraction: L = L₀ / γ.
    ///
    /// In the financial analogy: the "length" of a price move (e.g. a spread
    /// or range) contracts in the observer frame when the market is moving.
    ///
    /// # Arguments
    /// * `proper_length` — Rest-frame length (must be > 0)
    /// * `beta`          — Normalised market velocity
    ///
    /// # Returns
    /// - `Some(L)` with 0 < L ≤ proper_length
    /// - `None` if proper_length ≤ 0 or β is invalid
    [[nodiscard]] static std::optional<double>
    contractLength(double proper_length, BetaVelocity beta) noexcept;

    /// Compute rapidity: φ = atanh(β).
    ///
    /// Rapidity is additive under velocity composition:
    ///   φ(β₁ ⊕ β₂) = φ(β₁) + φ(β₂)
    ///
    /// This makes rapidity the natural coordinate for combining market
    /// velocity signals from multiple assets.
    ///
    /// # Returns
    /// - `Some(φ)` ∈ (−∞, +∞)
    /// - `None` if β is invalid (|β| ≥ 1 makes atanh undefined)
    [[nodiscard]] static std::optional<double>
    rapidity(BetaVelocity beta) noexcept;

    /// Compute relativistic energy: E = γ · m_eff · c²_market.
    ///
    /// Total relativistic energy (rest + kinetic) in the financial frame.
    /// Rest energy E₀ = m_eff · c²_market (baseline liquidity × volatility).
    ///
    /// # Arguments
    /// * `beta`           — Normalised market velocity
    /// * `effective_mass` — Liquidity-proxy mass (must be > 0)
    /// * `c_market`       — Speed of information (default = SPEED_OF_INFORMATION)
    ///
    /// # Returns
    /// - `Some(E)` ≥ E₀
    /// - `None` if effective_mass ≤ 0 or β is invalid
    [[nodiscard]] static std::optional<double>
    totalEnergy(BetaVelocity beta,
                double effective_mass,
                double c_market = constants::SPEED_OF_INFORMATION) noexcept;
};

} // namespace srfm::lorentz
