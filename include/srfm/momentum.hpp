#pragma once

/// @file include/srfm/momentum.hpp
/// @brief Momentum-Velocity Signal Processor — AGT-03 public API (implemented by AGT-06).
///
/// # Module: Momentum Processor
///
/// ## Responsibility
/// Apply relativistic momentum corrections to raw strategy signals.  In the
/// financial spacetime analogy, a signal generated during a high-β market regime
/// carries more "momentum" than the same signal in a quiet Newtonian market.
///
/// ## The Core Idea
/// Classical momentum: p = m·v
/// Relativistic momentum: p_rel = γ(β) · m_eff · v
///
/// Mapping to finance:
///   - v (velocity) → raw strategy signal value
///   - m_eff (effective mass) → liquidity proxy (e.g. ADV)
///   - β (velocity fraction) → normalised market velocity from BetaCalculator
///   - γ(β) → Lorentz factor ≥ 1; amplifies signals in fast-moving markets
///
/// ## Guarantees
/// - All methods are `noexcept` and return `std::optional` for fallible operations
/// - No dynamic allocation in hot path
/// - Thread-safe: MomentumProcessor is stateless

#include "srfm/types.hpp"
#include "srfm/constants.hpp"

#include <optional>
#include <span>
#include <vector>

namespace srfm::momentum {

// ─── MomentumSignal ───────────────────────────────────────────────────────────

/// Input descriptor for a single relativistic momentum computation.
struct MomentumSignal {
    double raw_value;       ///< Unmodified strategy signal (any finite double)
    BetaVelocity beta;      ///< Normalised market velocity β ∈ [0, BETA_MAX_SAFE)
    double effective_mass;  ///< Liquidity proxy m_eff > 0 (e.g. ADV normalised)
};

// ─── RelativisticMomentum ─────────────────────────────────────────────────────

/// Result of applying relativistic momentum correction to a single signal.
struct RelativisticMomentum {
    double raw_value;       ///< Original signal before correction
    double adjusted_value;  ///< γ · m_eff · raw_value
    LorentzFactor gamma;    ///< Lorentz factor used (≥ 1)
    BetaVelocity beta;      ///< Market velocity at the time of the signal
};

// ─── MomentumProcessor ────────────────────────────────────────────────────────

/// Stateless utility for applying relativistic momentum corrections.
///
/// The processor translates the BetaVelocity at each bar into a Lorentz factor
/// γ and returns the corrected signal p_rel = γ · m_eff · raw_signal.
class MomentumProcessor {
public:
    /// Apply relativistic momentum correction to a single signal.
    ///
    /// # Returns
    /// - `nullopt` if `effective_mass <= 0` or `beta` is invalid (|β| ≥ BETA_MAX_SAFE)
    /// - `RelativisticMomentum` with `adjusted_value = γ · m_eff · raw_value`
    [[nodiscard]] static std::optional<RelativisticMomentum>
    process(const MomentumSignal& signal) noexcept;

    /// Compute relativistic momentum magnitude: p_rel = γ(β) · mass · speed.
    ///
    /// # Arguments
    /// * `beta`  — Market velocity β
    /// * `mass`  — Effective mass m_eff > 0
    /// * `speed` — Speed component (|raw signal| or |price velocity|)
    ///
    /// # Returns
    /// p_rel ≥ 0, or `nullopt` on invalid inputs.
    [[nodiscard]] static std::optional<double>
    relativistic_momentum(BetaVelocity beta,
                          double mass,
                          double speed) noexcept;

    /// Apply corrections to a full series of signals in one call.
    ///
    /// Signals with invalid β fall back to the raw value (γ = 1 Newtonian limit).
    ///
    /// # Returns
    /// Vector of adjusted values, same length as `signals`.
    /// Returns `nullopt` if `signals` is empty.
    [[nodiscard]] static std::optional<std::vector<RelativisticMomentum>>
    process_series(std::span<const MomentumSignal> signals) noexcept;
};

}  // namespace srfm::momentum
