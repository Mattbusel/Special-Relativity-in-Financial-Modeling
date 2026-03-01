#pragma once
// Forward declaration: SIMD acceleration module needs friend access to
// construct LorentzFactor objects from pre-computed scalar values without
// an extra scalar sqrt per element.  This is a purely internal trust
// boundary; the public API is unchanged.
namespace srfm::simd { struct SimdGammaCompute; }

/**
 * @file  momentum.hpp
 * @brief Momentum-Velocity Signal Processor  (AGT-03 / SRFM)
 *
 * Module:  src/momentum/
 * Owner:   AGT-03  (Builder)  —  2026-02-28
 *
 * Responsibility
 * --------------
 * Compute relativistic momentum signals for financial time-series:
 *
 *   p_rel = γ · m_eff · v_market
 *
 * where
 *   β  (BetaVelocity)  = normalised market velocity,  |β| ∈ [0, BETA_MAX_SAFE)
 *   γ  (LorentzFactor) = 1/√(1−β²)  ≥ 1
 *   m_eff (EffectiveMass) = ADV / ADV_baseline  (liquidity proxy)
 *   v_market = raw signal value
 *
 * Higher liquidity → higher m_eff → momentum harder to shift; this mirrors
 * the relativistic mechanic where higher rest-mass resists acceleration.
 *
 * Design Constraints
 * ------------------
 *   • Zero raw pointers in the public API.
 *   • All fallible operations return std::optional (no exceptions thrown).
 *   • All public methods are noexcept.
 *   • Thread-safe: RelativisticSignalProcessor is stateless.
 *
 * NOT Responsible For
 * -------------------
 *   • Sourcing ADV data        (caller provides EffectiveMass)
 *   • Signal persistence       (stateless)
 *   • Cross-asset normalisation
 *
 * Note on upstream headers
 * ------------------------
 * AGT-01 (src/lorentz/) and AGT-02 (src/manifold/) had not yet landed in
 * this repository at the time AGT-03 shipped.  The Lorentz primitives are
 * therefore defined here.  When AGT-01's header is available, replace the
 * BetaVelocity / LorentzFactor definitions with an #include of that header.
 */

#include <cmath>
#include <optional>
#include <span>
#include <vector>

namespace srfm::momentum {

// ── Constants ─────────────────────────────────────────────────────────────────

/// Upper bound for safe β values.  At β = 1.0 γ → ∞; values at or above
/// this threshold are rejected to keep all arithmetic finite.
inline constexpr double BETA_MAX_SAFE = 0.9999;

// ── BetaVelocity ──────────────────────────────────────────────────────────────

/**
 * @brief Normalised market velocity β = price_velocity / c_market.
 *
 * Invariant: std::isfinite(value()) && std::abs(value()) < BETA_MAX_SAFE.
 * Construct exclusively via BetaVelocity::make().
 */
class BetaVelocity {
public:
    /**
     * @brief Validate and construct a BetaVelocity.
     * @param value  Candidate β value.
     * @return std::nullopt when |value| ≥ BETA_MAX_SAFE or value is non-finite.
     */
    [[nodiscard]] static std::optional<BetaVelocity>
    make(double value) noexcept;

    /// Returns the raw β value.
    [[nodiscard]] double value() const noexcept { return value_; }

private:
    explicit BetaVelocity(double v) noexcept : value_{v} {}
    double value_{0.0};
};

// ── LorentzFactor ─────────────────────────────────────────────────────────────

/**
 * @brief Pre-computed Lorentz factor γ = 1/√(1−β²).  Always ≥ 1.0.
 *
 * Obtained exclusively from lorentz_gamma(BetaVelocity).
 * Default value is 1.0 (the Newtonian limit at β = 0).
 */
class LorentzFactor {
public:
    /// Default-constructs to the Newtonian identity γ = 1.
    LorentzFactor() noexcept : value_{1.0} {}

    /// Returns the raw γ value.
    [[nodiscard]] double value() const noexcept { return value_; }

private:
    friend std::optional<LorentzFactor>
    lorentz_gamma(BetaVelocity beta) noexcept;

    /// Internal SIMD module — constructs LorentzFactor from a pre-validated
    /// gamma value without re-computing sqrt.  Caller guarantees v >= 1.0 and
    /// std::isfinite(v).
    friend struct srfm::simd::SimdGammaCompute;

    explicit LorentzFactor(double v) noexcept : value_{v} {}
    double value_{1.0};
};

// ── EffectiveMass ─────────────────────────────────────────────────────────────

/**
 * @brief ADV-based effective mass proxy.
 *
 * m_eff = adv / adv_baseline.
 * Invariant: value() > 0 and std::isfinite(value()).
 * Construct via EffectiveMass::make() or EffectiveMass::from_adv().
 */
class EffectiveMass {
public:
    /**
     * @brief Validate and construct an EffectiveMass.
     * @return std::nullopt when value ≤ 0 or non-finite.
     */
    [[nodiscard]] static std::optional<EffectiveMass>
    make(double value) noexcept;

    /**
     * @brief Construct from raw ADV and a baseline ADV.
     *
     * m_eff = adv / adv_baseline.
     * @return std::nullopt when either argument is non-positive or non-finite.
     */
    [[nodiscard]] static std::optional<EffectiveMass>
    from_adv(double adv, double adv_baseline) noexcept;

    /// Returns the raw m_eff value.
    [[nodiscard]] double value() const noexcept { return value_; }

private:
    explicit EffectiveMass(double v) noexcept : value_{v} {}
    double value_{1.0};
};

// ── Signal types ──────────────────────────────────────────────────────────────

/// A raw (pre-correction) market signal value.
struct RawSignal {
    double value{0.0};
};

/**
 * @brief A gamma-corrected relativistic momentum signal.
 *
 *   adjusted_value = γ · m_eff · raw_value
 */
struct RelativisticSignal {
    double       raw_value{0.0};
    LorentzFactor gamma{};            ///< Lorentz factor applied during correction.
    double       adjusted_value{0.0}; ///< γ · m_eff · raw_value
};

// ── Physics kernels ───────────────────────────────────────────────────────────

/**
 * @brief Compute Lorentz factor γ = 1/√(1−β²).
 *
 * @param beta  Validated normalised market velocity.
 * @return LorentzFactor with γ ≥ 1.0, or std::nullopt if result is
 *         non-finite (unreachable after BetaVelocity validation, but
 *         explicit for defence-in-depth).
 */
[[nodiscard]] std::optional<LorentzFactor>
lorentz_gamma(BetaVelocity beta) noexcept;

/**
 * @brief Apply relativistic momentum correction: p_rel = γ · m_eff · raw.
 *
 * Mirrors applyMomentumCorrection() from the SRFM C++ reference.
 *
 * @param raw_signal  Un-corrected market signal (any finite double).
 * @param beta        Validated normalised market velocity.
 * @param m_eff       ADV-based effective mass.
 * @return {adjusted_value, gamma} pair, or std::nullopt if γ fails.
 */
[[nodiscard]] std::optional<std::pair<double, LorentzFactor>>
apply_momentum_correction(double raw_signal,
                          BetaVelocity beta,
                          EffectiveMass m_eff) noexcept;

/**
 * @brief Relativistic velocity composition: β_result = (β₁+β₂)/(1+β₁β₂).
 *
 * Preserves the sub-luminal invariant.
 * @return std::nullopt if the composed result would be ≥ BETA_MAX_SAFE.
 */
[[nodiscard]] std::optional<BetaVelocity>
compose_velocities(BetaVelocity beta1, BetaVelocity beta2) noexcept;

/**
 * @brief Recover the proper (un-dilated) value: proper = dilated / γ.
 *
 * @return std::nullopt if γ computation fails.
 */
[[nodiscard]] std::optional<double>
inverse_transform(double dilated_value, BetaVelocity beta) noexcept;

// ── RelativisticSignalProcessor ───────────────────────────────────────────────

/**
 * @brief Converts raw market signals to gamma-corrected relativistic signals.
 *
 * Stateless.  All signals in a batch share the same beta / m_eff frame.
 * Thread-safe: process() and process_one() are const and noexcept.
 *
 * @example
 * @code
 *   auto beta  = BetaVelocity::make(0.6).value();
 *   auto m_eff = EffectiveMass::make(1.0).value();
 *   std::array<RawSignal,2> raw{{{100.0},{-50.0}}};
 *   RelativisticSignalProcessor proc;
 *   auto out = proc.process(raw, beta, m_eff);
 *   // out[0].adjusted_value ≈ 125.0 (γ=1.25, m_eff=1)
 * @endcode
 */
class RelativisticSignalProcessor {
public:
    RelativisticSignalProcessor() noexcept = default;

    /**
     * @brief Process a batch of raw signals.
     *
     * Gamma is computed once and reused for every signal in the batch.
     *
     * @param signals  Span of raw signal values (zero or more).
     * @param beta     Normalised market velocity for this processing frame.
     * @param m_eff    ADV-based effective mass for this frame.
     * @return Vector of RelativisticSignal (one per input), in order.
     *         Returns std::nullopt only if γ computation yields a non-finite
     *         result (unreachable with a valid BetaVelocity).
     */
    [[nodiscard]] std::optional<std::vector<RelativisticSignal>>
    process(std::span<const RawSignal> signals,
            BetaVelocity beta,
            EffectiveMass m_eff) const noexcept;

    /**
     * @brief Process a single raw signal (convenience wrapper).
     *
     * @return RelativisticSignal, or std::nullopt if γ fails.
     */
    [[nodiscard]] std::optional<RelativisticSignal>
    process_one(RawSignal signal,
                BetaVelocity beta,
                EffectiveMass m_eff) const noexcept;
};

} // namespace srfm::momentum
