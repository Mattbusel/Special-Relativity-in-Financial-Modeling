/// @file src/lorentz/lorentz_transform.cpp
/// @brief Lorentz Transform Engine — AGT-01 implementation.

#include "lorentz_transform.hpp"

#include <cmath>

namespace srfm::lorentz {

// ─── Validation ───────────────────────────────────────────────────────────────

bool LorentzTransform::isValidBeta(double beta) noexcept {
    // Must be finite and strictly within the safe subluminal bound.
    return std::isfinite(beta) && std::abs(beta) < constants::BETA_MAX_SAFE;
}

// ─── Core Transforms ──────────────────────────────────────────────────────────

std::optional<LorentzFactor>
LorentzTransform::gamma(BetaVelocity beta) noexcept {
    if (!isValidBeta(beta.value)) {
        return std::nullopt;
    }

    // γ = 1 / √(1 − β²)
    // With BETA_MAX_SAFE = 0.9999, the denominator is ≥ √(1 − 0.9999²) ≈ 0.014
    // — no risk of division by zero.
    const double beta2 = beta.value * beta.value;
    const double denom = std::sqrt(1.0 - beta2);

    if (denom <= 0.0) {
        // Should never reach here given isValidBeta, but guard defensively.
        return std::nullopt;
    }

    return LorentzFactor{1.0 / denom};
}

std::optional<double>
LorentzTransform::dilateTime(double proper_time,
                              BetaVelocity beta) noexcept {
    // Proper time must be non-negative (signal cannot have negative age).
    if (proper_time < 0.0) {
        return std::nullopt;
    }

    auto g = gamma(beta);
    if (!g) {
        return std::nullopt;
    }

    // t_dilated = γ · τ
    // γ ≥ 1, so dilated time is always ≥ proper time.
    return proper_time * g->value;
}

std::optional<RelativisticSignal>
LorentzTransform::applyMomentumCorrection(double raw_signal,
                                           BetaVelocity beta,
                                           double effective_mass) noexcept {
    // Effective mass is a liquidity proxy — must be strictly positive.
    if (effective_mass <= 0.0) {
        return std::nullopt;
    }

    auto g = gamma(beta);
    if (!g) {
        return std::nullopt;
    }

    // p_rel = γ · m_eff · raw_signal
    // Newtonian limit (β → 0, γ → 1): p_rel → m_eff · raw_signal (classical).
    const double adjusted = g->value * effective_mass * raw_signal;

    return RelativisticSignal{
        .raw_value      = raw_signal,
        .gamma          = *g,
        .adjusted_value = adjusted,
        .time           = {}  // caller sets timestamp if needed
    };
}

BetaVelocity
LorentzTransform::composeVelocities(BetaVelocity beta1,
                                     BetaVelocity beta2) noexcept {
    // Relativistic velocity addition: β₁₂ = (β₁ + β₂) / (1 + β₁β₂)
    //
    // Properties guaranteed by the formula:
    //  - |β₁₂| < 1 whenever |β₁| < 1 and |β₂| < 1
    //  - Commutative: β₁ ⊕ β₂ = β₂ ⊕ β₁
    //  - Identity element: β ⊕ 0 = β
    //
    // The denominator 1 + β₁β₂ is always > 0 when both inputs are < 1 in
    // magnitude, so there is no risk of division by zero here.
    const double num   = beta1.value + beta2.value;
    const double denom = 1.0 + beta1.value * beta2.value;
    return BetaVelocity{num / denom};
}

std::optional<double>
LorentzTransform::inverseTransform(double dilated_value,
                                    BetaVelocity beta) noexcept {
    auto g = gamma(beta);
    if (!g) {
        return std::nullopt;
    }

    // Inverse dilation: τ = t / γ
    // γ > 0 always (it's a positive real), so no division by zero.
    return dilated_value / g->value;
}

std::optional<double>
LorentzTransform::contractLength(double proper_length,
                                  BetaVelocity beta) noexcept {
    // Proper length must be strictly positive (a zero-length interval has no
    // physical meaning in this context).
    if (proper_length <= 0.0) {
        return std::nullopt;
    }

    auto g = gamma(beta);
    if (!g) {
        return std::nullopt;
    }

    // L = L₀ / γ
    // γ ≥ 1, so contracted length ≤ proper length. Always positive.
    return proper_length / g->value;
}

std::optional<double>
LorentzTransform::rapidity(BetaVelocity beta) noexcept {
    if (!isValidBeta(beta.value)) {
        return std::nullopt;
    }

    // φ = atanh(β)
    //
    // atanh is defined on the open interval (−1, 1).
    // isValidBeta ensures |β| < BETA_MAX_SAFE < 1, so this is always safe.
    //
    // Key property: rapidity is additive under relativistic velocity addition:
    //   φ(β₁ ⊕ β₂) = φ(β₁) + φ(β₂)
    return std::atanh(beta.value);
}

std::optional<double>
LorentzTransform::totalEnergy(BetaVelocity beta,
                               double effective_mass,
                               double c_market) noexcept {
    if (effective_mass <= 0.0) {
        return std::nullopt;
    }

    auto g = gamma(beta);
    if (!g) {
        return std::nullopt;
    }

    // E = γ · m_eff · c²_market
    // Rest energy E₀ = m_eff · c²_market (at β = 0, γ = 1).
    // Kinetic energy E_k = E − E₀ = (γ − 1) · m_eff · c²_market.
    return g->value * effective_mass * c_market * c_market;
}

} // namespace srfm::lorentz
