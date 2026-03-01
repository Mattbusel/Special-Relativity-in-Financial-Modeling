/**
 * @file  momentum.cpp
 * @brief Implementation of the Momentum-Velocity Signal Processor (AGT-03).
 *
 * See momentum.hpp for the full module contract and physics basis.
 */

#include "momentum.hpp"

#include <cmath>
#include <utility>

namespace srfm::momentum {

// ── BetaVelocity ──────────────────────────────────────────────────────────────

std::optional<BetaVelocity>
BetaVelocity::make(double value) noexcept {
    if (!std::isfinite(value))       return std::nullopt;
    if (std::abs(value) >= BETA_MAX_SAFE) return std::nullopt;
    return BetaVelocity{value};
}

// ── EffectiveMass ─────────────────────────────────────────────────────────────

std::optional<EffectiveMass>
EffectiveMass::make(double value) noexcept {
    if (!std::isfinite(value) || value <= 0.0) return std::nullopt;
    return EffectiveMass{value};
}

std::optional<EffectiveMass>
EffectiveMass::from_adv(double adv, double adv_baseline) noexcept {
    if (!std::isfinite(adv)          || adv          <= 0.0) return std::nullopt;
    if (!std::isfinite(adv_baseline) || adv_baseline <= 0.0) return std::nullopt;
    return make(adv / adv_baseline);
}

// ── Physics kernels ───────────────────────────────────────────────────────────

std::optional<LorentzFactor>
lorentz_gamma(BetaVelocity beta) noexcept {
    const double b         = beta.value();
    const double gamma_val = 1.0 / std::sqrt(1.0 - b * b);
    if (!std::isfinite(gamma_val)) return std::nullopt;
    return LorentzFactor{gamma_val};
}

std::optional<std::pair<double, LorentzFactor>>
apply_momentum_correction(double       raw_signal,
                          BetaVelocity beta,
                          EffectiveMass m_eff) noexcept {
    auto gamma_opt = lorentz_gamma(beta);
    if (!gamma_opt) return std::nullopt;
    const double adjusted = gamma_opt->value() * m_eff.value() * raw_signal;
    return std::make_pair(adjusted, *gamma_opt);
}

std::optional<BetaVelocity>
compose_velocities(BetaVelocity beta1, BetaVelocity beta2) noexcept {
    const double b1       = beta1.value();
    const double b2       = beta2.value();
    const double composed = (b1 + b2) / (1.0 + b1 * b2);
    return BetaVelocity::make(composed);
}

std::optional<double>
inverse_transform(double dilated_value, BetaVelocity beta) noexcept {
    auto gamma_opt = lorentz_gamma(beta);
    if (!gamma_opt) return std::nullopt;
    return dilated_value / gamma_opt->value();
}

// ── RelativisticSignalProcessor ───────────────────────────────────────────────

std::optional<std::vector<RelativisticSignal>>
RelativisticSignalProcessor::process(
    std::span<const RawSignal> signals,
    BetaVelocity               beta,
    EffectiveMass              m_eff) const noexcept {

    // Compute γ once; reuse for every signal in the batch.
    auto gamma_opt = lorentz_gamma(beta);
    if (!gamma_opt) return std::nullopt;
    const LorentzFactor gamma = *gamma_opt;
    const double        scale = gamma.value() * m_eff.value();

    std::vector<RelativisticSignal> out;
    out.reserve(signals.size());
    for (const auto& sig : signals) {
        out.push_back(RelativisticSignal{
            sig.value,
            gamma,
            scale * sig.value
        });
    }
    return out;
}

std::optional<RelativisticSignal>
RelativisticSignalProcessor::process_one(
    RawSignal    signal,
    BetaVelocity beta,
    EffectiveMass m_eff) const noexcept {

    auto gamma_opt = lorentz_gamma(beta);
    if (!gamma_opt) return std::nullopt;
    const double adjusted = gamma_opt->value() * m_eff.value() * signal.value;
    return RelativisticSignal{signal.value, *gamma_opt, adjusted};
}

} // namespace srfm::momentum
