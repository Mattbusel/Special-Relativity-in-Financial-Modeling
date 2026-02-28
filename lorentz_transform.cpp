#include "lorentz_transform.hpp"
#include <cmath>
#include <cassert>

namespace srfm::lorentz {

bool LorentzTransform::isValidBeta(double beta) noexcept {
    return std::abs(beta) < constants::BETA_MAX_SAFE && std::isfinite(beta);
}

std::optional<LorentzFactor>
LorentzTransform::gamma(BetaVelocity beta) noexcept {
    if (!isValidBeta(beta.value)) return std::nullopt;

    const double beta2 = beta.value * beta.value;
    const double denom = std::sqrt(1.0 - beta2);

    // denom should never be zero given BETA_MAX_SAFE < 1, but guard anyway
    if (denom <= 0.0) return std::nullopt;

    return LorentzFactor{1.0 / denom};
}

std::optional<double>
LorentzTransform::dilateTime(double proper_time,
                              BetaVelocity beta) noexcept {
    if (proper_time < 0.0) return std::nullopt;

    auto g = gamma(beta);
    if (!g) return std::nullopt;

    // Dilated time: t = γ * τ
    // Fast markets (high β) dilate signal time — signals age slower
    // relative to the market frame, making them heavier weights
    return proper_time * g->value;
}

std::optional<RelativisticSignal>
LorentzTransform::applyMomentumCorrection(double raw_signal,
                                           BetaVelocity beta,
                                           double effective_mass) noexcept {
    if (effective_mass <= 0.0) return std::nullopt;

    auto g = gamma(beta);
    if (!g) return std::nullopt;

    // Relativistic momentum analog: p = γ * m_eff * raw_signal
    // In the Newtonian limit (β → 0, γ → 1), reduces to classical momentum
    const double adjusted = g->value * effective_mass * raw_signal;

    return RelativisticSignal{
        .raw_value      = raw_signal,
        .gamma          = *g,
        .adjusted_value = adjusted,
        .time           = {}  // caller sets timestamp
    };
}

BetaVelocity
LorentzTransform::composeVelocities(BetaVelocity beta1,
                                     BetaVelocity beta2) noexcept {
    // Relativistic velocity addition: β_total = (β₁ + β₂) / (1 + β₁β₂)
    // Guarantees |β_total| < 1 when |β₁|, |β₂| < 1
    const double numerator   = beta1.value + beta2.value;
    const double denominator = 1.0 + beta1.value * beta2.value;

    // denominator is always > 0 when both betas are < 1
    return BetaVelocity{numerator / denominator};
}

std::optional<double>
LorentzTransform::inverseTransform(double dilated_value,
                                    BetaVelocity beta) noexcept {
    auto g = gamma(beta);
    if (!g || g->value == 0.0) return std::nullopt;

    // Inverse: recover proper value = dilated / γ
    return dilated_value / g->value;
}

} // namespace srfm::lorentz
