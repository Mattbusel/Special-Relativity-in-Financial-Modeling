/// @file src/momentum/momentum_processor.cpp
/// @brief MomentumProcessor — relativistic momentum corrections (AGT-06 stub).

#include "srfm/momentum.hpp"
#include "../lorentz/lorentz_transform.hpp"

#include <cmath>

namespace srfm::momentum {

// ─── MomentumProcessor::process ───────────────────────────────────────────────

std::optional<RelativisticMomentum>
MomentumProcessor::process(const MomentumSignal& signal) noexcept {
    if (signal.effective_mass <= 0.0) {
        return std::nullopt;
    }

    // Delegate γ computation to the Lorentz engine.
    auto rel = lorentz::LorentzTransform::applyMomentumCorrection(
        signal.raw_value, signal.beta, signal.effective_mass);
    if (!rel) {
        return std::nullopt;
    }

    return RelativisticMomentum{
        .raw_value      = rel->raw_value,
        .adjusted_value = rel->adjusted_value,
        .gamma          = rel->gamma,
        .beta           = signal.beta,
    };
}

// ─── MomentumProcessor::relativistic_momentum ────────────────────────────────

std::optional<double>
MomentumProcessor::relativistic_momentum(BetaVelocity beta,
                                          double       mass,
                                          double       speed) noexcept {
    if (mass <= 0.0 || !std::isfinite(speed)) {
        return std::nullopt;
    }

    auto g = lorentz::LorentzTransform::gamma(beta);
    if (!g) {
        return std::nullopt;
    }

    // p_rel = γ · m · |v|  (magnitude — speed is already absolute)
    return g->value * mass * std::abs(speed);
}

// ─── MomentumProcessor::process_series ───────────────────────────────────────

std::optional<std::vector<RelativisticMomentum>>
MomentumProcessor::process_series(
    std::span<const MomentumSignal> signals) noexcept {
    if (signals.empty()) {
        return std::nullopt;
    }

    std::vector<RelativisticMomentum> results;
    results.reserve(signals.size());

    for (const auto& sig : signals) {
        auto r = process(sig);
        if (r) {
            results.push_back(*r);
        } else {
            // Newtonian fallback: γ = 1, adjusted = raw
            results.push_back(RelativisticMomentum{
                .raw_value      = sig.raw_value,
                .adjusted_value = sig.raw_value,
                .gamma          = LorentzFactor{1.0},
                .beta           = sig.beta,
            });
        }
    }

    return results;
}

}  // namespace srfm::momentum
