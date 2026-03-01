/**
 * @file  beta_calculator.cpp
 * @brief Implementation of online BetaVelocity calculator (AGT-13 / SRFM).
 *
 * See beta_calculator.hpp for the full module contract.
 */

#include "beta_calculator.hpp"

#include <cmath>
#include <numeric>

namespace srfm::beta_calculator {

// ── rapidity ─────────────────────────────────────────────────────────────────

std::optional<double>
rapidity(BetaVelocity beta) noexcept {
    const double b = beta.value();
    // atanh domain: |b| < 1.  BetaVelocity guarantees |b| < BETA_MAX_SAFE < 1.
    const double phi = std::atanh(b);
    if (!std::isfinite(phi)) return std::nullopt;
    return phi;
}

// ── doppler_factor ────────────────────────────────────────────────────────────

std::optional<double>
doppler_factor(BetaVelocity beta) noexcept {
    const double b = beta.value();
    const double numerator   = 1.0 + b;
    const double denominator = 1.0 - b;
    // denominator > 0 because |b| < BETA_MAX_SAFE < 1
    if (denominator <= 0.0 || !std::isfinite(denominator)) return std::nullopt;
    const double ratio = numerator / denominator;
    if (ratio < 0.0 || !std::isfinite(ratio)) return std::nullopt;
    const double d = std::sqrt(ratio);
    if (!std::isfinite(d) || d <= 0.0) return std::nullopt;
    return d;
}

// ── full_beta_result ──────────────────────────────────────────────────────────

std::optional<BetaVelocityResult>
full_beta_result(double beta_value) noexcept {
    auto bv_opt = BetaVelocity::make(beta_value);
    if (!bv_opt) return std::nullopt;
    const BetaVelocity bv = *bv_opt;

    // Lorentz factor
    auto gamma_opt = momentum::lorentz_gamma(bv);
    if (!gamma_opt) return std::nullopt;

    // Rapidity
    auto phi_opt = rapidity(bv);
    if (!phi_opt) return std::nullopt;

    // Doppler
    auto d_opt = doppler_factor(bv);
    if (!d_opt) return std::nullopt;

    return BetaVelocityResult{
        bv.value(),
        gamma_opt->value(),
        *phi_opt,
        *d_opt
    };
}

// ── BetaCalculator ────────────────────────────────────────────────────────────

std::optional<BetaVelocityResult>
BetaCalculator::fromPriceVelocityOnline(
    const std::vector<double>& prices,
    double                     c_market) const noexcept {

    // Validate c_market
    if (!std::isfinite(c_market) || c_market <= 0.0) return std::nullopt;

    // Need at least 2 prices for 1 return
    if (prices.size() < 2) return std::nullopt;

    // Validate all prices
    for (const double p : prices) {
        if (!std::isfinite(p) || p <= 0.0) return std::nullopt;
    }

    // Compute log-return velocities: v_i = ln(p_{i+1} / p_i)
    const std::size_t n = prices.size() - 1u;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double ratio = prices[i + 1u] / prices[i];
        if (ratio <= 0.0 || !std::isfinite(ratio)) return std::nullopt;
        const double log_ret = std::log(ratio);
        if (!std::isfinite(log_ret)) return std::nullopt;
        sum += log_ret;
    }

    // Mean log-return velocity
    const double mean_velocity = sum / static_cast<double>(n);
    if (!std::isfinite(mean_velocity)) return std::nullopt;

    // Normalise: β = v̄ / c_market
    double beta_raw = mean_velocity / c_market;
    if (!std::isfinite(beta_raw)) return std::nullopt;

    // Clamp to safe sub-luminal range (never saturate, just cap)
    constexpr double CLAMP = momentum::BETA_MAX_SAFE - 1e-7;
    if (beta_raw >  CLAMP) beta_raw =  CLAMP;
    if (beta_raw < -CLAMP) beta_raw = -CLAMP;

    return full_beta_result(beta_raw);
}

} // namespace srfm::beta_calculator
