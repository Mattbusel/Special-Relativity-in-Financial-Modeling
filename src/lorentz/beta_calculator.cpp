/// @file src/lorentz/beta_calculator.cpp
/// @brief BetaCalculator — financial-to-physics velocity mapping (AGT-01).

#include "beta_calculator.hpp"
#include "lorentz_transform.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace srfm::lorentz {

// ─── Internal helpers ─────────────────────────────────────────────────────────

namespace {

/// Clamp a raw ratio to [0, BETA_MAX_SAFE).
/// Negative raw_ratio is treated as positive (speed has no sign here).
[[nodiscard]] BetaVelocity ratio_to_beta(double raw_ratio) noexcept {
    // Take absolute value first: speed is magnitude-only.
    const double abs_ratio = std::abs(raw_ratio);
    // Clamp to just below BETA_MAX_SAFE so the result is always valid.
    const double clamped   = std::min(abs_ratio, constants::BETA_MAX_SAFE - 0.0);
    return BetaVelocity{std::min(clamped, constants::BETA_MAX_SAFE - 1e-15)};
}

} // anonymous namespace

// ─── fromPriceVelocity ────────────────────────────────────────────────────────

std::optional<BetaVelocity>
BetaCalculator::fromPriceVelocity(double price_velocity,
                                   double max_velocity) noexcept {
    if (!std::isfinite(price_velocity)) {
        return std::nullopt;
    }
    if (max_velocity <= 0.0) {
        return std::nullopt;
    }

    // β = |price_velocity| / max_velocity, clamped to [0, BETA_MAX_SAFE)
    const double raw_beta = std::abs(price_velocity) / max_velocity;
    return ratio_to_beta(raw_beta);
}

// ─── fromReturn ───────────────────────────────────────────────────────────────

std::optional<BetaVelocity>
BetaCalculator::fromReturn(double period_return,
                            double max_return) noexcept {
    if (!std::isfinite(period_return)) {
        return std::nullopt;
    }
    if (max_return <= 0.0) {
        return std::nullopt;
    }

    // β = |return| / max_return, clamped to safe range
    const double raw_beta = std::abs(period_return) / max_return;
    return ratio_to_beta(raw_beta);
}

// ─── meanAbsVelocity ─────────────────────────────────────────────────────────

std::optional<double>
BetaCalculator::meanAbsVelocity(std::span<const double> prices,
                                 double time_delta) noexcept {
    if (prices.size() < 2) {
        return std::nullopt;
    }
    if (time_delta <= 0.0) {
        return std::nullopt;
    }

    // Check all prices are finite before we start.
    for (double p : prices) {
        if (!std::isfinite(p)) {
            return std::nullopt;
        }
    }

    const std::size_t n = prices.size();
    double total_abs_v  = 0.0;
    std::size_t count   = 0;

    for (std::size_t i = 0; i < n; ++i) {
        double v{};
        if (i == 0) {
            // Forward difference at the left boundary
            v = (prices[1] - prices[0]) / time_delta;
        } else if (i == n - 1) {
            // Backward difference at the right boundary
            v = (prices[n - 1] - prices[n - 2]) / time_delta;
        } else {
            // Central difference at interior points (O(h²) accuracy)
            v = (prices[i + 1] - prices[i - 1]) / (2.0 * time_delta);
        }
        total_abs_v += std::abs(v);
        ++count;
    }

    return total_abs_v / static_cast<double>(count);
}

// ─── fromRollingWindow ────────────────────────────────────────────────────────

std::optional<BetaVelocity>
BetaCalculator::fromRollingWindow(std::span<const double> prices,
                                   std::size_t             window,
                                   double                  max_velocity,
                                   double                  time_delta) noexcept {
    if (window < 2) {
        return std::nullopt;
    }
    if (window > prices.size()) {
        return std::nullopt;
    }
    if (max_velocity <= 0.0) {
        return std::nullopt;
    }

    // Take the most-recent `window` prices.
    const std::size_t offset = prices.size() - window;
    auto recent = prices.subspan(offset, window);

    auto vel = meanAbsVelocity(recent, time_delta);
    if (!vel) {
        return std::nullopt;
    }

    return fromPriceVelocity(*vel, max_velocity);
}

// ─── fromPriceVelocityOnline ──────────────────────────────────────────────────

std::optional<std::vector<BetaVelocity>>
BetaCalculator::fromPriceVelocityOnline(std::span<const double> prices,
                                          double time_delta) noexcept {
    if (prices.size() < 2) {
        return std::nullopt;
    }
    if (time_delta <= 0.0) {
        return std::nullopt;
    }

    // Guard: all prices must be finite before we begin.
    for (double p : prices) {
        if (!std::isfinite(p)) {
            return std::nullopt;
        }
    }

    const std::size_t n = prices.size();
    std::vector<BetaVelocity> result;
    result.reserve(n);

    double running_max = 0.0;  // running maximum of absolute instantaneous velocity

    for (std::size_t i = 0; i < n; ++i) {
        // Instantaneous velocity at bar i using one-sided or central differences.
        // We use only prices[0..i] (i.e., no look-ahead).
        double v{};
        if (i == 0) {
            // Forward difference (only one future point available — still causal
            // because we treat this as the velocity between bar 0 and bar 1,
            // which the strategy knows at bar 1).  At bar 0 we can only estimate
            // from a single lag: use backward difference from bar 1 retroactively.
            // By convention, adopt the forward difference here since bar 1 is the
            // next observation we receive:
            v = std::abs((prices[1] - prices[0]) / time_delta);
        } else if (i == 1) {
            // With two points we can only use backward difference.
            v = std::abs((prices[1] - prices[0]) / time_delta);
        } else {
            // Central difference at bar i using prices[i-1] and prices[i].
            // (prices[i+1] would be look-ahead, so we use backward difference.)
            v = std::abs((prices[i] - prices[i - 1]) / time_delta);
        }

        // Update running max — monotonically non-decreasing.
        if (v > running_max) {
            running_max = v;
        }

        // Compute β_i using only the running_max up to bar i.
        BetaVelocity beta{};
        if (running_max < 1e-15) {
            // No velocity observed yet — Newtonian / stationary market.
            beta = BetaVelocity{0.0};
        } else {
            const double raw_beta = v / running_max;
            beta = ratio_to_beta(raw_beta);
        }

        result.push_back(beta);
    }

    return result;
}

// ─── Classification ───────────────────────────────────────────────────────────

bool BetaCalculator::isNewtonian(BetaVelocity beta) noexcept {
    return std::abs(beta.value) < constants::BETA_NEWTONIAN_THRESHOLD;
}

bool BetaCalculator::isRelativistic(BetaVelocity beta) noexcept {
    return !isNewtonian(beta);
}

bool BetaCalculator::isValid(BetaVelocity beta) noexcept {
    return LorentzTransform::isValidBeta(beta.value);
}

// ─── clamp ────────────────────────────────────────────────────────────────────

BetaVelocity BetaCalculator::clamp(double raw_beta) noexcept {
    // Preserve sign — market direction matters here (unlike speed-only contexts).
    const double limit  = constants::BETA_MAX_SAFE;
    const double clamped = std::clamp(raw_beta, -limit, limit);

    // If the input was exactly ±BETA_MAX_SAFE, nudge it just inside the boundary
    // so isValidBeta returns true for the result.
    if (clamped >= limit) {
        return BetaVelocity{limit - 1e-15};
    }
    if (clamped <= -limit) {
        return BetaVelocity{-limit + 1e-15};
    }
    return BetaVelocity{clamped};
}

// ─── kineticEnergy ────────────────────────────────────────────────────────────

std::optional<double>
BetaCalculator::kineticEnergy(BetaVelocity beta,
                               double effective_mass,
                               double c_market) noexcept {
    if (effective_mass <= 0.0) {
        return std::nullopt;
    }

    auto g = LorentzTransform::gamma(beta);
    if (!g) {
        return std::nullopt;
    }

    // E_k = (γ − 1) · m_eff · c²_market
    // At β = 0: E_k = 0 (no kinetic energy at rest).
    // Newtonian expansion: E_k ≈ ½ m_eff (β·c)² for small β (reproduces
    // classical ½mv²).
    return (g->value - 1.0) * effective_mass * c_market * c_market;
}

// ─── dopplerFactor ────────────────────────────────────────────────────────────

std::optional<double>
BetaCalculator::dopplerFactor(BetaVelocity beta) noexcept {
    if (!LorentzTransform::isValidBeta(beta.value)) {
        return std::nullopt;
    }

    // D = √((1 + β) / (1 − β))
    //
    // β > 0 (approaching): D > 1 — blue-shift (higher observed frequency)
    // β < 0 (receding):    D < 1 — red-shift  (lower observed frequency)
    // β = 0:               D = 1 — no shift
    //
    // isValidBeta guarantees |β| < BETA_MAX_SAFE < 1, so (1 − β) > 0 always.
    const double numerator   = 1.0 + beta.value;
    const double denominator = 1.0 - beta.value;

    // Extra guard: both factors must be positive for a real Doppler factor.
    if (denominator <= 0.0 || numerator <= 0.0) {
        return std::nullopt;
    }

    return std::sqrt(numerator / denominator);
}

} // namespace srfm::lorentz
