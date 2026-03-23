/**
 * @file  src/minkowski_momentum.cpp
 * @brief Implementation of the Minkowski Momentum module — Round 6.
 *
 * See include/srfm/minkowski_momentum.hpp for the full API contract,
 * concept description, and formula derivations.
 */

#include "srfm/minkowski_momentum.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <span>
#include <vector>

namespace srfm::minkowski_momentum {

// ─── MinkowskiMomentum ────────────────────────────────────────────────────────

double MinkowskiMomentum::invariant_mass_sq(const FourMomentum& p) noexcept {
    return p.energy * p.energy
         - p.px * p.px
         - p.py * p.py
         - p.pz * p.pz;
}

std::optional<double>
MinkowskiMomentum::invariant_mass(const FourMomentum& p) noexcept {
    if (!std::isfinite(p.energy) ||
        !std::isfinite(p.px)     ||
        !std::isfinite(p.py)     ||
        !std::isfinite(p.pz)) {
        return std::nullopt;
    }

    const double m2 = invariant_mass_sq(p);
    // Return signed sqrt: positive for time-like (m2 > 0), negative for space-like.
    if (m2 >= 0.0) {
        return std::sqrt(m2);
    } else {
        return -std::sqrt(-m2);
    }
}

std::optional<double>
MinkowskiMomentum::rapidity(const FourMomentum& p) noexcept {
    if (!std::isfinite(p.energy) || !std::isfinite(p.px)) {
        return std::nullopt;
    }

    const double num = p.energy + p.px;
    const double den = p.energy - p.px;

    if (den <= 0.0 || num <= 0.0) {
        // Rapidity undefined when |p_x| >= E
        return std::nullopt;
    }

    return 0.5 * std::log(num / den);
}

double MinkowskiMomentum::transverse_momentum(const FourMomentum& p) noexcept {
    return std::sqrt(p.py * p.py + p.pz * p.pz);
}

double MinkowskiMomentum::spatial_magnitude(const FourMomentum& p) noexcept {
    return std::sqrt(p.px * p.px + p.py * p.py + p.pz * p.pz);
}

// ─── FourMomentumConservation ─────────────────────────────────────────────────

FourMomentum FourMomentumConservation::sum(
    std::span<const FourMomentum> momenta) noexcept
{
    FourMomentum result{0.0, 0.0, 0.0, 0.0};
    for (const auto& p : momenta) {
        result.energy += p.energy;
        result.px     += p.px;
        result.py     += p.py;
        result.pz     += p.pz;
    }
    return result;
}

bool FourMomentumConservation::conserves(
    std::span<const FourMomentum> trades,
    const FourMomentum& reference,
    double tolerance) noexcept
{
    const FourMomentum net = sum(trades);
    return std::abs(net.energy - reference.energy) <= tolerance &&
           std::abs(net.px     - reference.px)     <= tolerance &&
           std::abs(net.py     - reference.py)     <= tolerance &&
           std::abs(net.pz     - reference.pz)     <= tolerance;
}

// ─── MomentumPortfolioOptimizer ───────────────────────────────────────────────

FourMomentum MomentumPortfolioOptimizer::_portfolio_momentum(
    std::span<const double>              weights,
    std::span<const double>              returns,
    std::span<const std::array<double,3>> exposures) noexcept
{
    FourMomentum p{0.0, 0.0, 0.0, 0.0};
    const std::size_t n = weights.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double w = weights[i];
        p.energy += w * returns[i];
        p.px     += w * exposures[i][0];
        p.py     += w * exposures[i][1];
        p.pz     += w * exposures[i][2];
    }
    return p;
}

std::optional<MomentumPortfolioOptimizer::Result>
MomentumPortfolioOptimizer::optimize(
    std::span<const double>              returns,
    std::span<const std::array<double,3>> exposures,
    const Config& cfg) noexcept
{
    const std::size_t n = returns.size();
    if (n == 0 || n != exposures.size()) {
        return std::nullopt;
    }
    if (cfg.max_iterations <= 0 || cfg.learning_rate <= 0.0) {
        return std::nullopt;
    }

    // Initialise to equal weights.
    std::vector<double> w(n, 1.0 / static_cast<double>(n));

    auto clamp_and_normalise = [&]() {
        // Clamp each weight to [min_weight, max_weight].
        for (auto& wi : w) {
            wi = std::clamp(wi, cfg.min_weight, cfg.max_weight);
        }
        // Re-normalise to sum to 1.
        const double total = std::accumulate(w.begin(), w.end(), 0.0);
        if (total <= 0.0) {
            std::fill(w.begin(), w.end(), 1.0 / static_cast<double>(n));
        } else {
            for (auto& wi : w) wi /= total;
        }
    };

    clamp_and_normalise();

    double prev_m2 = MinkowskiMomentum::invariant_mass_sq(
        _portfolio_momentum(w, returns, exposures));

    int iters = 0;
    bool converged = false;

    for (int iter = 0; iter < cfg.max_iterations; ++iter) {
        ++iters;

        // Numerical gradient of m² with respect to each weight w_i.
        // Gradient_i ~ [m²(w + delta_i*e_i) - m²(w)] / delta_i
        // using central difference for accuracy.
        constexpr double DELTA = 1e-6;
        std::vector<double> grad(n, 0.0);

        for (std::size_t i = 0; i < n; ++i) {
            // Forward perturb
            w[i] += DELTA;
            const double m2_fwd = MinkowskiMomentum::invariant_mass_sq(
                _portfolio_momentum(w, returns, exposures));
            w[i] -= 2.0 * DELTA;
            const double m2_bwd = MinkowskiMomentum::invariant_mass_sq(
                _portfolio_momentum(w, returns, exposures));
            w[i] += DELTA;  // restore

            grad[i] = (m2_fwd - m2_bwd) / (2.0 * DELTA);
        }

        // Gradient ascent step.
        for (std::size_t i = 0; i < n; ++i) {
            w[i] += cfg.learning_rate * grad[i];
        }

        clamp_and_normalise();

        const double new_m2 = MinkowskiMomentum::invariant_mass_sq(
            _portfolio_momentum(w, returns, exposures));

        if (std::abs(new_m2 - prev_m2) < cfg.tolerance) {
            prev_m2 = new_m2;
            converged = true;
            break;
        }
        prev_m2 = new_m2;
    }

    return Result{
        std::move(w),
        prev_m2,
        iters,
        converged,
    };
}

}  // namespace srfm::minkowski_momentum
