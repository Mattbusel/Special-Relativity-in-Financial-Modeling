/**
 * @file  src/proper_time.cpp
 * @brief Implementation of the Proper Time Portfolio module — Round 7.
 *
 * See include/srfm/proper_time.hpp for the full API contract and concept
 * description.
 */

#include "srfm/proper_time.hpp"
#include "srfm/constants.hpp"

#include <algorithm>
#include <cmath>

namespace srfm::proper_time {

// ─── ProperTime ───────────────────────────────────────────────────────────────

double ProperTime::gamma_factor(double velocity) noexcept {
    // Clamp to safe range to avoid numerical singularity
    const double beta = std::clamp(velocity, 0.0, constants::BETA_MAX_SAFE);
    const double beta_sq = beta * beta;
    return 1.0 / std::sqrt(1.0 - beta_sq);
}

double ProperTime::to_velocity(
    double portfolio_vol,
    double max_vol) noexcept
{
    if (max_vol <= 0.0) {
        return 0.0;
    }
    const double beta = portfolio_vol / max_vol;
    return std::clamp(beta, 0.0, constants::BETA_MAX_SAFE);
}

ProperTimeValue ProperTime::compute(
    double coordinate_time,
    double velocity) noexcept
{
    ProperTimeValue result;
    result.velocity = std::clamp(velocity, 0.0, constants::BETA_MAX_SAFE);
    result.gamma    = gamma_factor(result.velocity);
    // τ = t / γ  (proper time is always ≤ coordinate time for v > 0)
    result.tau      = coordinate_time / result.gamma;
    return result;
}

// ─── ProperTimeClock ──────────────────────────────────────────────────────────

ProperTimeClock::ProperTimeClock(double max_vol)
    : max_vol_(max_vol > 0.0 ? max_vol : DEFAULT_MAX_VOL)
{
}

void ProperTimeClock::tick(double new_vol, double dt_days) noexcept {
    if (dt_days <= 0.0) {
        return;
    }

    // Compute velocity β = vol / max_vol, clamped to [0, BETA_MAX_SAFE]
    current_beta_  = ProperTime::to_velocity(new_vol, max_vol_);
    current_gamma_ = ProperTime::gamma_factor(current_beta_);

    // Integrate: dτ = dt / γ
    const double dtau = dt_days / current_gamma_;

    elapsed_t_   += dt_days;
    elapsed_tau_ += dtau;
    ++tick_count_;
}

void ProperTimeClock::reset() noexcept {
    elapsed_tau_   = 0.0;
    elapsed_t_     = 0.0;
    current_gamma_ = 1.0;
    current_beta_  = 0.0;
    tick_count_    = 0;
}

// ─── PortfolioAgingModel ──────────────────────────────────────────────────────

PortfolioAgingModel::PortfolioAgingModel(double max_vol)
    : max_vol_(max_vol > 0.0 ? max_vol : DEFAULT_MAX_VOL)
{
}

PortfolioAgingModel::AgingResult PortfolioAgingModel::compute(
    double coordinate_age,
    double portfolio_vol,
    double sharpe) const noexcept
{
    AgingResult result;
    result.coordinate_age = coordinate_age;

    const double beta = ProperTime::to_velocity(portfolio_vol, max_vol_);
    result.gamma        = ProperTime::gamma_factor(beta);

    // Effective age: high-vol portfolios "age faster" in coordinate time
    result.effective_age = coordinate_age * result.gamma;

    // Adjusted Sharpe: penalise by √(effective_age)
    if (result.effective_age > 0.0) {
        result.adj_sharpe = sharpe / std::sqrt(result.effective_age);
    } else {
        result.adj_sharpe = 0.0;
    }

    return result;
}

// ─── RelativisticRebalanceTimer ───────────────────────────────────────────────

RelativisticRebalanceTimer::RelativisticRebalanceTimer(
    double threshold_tau,
    double max_vol)
    : threshold_tau_(threshold_tau > 0.0 ? threshold_tau : 5.0)
    , max_vol_(max_vol > 0.0 ? max_vol : DEFAULT_MAX_VOL)
    , clock_(max_vol_)
{
}

bool RelativisticRebalanceTimer::tick(double new_vol, double dt_days) noexcept {
    if (dt_days <= 0.0) {
        return false;
    }

    const double beta        = ProperTime::to_velocity(new_vol, max_vol_);
    const double gamma       = ProperTime::gamma_factor(beta);
    const double dtau        = dt_days / gamma;

    tau_since_rebalance_ += dtau;
    clock_.tick(new_vol, dt_days);

    if (tau_since_rebalance_ >= threshold_tau_) {
        tau_since_rebalance_ = 0.0;
        ++rebalance_count_;
        return true;
    }

    return false;
}

void RelativisticRebalanceTimer::reset_timer() noexcept {
    tau_since_rebalance_ = 0.0;
}

} // namespace srfm::proper_time
