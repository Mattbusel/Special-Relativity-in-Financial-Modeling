/**
 * @file  tests/portfolio/test_proper_time.cpp
 * @brief GTest unit tests for the ProperTime Portfolio module (Round 7).
 *
 * Covers ProperTime static helpers, ProperTimeClock integration,
 * PortfolioAgingModel, and RelativisticRebalanceTimer.
 */

#include <gtest/gtest.h>

#include "srfm/proper_time.hpp"
#include "srfm/constants.hpp"

#include <cmath>
#include <numeric>
#include <vector>

using namespace srfm::proper_time;

// ============================================================================
// ProperTime static helpers
// ============================================================================

// --- gamma_factor -----------------------------------------------------------

TEST(ProperTimeGamma, ZeroVelocityGammaIsOne) {
    EXPECT_DOUBLE_EQ(ProperTime::gamma_factor(0.0), 1.0);
}

TEST(ProperTimeGamma, HighVelocityGammaExceedsOne) {
    EXPECT_GT(ProperTime::gamma_factor(0.5), 1.0);
}

TEST(ProperTimeGamma, NegativeVelocityClampedToZero) {
    // Negative velocity should be clamped to 0; gamma = 1.0
    EXPECT_DOUBLE_EQ(ProperTime::gamma_factor(-0.5), 1.0);
}

TEST(ProperTimeGamma, VelocityAboveBetaMaxClampedSafely) {
    // Should not produce NaN or infinity
    const double g = ProperTime::gamma_factor(1.5);
    EXPECT_TRUE(std::isfinite(g));
    EXPECT_GT(g, 1.0);
}

TEST(ProperTimeGamma, FormulaCorrectness) {
    // γ = 1/√(1 - β²)
    const double beta = 0.6;
    const double expected = 1.0 / std::sqrt(1.0 - beta * beta);
    EXPECT_NEAR(ProperTime::gamma_factor(beta), expected, 1e-12);
}

// --- to_velocity ------------------------------------------------------------

TEST(ProperTimeVelocity, ZeroVolGivesZeroBeta) {
    EXPECT_DOUBLE_EQ(ProperTime::to_velocity(0.0, 0.5), 0.0);
}

TEST(ProperTimeVelocity, VolEqualsMaxGivesBetaMaxSafe) {
    const double beta = ProperTime::to_velocity(0.5, 0.5);
    EXPECT_LE(beta, srfm::constants::BETA_MAX_SAFE);
    EXPECT_NEAR(beta, srfm::constants::BETA_MAX_SAFE, 1e-9);
}

TEST(ProperTimeVelocity, VolHalfMaxGivesHalfBeta) {
    EXPECT_NEAR(ProperTime::to_velocity(0.25, 0.5), 0.5, 1e-12);
}

TEST(ProperTimeVelocity, ZeroMaxVolReturnsSafeZero) {
    // Edge case: max_vol = 0 should not crash
    const double beta = ProperTime::to_velocity(0.3, 0.0);
    EXPECT_DOUBLE_EQ(beta, 0.0);
}

TEST(ProperTimeVelocity, ExcessVolClamped) {
    const double beta = ProperTime::to_velocity(1.0, 0.5);
    EXPECT_LE(beta, srfm::constants::BETA_MAX_SAFE);
}

// --- compute ----------------------------------------------------------------

TEST(ProperTimeCompute, ZeroVelocityTauEqualsT) {
    auto pt = ProperTime::compute(1.0, 0.0);
    EXPECT_DOUBLE_EQ(pt.tau, 1.0);
    EXPECT_DOUBLE_EQ(pt.gamma, 1.0);
    EXPECT_DOUBLE_EQ(pt.velocity, 0.0);
}

TEST(ProperTimeCompute, HighVelocityTauLessThanT) {
    auto pt = ProperTime::compute(1.0, 0.8);
    EXPECT_LT(pt.tau, 1.0);
    EXPECT_GT(pt.gamma, 1.0);
}

TEST(ProperTimeCompute, TauPositiveForPositiveT) {
    auto pt = ProperTime::compute(10.0, 0.5);
    EXPECT_GT(pt.tau, 0.0);
}

TEST(ProperTimeCompute, InvariantTauTimeGamma) {
    // τ * γ == t
    auto pt = ProperTime::compute(5.0, 0.6);
    EXPECT_NEAR(pt.tau * pt.gamma, 5.0, 1e-10);
}

// ============================================================================
// ProperTimeClock
// ============================================================================

TEST(ProperTimeClock, InitialStateZero) {
    ProperTimeClock clk;
    EXPECT_DOUBLE_EQ(clk.elapsed_proper_time(), 0.0);
    EXPECT_DOUBLE_EQ(clk.elapsed_coordinate_time(), 0.0);
    EXPECT_EQ(clk.tick_count(), 0u);
}

TEST(ProperTimeClock, SingleTickAtZeroVol) {
    ProperTimeClock clk;
    clk.tick(0.0, 1.0);
    // v=0 → γ=1 → dτ = dt = 1.0
    EXPECT_NEAR(clk.elapsed_proper_time(), 1.0, 1e-12);
    EXPECT_NEAR(clk.elapsed_coordinate_time(), 1.0, 1e-12);
}

TEST(ProperTimeClock, HighVolProperTimeLessThanCoord) {
    ProperTimeClock clk(0.5);
    for (int i = 0; i < 10; ++i) {
        clk.tick(0.4, 1.0);  // beta = 0.8
    }
    EXPECT_LT(clk.elapsed_proper_time(), clk.elapsed_coordinate_time());
}

TEST(ProperTimeClock, TickCountAccumulates) {
    ProperTimeClock clk;
    clk.tick(0.1, 1.0);
    clk.tick(0.2, 1.0);
    clk.tick(0.3, 1.0);
    EXPECT_EQ(clk.tick_count(), 3u);
}

TEST(ProperTimeClock, ResetClearsAll) {
    ProperTimeClock clk;
    clk.tick(0.2, 5.0);
    clk.reset();
    EXPECT_DOUBLE_EQ(clk.elapsed_proper_time(), 0.0);
    EXPECT_DOUBLE_EQ(clk.elapsed_coordinate_time(), 0.0);
    EXPECT_EQ(clk.tick_count(), 0u);
}

TEST(ProperTimeClock, NegativeDtIsIgnored) {
    ProperTimeClock clk;
    clk.tick(0.1, -1.0);
    EXPECT_DOUBLE_EQ(clk.elapsed_proper_time(), 0.0);
}

TEST(ProperTimeClock, GammaAndVelocityUpdatedAfterTick) {
    ProperTimeClock clk(0.5);
    clk.tick(0.25, 1.0);  // beta = 0.5
    EXPECT_GT(clk.current_gamma(), 1.0);
    EXPECT_NEAR(clk.current_velocity(), 0.5, 1e-9);
}

// ============================================================================
// PortfolioAgingModel
// ============================================================================

TEST(PortfolioAgingModel, ZeroVolNoAging) {
    PortfolioAgingModel model;
    auto r = model.compute(1.0, 0.0, 1.5);
    EXPECT_NEAR(r.effective_age, 1.0, 1e-12);
    EXPECT_NEAR(r.gamma, 1.0, 1e-12);
}

TEST(PortfolioAgingModel, HighVolEffectiveAgeExceedsCoordAge) {
    PortfolioAgingModel model(0.5);
    auto r = model.compute(1.0, 0.4, 1.5);
    EXPECT_GT(r.effective_age, 1.0);
}

TEST(PortfolioAgingModel, AdjSharpePositiveForPositiveAge) {
    PortfolioAgingModel model;
    auto r = model.compute(2.0, 0.1, 1.0);
    EXPECT_GT(r.adj_sharpe, 0.0);
}

TEST(PortfolioAgingModel, AdjSharpeZeroForZeroAge) {
    PortfolioAgingModel model;
    auto r = model.compute(0.0, 0.2, 1.5);
    EXPECT_DOUBLE_EQ(r.adj_sharpe, 0.0);
}

TEST(PortfolioAgingModel, AdjSharpeFormula) {
    // adj_sharpe = sharpe / sqrt(effective_age)
    PortfolioAgingModel model(0.5);
    const double vol = 0.25;
    const double age = 4.0;
    const double sharpe = 2.0;
    auto r = model.compute(age, vol, sharpe);
    const double expected = sharpe / std::sqrt(r.effective_age);
    EXPECT_NEAR(r.adj_sharpe, expected, 1e-10);
}

TEST(PortfolioAgingModel, CoordinateAgePreserved) {
    PortfolioAgingModel model;
    auto r = model.compute(3.0, 0.2, 1.0);
    EXPECT_DOUBLE_EQ(r.coordinate_age, 3.0);
}

// ============================================================================
// RelativisticRebalanceTimer
// ============================================================================

TEST(RelativisticRebalanceTimer, NoRebalanceBeforeThreshold) {
    RelativisticRebalanceTimer timer(10.0);
    bool fired = timer.tick(0.0, 1.0);  // v=0 → dτ = 1.0; threshold = 10
    EXPECT_FALSE(fired);
}

TEST(RelativisticRebalanceTimer, RebalanceFiredAtThreshold) {
    RelativisticRebalanceTimer timer(5.0);
    int fired_count = 0;
    for (int i = 0; i < 6; ++i) {
        if (timer.tick(0.0, 1.0)) {
            ++fired_count;
        }
    }
    EXPECT_GE(fired_count, 1);
}

TEST(RelativisticRebalanceTimer, HighVolFiresLessOften) {
    // With high vol, dτ < dt, so more calendar days are needed per rebalance
    RelativisticRebalanceTimer timer_low(5.0, 0.5);
    RelativisticRebalanceTimer timer_high(5.0, 0.5);

    int low_fires = 0, high_fires = 0;
    for (int i = 0; i < 30; ++i) {
        if (timer_low.tick(0.0, 1.0))    ++low_fires;   // β = 0
        if (timer_high.tick(0.4, 1.0))   ++high_fires;  // β = 0.8
    }
    EXPECT_GE(low_fires, high_fires);
}

TEST(RelativisticRebalanceTimer, RebalanceCountIncrements) {
    RelativisticRebalanceTimer timer(1.0);
    for (int i = 0; i < 5; ++i) {
        timer.tick(0.0, 1.0);
    }
    EXPECT_GE(timer.rebalance_count(), 1u);
}

TEST(RelativisticRebalanceTimer, TauSinceRebalanceResetsAfterFire) {
    RelativisticRebalanceTimer timer(1.0);
    timer.tick(0.0, 1.0);  // fires (τ accumulated = 1.0)
    EXPECT_NEAR(timer.tau_since_rebalance(), 0.0, 1e-12);
}

TEST(RelativisticRebalanceTimer, ThresholdPreserved) {
    RelativisticRebalanceTimer timer(7.5);
    EXPECT_DOUBLE_EQ(timer.threshold_tau(), 7.5);
}

TEST(RelativisticRebalanceTimer, ResetTimerClearsTau) {
    RelativisticRebalanceTimer timer(10.0);
    timer.tick(0.0, 3.0);
    timer.reset_timer();
    EXPECT_DOUBLE_EQ(timer.tau_since_rebalance(), 0.0);
}

TEST(RelativisticRebalanceTimer, NegativeDtIgnored) {
    RelativisticRebalanceTimer timer(1.0);
    bool fired = timer.tick(0.1, -1.0);
    EXPECT_FALSE(fired);
    EXPECT_DOUBLE_EQ(timer.tau_since_rebalance(), 0.0);
}
