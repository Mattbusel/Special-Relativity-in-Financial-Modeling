/// @file tests/lorentz/test_lorentz_portfolio.cpp
/// @brief GTest unit tests for srfm::portfolio — 20+ tests.

#include "srfm/lorentz_portfolio.hpp"

#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

using namespace srfm::portfolio;

// ─── Helper constants ─────────────────────────────────────────────────────────

static constexpr double kEps = 1e-9;

static PortfolioFourVector make_pf(double ret, double vol,
                                   double skew = 0.0, double kurt = 0.0) {
    PortfolioFourVector pf;
    pf.ret  = ret;
    pf.vol  = vol;
    pf.skew = skew;
    pf.kurt = kurt;
    return pf;
}

// ─── LorentzFactor ────────────────────────────────────────────────────────────

TEST(LorentzFactor, ZeroBetaGivesOne) {
    LorentzFactor lf(0.0);
    EXPECT_NEAR(lf.gamma, 1.0, kEps);
}

TEST(LorentzFactor, HalfBetaGivesCorrectGamma) {
    LorentzFactor lf(0.5);
    double expected = 1.0 / std::sqrt(1.0 - 0.25);
    EXPECT_NEAR(lf.gamma, expected, kEps);
}

TEST(LorentzFactor, NegativeBetaGivesSameGammaAsPositive) {
    LorentzFactor lf_pos(0.6);
    LorentzFactor lf_neg(-0.6);
    EXPECT_NEAR(lf_pos.gamma, lf_neg.gamma, kEps);
}

TEST(LorentzFactor, BetaAtPlusOneThrows) {
    EXPECT_THROW(LorentzFactor(1.0), std::domain_error);
}

TEST(LorentzFactor, BetaAtMinusOneThrows) {
    EXPECT_THROW(LorentzFactor(-1.0), std::domain_error);
}

TEST(LorentzFactor, BetaAboveOneThrows) {
    EXPECT_THROW(LorentzFactor(1.5), std::domain_error);
}

TEST(LorentzFactor, GammaAlwaysGEOne) {
    for (double b : {0.0, 0.1, 0.5, 0.9, 0.99}) {
        LorentzFactor lf(b);
        EXPECT_GE(lf.gamma, 1.0);
    }
}

// ─── PortfolioFourVector ──────────────────────────────────────────────────────

TEST(PortfolioFourVector, SharpeRatioBasic) {
    auto pf = make_pf(0.12, 0.10);
    EXPECT_NEAR(pf.sharpe(), 1.2, kEps);
}

TEST(PortfolioFourVector, SharpeZeroVolReturnsZero) {
    auto pf = make_pf(0.12, 0.0);
    EXPECT_NEAR(pf.sharpe(), 0.0, kEps);
}

TEST(PortfolioFourVector, NegativeReturnNegativeSharpe) {
    auto pf = make_pf(-0.05, 0.10);
    EXPECT_NEAR(pf.sharpe(), -0.5, kEps);
}

// ─── LorentzBoost ─────────────────────────────────────────────────────────────

TEST(LorentzBoost, ZeroBetaIsIdentity) {
    auto pf = make_pf(0.12, 0.08, 0.3, 1.5);
    auto boosted = LorentzBoost::transform(pf, 0.0);
    EXPECT_NEAR(boosted.ret,  pf.ret,  kEps);
    EXPECT_NEAR(boosted.vol,  pf.vol,  kEps);
    EXPECT_NEAR(boosted.skew, pf.skew, kEps);
    EXPECT_NEAR(boosted.kurt, pf.kurt, kEps);
}

TEST(LorentzBoost, TransverseComponentsUnchanged) {
    auto pf = make_pf(0.10, 0.12, 0.5, 2.0);
    auto boosted = LorentzBoost::transform(pf, 0.5);
    EXPECT_NEAR(boosted.skew, pf.skew, kEps);
    EXPECT_NEAR(boosted.kurt, pf.kurt, kEps);
}

TEST(LorentzBoost, ReturnTransformFormula) {
    double beta = 0.6;
    double g    = 1.0 / std::sqrt(1.0 - beta * beta);
    auto pf     = make_pf(0.15, 0.10);
    auto boosted = LorentzBoost::transform(pf, beta);
    double expected_ret = g * (0.15 - beta * 0.10);
    EXPECT_NEAR(boosted.ret, expected_ret, kEps);
}

TEST(LorentzBoost, VolTransformFormula) {
    double beta = 0.6;
    double g    = 1.0 / std::sqrt(1.0 - beta * beta);
    auto pf     = make_pf(0.15, 0.10);
    auto boosted = LorentzBoost::transform(pf, beta);
    double expected_vol = g * (0.10 - beta * 0.15);
    EXPECT_NEAR(boosted.vol, expected_vol, kEps);
}

TEST(LorentzBoost, NegativeBetaInversesPositiveBeta) {
    auto pf      = make_pf(0.12, 0.10);
    auto b_pos   = LorentzBoost::transform(pf, 0.4);
    auto b_neg   = LorentzBoost::transform(b_pos, -0.4);
    // Two boosts with opposite β should return close to original
    EXPECT_NEAR(b_neg.ret, pf.ret, 1e-9);
    EXPECT_NEAR(b_neg.vol, pf.vol, 1e-9);
}

TEST(LorentzBoost, BetaAtBoundaryThrows) {
    auto pf = make_pf(0.10, 0.08);
    EXPECT_THROW(LorentzBoost::transform(pf, 1.0),  std::domain_error);
    EXPECT_THROW(LorentzBoost::transform(pf, -1.0), std::domain_error);
}

// ─── PortfolioInvariant ───────────────────────────────────────────────────────

TEST(PortfolioInvariant, BasicComputation) {
    auto pf = make_pf(3.0, 1.0, 1.0, 1.0);
    // I = 9 - 1 - 1 - 1 = 6
    EXPECT_NEAR(PortfolioInvariant::compute(pf), 6.0, kEps);
}

TEST(PortfolioInvariant, CanBeNegative) {
    auto pf = make_pf(0.0, 2.0, 1.0, 1.0);
    // I = 0 - 4 - 1 - 1 = -6
    EXPECT_NEAR(PortfolioInvariant::compute(pf), -6.0, kEps);
}

TEST(PortfolioInvariant, InvariantUnderBoost) {
    auto pf = make_pf(0.20, 0.12, 0.3, 1.5);
    double I_orig = PortfolioInvariant::compute(pf);

    for (double beta : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        auto boosted = LorentzBoost::transform(pf, beta);
        double I_boosted = PortfolioInvariant::compute(boosted);
        EXPECT_NEAR(I_boosted, I_orig, 1e-8)
            << "Invariant violated for beta=" << beta;
    }
}

TEST(PortfolioInvariant, InvariantUnderNegativeBoost) {
    auto pf = make_pf(0.15, 0.10, 0.5, 2.0);
    double I_orig = PortfolioInvariant::compute(pf);

    for (double beta : {-0.1, -0.5, -0.8, -0.99}) {
        auto boosted = LorentzBoost::transform(pf, beta);
        EXPECT_NEAR(PortfolioInvariant::compute(boosted), I_orig, 1e-7)
            << "Invariant violated for beta=" << beta;
    }
}

TEST(PortfolioInvariant, ZeroPortfolioZeroInvariant) {
    auto pf = make_pf(0.0, 0.0, 0.0, 0.0);
    EXPECT_NEAR(PortfolioInvariant::compute(pf), 0.0, kEps);
}

// ─── OptimalBoost ─────────────────────────────────────────────────────────────

TEST(OptimalBoost, ReturnsValueInRange) {
    auto pf = make_pf(0.12, 0.10);
    double beta = OptimalBoost::find(1.5, pf);
    EXPECT_GE(beta, -0.99);
    EXPECT_LE(beta, 0.99);
}

TEST(OptimalBoost, PositiveReturnFavorsPositiveBeta) {
    // If ret > vol, a positive boost should improve Sharpe
    auto pf = make_pf(0.20, 0.10);  // Sharpe = 2.0
    double beta = OptimalBoost::find(3.0, pf);
    // Boosted Sharpe should be >= original Sharpe
    auto boosted = LorentzBoost::transform(pf, beta);
    EXPECT_GE(boosted.sharpe(), 2.0 - 1e-6);
}

TEST(OptimalBoost, CustomStepWorks) {
    auto pf = make_pf(0.10, 0.10);
    double beta_coarse = OptimalBoost::find(1.0, pf, 0.1);
    double beta_fine   = OptimalBoost::find(1.0, pf, 0.01);
    EXPECT_GE(beta_coarse, -0.99);
    EXPECT_GE(beta_fine,   -0.99);
}

TEST(OptimalBoost, ZeroReturnZeroVol) {
    auto pf = make_pf(0.0, 0.0);
    double beta = OptimalBoost::find(0.0, pf);
    // No crash, returns something in valid range
    EXPECT_GE(beta, -0.99);
    EXPECT_LE(beta, 0.99);
}
