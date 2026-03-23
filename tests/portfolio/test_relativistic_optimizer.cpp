/// @file tests/portfolio/test_relativistic_optimizer.cpp
/// @brief GTest unit tests for RelativisticPortfolio optimizer.

#include "relativistic_optimizer.hpp"

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <numeric>

using namespace srfm::portfolio;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static AssetEvent make_event(const std::string& id,
                              double t, double P, double V = 1e6,
                              double M = 1e9) {
    return AssetEvent{id, t, P, V, M};
}

// ─────────────────────────────────────────────────────────────────────────────
// Basic construction and property checks
// ─────────────────────────────────────────────────────────────────────────────

TEST(RelativisticPortfolio, EmptyPortfolioNullopt) {
    RelativisticPortfolio rp;
    EXPECT_EQ(rp.n_assets(), 0u);
    EXPECT_FALSE(rp.optimize_weights(0.05).has_value());
    EXPECT_FALSE(rp.relativistic_returns().has_value());
    EXPECT_FALSE(rp.spacetime_covariance().has_value());
}

TEST(RelativisticPortfolio, SingleAssetNullopt) {
    RelativisticPortfolio rp;
    rp.add_asset(make_event("A", 1.0, 100.0), 0.10);
    EXPECT_EQ(rp.n_assets(), 1u);
    EXPECT_FALSE(rp.optimize_weights(0.05).has_value());
}

TEST(RelativisticPortfolio, ClearResetsPortfolio) {
    RelativisticPortfolio rp;
    rp.add_asset(make_event("A", 1.0, 100.0), 0.10);
    rp.add_asset(make_event("B", 1.0, 200.0), 0.08);
    EXPECT_EQ(rp.n_assets(), 2u);
    rp.clear();
    EXPECT_EQ(rp.n_assets(), 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Relativistic returns
// ─────────────────────────────────────────────────────────────────────────────

TEST(RelativisticPortfolio, RelativisticReturnsSize) {
    RelativisticPortfolio rp;
    rp.add_asset(make_event("A", 1.0, 10.0), 0.10);
    rp.add_asset(make_event("B", 2.0, 20.0), 0.12);
    rp.add_asset(make_event("C", 3.0, 30.0), 0.08);

    auto mu_opt = rp.relativistic_returns();
    ASSERT_TRUE(mu_opt.has_value());
    EXPECT_EQ(mu_opt->size(), 3);
}

TEST(RelativisticPortfolio, RelativisticReturnsNonNegative) {
    // Positive expected returns should remain positive after γ correction.
    RelativisticPortfolio rp;
    rp.add_asset(make_event("A", 1.0, 50.0), 0.10);
    rp.add_asset(make_event("B", 1.0, 60.0), 0.12);

    auto mu_opt = rp.relativistic_returns();
    ASSERT_TRUE(mu_opt.has_value());

    for (int i = 0; i < mu_opt->size(); ++i) {
        EXPECT_GT((*mu_opt)(i), 0.0)
            << "Relativistic return[" << i << "] should be positive";
    }
}

TEST(RelativisticPortfolio, StaticAssetGammaIsOne) {
    // Asset at t=0, P=0 → β=0 → γ=1 → μ_rel = μ_raw.
    RelativisticPortfolio rp;
    rp.add_asset(make_event("A", 0.0, 0.0), 0.10);
    rp.add_asset(make_event("B", 0.0, 0.0), 0.12);

    auto mu_opt = rp.relativistic_returns();
    ASSERT_TRUE(mu_opt.has_value());

    EXPECT_NEAR((*mu_opt)(0), 0.10, 1e-14);
    EXPECT_NEAR((*mu_opt)(1), 0.12, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// Spacetime covariance
// ─────────────────────────────────────────────────────────────────────────────

TEST(RelativisticPortfolio, CovarianceDiagonalIsOne) {
    RelativisticPortfolio rp;
    rp.add_asset(make_event("A", 1.0, 100.0), 0.10);
    rp.add_asset(make_event("B", 2.0, 200.0), 0.08);

    auto cov_opt = rp.spacetime_covariance();
    ASSERT_TRUE(cov_opt.has_value());

    EXPECT_DOUBLE_EQ((*cov_opt)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*cov_opt)(1, 1), 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// optimize_weights: constraint satisfaction
// ─────────────────────────────────────────────────────────────────────────────

TEST(RelativisticPortfolio, WeightsSumToOne) {
    RelativisticPortfolio rp;
    rp.add_asset(make_event("A", 1.0, 100.0), 0.10);
    rp.add_asset(make_event("B", 1.0, 200.0), 0.08);
    rp.add_asset(make_event("C", 1.0, 150.0), 0.12);

    auto result = rp.optimize_weights(0.05);
    ASSERT_TRUE(result.has_value());

    double sum = result->weights.sum();
    EXPECT_NEAR(sum, 1.0, 1e-8) << "Weights do not sum to 1";
}

TEST(RelativisticPortfolio, WeightsNonNegative) {
    RelativisticPortfolio rp;
    rp.add_asset(make_event("A", 1.0, 100.0), 0.10);
    rp.add_asset(make_event("B", 1.0, 200.0), 0.08);
    rp.add_asset(make_event("C", 1.0, 150.0), 0.12);

    auto result = rp.optimize_weights(0.05);
    ASSERT_TRUE(result.has_value());

    for (int i = 0; i < result->weights.size(); ++i) {
        EXPECT_GE(result->weights(i), -1e-9)
            << "Weight[" << i << "] is negative: " << result->weights(i);
    }
}

TEST(RelativisticPortfolio, GeodesicRiskNonNegative) {
    RelativisticPortfolio rp;
    rp.add_asset(make_event("A", 1.0, 100.0), 0.10);
    rp.add_asset(make_event("B", 2.0, 300.0), 0.15);

    auto result = rp.optimize_weights(0.08);
    ASSERT_TRUE(result.has_value());

    EXPECT_GE(result->geodesic_risk, 0.0);
}

TEST(RelativisticPortfolio, IterationCountPositive) {
    RelativisticPortfolio rp;
    rp.add_asset(make_event("A", 1.0, 100.0), 0.10);
    rp.add_asset(make_event("B", 1.0, 200.0), 0.08);

    auto result = rp.optimize_weights(0.05);
    ASSERT_TRUE(result.has_value());

    EXPECT_GT(result->iterations, 0);
}

TEST(RelativisticPortfolio, ImpossibleTargetStillReturnsFeasible) {
    // Target return much higher than any individual asset — optimizer should
    // still converge to a feasible (simplex-projected) weight vector.
    RelativisticPortfolio rp;
    rp.add_asset(make_event("A", 1.0, 100.0), 0.05);
    rp.add_asset(make_event("B", 1.0, 100.0), 0.06);

    auto result = rp.optimize_weights(999.0);  // impossible target
    ASSERT_TRUE(result.has_value());

    // Weights must still be valid.
    EXPECT_NEAR(result->weights.sum(), 1.0, 1e-8);
    for (int i = 0; i < result->weights.size(); ++i) {
        EXPECT_GE(result->weights(i), -1e-9);
    }
}

TEST(RelativisticPortfolio, TwoEqualAssetsApproxEqual) {
    // Two identical assets: unconstrained optimum is w = [0.5, 0.5].
    RelativisticPortfolio rp;
    rp.add_asset(make_event("A", 0.0, 0.0), 0.10);
    rp.add_asset(make_event("B", 0.0, 0.0), 0.10);

    OptimizerConfig cfg;
    cfg.max_iterations = 2000;
    cfg.step_size = 1e-3;

    RelativisticPortfolio rp2(cfg);
    rp2.add_asset(make_event("A", 0.0, 0.0), 0.10);
    rp2.add_asset(make_event("B", 0.0, 0.0), 0.10);

    auto result = rp2.optimize_weights(0.05);
    ASSERT_TRUE(result.has_value());

    // With symmetric inputs the optimizer should converge near equal weights.
    EXPECT_NEAR(result->weights(0), result->weights(1), 0.1)
        << "Expected approximately equal weights for identical assets";
}
