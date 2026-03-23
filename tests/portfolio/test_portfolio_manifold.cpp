/// @file tests/portfolio/test_portfolio_manifold.cpp
/// @brief GTest unit tests for MinkowskiCovariance and SpacetimeCausalGraph.

#include "portfolio_manifold.hpp"

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <string>

using namespace srfm::portfolio;
using namespace srfm::manifold;

// ─────────────────────────────────────────────────────────────────────────────
// AssetEvent helpers
// ─────────────────────────────────────────────────────────────────────────────

TEST(AssetEvent, FourVector) {
    AssetEvent e{"AAPL", 1.0, 150.0, 1e6, 2.4e9};
    auto v = e.to_four_vector();
    EXPECT_DOUBLE_EQ(v(0), 1.0);
    EXPECT_DOUBLE_EQ(v(1), 150.0);
    EXPECT_DOUBLE_EQ(v(2), 1e6);
    EXPECT_DOUBLE_EQ(v(3), 2.4e9);
}

TEST(AssetEvent, ToNAssetEvent) {
    AssetEvent e{"TEST", 2.0, 100.0, 5e5, 1e9};
    auto ne = e.to_nasset_event();
    EXPECT_DOUBLE_EQ(ne.t, 2.0);
    ASSERT_EQ(ne.prices.size(), 3);
    EXPECT_DOUBLE_EQ(ne.prices(0), 100.0);
    EXPECT_DOUBLE_EQ(ne.prices(1), 5e5);
    EXPECT_DOUBLE_EQ(ne.prices(2), 1e9);
}

// ─────────────────────────────────────────────────────────────────────────────
// MinkowskiCovariance — basic operations
// ─────────────────────────────────────────────────────────────────────────────

TEST(MinkowskiCovariance, EmptyReturnsNullopt) {
    MinkowskiCovariance mc;
    EXPECT_EQ(mc.size(), 0u);
    EXPECT_FALSE(mc.compute_spacetime_covariance().has_value());
}

TEST(MinkowskiCovariance, SingleAssetReturnsNullopt) {
    MinkowskiCovariance mc;
    mc.add_asset(AssetEvent{"A", 0.0, 100.0, 1e6, 1e9});
    EXPECT_EQ(mc.size(), 1u);
    EXPECT_FALSE(mc.compute_spacetime_covariance().has_value());
}

TEST(MinkowskiCovariance, TwoAssetsCovarianceDiagonalIsOne) {
    MinkowskiCovariance mc;
    mc.add_asset(AssetEvent{"A", 0.0, 100.0, 1e6, 1e9});
    mc.add_asset(AssetEvent{"B", 1.0, 200.0, 2e6, 2e9});

    auto cov_opt = mc.compute_spacetime_covariance();
    ASSERT_TRUE(cov_opt.has_value());

    const auto& cov = *cov_opt;
    ASSERT_EQ(cov.rows(), 2);
    ASSERT_EQ(cov.cols(), 2);

    // Diagonal must be 1.0.
    EXPECT_DOUBLE_EQ(cov(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(cov(1, 1), 1.0);
}

TEST(MinkowskiCovariance, OffDiagonalIsBetweenZeroAndOne) {
    MinkowskiCovariance mc;
    mc.add_asset(AssetEvent{"A", 0.0, 100.0, 1e4, 1e6});
    mc.add_asset(AssetEvent{"B", 0.5, 101.0, 1.1e4, 1.05e6});

    auto cov_opt = mc.compute_spacetime_covariance();
    ASSERT_TRUE(cov_opt.has_value());

    const double off_diag = (*cov_opt)(0, 1);
    EXPECT_GT(off_diag, 0.0);
    EXPECT_LE(off_diag, 1.0);
}

TEST(MinkowskiCovariance, CovarianceMatrixIsSymmetric) {
    MinkowskiCovariance mc;
    mc.add_asset(AssetEvent{"A", 0.0, 100.0, 1e4, 1e6});
    mc.add_asset(AssetEvent{"B", 0.5, 110.0, 2e4, 2e6});
    mc.add_asset(AssetEvent{"C", 1.0, 120.0, 3e4, 3e6});

    auto cov_opt = mc.compute_spacetime_covariance();
    ASSERT_TRUE(cov_opt.has_value());
    const auto& cov = *cov_opt;

    for (int i = 0; i < cov.rows(); ++i) {
        for (int j = 0; j < cov.cols(); ++j) {
            EXPECT_NEAR(cov(i, j), cov(j, i), 1e-14)
                << "Covariance matrix not symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST(MinkowskiCovariance, IdenticalEventsMaxCorrelation) {
    // Two identical events at the same spacetime point: ds²=0 → C = exp(0) = 1.
    MinkowskiCovariance mc;
    AssetEvent e{"X", 1.0, 100.0, 1e6, 1e9};
    mc.add_asset(e);
    mc.add_asset(e);  // exact duplicate

    auto cov_opt = mc.compute_spacetime_covariance();
    ASSERT_TRUE(cov_opt.has_value());
    // Off-diagonal should be exp(-|0|) = 1.
    EXPECT_DOUBLE_EQ((*cov_opt)(0, 1), 1.0);
    EXPECT_DOUBLE_EQ((*cov_opt)(1, 0), 1.0);
}

TEST(MinkowskiCovariance, IntervalCorrelationOutOfRange) {
    MinkowskiCovariance mc;
    mc.add_asset(AssetEvent{"A", 0.0, 100.0, 1e4, 1e6});
    EXPECT_FALSE(mc.interval_correlation(0, 5).has_value());
    EXPECT_FALSE(mc.interval_correlation(5, 0).has_value());
}

TEST(MinkowskiCovariance, ClassifyPair) {
    MinkowskiCovariance mc(1.0);
    // Asset A at t=0, B at t=1; large Δt makes ds² negative → TIMELIKE.
    mc.add_asset(AssetEvent{"A", 0.0, 0.0, 0.0, 0.0});
    mc.add_asset(AssetEvent{"B", 100.0, 0.0, 0.0, 0.0});

    auto type = mc.classify_pair(0, 1);
    ASSERT_TRUE(type.has_value());
    EXPECT_EQ(*type, IntervalType::TIMELIKE);
}

TEST(MinkowskiCovariance, Clear) {
    MinkowskiCovariance mc;
    mc.add_asset(AssetEvent{"A", 0.0, 100.0, 1e4, 1e6});
    mc.add_asset(AssetEvent{"B", 1.0, 200.0, 2e4, 2e6});
    EXPECT_EQ(mc.size(), 2u);
    mc.clear();
    EXPECT_EQ(mc.size(), 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// SpacetimeCausalGraph
// ─────────────────────────────────────────────────────────────────────────────

TEST(SpacetimeCausalGraph, BuildRequiresTwoAssets) {
    MinkowskiCovariance mc;
    mc.add_asset(AssetEvent{"A", 0.0, 100.0, 1e4, 1e6});
    // Only one asset — should fail.
    EXPECT_FALSE(SpacetimeCausalGraph::build(mc).has_value());
}

TEST(SpacetimeCausalGraph, NoSelfEdges) {
    MinkowskiCovariance mc;
    mc.add_asset(AssetEvent{"A", 0.0, 0.0, 0.0, 0.0});
    mc.add_asset(AssetEvent{"B", 1.0, 0.0, 0.0, 0.0});

    auto g = SpacetimeCausalGraph::build(mc);
    ASSERT_TRUE(g.has_value());

    const auto& adj = g->adjacency_matrix();
    for (std::size_t i = 0; i < g->n_assets(); ++i) {
        EXPECT_EQ(adj(static_cast<int>(i), static_cast<int>(i)), 0)
            << "Self-edge at node " << i;
    }
}

TEST(SpacetimeCausalGraph, TimelikeSeparationCreatesEdge) {
    // Asset A at t=0, B at t=10 with identical spatial coords → ds²<0 → TIMELIKE.
    MinkowskiCovariance mc(1.0);
    mc.add_asset(AssetEvent{"A", 0.0, 0.0, 0.0, 0.0});
    mc.add_asset(AssetEvent{"B", 10.0, 0.0, 0.0, 0.0});

    auto g = SpacetimeCausalGraph::build(mc);
    ASSERT_TRUE(g.has_value());

    // ds²(0,1) = -1·100 + 0 = -100 → TIMELIKE → edge 0→1 should exist.
    EXPECT_TRUE(g->has_edge(0, 1))
        << "Expected TIMELIKE edge 0→1 for large Δt";
}

TEST(SpacetimeCausalGraph, SpacelikeSeparationNoEdge) {
    // Same time, huge spatial separation → ds² > 0 → SPACELIKE → no edge.
    MinkowskiCovariance mc(1.0);
    mc.add_asset(AssetEvent{"A", 0.0, 0.0, 0.0, 0.0});
    mc.add_asset(AssetEvent{"B", 0.0, 1000.0, 0.0, 0.0});

    auto g = SpacetimeCausalGraph::build(mc);
    ASSERT_TRUE(g.has_value());

    EXPECT_FALSE(g->has_edge(0, 1))
        << "Did not expect edge for SPACELIKE separation";
    EXPECT_FALSE(g->has_edge(1, 0))
        << "Did not expect edge for SPACELIKE separation";
}

TEST(SpacetimeCausalGraph, AssetIds) {
    MinkowskiCovariance mc;
    mc.add_asset(AssetEvent{"AAPL", 0.0, 150.0, 1e6, 2e9});
    mc.add_asset(AssetEvent{"MSFT", 1.0, 290.0, 8e5, 2.1e9});

    auto g = SpacetimeCausalGraph::build(mc);
    ASSERT_TRUE(g.has_value());

    const auto& ids = g->asset_ids();
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], "AAPL");
    EXPECT_EQ(ids[1], "MSFT");
}

TEST(SpacetimeCausalGraph, OutInDegree) {
    // Build a 3-asset graph where A→B and A→C (both TIMELIKE from A).
    MinkowskiCovariance mc(1.0);
    mc.add_asset(AssetEvent{"A", 0.0, 0.0, 0.0, 0.0});
    mc.add_asset(AssetEvent{"B", 10.0, 0.0, 0.0, 0.0});
    mc.add_asset(AssetEvent{"C", 10.0, 0.0, 0.0, 0.0});

    auto g = SpacetimeCausalGraph::build(mc);
    ASSERT_TRUE(g.has_value());

    // A has edges to B and C (both TIMELIKE).
    EXPECT_GE(g->out_degree(0), 0);
    EXPECT_GE(g->in_degree(1), 0);
}
