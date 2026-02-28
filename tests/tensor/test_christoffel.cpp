#include <gtest/gtest.h>
#include "srfm/tensor.hpp"
#include "srfm/constants.hpp"
#include <Eigen/Dense>
#include <cmath>

using namespace srfm;
using namespace srfm::tensor;
using namespace srfm::constants;

static SpacetimePoint origin() { return SpacetimePoint::Zero(); }

// ─── Flat Metric → Zero Christoffel ──────────────────────────────────────────
//
// Key mathematical fact: Γ^λ_μν = 0 everywhere for a constant (flat) metric.
// Partial derivatives ∂_σ g_μν = 0 → all Christoffel symbols vanish.

TEST(Christoffel_FlatMetric, Minkowski_AllSymbolsZeroAtOrigin) {
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    ChristoffelSymbols cs(g);
    auto gamma = cs.compute(origin());

    for (int l = 0; l < SPACETIME_DIM; ++l)
        for (int mu = 0; mu < SPACETIME_DIM; ++mu)
            for (int nu = 0; nu < SPACETIME_DIM; ++nu)
                EXPECT_NEAR(gamma[l](mu, nu), 0.0, 1e-8)
                    << "Gamma[" << l << "](" << mu << "," << nu
                    << ") != 0 for flat metric";
}

TEST(Christoffel_FlatMetric, Diagonal_AllSymbolsZeroAtArbitraryPoint) {
    auto g = MetricTensor::make_diagonal(1.0, {0.2, 0.3, 0.4});
    ChristoffelSymbols cs(g);

    SpacetimePoint p;
    p << 1.0, -2.0, 3.5, 0.7;
    auto gamma = cs.compute(p);

    for (int l = 0; l < SPACETIME_DIM; ++l)
        for (int mu = 0; mu < SPACETIME_DIM; ++mu)
            for (int nu = 0; nu < SPACETIME_DIM; ++nu)
                EXPECT_NEAR(gamma[l](mu, nu), 0.0, 1e-7)
                    << "Gamma[" << l << "](" << mu << "," << nu << ") at non-origin";
}

// ─── Symmetry: Γ^λ_μν = Γ^λ_νμ ──────────────────────────────────────────────
//
// The Christoffel symbols of the Levi-Civita connection are symmetric
// in the lower two indices.

TEST(Christoffel_Symmetry, FlatMetric_LowerIndexSymmetric) {
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    ChristoffelSymbols cs(g);
    auto gamma = cs.compute(origin());

    for (int l = 0; l < SPACETIME_DIM; ++l)
        for (int mu = 0; mu < SPACETIME_DIM; ++mu)
            for (int nu = 0; nu < SPACETIME_DIM; ++nu)
                EXPECT_NEAR(gamma[l](mu, nu), gamma[l](nu, mu), 1e-10)
                    << "Symmetry broken: Gamma[" << l << "](" << mu << "," << nu << ")";
}

TEST(Christoffel_Symmetry, CurvedMetric_LowerIndexSymmetric) {
    // Position-dependent metric: g₁₁ varies with x¹ → nonzero Christoffel
    auto g = MetricTensor([](const SpacetimePoint& x) {
        MetricMatrix m = MetricMatrix::Zero();
        m(0, 0) = -1.0;
        m(1, 1) = 1.0 + 0.5 * x(1) * x(1); // curved spatial component
        m(2, 2) = 1.0;
        m(3, 3) = 1.0;
        return m;
    });

    ChristoffelSymbols cs(g, 1e-6);
    SpacetimePoint p;
    p << 0.0, 1.0, 0.0, 0.0;
    auto gamma = cs.compute(p);

    for (int l = 0; l < SPACETIME_DIM; ++l)
        for (int mu = 0; mu < SPACETIME_DIM; ++mu)
            for (int nu = 0; nu < SPACETIME_DIM; ++nu)
                EXPECT_NEAR(gamma[l](mu, nu), gamma[l](nu, mu), 1e-7)
                    << "Symmetry violated for curved metric at Gamma["
                    << l << "](" << mu << "," << nu << ")";
}

// ─── Nonzero Christoffel for Known Curved Metric ─────────────────────────────
//
// Use a metric with known analytic Christoffel symbols to verify numerics.
//
// Metric: g = diag(-1, f(x¹), 1, 1) where f(x¹) = exp(x¹)
//
// g₁₁ = e^x¹  →  ∂₁g₁₁ = e^x¹
//
// Non-zero symbol: Γ¹₁₁ = (1/2) g^11 ∂₁g₁₁ = (1/2)(e^{-x¹})(e^x¹) = 1/2
//
// All others zero (metric is diagonal and only g₁₁ varies).

TEST(Christoffel_Curved, ExponentialMetric_Gamma111_IsHalf) {
    auto g = MetricTensor([](const SpacetimePoint& x) {
        MetricMatrix m = MetricMatrix::Zero();
        m(0, 0) = -1.0;
        m(1, 1) = std::exp(x(1));
        m(2, 2) = 1.0;
        m(3, 3) = 1.0;
        return m;
    });

    ChristoffelSymbols cs(g, 1e-5);

    // Γ¹₁₁ should equal 0.5 at any point (it's position-independent)
    SpacetimePoint p;
    p << 0.0, 0.5, 0.0, 0.0;
    auto gamma = cs.compute(p);

    EXPECT_NEAR(gamma[1](1, 1), 0.5, 1e-5)
        << "Gamma^1_11 for e^x metric should be 0.5";
}

TEST(Christoffel_Curved, ExponentialMetric_OtherDiagonalSymbols_NearZero) {
    auto g = MetricTensor([](const SpacetimePoint& x) {
        MetricMatrix m = MetricMatrix::Zero();
        m(0, 0) = -1.0;
        m(1, 1) = std::exp(x(1));
        m(2, 2) = 1.0;
        m(3, 3) = 1.0;
        return m;
    });

    ChristoffelSymbols cs(g, 1e-5);
    SpacetimePoint p;
    p << 0.0, 0.5, 0.0, 0.0;
    auto gamma = cs.compute(p);

    // All diagonal symbols except Gamma^1_11 should vanish
    for (int l = 0; l < SPACETIME_DIM; ++l) {
        if (l == 1) continue;
        EXPECT_NEAR(gamma[l](l, l), 0.0, 1e-5)
            << "Gamma^" << l << "_" << l << l << " should be zero";
    }
}

// ─── Contract: Γ^λ_μν u^μ u^ν ────────────────────────────────────────────────

TEST(Christoffel_Contract, FlatMetric_ContractWithAnyVelocity_IsZero) {
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    ChristoffelSymbols cs(g);
    auto gamma = cs.compute(origin());

    FourVelocity u;
    u << 1.0, 0.3, -0.2, 0.7;

    FourVelocity result = cs.contract(gamma, u);
    EXPECT_TRUE(result.isZero(1e-8))
        << "Contraction on flat metric should be zero, got: " << result.transpose();
}

TEST(Christoffel_Contract, ZeroVelocity_ContractIsAlwaysZero) {
    // Regardless of Christoffel symbols, u=0 → Γu u = 0
    auto g = MetricTensor([](const SpacetimePoint& x) {
        MetricMatrix m = MetricMatrix::Zero();
        m(0, 0) = -1.0;
        m(1, 1) = 1.0 + x(1) * x(1);
        m(2, 2) = 1.0;
        m(3, 3) = 1.0;
        return m;
    });

    ChristoffelSymbols cs(g);
    SpacetimePoint p;
    p << 0.0, 1.0, 0.0, 0.0;
    auto gamma = cs.compute(p);

    FourVelocity u = FourVelocity::Zero();
    FourVelocity result = cs.contract(gamma, u);
    EXPECT_TRUE(result.isZero(1e-14));
}

TEST(Christoffel_Contract, ExponentialMetric_ContractMatchesAnalytic) {
    // For g = diag(-1, e^x¹, 1, 1), Γ¹₁₁ = 0.5.
    // Contract with u = (0, v, 0, 0):
    //   result^1 = Γ¹₁₁ v² = 0.5 v²
    //   result^λ = 0 for λ ≠ 1
    auto g = MetricTensor([](const SpacetimePoint& x) {
        MetricMatrix m = MetricMatrix::Zero();
        m(0, 0) = -1.0;
        m(1, 1) = std::exp(x(1));
        m(2, 2) = 1.0;
        m(3, 3) = 1.0;
        return m;
    });

    ChristoffelSymbols cs(g, 1e-5);
    SpacetimePoint p;
    p << 0.0, 0.5, 0.0, 0.0;
    auto gamma = cs.compute(p);

    double v = 2.0;
    FourVelocity u;
    u << 0.0, v, 0.0, 0.0;

    FourVelocity result = cs.contract(gamma, u);
    EXPECT_NEAR(result(1), 0.5 * v * v, 1e-4);
    EXPECT_NEAR(result(0), 0.0, 1e-5);
    EXPECT_NEAR(result(2), 0.0, 1e-5);
    EXPECT_NEAR(result(3), 0.0, 1e-5);
}

// ─── Financial Interpretation: Covariance Metric ─────────────────────────────
//
// A covariance-based metric with correlated assets should produce nonzero
// Christoffel symbols when volatilities vary through market space.

TEST(Christoffel_Financial, TimeVaryingVol_ProducesNonzeroSymbols) {
    // Simulate a market where asset 1 volatility σ(t) = 0.2 + 0.1 * t
    // (volatility increases with time — heteroskedastic market)
    auto g = MetricTensor([](const SpacetimePoint& x) {
        MetricMatrix m = MetricMatrix::Zero();
        double t = x(0);
        double sigma1 = 0.2 + 0.1 * t;
        m(0, 0) = -1.0;
        m(1, 1) = sigma1 * sigma1;
        m(2, 2) = 0.09;  // 0.3² fixed
        m(3, 3) = 0.04;  // 0.2² fixed
        return m;
    });

    ChristoffelSymbols cs(g, 1e-6);
    SpacetimePoint p;
    p << 1.0, 0.0, 0.0, 0.0;  // at time t=1
    auto gamma = cs.compute(p);

    // Γ^1_01 should be nonzero (metric changes with time)
    // At t=1: σ₁ = 0.3, ∂_t(σ₁²) = 2σ₁ * 0.1 = 0.06
    // Γ^1_01 = (1/2) g^11 ∂_0 g_11 = (1/2)(1/σ₁²)(0.06)
    double sigma1_at_t1  = 0.3;
    double d_g11_dt      = 0.06;
    double expected_G101 = 0.5 * (1.0 / (sigma1_at_t1 * sigma1_at_t1)) * d_g11_dt;

    EXPECT_NEAR(gamma[1](0, 1), expected_G101, 1e-4)
        << "Time-varying vol should produce nonzero Gamma^1_01";
    EXPECT_NEAR(gamma[1](1, 0), expected_G101, 1e-4)
        << "Symmetry: Gamma^1_10 should equal Gamma^1_01";
}

// ─── Index Dimension Checks ───────────────────────────────────────────────────

TEST(Christoffel_Structure, ResultHasCorrectDimensions) {
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    ChristoffelSymbols cs(g);
    auto gamma = cs.compute(origin());

    EXPECT_EQ(static_cast<int>(gamma.size()), SPACETIME_DIM);
    for (int l = 0; l < SPACETIME_DIM; ++l) {
        EXPECT_EQ(gamma[l].rows(), SPACETIME_DIM);
        EXPECT_EQ(gamma[l].cols(), SPACETIME_DIM);
    }
}
