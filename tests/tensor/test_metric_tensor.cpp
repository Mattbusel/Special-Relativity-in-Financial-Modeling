#include <gtest/gtest.h>
#include "srfm/tensor.hpp"
#include "srfm/constants.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <array>

using namespace srfm;
using namespace srfm::tensor;
using namespace srfm::constants;

// ─── Helpers ──────────────────────────────────────────────────────────────────

static SpacetimePoint origin() { return SpacetimePoint::Zero(); }

// ─── Minkowski Factory ────────────────────────────────────────────────────────

TEST(MetricTensor_Minkowski, DefaultScale_DiagonalMinusOnePlusOne) {
    auto g   = MetricTensor::make_minkowski(1.0, 1.0);
    auto gx  = g.evaluate(origin());

    // Signature: diag(-1, 1, 1, 1)
    EXPECT_NEAR(gx(0, 0), -1.0, FLOAT_EPSILON);
    EXPECT_NEAR(gx(1, 1),  1.0, FLOAT_EPSILON);
    EXPECT_NEAR(gx(2, 2),  1.0, FLOAT_EPSILON);
    EXPECT_NEAR(gx(3, 3),  1.0, FLOAT_EPSILON);
}

TEST(MetricTensor_Minkowski, DefaultScale_OffDiagonalAreZero) {
    auto g  = MetricTensor::make_minkowski(1.0, 1.0);
    auto gx = g.evaluate(origin());

    for (int mu = 0; mu < SPACETIME_DIM; ++mu)
        for (int nu = 0; nu < SPACETIME_DIM; ++nu)
            if (mu != nu)
                EXPECT_NEAR(gx(mu, nu), 0.0, FLOAT_EPSILON)
                    << "Off-diagonal non-zero at (" << mu << "," << nu << ")";
}

TEST(MetricTensor_Minkowski, ScaledTime_TimeTimeEntry) {
    double c = 3.0;
    auto g   = MetricTensor::make_minkowski(c, 1.0);
    auto gx  = g.evaluate(origin());

    EXPECT_NEAR(gx(0, 0), -(c * c), FLOAT_EPSILON);
}

TEST(MetricTensor_Minkowski, ScaledSpatial_SpatialEntries) {
    double sigma = 0.2;
    auto g   = MetricTensor::make_minkowski(1.0, sigma);
    auto gx  = g.evaluate(origin());

    for (int i = 1; i < SPACETIME_DIM; ++i)
        EXPECT_NEAR(gx(i, i), sigma * sigma, FLOAT_EPSILON)
            << "Spatial entry wrong at index " << i;
}

TEST(MetricTensor_Minkowski, IsLorentzian) {
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    EXPECT_TRUE(g.is_lorentzian(origin()));
}

TEST(MetricTensor_Minkowski, IsPositionIndependent) {
    // Flat metric must be the same at every point
    auto g = MetricTensor::make_minkowski(1.0, 0.2);

    SpacetimePoint p1, p2;
    p1 << 1.0, 2.0, 3.0, 4.0;
    p2 << -5.0, 100.0, 0.0, -1.0;

    auto gp1 = g.evaluate(p1);
    auto gp2 = g.evaluate(p2);

    EXPECT_TRUE((gp1 - gp2).norm() < FLOAT_EPSILON);
}

// ─── Diagonal Factory ─────────────────────────────────────────────────────────

TEST(MetricTensor_Diagonal, TimeEntry_MatchesTimescale) {
    auto g  = MetricTensor::make_diagonal(2.0, {0.1, 0.2, 0.3});
    auto gx = g.evaluate(origin());

    EXPECT_NEAR(gx(0, 0), -4.0, FLOAT_EPSILON);
}

TEST(MetricTensor_Diagonal, SpatialEntries_MatchVolatilities) {
    std::array<double, 3> vol = {0.1, 0.2, 0.3};
    auto g  = MetricTensor::make_diagonal(1.0, vol);
    auto gx = g.evaluate(origin());

    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(gx(i + 1, i + 1), vol[i] * vol[i], FLOAT_EPSILON)
            << "Spatial volatility wrong at asset " << i;
}

TEST(MetricTensor_Diagonal, OffDiagonalAreZero) {
    auto g  = MetricTensor::make_diagonal(1.0, {0.1, 0.2, 0.3});
    auto gx = g.evaluate(origin());

    for (int mu = 0; mu < SPACETIME_DIM; ++mu)
        for (int nu = 0; nu < SPACETIME_DIM; ++nu)
            if (mu != nu)
                EXPECT_NEAR(gx(mu, nu), 0.0, FLOAT_EPSILON)
                    << "Off-diagonal at (" << mu << "," << nu << ")";
}

TEST(MetricTensor_Diagonal, IsLorentzian) {
    auto g = MetricTensor::make_diagonal(1.0, {0.1, 0.2, 0.3});
    EXPECT_TRUE(g.is_lorentzian(origin()));
}

// ─── Covariance Factory ───────────────────────────────────────────────────────

TEST(MetricTensor_Covariance, SpatialBlock_MatchesCovMatrix) {
    Eigen::Matrix3d cov;
    cov << 0.04, 0.01, 0.005,
           0.01, 0.09, 0.02,
           0.005, 0.02, 0.01;

    auto g  = MetricTensor::make_from_covariance(1.0, cov);
    auto gx = g.evaluate(origin());

    // Extract the 3x3 spatial block
    Eigen::Matrix3d spatial = gx.block<3, 3>(1, 1);
    EXPECT_TRUE((spatial - cov).norm() < FLOAT_EPSILON * 10)
        << "Spatial block mismatch:\n" << spatial << "\n!=\n" << cov;
}

TEST(MetricTensor_Covariance, TimeEntry_NegativeTimescaleSquared) {
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();
    auto g  = MetricTensor::make_from_covariance(2.0, cov);
    auto gx = g.evaluate(origin());
    EXPECT_NEAR(gx(0, 0), -4.0, FLOAT_EPSILON);
}

TEST(MetricTensor_Covariance, TimeSpaceCrossTerms_AreZero) {
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();
    auto g  = MetricTensor::make_from_covariance(1.0, cov);
    auto gx = g.evaluate(origin());

    for (int i = 1; i < SPACETIME_DIM; ++i) {
        EXPECT_NEAR(gx(0, i), 0.0, FLOAT_EPSILON) << "g(0," << i << ") != 0";
        EXPECT_NEAR(gx(i, 0), 0.0, FLOAT_EPSILON) << "g(" << i << ",0) != 0";
    }
}

TEST(MetricTensor_Covariance, IsLorentzian_PositiveDefiniteCov) {
    Eigen::Matrix3d cov;
    cov << 1.0, 0.3, 0.1,
           0.3, 1.0, 0.2,
           0.1, 0.2, 1.0;

    auto g = MetricTensor::make_from_covariance(1.0, cov);
    EXPECT_TRUE(g.is_lorentzian(origin()));
}

// ─── Inverse ──────────────────────────────────────────────────────────────────

TEST(MetricTensor_Inverse, Minkowski_InverseIsItself_NormalisedScale) {
    // For diag(-1,1,1,1), inverse = diag(-1,1,1,1)
    auto g    = MetricTensor::make_minkowski(1.0, 1.0);
    auto ginv = g.inverse(origin());

    ASSERT_TRUE(ginv.has_value());
    EXPECT_NEAR((*ginv)(0, 0), -1.0, FLOAT_EPSILON);
    for (int i = 1; i < SPACETIME_DIM; ++i)
        EXPECT_NEAR((*ginv)(i, i), 1.0, FLOAT_EPSILON);
}

TEST(MetricTensor_Inverse, Product_GTimesGinv_IsIdentity) {
    auto g    = MetricTensor::make_diagonal(1.0, {0.2, 0.3, 0.4});
    auto ginv = g.inverse(origin());

    ASSERT_TRUE(ginv.has_value());

    MetricMatrix product = g.evaluate(origin()) * (*ginv);
    EXPECT_TRUE(product.isApprox(MetricMatrix::Identity(), 1e-10))
        << "g * g_inv not identity:\n" << product;
}

TEST(MetricTensor_Inverse, ScaledMinkowski_InverseHasReciprocal) {
    // diag(-c², σ², σ², σ²) → inverse = diag(-1/c², 1/σ², ...)
    double c = 2.0, s = 0.5;
    auto g    = MetricTensor::make_minkowski(c, s);
    auto ginv = g.inverse(origin());

    ASSERT_TRUE(ginv.has_value());
    EXPECT_NEAR((*ginv)(0, 0), -1.0 / (c * c), FLOAT_EPSILON);
    for (int i = 1; i < SPACETIME_DIM; ++i)
        EXPECT_NEAR((*ginv)(i, i), 1.0 / (s * s), FLOAT_EPSILON);
}

// ─── Spacetime Interval ───────────────────────────────────────────────────────

TEST(MetricTensor_Interval, NullVector_ZeroInterval) {
    // For g = diag(-1,1,1,1), dx = (1,1,0,0): ds² = -1 + 1 = 0 (null)
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    FourVelocity dx;
    dx << 1.0, 1.0, 0.0, 0.0;
    EXPECT_NEAR(g.spacetime_interval(origin(), dx), 0.0, FLOAT_EPSILON);
}

TEST(MetricTensor_Interval, TimelikeVector_NegativeInterval) {
    // dx = (1,0,0,0): ds² = -1 (purely temporal)
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    FourVelocity dx;
    dx << 1.0, 0.0, 0.0, 0.0;
    EXPECT_LT(g.spacetime_interval(origin(), dx), 0.0);
}

TEST(MetricTensor_Interval, SpacelikeVector_PositiveInterval) {
    // dx = (0,1,0,0): ds² = +1 (purely spatial)
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    FourVelocity dx;
    dx << 0.0, 1.0, 0.0, 0.0;
    EXPECT_GT(g.spacetime_interval(origin(), dx), 0.0);
}

TEST(MetricTensor_Interval, Bilinear_ScaledVector) {
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    FourVelocity dx;
    dx << 2.0, 0.0, 0.0, 0.0;
    double ds2 = g.spacetime_interval(origin(), dx);
    EXPECT_NEAR(ds2, -4.0, FLOAT_EPSILON);  // g(dx, dx) = -1 * 4 = -4
}

// ─── Custom MetricFunction ────────────────────────────────────────────────────

TEST(MetricTensor_Custom, LambdaMetric_EvaluatesCorrectly) {
    // Metric whose g₁₁ grows linearly with x¹
    auto g = MetricTensor([](const SpacetimePoint& x) {
        MetricMatrix m = MetricMatrix::Zero();
        m(0, 0) = -1.0;
        m(1, 1) = 1.0 + x(1); // position-dependent
        m(2, 2) = 1.0;
        m(3, 3) = 1.0;
        return m;
    });

    SpacetimePoint p = SpacetimePoint::Zero();
    p(1) = 2.0;
    auto gx = g.evaluate(p);

    EXPECT_NEAR(gx(1, 1), 3.0, FLOAT_EPSILON);
}
