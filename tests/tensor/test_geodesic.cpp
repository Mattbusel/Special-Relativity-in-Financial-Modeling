#include <gtest/gtest.h>
#include "srfm/tensor.hpp"
#include "srfm/constants.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <vector>

using namespace srfm;
using namespace srfm::tensor;
using namespace srfm::constants;

static SpacetimePoint origin() { return SpacetimePoint::Zero(); }

// ─── GeodesicState Operators ──────────────────────────────────────────────────

TEST(GeodesicState_Operators, Addition_SumsPositionAndVelocity) {
    GeodesicState a{SpacetimePoint::Ones(), FourVelocity::Ones()};
    GeodesicState b{2.0 * SpacetimePoint::Ones(), 3.0 * FourVelocity::Ones()};
    GeodesicState c = a + b;

    EXPECT_TRUE(c.position.isApprox(3.0 * SpacetimePoint::Ones(), 1e-14));
    EXPECT_TRUE(c.velocity.isApprox(4.0 * FourVelocity::Ones(), 1e-14));
}

TEST(GeodesicState_Operators, ScalarMultiply_ScalesPositionAndVelocity) {
    GeodesicState s{SpacetimePoint::Ones(), FourVelocity::Ones()};
    GeodesicState r = 3.5 * s;

    EXPECT_TRUE(r.position.isApprox(3.5 * SpacetimePoint::Ones(), 1e-14));
    EXPECT_TRUE(r.velocity.isApprox(3.5 * FourVelocity::Ones(), 1e-14));
}

TEST(GeodesicState_Operators, ZeroScalar_ProducesZeroState) {
    GeodesicState s{SpacetimePoint::Ones(), FourVelocity::Ones()};
    GeodesicState r = 0.0 * s;

    EXPECT_TRUE(r.position.isZero(1e-14));
    EXPECT_TRUE(r.velocity.isZero(1e-14));
}

// ─── Flat Spacetime: Straight-Line Geodesics ─────────────────────────────────
//
// In flat (Minkowski) spacetime Γ^λ_μν = 0 everywhere.
// The geodesic equation reduces to d²x^λ/dτ² = 0 → x^λ(τ) = x^λ₀ + u^λ τ.
// Trajectory must be a straight line in spacetime.

TEST(Geodesic_Flat, PureTimelike_PositionAdvancesByVelocityTimesStep) {
    auto g     = MetricTensor::make_minkowski(1.0, 1.0);
    GeodesicSolver sol(g, 0.1);

    FourVelocity u0;
    u0 << 1.0, 0.0, 0.0, 0.0;

    auto traj = sol.integrate(origin(), u0, 10);

    ASSERT_EQ(static_cast<int>(traj.size()), 11);

    // After N steps of dτ=0.1, time component x⁰ = N * dτ * u⁰ = N * 0.1
    for (int i = 0; i <= 10; ++i) {
        EXPECT_NEAR(traj[i].position(0), i * 0.1, 1e-8)
            << "x⁰ wrong at step " << i;
        EXPECT_NEAR(traj[i].position(1), 0.0, 1e-8) << "x¹ wrong at step " << i;
    }
}

TEST(Geodesic_Flat, DiagonalVelocity_AllComponentsAdvanceLinearly) {
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    GeodesicSolver sol(g, 0.05);

    FourVelocity u0;
    u0 << 1.0, 0.2, -0.3, 0.1;

    auto traj = sol.integrate(origin(), u0, 20);

    double dtau = 0.05;
    for (int i = 1; i <= 20; ++i) {
        for (int mu = 0; mu < SPACETIME_DIM; ++mu) {
            double expected = u0(mu) * dtau * i;
            EXPECT_NEAR(traj[i].position(mu), expected, 1e-6)
                << "Component " << mu << " at step " << i;
        }
    }
}

TEST(Geodesic_Flat, Velocity_IsPreservedAlongFlatGeodesic) {
    // In flat spacetime, four-velocity is parallel transported → constant.
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    GeodesicSolver sol(g, 0.01);

    FourVelocity u0;
    u0 << 1.0, 0.3, 0.0, -0.1;

    auto traj = sol.integrate(origin(), u0, 50);

    for (int i = 1; i <= 50; ++i)
        EXPECT_TRUE(traj[i].velocity.isApprox(u0, 1e-8))
            << "Velocity changed at step " << i;
}

TEST(Geodesic_Flat, InitialState_IsFirstElement) {
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    GeodesicSolver sol(g, 0.1);

    SpacetimePoint x0;
    x0 << 1.0, 2.0, 3.0, 4.0;
    FourVelocity u0;
    u0 << 0.5, 0.1, 0.2, 0.3;

    auto traj = sol.integrate(x0, u0, 5);

    ASSERT_GE(static_cast<int>(traj.size()), 1);
    EXPECT_TRUE(traj[0].position.isApprox(x0, FLOAT_EPSILON));
    EXPECT_TRUE(traj[0].velocity.isApprox(u0, FLOAT_EPSILON));
}

TEST(Geodesic_Flat, ZeroSteps_ReturnsOnlyInitialState) {
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    GeodesicSolver sol(g, 0.1);

    auto traj = sol.integrate(origin(), FourVelocity::Zero(), 0);
    ASSERT_EQ(static_cast<int>(traj.size()), 1);
}

// ─── Curved Spacetime: Known Analytic Solution ───────────────────────────────
//
// Metric: g = diag(-1, e^x¹, 1, 1)
// Γ¹₁₁ = 0.5 (the only nonzero symbol)
//
// For motion with u = (0, u¹, 0, 0):
//   du¹/dτ = -Γ¹₁₁ (u¹)² = -0.5 (u¹)²
//   dx¹/dτ = u¹
//
// This is a Bernoulli ODE. With u¹(0) = v₀:
//   u¹(τ) = v₀ / (1 + 0.5 v₀ τ)
//   x¹(τ) = (2/v₀) * ln(1 + 0.5 v₀ τ)  [from integrating]
// For short τ and small v₀, test numerical vs analytic to ~1% accuracy.

TEST(Geodesic_Curved, ExponentialMetric_VelocityDeceleration) {
    auto g = MetricTensor([](const SpacetimePoint& x) {
        MetricMatrix m = MetricMatrix::Zero();
        m(0, 0) = -1.0;
        m(1, 1) = std::exp(x(1));
        m(2, 2) = 1.0;
        m(3, 3) = 1.0;
        return m;
    });

    GeodesicSolver sol(g, 0.001, 1e-6);

    FourVelocity u0 = FourVelocity::Zero();
    u0(1) = 1.0; // v₀ = 1

    auto traj = sol.integrate(origin(), u0, 100);

    // After τ = 0.1 (100 steps × 0.001):
    // u¹(0.1) = 1 / (1 + 0.5 * 0.1) = 1 / 1.05 ≈ 0.9524
    double tau   = 0.1;
    double v0    = 1.0;
    double u1_analytic = v0 / (1.0 + 0.5 * v0 * tau);
    double u1_numeric  = traj.back().velocity(1);

    EXPECT_NEAR(u1_numeric, u1_analytic, 1e-3)
        << "Velocity decelerates incorrectly in curved metric";
}

TEST(Geodesic_Curved, ExponentialMetric_UnperturbedComponents_Unchanged) {
    // Components 0, 2, 3 have no Christoffel force → must stay constant
    auto g = MetricTensor([](const SpacetimePoint& x) {
        MetricMatrix m = MetricMatrix::Zero();
        m(0, 0) = -1.0;
        m(1, 1) = std::exp(x(1));
        m(2, 2) = 1.0;
        m(3, 3) = 1.0;
        return m;
    });

    GeodesicSolver sol(g, 0.001, 1e-6);
    FourVelocity u0 = FourVelocity::Zero();
    u0(1) = 0.5;

    auto traj = sol.integrate(origin(), u0, 100);

    for (int i = 1; i <= 100; ++i) {
        EXPECT_NEAR(traj[i].velocity(0), 0.0, 1e-8)
            << "u⁰ changed at step " << i;
        EXPECT_NEAR(traj[i].velocity(2), 0.0, 1e-8)
            << "u² changed at step " << i;
        EXPECT_NEAR(traj[i].velocity(3), 0.0, 1e-8)
            << "u³ changed at step " << i;
    }
}

// ─── norm_squared ─────────────────────────────────────────────────────────────

TEST(Geodesic_NormSquared, Timelike_IsNegative) {
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    GeodesicSolver sol(g);

    FourVelocity u;
    u << 1.0, 0.0, 0.0, 0.0;

    EXPECT_LT(sol.norm_squared(origin(), u), 0.0);
}

TEST(Geodesic_NormSquared, Null_IsZero) {
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    GeodesicSolver sol(g);

    // ds² = -t² + x² = 0 when |t| = |x|
    FourVelocity u;
    u << 1.0, 1.0, 0.0, 0.0;

    EXPECT_NEAR(sol.norm_squared(origin(), u), 0.0, FLOAT_EPSILON);
}

TEST(Geodesic_NormSquared, Spacelike_IsPositive) {
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    GeodesicSolver sol(g);

    FourVelocity u;
    u << 0.0, 1.0, 0.0, 0.0;

    EXPECT_GT(sol.norm_squared(origin(), u), 0.0);
}

TEST(Geodesic_NormSquared, FlatMetric_PreservedAlongGeodesic) {
    // On a flat-metric geodesic the norm of u is a constant of motion.
    auto g = MetricTensor::make_minkowski(1.0, 1.0);
    GeodesicSolver sol(g, 0.01);

    FourVelocity u0;
    u0 << 1.0, 0.3, -0.1, 0.05;
    double n0 = sol.norm_squared(origin(), u0);

    auto traj = sol.integrate(origin(), u0, 100);

    for (int i = 1; i <= 100; ++i) {
        double ni = sol.norm_squared(traj[i].position, traj[i].velocity);
        EXPECT_NEAR(ni, n0, 1e-8) << "norm_squared changed at step " << i;
    }
}

// ─── Financial Interpretation ─────────────────────────────────────────────────
//
// With a covariance-based metric, the geodesic describes the natural
// drift path of a price trajectory that "follows the curvature" of the
// market's correlation structure. We verify basic sanity: the trajectory
// moves, is finite, and respects the initial conditions.

TEST(Geodesic_Financial, CovarianceMetric_FiniteTrajectory) {
    Eigen::Matrix3d cov;
    cov << 0.04, 0.01, 0.002,
           0.01, 0.09, 0.015,
           0.002, 0.015, 0.0225;

    auto g = MetricTensor::make_from_covariance(1.0, cov);
    GeodesicSolver sol(g, 0.01);

    FourVelocity u0;
    u0 << 1.0, 0.05, -0.03, 0.02;

    auto traj = sol.integrate(origin(), u0, 50);
    ASSERT_EQ(static_cast<int>(traj.size()), 51);

    for (const auto& state : traj) {
        EXPECT_TRUE(state.position.allFinite())
            << "Position became non-finite";
        EXPECT_TRUE(state.velocity.allFinite())
            << "Velocity became non-finite";
    }
}

TEST(Geodesic_Financial, CovarianceMetric_TimeComponentAdvances) {
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 0.04;
    auto g = MetricTensor::make_from_covariance(1.0, cov);
    GeodesicSolver sol(g, 0.1);

    FourVelocity u0;
    u0 << 1.0, 0.0, 0.0, 0.0;

    auto traj = sol.integrate(origin(), u0, 10);

    // t must increase monotonically (causality)
    for (int i = 1; i <= 10; ++i)
        EXPECT_GT(traj[i].position(0), traj[i - 1].position(0))
            << "Time component did not advance at step " << i;
}
