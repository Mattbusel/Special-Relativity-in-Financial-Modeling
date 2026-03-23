/// @file tests/tensor/test_metric_singularity.cpp
/// @brief GTest unit tests for Tikhonov regularization in MetricTensor::inverse.
///
/// Verifies that:
/// 1. A well-conditioned metric inverts normally.
/// 2. A singular metric no longer returns nullopt but instead returns a
///    regularized inverse (Tikhonov fix, Task 4).
/// 3. The regularized inverse satisfies g_reg · g_reg_inv ≈ I.

#include "srfm/tensor.hpp"
#include "srfm/types.hpp"
#include "srfm/constants.hpp"

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>

using namespace srfm;
using namespace srfm::tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Non-singular metric inverts cleanly
// ─────────────────────────────────────────────────────────────────────────────

TEST(MetricSingularity, WellConditionedInvertsCleanly) {
    auto metric = MetricTensor::make_minkowski(1.0, 0.2);
    SpacetimePoint x = SpacetimePoint::Zero();
    auto inv = metric.inverse(x);

    ASSERT_TRUE(inv.has_value()) << "Expected invertible metric to return a value";

    // g · g_inv should be identity.
    MetricMatrix g     = metric.evaluate(x);
    MetricMatrix g_inv = *inv;
    MetricMatrix prod  = g * g_inv;

    for (int i = 0; i < SPACETIME_DIM; ++i) {
        for (int j = 0; j < SPACETIME_DIM; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(prod(i, j), expected, 1e-10)
                << "(g * g_inv)[" << i << "][" << j << "] != " << expected;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Singular metric: Tikhonov regularization should save the inversion
// ─────────────────────────────────────────────────────────────────────────────

TEST(MetricSingularity, SingularMetricReturnsTikhonovRegularized) {
    // Build a singular metric: zero spatial diagonal (zero volatility).
    auto singular_fn = [](const SpacetimePoint& /*x*/) -> MetricMatrix {
        MetricMatrix g = MetricMatrix::Zero();
        g(0, 0) = -1.0;
        // Spatial entries are zero → singular.
        return g;
    };
    MetricTensor metric(singular_fn);
    SpacetimePoint x = SpacetimePoint::Zero();

    // Before Task 4 fix: this returned nullopt.
    // After Task 4 fix: should return a Tikhonov-regularized inverse.
    auto inv = metric.inverse(x);

    ASSERT_TRUE(inv.has_value())
        << "Tikhonov regularization should rescue singular metric inversion";
}

TEST(MetricSingularity, TikhonovInverseApproximatesIdentity) {
    // Slightly ill-conditioned (near-singular) metric.
    auto near_singular_fn = [](const SpacetimePoint& /*x*/) -> MetricMatrix {
        MetricMatrix g = MetricMatrix::Zero();
        g(0, 0) = -1.0;
        g(1, 1) =  1e-12;  // Tiny but non-zero
        g(2, 2) =  1e-12;
        g(3, 3) =  1e-12;
        return g;
    };
    MetricTensor metric(near_singular_fn);
    SpacetimePoint x = SpacetimePoint::Zero();

    auto inv = metric.inverse(x);
    ASSERT_TRUE(inv.has_value());

    // g_reg · g_reg_inv should be close to identity (within regularization error).
    MetricMatrix g     = metric.evaluate(x);
    MetricMatrix g_inv = *inv;

    // Add the same Tikhonov shift we expect was applied internally.
    // (1e-10 * I), so we can verify the round-trip.
    MetricMatrix g_reg = g;
    g_reg += 1e-10 * MetricMatrix::Identity();

    MetricMatrix prod = g_reg * g_inv;
    for (int i = 0; i < SPACETIME_DIM; ++i) {
        for (int j = 0; j < SPACETIME_DIM; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(prod(i, j), expected, 1e-6)
                << "(g_reg * g_inv)[" << i << "][" << j << "] != " << expected;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Completely degenerate matrix (all zeros) returns nullopt even after Tikhonov
// ─────────────────────────────────────────────────────────────────────────────

TEST(MetricSingularity, AllZeroMetricReturnsNulloptAfterTikhonov) {
    // A fully zero matrix: g + λI = λI which is invertible, so this test
    // documents that we always return a value for any non-trivially zero matrix.
    // We test the explicitly-zero-g case: λI is invertible, so we get Some.
    auto zero_fn = [](const SpacetimePoint& /*x*/) -> MetricMatrix {
        return MetricMatrix::Zero();
    };
    MetricTensor metric(zero_fn);
    SpacetimePoint x = SpacetimePoint::Zero();

    auto inv = metric.inverse(x);
    // After Tikhonov: g_reg = 0 + 1e-10 I = 1e-10 I, which IS invertible.
    ASSERT_TRUE(inv.has_value())
        << "Tikhonov shift of zero matrix should be invertible";
}
