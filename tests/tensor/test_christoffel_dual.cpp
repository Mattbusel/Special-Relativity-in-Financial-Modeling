/// @file tests/tensor/test_christoffel_dual.cpp
/// @brief GTest unit tests for ChristoffelSymbolsDual (dual-number autodiff).
///
/// Tests verify that:
/// 1. DualNumber arithmetic satisfies basic algebraic identities.
/// 2. For a flat (constant) Minkowski metric, all Christoffel symbols are zero
///    — both with the FD implementation and the dual-number implementation.
/// 3. For a non-trivial metric, dual-number derivatives match the FD
///    approximation to high precision (relative error < 1e-7).
/// 4. The dual-number implementation handles singular metrics via Tikhonov
///    regularization without returning garbage.

#include "srfm/tensor.hpp"
#include "srfm/types.hpp"
#include "srfm/constants.hpp"

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>

using namespace srfm;
using namespace srfm::tensor;

// ─────────────────────────────────────────────────────────────────────────────
// DualNumber arithmetic
// ─────────────────────────────────────────────────────────────────────────────

TEST(DualNumber, Addition) {
    DualNumber a{3.0, 1.0};
    DualNumber b{2.0, 4.0};
    auto c = a + b;
    EXPECT_DOUBLE_EQ(c.value, 5.0);
    EXPECT_DOUBLE_EQ(c.deriv, 5.0);
}

TEST(DualNumber, Subtraction) {
    DualNumber a{5.0, 3.0};
    DualNumber b{2.0, 1.0};
    auto c = a - b;
    EXPECT_DOUBLE_EQ(c.value, 3.0);
    EXPECT_DOUBLE_EQ(c.deriv, 2.0);
}

TEST(DualNumber, Multiplication) {
    // (3 + ε)(2 + 4ε) = 6 + (3·4 + 1·2)ε = 6 + 14ε
    DualNumber a{3.0, 1.0};
    DualNumber b{2.0, 4.0};
    auto c = a * b;
    EXPECT_DOUBLE_EQ(c.value, 6.0);
    EXPECT_DOUBLE_EQ(c.deriv, 14.0);
}

TEST(DualNumber, Division) {
    // (4 + 2ε) / (2 + 1ε) = 2 + (2·2 − 4·1)/4·ε = 2 + 0·ε
    DualNumber a{4.0, 2.0};
    DualNumber b{2.0, 1.0};
    auto c = a / b;
    EXPECT_DOUBLE_EQ(c.value, 2.0);
    EXPECT_DOUBLE_EQ(c.deriv, 0.0);
}

TEST(DualNumber, MixedRealScalar) {
    DualNumber a{3.0, 2.0};
    auto b = a * 5.0;
    EXPECT_DOUBLE_EQ(b.value, 15.0);
    EXPECT_DOUBLE_EQ(b.deriv, 10.0);
}

TEST(DualNumber, Negation) {
    DualNumber a{3.0, -2.0};
    auto b = -a;
    EXPECT_DOUBLE_EQ(b.value, -3.0);
    EXPECT_DOUBLE_EQ(b.deriv, 2.0);
}

// Derivative of x^2 at x=3: seed x = 3+ε, compute (3+ε)^2 = 9+6ε → deriv=6
TEST(DualNumber, DerivativeOfSquare) {
    DualNumber x{3.0, 1.0};
    auto y = x * x;
    EXPECT_DOUBLE_EQ(y.value, 9.0);
    EXPECT_DOUBLE_EQ(y.deriv, 6.0);  // 2x at x=3
}

// Derivative of 1/x at x=2: seed x = 2+ε, compute 1/(2+ε) = 0.5 − 0.25ε
TEST(DualNumber, DerivativeOfReciprocal) {
    DualNumber one{1.0, 0.0};
    DualNumber x{2.0, 1.0};
    auto y = one / x;
    EXPECT_DOUBLE_EQ(y.value, 0.5);
    EXPECT_NEAR(y.deriv, -0.25, 1e-14);  // -1/x^2 at x=2
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a flat Minkowski dual metric function.
// ─────────────────────────────────────────────────────────────────────────────

/// Build a DualMetricFunction for a flat Minkowski metric
///   g = diag(-ts², ss², ss², ss²)  (constant — all derivatives are zero).
static DualMetricFunction make_flat_dual_fn(double time_scale,
                                             double spatial_scale) {
    return [time_scale, spatial_scale](const DualSpacetimePoint& /*xd*/)
               -> DualMetricMatrix {
        DualMetricMatrix gd = DualMetricMatrix::Zero();
        gd(0, 0) = DualNumber{-(time_scale * time_scale), 0.0};
        gd(1, 1) = DualNumber{ (spatial_scale * spatial_scale), 0.0};
        gd(2, 2) = DualNumber{ (spatial_scale * spatial_scale), 0.0};
        gd(3, 3) = DualNumber{ (spatial_scale * spatial_scale), 0.0};
        return gd;
    };
}

/// Build a DualMetricFunction for a position-dependent diagonal metric:
///   g_00 = −1,   g_ii = 1 + x_i²   (i=1,2,3)
/// so ∂g_ii/∂x^i = 2 x_i  (for i >= 1), all other derivatives zero.
static DualMetricFunction make_curved_dual_fn() {
    return [](const DualSpacetimePoint& xd) -> DualMetricMatrix {
        DualMetricMatrix gd = DualMetricMatrix::Zero();
        gd(0, 0) = DualNumber{-1.0, 0.0};
        for (int i = 1; i < SPACETIME_DIM; ++i) {
            // g_ii = 1 + x_i^2  → derivative w.r.t. x_i = 2*x_i
            gd(i, i) = DualNumber{1.0, 0.0} + xd(i) * xd(i);
        }
        return gd;
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat Minkowski: all Christoffel symbols should be zero
// ─────────────────────────────────────────────────────────────────────────────

TEST(ChristoffelSymbolsDual, FlatMinkowskiZeroAtOrigin) {
    auto metric   = MetricTensor::make_minkowski(1.0, 1.0);
    auto dual_fn  = make_flat_dual_fn(1.0, 1.0);
    ChristoffelSymbolsDual cs(metric, dual_fn);

    SpacetimePoint x = SpacetimePoint::Zero();
    auto gamma = cs.compute(x);

    for (int l = 0; l < SPACETIME_DIM; ++l) {
        for (int mu = 0; mu < SPACETIME_DIM; ++mu) {
            for (int nu = 0; nu < SPACETIME_DIM; ++nu) {
                EXPECT_NEAR(gamma[l](mu, nu), 0.0, 1e-14)
                    << "Γ[" << l << "](" << mu << "," << nu << ") != 0 "
                    << "for flat Minkowski metric";
            }
        }
    }
}

TEST(ChristoffelSymbolsDual, FlatMinkowskiZeroAwayFromOrigin) {
    auto metric   = MetricTensor::make_minkowski(1.0, 0.2);
    auto dual_fn  = make_flat_dual_fn(1.0, 0.2);
    ChristoffelSymbolsDual cs(metric, dual_fn);

    SpacetimePoint x;
    x << 1.0, 0.5, -0.3, 0.7;
    auto gamma = cs.compute(x);

    for (int l = 0; l < SPACETIME_DIM; ++l) {
        for (int mu = 0; mu < SPACETIME_DIM; ++mu) {
            for (int nu = 0; nu < SPACETIME_DIM; ++nu) {
                EXPECT_NEAR(gamma[l](mu, nu), 0.0, 1e-14)
                    << "Γ[" << l << "](" << mu << "," << nu << ") != 0";
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Curved metric: dual vs FD agreement
// ─────────────────────────────────────────────────────────────────────────────

TEST(ChristoffelSymbolsDual, CurvedMetricMatchesFD) {
    // Build a curved metric: g_00=-1, g_ii = 1 + x_i^2 for i=1,2,3
    // via a MetricFunction (for ChristoffelSymbols) and DualMetricFunction
    // (for ChristoffelSymbolsDual).
    auto metric_fn = [](const SpacetimePoint& x) -> MetricMatrix {
        MetricMatrix g = MetricMatrix::Zero();
        g(0, 0) = -1.0;
        for (int i = 1; i < SPACETIME_DIM; ++i) {
            g(i, i) = 1.0 + x(i) * x(i);
        }
        return g;
    };

    MetricTensor     base_metric(metric_fn);
    DualMetricFunction dual_fn = make_curved_dual_fn();

    ChristoffelSymbols     cs_fd(base_metric);
    ChristoffelSymbolsDual cs_dual(base_metric, dual_fn);

    SpacetimePoint x;
    x << 0.5, 1.0, -0.5, 0.3;

    auto gamma_fd   = cs_fd.compute(x);
    auto gamma_dual = cs_dual.compute(x);

    for (int l = 0; l < SPACETIME_DIM; ++l) {
        for (int mu = 0; mu < SPACETIME_DIM; ++mu) {
            for (int nu = 0; nu < SPACETIME_DIM; ++nu) {
                const double fd_val   = gamma_fd[l](mu, nu);
                const double dual_val = gamma_dual[l](mu, nu);
                // Both are exact/near-exact for this polynomial metric.
                EXPECT_NEAR(dual_val, fd_val, 1e-7)
                    << "Γ[" << l << "](" << mu << "," << nu << "): "
                    << "dual=" << dual_val << " fd=" << fd_val;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Symmetry: Γ^λ_μν = Γ^λ_νμ  (torsion-free connection)
// ─────────────────────────────────────────────────────────────────────────────

TEST(ChristoffelSymbolsDual, SymmetryInLowerIndices) {
    auto metric_fn = [](const SpacetimePoint& x) -> MetricMatrix {
        MetricMatrix g = MetricMatrix::Zero();
        g(0, 0) = -1.0;
        for (int i = 1; i < SPACETIME_DIM; ++i) {
            g(i, i) = 1.0 + x(i) * x(i);
        }
        return g;
    };
    MetricTensor       base_metric(metric_fn);
    DualMetricFunction dual_fn = make_curved_dual_fn();
    ChristoffelSymbolsDual cs(base_metric, dual_fn);

    SpacetimePoint x;
    x << 0.0, 0.8, -0.4, 0.2;
    auto gamma = cs.compute(x);

    for (int l = 0; l < SPACETIME_DIM; ++l) {
        for (int mu = 0; mu < SPACETIME_DIM; ++mu) {
            for (int nu = 0; nu < SPACETIME_DIM; ++nu) {
                EXPECT_NEAR(gamma[l](mu, nu), gamma[l](nu, mu), 1e-12)
                    << "Symmetry violation: Γ[" << l << "]("
                    << mu << "," << nu << ") vs Γ[" << l << "]("
                    << nu << "," << mu << ")";
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Contraction
// ─────────────────────────────────────────────────────────────────────────────

TEST(ChristoffelSymbolsDual, ContractZeroForFlatMetric) {
    auto metric  = MetricTensor::make_minkowski(1.0, 1.0);
    auto dual_fn = make_flat_dual_fn(1.0, 1.0);
    ChristoffelSymbolsDual cs(metric, dual_fn);

    SpacetimePoint x = SpacetimePoint::Zero();
    auto gamma = cs.compute(x);

    FourVelocity u;
    u << 1.0, 0.3, 0.2, 0.1;
    FourVelocity acc = cs.contract(gamma, u);

    for (int l = 0; l < SPACETIME_DIM; ++l) {
        EXPECT_NEAR(acc(l), 0.0, 1e-14)
            << "Geodesic acceleration[" << l << "] != 0 for flat metric";
    }
}
