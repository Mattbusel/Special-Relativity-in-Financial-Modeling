/**
 * @file  test_christoffel_n.cpp
 * @brief Tests for ChristoffelN — Stage 4 N-Asset Manifold.
 *
 * Coverage:
 *   - Flat (constant metric) → all Christoffel symbols exactly zero
 *   - Symmetry Γ^λ_μν = Γ^λ_νμ for multiple N
 *   - Dimension consistency of all_symbols()
 *   - verify_symmetry() helper
 *   - Boundary index validation
 *   - Parameter sweeps over N = 1..10
 *   - FD step convergence (flat case stays zero regardless of h)
 *   - Large N stress test (N=50)
 */

#include "srfm_test_n_asset.hpp"
#include "../../include/srfm/tensor/christoffel_n.hpp"

#include <cmath>
#include <vector>

using namespace srfm::tensor;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Eigen::MatrixXd diag_cov_ch(int n, double var = 0.04) {
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) { cov(i, i) = var * (i + 1); }
    return cov;
}

static Eigen::VectorXd zero_coord(int D) {
    return Eigen::VectorXd::Zero(D);
}

// ── Flat manifold → all Christoffel symbols are zero ─────────────────────────

static void test_christoffel_flat_minkowski_all_zero_n1() {
    auto cov = diag_cov_ch(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    Eigen::VectorXd x = zero_coord(2);
    // For constant metric all derivatives are zero → all Γ = 0.
    for (int lambda = 0; lambda < 2; ++lambda) {
        for (int mu = 0; mu < 2; ++mu) {
            for (int nu = 0; nu < 2; ++nu) {
                auto s = ch.symbol(lambda, mu, nu, x);
                SRFM_HAS_VALUE(s);
                SRFM_CHECK_NEAR(*s, 0.0, 1e-8);
            }
        }
    }
}

static void test_christoffel_flat_all_zero_n2() {
    auto cov = diag_cov_ch(2);
    auto m   = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    Eigen::VectorXd x = zero_coord(3);
    for (int l = 0; l < 3; ++l) {
        for (int mu = 0; mu < 3; ++mu) {
            for (int nu = 0; nu < 3; ++nu) {
                auto s = ch.symbol(l, mu, nu, x);
                SRFM_HAS_VALUE(s);
                SRFM_CHECK_NEAR(*s, 0.0, 1e-8);
            }
        }
    }
}

static void test_christoffel_flat_all_zero_n3() {
    auto cov = diag_cov_ch(3);
    auto m   = NAssetManifold::make(3, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    Eigen::VectorXd x = zero_coord(4);
    int D = 4;
    for (int l = 0; l < D; ++l) {
        for (int mu = 0; mu < D; ++mu) {
            for (int nu = 0; nu < D; ++nu) {
                auto s = ch.symbol(l, mu, nu, x);
                SRFM_HAS_VALUE(s);
                SRFM_CHECK_NEAR(*s, 0.0, 1e-8);
            }
        }
    }
}

static void test_christoffel_flat_all_zero_n4() {
    auto cov = diag_cov_ch(4);
    auto m   = NAssetManifold::make(4, cov, 1.5);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    Eigen::VectorXd x = zero_coord(5);
    int D = 5;
    for (int l = 0; l < D; ++l) {
        for (int mu = 0; mu < D; ++mu) {
            for (int nu = 0; nu < D; ++nu) {
                auto s = ch.symbol(l, mu, nu, x);
                SRFM_HAS_VALUE(s);
                SRFM_CHECK_NEAR(*s, 0.0, 1e-7);
            }
        }
    }
}

static void test_christoffel_zero_on_constant_metric_n5() {
    auto cov = diag_cov_ch(5);
    auto m   = NAssetManifold::make(5, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    // Test at a non-zero point — should still be zero for constant metric.
    Eigen::VectorXd x = Eigen::VectorXd::Constant(6, 42.7);
    int D = 6;
    for (int l = 0; l < D; ++l) {
        for (int mu = 0; mu < D; ++mu) {
            for (int nu = 0; nu < D; ++nu) {
                auto s = ch.symbol(l, mu, nu, x);
                SRFM_HAS_VALUE(s);
                SRFM_CHECK_NEAR(*s, 0.0, 1e-7);
            }
        }
    }
}

// ── all_symbols() dimension consistency ──────────────────────────────────────

static void test_christoffel_dimension_consistency_n1() {
    auto cov = diag_cov_ch(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    auto syms = ch.all_symbols(zero_coord(2));
    SRFM_HAS_VALUE(syms);
    SRFM_CHECK(static_cast<int>(syms->size()) == 2); // lambda dimension
    for (const auto& row : *syms) {
        SRFM_CHECK(static_cast<int>(row.size()) == 2); // mu dimension
        for (const auto& col : row) {
            SRFM_CHECK(static_cast<int>(col.size()) == 2); // nu dimension
        }
    }
}

static void test_christoffel_dimension_consistency_n4() {
    auto cov = diag_cov_ch(4);
    auto m   = NAssetManifold::make(4, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    auto syms = ch.all_symbols(zero_coord(5));
    SRFM_HAS_VALUE(syms);
    SRFM_CHECK(static_cast<int>(syms->size()) == 5);
    for (const auto& row : *syms) {
        SRFM_CHECK(static_cast<int>(row.size()) == 5);
        for (const auto& col : row) {
            SRFM_CHECK(static_cast<int>(col.size()) == 5);
        }
    }
}

static void test_christoffel_dimension_consistency_n10() {
    auto cov = diag_cov_ch(10);
    auto m   = NAssetManifold::make(10, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    auto syms = ch.all_symbols(zero_coord(11));
    SRFM_HAS_VALUE(syms);
    SRFM_CHECK(static_cast<int>(syms->size()) == 11);
}

// ── Symmetry tests ────────────────────────────────────────────────────────────

static void test_christoffel_symmetry_lower_indices_n1() {
    auto cov = diag_cov_ch(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    Eigen::VectorXd x = zero_coord(2);
    int D = 2;
    for (int l = 0; l < D; ++l) {
        for (int mu = 0; mu < D; ++mu) {
            for (int nu = 0; nu < D; ++nu) {
                auto s_mn = ch.symbol(l, mu, nu, x);
                auto s_nm = ch.symbol(l, nu, mu, x);
                SRFM_HAS_VALUE(s_mn);
                SRFM_HAS_VALUE(s_nm);
                SRFM_CHECK_NEAR(*s_mn, *s_nm, 1e-10);
            }
        }
    }
}

static void test_christoffel_symmetry_lower_indices_n2() {
    auto cov = diag_cov_ch(2);
    auto m   = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    Eigen::VectorXd x = zero_coord(3);
    int D = 3;
    for (int l = 0; l < D; ++l) {
        for (int mu = 0; mu < D; ++mu) {
            for (int nu = 0; nu < D; ++nu) {
                auto s_mn = ch.symbol(l, mu, nu, x);
                auto s_nm = ch.symbol(l, nu, mu, x);
                SRFM_HAS_VALUE(s_mn);
                SRFM_HAS_VALUE(s_nm);
                SRFM_CHECK_NEAR(*s_mn, *s_nm, 1e-10);
            }
        }
    }
}

static void test_christoffel_symmetry_lower_indices_n5() {
    auto cov = diag_cov_ch(5);
    auto m   = NAssetManifold::make(5, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    Eigen::VectorXd x = zero_coord(6);
    int D = 6;
    for (int l = 0; l < D; ++l) {
        for (int mu = 0; mu < D; ++mu) {
            for (int nu = mu; nu < D; ++nu) {
                auto s_mn = ch.symbol(l, mu, nu, x);
                auto s_nm = ch.symbol(l, nu, mu, x);
                SRFM_HAS_VALUE(s_mn);
                SRFM_HAS_VALUE(s_nm);
                SRFM_CHECK_NEAR(*s_mn, *s_nm, 1e-9);
            }
        }
    }
}

static void test_christoffel_verify_symmetry_n1_returns_true() {
    auto cov = diag_cov_ch(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    SRFM_CHECK(ch.verify_symmetry(zero_coord(2)) == true);
}

static void test_christoffel_verify_symmetry_n3_returns_true() {
    auto cov = diag_cov_ch(3);
    auto m   = NAssetManifold::make(3, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    SRFM_CHECK(ch.verify_symmetry(zero_coord(4)) == true);
}

static void test_christoffel_n3_verify_symmetry_all_points() {
    auto cov = diag_cov_ch(3);
    auto m   = NAssetManifold::make(3, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    // Check at several different points.
    std::vector<double> test_vals = {0.0, 1.0, -1.5, 10.0, -100.0};
    for (double v : test_vals) {
        Eigen::VectorXd x = Eigen::VectorXd::Constant(4, v);
        SRFM_CHECK(ch.verify_symmetry(x, 1e-8) == true);
    }
}

static void test_christoffel_n10_symmetry() {
    auto cov = diag_cov_ch(10);
    auto m   = NAssetManifold::make(10, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    SRFM_CHECK(ch.verify_symmetry(zero_coord(11)) == true);
}

// ── Index out of range returns nullopt ───────────────────────────────────────

static void test_christoffel_invalid_lambda_returns_nullopt() {
    auto cov = diag_cov_ch(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    auto s = ch.symbol(5, 0, 0, zero_coord(2)); // lambda=5 out of range
    SRFM_NO_VALUE(s);
}

static void test_christoffel_invalid_mu_returns_nullopt() {
    auto cov = diag_cov_ch(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    auto s = ch.symbol(0, 5, 0, zero_coord(2));
    SRFM_NO_VALUE(s);
}

static void test_christoffel_invalid_nu_returns_nullopt() {
    auto cov = diag_cov_ch(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    auto s = ch.symbol(0, 0, 5, zero_coord(2));
    SRFM_NO_VALUE(s);
}

static void test_christoffel_wrong_x_dim_returns_nullopt() {
    auto cov = diag_cov_ch(2);
    auto m   = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    // dim=3 but passing x of size 10
    auto s = ch.symbol(0, 0, 0, Eigen::VectorXd::Zero(10));
    SRFM_NO_VALUE(s);
}

static void test_christoffel_all_symbols_wrong_x_dim_nullopt() {
    auto cov = diag_cov_ch(2);
    auto m   = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    auto syms = ch.all_symbols(Eigen::VectorXd::Zero(10));
    SRFM_NO_VALUE(syms);
}

// ── dim() accessor ────────────────────────────────────────────────────────────

static void test_christoffel_dim_accessor_n1() {
    auto cov = diag_cov_ch(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    SRFM_CHECK(ch.dim() == 2);
}

static void test_christoffel_dim_accessor_n7() {
    auto cov = diag_cov_ch(7);
    auto m   = NAssetManifold::make(7, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    SRFM_CHECK(ch.dim() == 8);
}

// ── all_symbols() all zero for flat metric ────────────────────────────────────

static void test_christoffel_all_symbols_zero_flat_n2() {
    auto cov = diag_cov_ch(2);
    auto m   = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    auto syms = ch.all_symbols(zero_coord(3));
    SRFM_HAS_VALUE(syms);
    for (const auto& row_l : *syms) {
        for (const auto& row_mu : row_l) {
            for (double v : row_mu) {
                SRFM_CHECK_NEAR(v, 0.0, 1e-8);
            }
        }
    }
}

static void test_christoffel_all_symbols_zero_flat_n4() {
    auto cov = diag_cov_ch(4);
    auto m   = NAssetManifold::make(4, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    auto syms = ch.all_symbols(zero_coord(5));
    SRFM_HAS_VALUE(syms);
    for (const auto& row_l : *syms) {
        for (const auto& row_mu : row_l) {
            for (double v : row_mu) {
                SRFM_CHECK_NEAR(v, 0.0, 1e-7);
            }
        }
    }
}

// ── Large N stress test ───────────────────────────────────────────────────────

static void test_christoffel_n50_constructs_and_zero() {
    int N = 50;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(N, N);
    for (int i = 0; i < N; ++i) { cov(i, i) = 0.001 * (i + 1); }
    auto m = NAssetManifold::make(N, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    Eigen::VectorXd x = zero_coord(N + 1);
    // Sample a few symbols — all should be near zero.
    auto s0 = ch.symbol(0, 0, 0, x);
    SRFM_HAS_VALUE(s0);
    SRFM_CHECK_NEAR(*s0, 0.0, 1e-6);

    auto s1 = ch.symbol(N, N, N, x);
    SRFM_HAS_VALUE(s1);
    SRFM_CHECK_NEAR(*s1, 0.0, 1e-6);
}

// ── Symmetry sweep over multiple N ───────────────────────────────────────────

static void test_christoffel_symmetry_sweep_n1_to_8() {
    for (int n = 1; n <= 8; ++n) {
        auto cov = diag_cov_ch(n);
        auto m   = NAssetManifold::make(n, cov, 1.0);
        SRFM_HAS_VALUE(m);
        ChristoffelN ch(*m);
        SRFM_CHECK(ch.verify_symmetry(zero_coord(n + 1)) == true);
    }
}

// ── FD step convergence on flat manifold ─────────────────────────────────────

static void test_christoffel_fd_step_flat_stays_zero() {
    // For a constant metric, all derivatives should be zero regardless of h.
    // (The FD_STEP = 1e-5 is fixed, but we can verify the output stays zero.)
    auto cov = diag_cov_ch(3);
    auto m   = NAssetManifold::make(3, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    std::vector<Eigen::VectorXd> test_points = {
        zero_coord(4),
        Eigen::VectorXd::Constant(4, 1.0),
        Eigen::VectorXd::Constant(4, -5.0),
        Eigen::VectorXd::Constant(4, 1000.0),
    };
    for (const auto& x : test_points) {
        for (int l = 0; l < 4; ++l) {
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    auto s = ch.symbol(l, mu, nu, x);
                    SRFM_HAS_VALUE(s);
                    SRFM_CHECK_NEAR(*s, 0.0, 1e-7);
                }
            }
        }
    }
}

// ── Verify symmetry with tight and loose tolerances ──────────────────────────

static void test_christoffel_verify_symmetry_tight_tol() {
    auto cov = diag_cov_ch(2);
    auto m   = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    // Tight tolerance: still passes for flat metric.
    SRFM_CHECK(ch.verify_symmetry(zero_coord(3), 1e-12) == true);
}

static void test_christoffel_verify_symmetry_loose_tol() {
    auto cov = diag_cov_ch(5);
    auto m   = NAssetManifold::make(5, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    SRFM_CHECK(ch.verify_symmetry(zero_coord(6), 1.0) == true);
}

// ── all_symbols symmetry cross-check ─────────────────────────────────────────

static void test_christoffel_all_symbols_symmetry_n3() {
    auto cov = diag_cov_ch(3);
    auto m   = NAssetManifold::make(3, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    auto syms = ch.all_symbols(zero_coord(4));
    SRFM_HAS_VALUE(syms);
    int D = 4;
    for (int l = 0; l < D; ++l) {
        for (int mu = 0; mu < D; ++mu) {
            for (int nu = 0; nu < D; ++nu) {
                SRFM_CHECK_NEAR((*syms)[static_cast<std::size_t>(l)]
                                        [static_cast<std::size_t>(mu)]
                                        [static_cast<std::size_t>(nu)],
                                (*syms)[static_cast<std::size_t>(l)]
                                        [static_cast<std::size_t>(nu)]
                                        [static_cast<std::size_t>(mu)],
                                1e-10);
            }
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    SRFM_SUITE("ChristoffelN Flat Manifold Zero",
        test_christoffel_flat_minkowski_all_zero_n1,
        test_christoffel_flat_all_zero_n2,
        test_christoffel_flat_all_zero_n3,
        test_christoffel_flat_all_zero_n4,
        test_christoffel_zero_on_constant_metric_n5
    );
    SRFM_SUITE("ChristoffelN Dimension Consistency",
        test_christoffel_dimension_consistency_n1,
        test_christoffel_dimension_consistency_n4,
        test_christoffel_dimension_consistency_n10,
        test_christoffel_dim_accessor_n1,
        test_christoffel_dim_accessor_n7
    );
    SRFM_SUITE("ChristoffelN Symmetry",
        test_christoffel_symmetry_lower_indices_n1,
        test_christoffel_symmetry_lower_indices_n2,
        test_christoffel_symmetry_lower_indices_n5,
        test_christoffel_verify_symmetry_n1_returns_true,
        test_christoffel_verify_symmetry_n3_returns_true,
        test_christoffel_n3_verify_symmetry_all_points,
        test_christoffel_n10_symmetry,
        test_christoffel_symmetry_sweep_n1_to_8,
        test_christoffel_all_symbols_symmetry_n3
    );
    SRFM_SUITE("ChristoffelN Validation",
        test_christoffel_invalid_lambda_returns_nullopt,
        test_christoffel_invalid_mu_returns_nullopt,
        test_christoffel_invalid_nu_returns_nullopt,
        test_christoffel_wrong_x_dim_returns_nullopt,
        test_christoffel_all_symbols_wrong_x_dim_nullopt
    );
    SRFM_SUITE("ChristoffelN All Symbols",
        test_christoffel_all_symbols_zero_flat_n2,
        test_christoffel_all_symbols_zero_flat_n4
    );
    SRFM_SUITE("ChristoffelN Large N",
        test_christoffel_n50_constructs_and_zero
    );
    SRFM_SUITE("ChristoffelN FD Convergence",
        test_christoffel_fd_step_flat_stays_zero
    );
    SRFM_SUITE("ChristoffelN Tolerances",
        test_christoffel_verify_symmetry_tight_tol,
        test_christoffel_verify_symmetry_loose_tol
    );
    return srfm_test::report();
}
