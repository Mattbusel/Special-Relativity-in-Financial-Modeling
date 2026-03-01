/**
 * @file  test_n_asset_manifold.cpp
 * @brief Tests for NAssetManifold — Stage 4 N-Asset Manifold.
 *
 * Coverage:
 *   - Construction (valid and invalid inputs)
 *   - Metric signature and structure
 *   - Metric symmetry
 *   - Metric inverse correctness
 *   - Line element computation
 *   - is_flat() predicate
 *   - dim() and n_assets() accessors
 *   - covariance() round-trip
 *   - reduces_to_4d() compatibility check
 *   - Parameter sweeps over N = 1..50
 *   - Multiple random-deterministic covariance matrices
 */

#include "srfm_test_n_asset.hpp"
#include "../../include/srfm/tensor/n_asset_manifold.hpp"

#include <cmath>
#include <vector>

using namespace srfm::tensor;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a diagonal covariance matrix with given variances.
static Eigen::MatrixXd diag_cov(const std::vector<double>& variances) {
    int N = static_cast<int>(variances.size());
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(N, N);
    for (int i = 0; i < N; ++i) {
        cov(i, i) = variances[i];
    }
    return cov;
}

/// Build a 2×2 covariance matrix from σ1, σ2 and correlation ρ.
static Eigen::MatrixXd cov_2x2(double s1, double s2, double rho) {
    Eigen::MatrixXd c(2, 2);
    c(0, 0) = s1 * s1;
    c(0, 1) = rho * s1 * s2;
    c(1, 0) = rho * s1 * s2;
    c(1, 1) = s2 * s2;
    return c;
}

/// Return a coordinate vector of the given dimension, filled with value v.
static Eigen::VectorXd coord(int D, double v = 0.0) {
    return Eigen::VectorXd::Constant(D, v);
}

// ── Basic construction tests ──────────────────────────────────────────────────

static void test_manifold_make_valid_1asset() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04; // σ = 0.2
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    SRFM_CHECK(m->n_assets() == 1);
    SRFM_CHECK(m->dim() == 2);
}

static void test_manifold_make_valid_2asset() {
    auto cov = cov_2x2(0.2, 0.3, 0.5);
    auto m = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    SRFM_CHECK(m->n_assets() == 2);
    SRFM_CHECK(m->dim() == 3);
}

static void test_manifold_make_valid_3asset() {
    auto cov = diag_cov({0.01, 0.04, 0.09});
    auto m = NAssetManifold::make(3, cov, 1.0);
    SRFM_HAS_VALUE(m);
    SRFM_CHECK(m->n_assets() == 3);
    SRFM_CHECK(m->dim() == 4);
}

static void test_manifold_make_valid_4asset() {
    auto cov = diag_cov({0.01, 0.04, 0.09, 0.16});
    auto m = NAssetManifold::make(4, cov, 1.0);
    SRFM_HAS_VALUE(m);
    SRFM_CHECK(m->n_assets() == 4);
    SRFM_CHECK(m->dim() == 5);
}

static void test_manifold_make_valid_10asset() {
    std::vector<double> vars(10);
    for (int i = 0; i < 10; ++i) { vars[static_cast<std::size_t>(i)] = 0.01 * (i + 1); }
    auto cov = diag_cov(vars);
    auto m = NAssetManifold::make(10, cov, 1.0);
    SRFM_HAS_VALUE(m);
    SRFM_CHECK(m->n_assets() == 10);
    SRFM_CHECK(m->dim() == 11);
}

static void test_manifold_make_valid_50asset() {
    std::vector<double> vars(50);
    for (int i = 0; i < 50; ++i) { vars[static_cast<std::size_t>(i)] = 0.001 * (i + 1); }
    auto cov = diag_cov(vars);
    auto m = NAssetManifold::make(50, cov, 2.5);
    SRFM_HAS_VALUE(m);
    SRFM_CHECK(m->n_assets() == 50);
    SRFM_CHECK(m->dim() == 51);
}

static void test_manifold_make_invalid_zero_n() {
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(1, 1);
    auto m = NAssetManifold::make(0, cov, 1.0);
    SRFM_NO_VALUE(m);
}

static void test_manifold_make_invalid_negative_n() {
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(1, 1);
    auto m = NAssetManifold::make(-1, cov, 1.0);
    SRFM_NO_VALUE(m);
}

static void test_manifold_make_invalid_singular_cov() {
    // Rank-deficient covariance (row 2 = row 1).
    Eigen::MatrixXd cov(2, 2);
    cov << 1.0, 1.0,
           1.0, 1.0;
    auto m = NAssetManifold::make(2, cov, 1.0);
    SRFM_NO_VALUE(m);
}

static void test_manifold_make_invalid_c_market_zero() {
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2) * 0.04;
    auto m = NAssetManifold::make(2, cov, 0.0);
    SRFM_NO_VALUE(m);
}

static void test_manifold_make_invalid_c_market_negative() {
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2) * 0.04;
    auto m = NAssetManifold::make(2, cov, -1.0);
    SRFM_NO_VALUE(m);
}

static void test_manifold_make_invalid_wrong_size_cov() {
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(3, 3) * 0.04;
    auto m = NAssetManifold::make(2, cov, 1.0); // n=2 but cov is 3×3
    SRFM_NO_VALUE(m);
}

static void test_manifold_make_invalid_asymmetric_cov() {
    Eigen::MatrixXd cov(2, 2);
    cov << 1.0, 0.5,
           0.0, 1.0; // not symmetric
    auto m = NAssetManifold::make(2, cov, 1.0);
    SRFM_NO_VALUE(m);
}

// ── Metric diagonal signature tests ──────────────────────────────────────────

static void test_manifold_metric_diagonal_signature_n1() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto g = m->metric_at(coord(2));
    SRFM_HAS_VALUE(g);
    // g_00 = -c² = -1
    SRFM_CHECK_NEAR((*g)(0, 0), -1.0, 1e-12);
    // g_11 = σ² = 0.04
    SRFM_CHECK_NEAR((*g)(1, 1), 0.04, 1e-12);
}

static void test_manifold_metric_diagonal_signature_n4() {
    std::vector<double> vars = {0.01, 0.04, 0.09, 0.16};
    auto cov = diag_cov(vars);
    auto m = NAssetManifold::make(4, cov, 2.0);
    SRFM_HAS_VALUE(m);
    auto g = m->metric_at(coord(5));
    SRFM_HAS_VALUE(g);
    // g_00 = -c² = -4
    SRFM_CHECK_NEAR((*g)(0, 0), -4.0, 1e-12);
    for (int i = 1; i <= 4; ++i) {
        SRFM_CHECK_NEAR((*g)(i, i), vars[static_cast<std::size_t>(i - 1)], 1e-12);
    }
}

static void test_manifold_metric_time_space_cross_terms_zero_n2() {
    auto cov = cov_2x2(0.2, 0.3, 0.5);
    auto m = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto g = m->metric_at(coord(3));
    SRFM_HAS_VALUE(g);
    // g_01 = g_10 = 0
    SRFM_CHECK_NEAR((*g)(0, 1), 0.0, 1e-12);
    SRFM_CHECK_NEAR((*g)(1, 0), 0.0, 1e-12);
    // g_02 = g_20 = 0
    SRFM_CHECK_NEAR((*g)(0, 2), 0.0, 1e-12);
    SRFM_CHECK_NEAR((*g)(2, 0), 0.0, 1e-12);
}

static void test_manifold_metric_off_diagonal_correlation_n2() {
    double s1 = 0.2, s2 = 0.3, rho = 0.6;
    auto cov = cov_2x2(s1, s2, rho);
    auto m = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto g = m->metric_at(coord(3));
    SRFM_HAS_VALUE(g);
    // g_12 = ρ σ1 σ2
    double expected_off = rho * s1 * s2;
    SRFM_CHECK_NEAR((*g)(1, 2), expected_off, 1e-12);
    SRFM_CHECK_NEAR((*g)(2, 1), expected_off, 1e-12);
}

// ── Metric symmetry ───────────────────────────────────────────────────────────

static void test_manifold_metric_symmetry_n1() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto g = m->metric_at(coord(2));
    SRFM_HAS_VALUE(g);
    int D = 2;
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            SRFM_CHECK_NEAR((*g)(i, j), (*g)(j, i), 1e-12);
        }
    }
}

static void test_manifold_metric_symmetry_n3() {
    auto cov = diag_cov({0.01, 0.04, 0.09});
    auto m = NAssetManifold::make(3, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto g = m->metric_at(coord(4));
    SRFM_HAS_VALUE(g);
    int D = 4;
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            SRFM_CHECK_NEAR((*g)(i, j), (*g)(j, i), 1e-12);
        }
    }
}

static void test_manifold_metric_symmetry_n10() {
    std::vector<double> vars(10);
    for (int i = 0; i < 10; ++i) { vars[static_cast<std::size_t>(i)] = 0.01 * (i + 1); }
    auto cov = diag_cov(vars);
    auto m = NAssetManifold::make(10, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto g = m->metric_at(coord(11));
    SRFM_HAS_VALUE(g);
    int D = 11;
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            SRFM_CHECK_NEAR((*g)(i, j), (*g)(j, i), 1e-12);
        }
    }
}

// ── Metric inverse tests ──────────────────────────────────────────────────────

static void test_manifold_metric_inverse_identity_n1() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto g    = m->metric_at(coord(2));
    auto ginv = m->inverse_metric_at(coord(2));
    SRFM_HAS_VALUE(g);
    SRFM_HAS_VALUE(ginv);
    Eigen::MatrixXd prod = (*g) * (*ginv);
    Eigen::MatrixXd I    = Eigen::MatrixXd::Identity(2, 2);
    SRFM_CHECK((prod - I).norm() < 1e-12);
}

static void test_manifold_metric_inverse_identity_n2() {
    auto cov = cov_2x2(0.2, 0.3, 0.5);
    auto m = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto g    = m->metric_at(coord(3));
    auto ginv = m->inverse_metric_at(coord(3));
    SRFM_HAS_VALUE(g);
    SRFM_HAS_VALUE(ginv);
    Eigen::MatrixXd prod = (*g) * (*ginv);
    Eigen::MatrixXd I    = Eigen::MatrixXd::Identity(3, 3);
    SRFM_CHECK((prod - I).norm() < 1e-10);
}

static void test_manifold_metric_inverse_identity_n4() {
    auto cov = diag_cov({0.01, 0.04, 0.09, 0.16});
    auto m = NAssetManifold::make(4, cov, 1.5);
    SRFM_HAS_VALUE(m);
    auto g    = m->metric_at(coord(5));
    auto ginv = m->inverse_metric_at(coord(5));
    SRFM_HAS_VALUE(g);
    SRFM_HAS_VALUE(ginv);
    Eigen::MatrixXd prod = (*g) * (*ginv);
    Eigen::MatrixXd I    = Eigen::MatrixXd::Identity(5, 5);
    SRFM_CHECK((prod - I).norm() < 1e-10);
}

static void test_manifold_metric_inverse_identity_n10() {
    std::vector<double> vars(10);
    for (int i = 0; i < 10; ++i) { vars[static_cast<std::size_t>(i)] = 0.01 * (i + 1); }
    auto cov = diag_cov(vars);
    auto m = NAssetManifold::make(10, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto g    = m->metric_at(coord(11));
    auto ginv = m->inverse_metric_at(coord(11));
    SRFM_HAS_VALUE(g);
    SRFM_HAS_VALUE(ginv);
    Eigen::MatrixXd prod = (*g) * (*ginv);
    Eigen::MatrixXd I    = Eigen::MatrixXd::Identity(11, 11);
    SRFM_CHECK((prod - I).norm() < 1e-9);
}

static void test_manifold_inverse_metric_nonsingular_n1() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 1.0;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto ginv = m->inverse_metric_at(coord(2));
    SRFM_HAS_VALUE(ginv);
    // Inverse of [[−1,0],[0,1]] is [[−1,0],[0,1]].
    SRFM_CHECK_NEAR((*ginv)(0, 0), -1.0, 1e-12);
    SRFM_CHECK_NEAR((*ginv)(1, 1),  1.0, 1e-12);
}

// ── Line element tests ────────────────────────────────────────────────────────

static void test_manifold_line_element_pure_time_negative_n1() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    // dx = (dt, 0): pure time displacement.
    Eigen::VectorXd dx(2);
    dx(0) = 1.0; dx(1) = 0.0;
    auto ds2 = m->line_element_sq(coord(2), dx);
    SRFM_HAS_VALUE(ds2);
    SRFM_CHECK(*ds2 < 0.0); // timelike
}

static void test_manifold_line_element_pure_spatial_positive_n1() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    // dx = (0, Δp): pure spatial displacement.
    Eigen::VectorXd dx(2);
    dx(0) = 0.0; dx(1) = 1.0;
    auto ds2 = m->line_element_sq(coord(2), dx);
    SRFM_HAS_VALUE(ds2);
    SRFM_CHECK(*ds2 > 0.0); // spacelike
}

static void test_manifold_line_element_flat_minkowski_n1() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 1.0; // σ² = 1 so c_market = 1 gives standard Minkowski
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    Eigen::VectorXd dx(2);
    dx(0) = 2.0; dx(1) = 1.0;
    // ds² = -c²(dt)² + σ²(dp)² = -4 + 1 = -3
    auto ds2 = m->line_element_sq(coord(2), dx);
    SRFM_HAS_VALUE(ds2);
    SRFM_CHECK_NEAR(*ds2, -3.0, 1e-12);
}

static void test_manifold_line_element_spacelike_n2() {
    auto cov = cov_2x2(0.2, 0.3, 0.0);
    auto m = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    // dx = (0.01, 1.0, 1.0) — small time step, large price moves.
    Eigen::VectorXd dx(3);
    dx(0) = 0.001; dx(1) = 1.0; dx(2) = 1.0;
    auto ds2 = m->line_element_sq(coord(3), dx);
    SRFM_HAS_VALUE(ds2);
    SRFM_CHECK(*ds2 > 0.0);
}

static void test_manifold_line_element_timelike_n2() {
    auto cov = cov_2x2(0.2, 0.3, 0.0);
    auto m = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    // dx = (10.0, 0.001, 0.001) — large time step, tiny price moves.
    Eigen::VectorXd dx(3);
    dx(0) = 10.0; dx(1) = 0.001; dx(2) = 0.001;
    auto ds2 = m->line_element_sq(coord(3), dx);
    SRFM_HAS_VALUE(ds2);
    SRFM_CHECK(*ds2 < 0.0);
}

static void test_manifold_line_element_wrong_dim_returns_nullopt() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    Eigen::VectorXd dx(5); // wrong size
    auto ds2 = m->line_element_sq(coord(2), dx);
    SRFM_NO_VALUE(ds2);
}

// ── c_market scaling test ─────────────────────────────────────────────────────

static void test_manifold_c_market_scales_time_component() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    double c = 3.7;
    auto m = NAssetManifold::make(1, cov, c);
    SRFM_HAS_VALUE(m);
    auto g = m->metric_at(coord(2));
    SRFM_HAS_VALUE(g);
    SRFM_CHECK_NEAR((*g)(0, 0), -(c * c), 1e-12);
}

static void test_manifold_c_market_sweeps() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    std::vector<double> cs = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0};
    for (double c : cs) {
        auto m = NAssetManifold::make(1, cov, c);
        SRFM_HAS_VALUE(m);
        auto g = m->metric_at(coord(2));
        SRFM_HAS_VALUE(g);
        SRFM_CHECK_NEAR((*g)(0, 0), -(c * c), 1e-10);
        SRFM_CHECK(m->c_market() == c);
    }
}

// ── is_flat() tests ───────────────────────────────────────────────────────────

static void test_manifold_is_flat_returns_true_n1() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    SRFM_CHECK(m->is_flat() == true);
}

static void test_manifold_is_flat_returns_true_n10() {
    std::vector<double> vars(10);
    for (int i = 0; i < 10; ++i) { vars[static_cast<std::size_t>(i)] = 0.01 * (i + 1); }
    auto cov = diag_cov(vars);
    auto m = NAssetManifold::make(10, cov, 1.0);
    SRFM_HAS_VALUE(m);
    SRFM_CHECK(m->is_flat() == true);
}

// ── dim() and n_assets() sweep ────────────────────────────────────────────────

static void test_manifold_dim_returns_n_plus_1_sweep() {
    for (int n = 1; n <= 20; ++n) {
        std::vector<double> vars(static_cast<std::size_t>(n), 0.01);
        auto cov = diag_cov(vars);
        auto m = NAssetManifold::make(n, cov, 1.0);
        SRFM_HAS_VALUE(m);
        SRFM_CHECK(m->dim() == n + 1);
        SRFM_CHECK(m->n_assets() == n);
    }
}

// ── covariance() round-trip ───────────────────────────────────────────────────

static void test_manifold_covariance_roundtrip_n1() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.0625;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto cov_out = m->covariance();
    SRFM_HAS_VALUE(cov_out);
    SRFM_CHECK_NEAR((*cov_out)(0, 0), 0.0625, 1e-12);
}

static void test_manifold_covariance_roundtrip_n3() {
    auto cov = diag_cov({0.01, 0.04, 0.09});
    auto m = NAssetManifold::make(3, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto cov_out = m->covariance();
    SRFM_HAS_VALUE(cov_out);
    SRFM_CHECK_NEAR((*cov_out)(0, 0), 0.01, 1e-12);
    SRFM_CHECK_NEAR((*cov_out)(1, 1), 0.04, 1e-12);
    SRFM_CHECK_NEAR((*cov_out)(2, 2), 0.09, 1e-12);
}

static void test_manifold_covariance_roundtrip_with_correlation() {
    double s1 = 0.2, s2 = 0.4, rho = 0.7;
    auto cov = cov_2x2(s1, s2, rho);
    auto m = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto cov_out = m->covariance();
    SRFM_HAS_VALUE(cov_out);
    SRFM_CHECK_NEAR((*cov_out)(0, 0), s1 * s1, 1e-12);
    SRFM_CHECK_NEAR((*cov_out)(1, 1), s2 * s2, 1e-12);
    SRFM_CHECK_NEAR((*cov_out)(0, 1), rho * s1 * s2, 1e-12);
}

// ── reduces_to_4d() compatibility ─────────────────────────────────────────────

static void test_manifold_reduces_to_4d_compatibility_n3_n3() {
    auto cov3 = diag_cov({0.01, 0.04, 0.09});
    auto m3 = NAssetManifold::make(3, cov3, 1.0);
    auto m3b = NAssetManifold::make(3, cov3, 1.0);
    SRFM_HAS_VALUE(m3);
    SRFM_HAS_VALUE(m3b);
    SRFM_CHECK(m3->reduces_to_4d(*m3b) == true);
}

static void test_manifold_reduces_to_4d_compatibility_n5_n3() {
    auto cov5 = diag_cov({0.01, 0.04, 0.09, 0.16, 0.25});
    auto cov3 = diag_cov({0.01, 0.04, 0.09});
    auto m5 = NAssetManifold::make(5, cov5, 1.0);
    auto m3 = NAssetManifold::make(3, cov3, 1.0);
    SRFM_HAS_VALUE(m5);
    SRFM_HAS_VALUE(m3);
    SRFM_CHECK(m5->reduces_to_4d(*m3) == true);
}

static void test_manifold_reduces_to_4d_incompatible_n1_n3() {
    Eigen::MatrixXd cov1(1, 1);
    cov1(0, 0) = 0.04;
    auto cov3 = diag_cov({0.01, 0.04, 0.09});
    auto m1 = NAssetManifold::make(1, cov1, 1.0);
    auto m3 = NAssetManifold::make(3, cov3, 1.0);
    SRFM_HAS_VALUE(m1);
    SRFM_HAS_VALUE(m3);
    // m1 has n_assets=1 < 3, so should NOT reduce to 4D.
    SRFM_CHECK(m1->reduces_to_4d(*m3) == false);
}

static void test_manifold_reduces_to_4d_other_n2_returns_false() {
    auto cov2 = diag_cov({0.01, 0.04});
    auto cov3 = diag_cov({0.01, 0.04, 0.09});
    auto m3a = NAssetManifold::make(3, cov3, 1.0);
    auto m2  = NAssetManifold::make(2, cov2, 1.0);
    SRFM_HAS_VALUE(m3a);
    SRFM_HAS_VALUE(m2);
    // other.n_assets() == 2 (not 3) → false.
    SRFM_CHECK(m3a->reduces_to_4d(*m2) == false);
}

// ── N=3 matches 4D block ──────────────────────────────────────────────────────

static void test_manifold_n3_matches_diagonal_block() {
    auto cov = diag_cov({0.01, 0.04, 0.09});
    auto m = NAssetManifold::make(3, cov, 1.0);
    SRFM_HAS_VALUE(m);
    SRFM_CHECK(m->dim() == 4);
    auto g = m->metric_at(coord(4));
    SRFM_HAS_VALUE(g);
    SRFM_CHECK_NEAR((*g)(0, 0), -1.0, 1e-12);
    SRFM_CHECK_NEAR((*g)(1, 1), 0.01, 1e-12);
    SRFM_CHECK_NEAR((*g)(2, 2), 0.04, 1e-12);
    SRFM_CHECK_NEAR((*g)(3, 3), 0.09, 1e-12);
}

// ── Zero-correlation (diagonal) block test ────────────────────────────────────

static void test_manifold_zero_correlation_diagonal_n2() {
    double s1 = 0.15, s2 = 0.25;
    auto cov = cov_2x2(s1, s2, 0.0); // ρ = 0
    auto m = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto g = m->metric_at(coord(3));
    SRFM_HAS_VALUE(g);
    // Off-diagonal spatial: g_12 = g_21 = 0
    SRFM_CHECK_NEAR((*g)(1, 2), 0.0, 1e-12);
    SRFM_CHECK_NEAR((*g)(2, 1), 0.0, 1e-12);
    // Diagonal
    SRFM_CHECK_NEAR((*g)(1, 1), s1 * s1, 1e-12);
    SRFM_CHECK_NEAR((*g)(2, 2), s2 * s2, 1e-12);
}

// ── Full correlation block test ───────────────────────────────────────────────

static void test_manifold_full_correlation_block_n2() {
    double s1 = 0.2, s2 = 0.3;
    // ρ = 0.99 — near-singular but still PD.
    auto cov = cov_2x2(s1, s2, 0.99);
    auto m = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto g = m->metric_at(coord(3));
    SRFM_HAS_VALUE(g);
    SRFM_CHECK_NEAR((*g)(1, 2), 0.99 * s1 * s2, 1e-12);
}

// ── Large N stress tests ──────────────────────────────────────────────────────

static void test_manifold_large_n_50_constructs() {
    int N = 50;
    std::vector<double> vars(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) { vars[static_cast<std::size_t>(i)] = 0.001 * (i + 1); }
    auto cov = diag_cov(vars);
    auto m = NAssetManifold::make(N, cov, 1.0);
    SRFM_HAS_VALUE(m);
    SRFM_CHECK(m->dim() == N + 1);
    auto g = m->metric_at(coord(N + 1));
    SRFM_HAS_VALUE(g);
    SRFM_CHECK(g->rows() == N + 1);
    SRFM_CHECK(g->cols() == N + 1);
}

static void test_manifold_large_n_100_constructs() {
    int N = 100;
    std::vector<double> vars(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) { vars[static_cast<std::size_t>(i)] = 0.0001 * (i + 1); }
    auto cov = diag_cov(vars);
    auto m = NAssetManifold::make(N, cov, 1.0);
    SRFM_HAS_VALUE(m);
    SRFM_CHECK(m->n_assets() == N);
}

// ── metric_at dimension validation ────────────────────────────────────────────

static void test_manifold_metric_at_wrong_dim_nullopt() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto g = m->metric_at(coord(5)); // dim=2, passing 5
    SRFM_NO_VALUE(g);
}

static void test_manifold_inverse_metric_at_wrong_dim_nullopt() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    auto ginv = m->inverse_metric_at(coord(5));
    SRFM_NO_VALUE(ginv);
}

// ── Multi-N parameter sweep metric check ──────────────────────────────────────

static void test_manifold_n_sweep_metric_time_component() {
    for (int n = 1; n <= 15; ++n) {
        std::vector<double> vars(static_cast<std::size_t>(n), 0.04);
        auto cov = diag_cov(vars);
        double c = 1.0 + 0.1 * n;
        auto m = NAssetManifold::make(n, cov, c);
        SRFM_HAS_VALUE(m);
        auto g = m->metric_at(coord(n + 1));
        SRFM_HAS_VALUE(g);
        SRFM_CHECK_NEAR((*g)(0, 0), -(c * c), 1e-10);
    }
}

static void test_manifold_n_sweep_spatial_diagonal() {
    for (int n = 1; n <= 10; ++n) {
        std::vector<double> vars(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            vars[static_cast<std::size_t>(i)] = 0.01 * (i + 1);
        }
        auto cov = diag_cov(vars);
        auto m = NAssetManifold::make(n, cov, 1.0);
        SRFM_HAS_VALUE(m);
        auto g = m->metric_at(coord(n + 1));
        SRFM_HAS_VALUE(g);
        for (int i = 0; i < n; ++i) {
            SRFM_CHECK_NEAR((*g)(i + 1, i + 1), vars[static_cast<std::size_t>(i)], 1e-12);
        }
    }
}

// ── Line element value verification N=2 ──────────────────────────────────────

static void test_manifold_line_element_analytic_n2() {
    // Uncorrelated: g = diag(-1, 0.04, 0.09)
    double s1sq = 0.04, s2sq = 0.09;
    Eigen::MatrixXd cov(2, 2);
    cov << s1sq, 0.0, 0.0, s2sq;
    auto m = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    Eigen::VectorXd dx(3);
    dx(0) = 1.0; dx(1) = 2.0; dx(2) = 3.0;
    // ds² = -1*(1)² + 0.04*(2)² + 0.09*(3)²
    //     = -1 + 0.16 + 0.81 = -0.03
    auto ds2 = m->line_element_sq(coord(3), dx);
    SRFM_HAS_VALUE(ds2);
    double expected = -1.0 + s1sq * 4.0 + s2sq * 9.0;
    SRFM_CHECK_NEAR(*ds2, expected, 1e-12);
}

// ── Metric independence of point x ───────────────────────────────────────────

static void test_manifold_metric_constant_different_points() {
    auto cov = diag_cov({0.01, 0.04, 0.09});
    auto m = NAssetManifold::make(3, cov, 1.0);
    SRFM_HAS_VALUE(m);
    Eigen::VectorXd x1 = coord(4, 0.0);
    Eigen::VectorXd x2 = coord(4, 100.0);
    auto g1 = m->metric_at(x1);
    auto g2 = m->metric_at(x2);
    SRFM_HAS_VALUE(g1);
    SRFM_HAS_VALUE(g2);
    SRFM_CHECK((*g1 - *g2).norm() < 1e-14);
}

// ── Additional correlation sweep ──────────────────────────────────────────────

static void test_manifold_correlation_sweep_2d() {
    double s1 = 0.2, s2 = 0.3;
    std::vector<double> rhos = {-0.9, -0.5, 0.0, 0.3, 0.5, 0.8, 0.9};
    for (double rho : rhos) {
        auto cov = cov_2x2(s1, s2, rho);
        auto m = NAssetManifold::make(2, cov, 1.0);
        SRFM_HAS_VALUE(m);
        auto g = m->metric_at(coord(3));
        SRFM_HAS_VALUE(g);
        SRFM_CHECK_NEAR((*g)(1, 2), rho * s1 * s2, 1e-12);
        SRFM_CHECK_NEAR((*g)(2, 1), rho * s1 * s2, 1e-12);
    }
}

// ── Metric inverse consistency sweep ─────────────────────────────────────────

static void test_manifold_inverse_sweep_n_1_to_8() {
    for (int n = 1; n <= 8; ++n) {
        std::vector<double> vars(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            vars[static_cast<std::size_t>(i)] = 0.01 + 0.03 * i;
        }
        auto cov = diag_cov(vars);
        auto m = NAssetManifold::make(n, cov, 1.0);
        SRFM_HAS_VALUE(m);
        auto g    = m->metric_at(coord(n + 1));
        auto ginv = m->inverse_metric_at(coord(n + 1));
        SRFM_HAS_VALUE(g);
        SRFM_HAS_VALUE(ginv);
        Eigen::MatrixXd prod = (*g) * (*ginv);
        Eigen::MatrixXd I    = Eigen::MatrixXd::Identity(n + 1, n + 1);
        SRFM_CHECK((prod - I).norm() < 1e-8);
    }
}

// ── Direct metric accessor ────────────────────────────────────────────────────

static void test_manifold_direct_metric_accessor() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    const Eigen::MatrixXd& g = m->metric();
    SRFM_CHECK(g.rows() == 2);
    SRFM_CHECK(g.cols() == 2);
    SRFM_CHECK_NEAR(g(0, 0), -1.0, 1e-12);
    SRFM_CHECK_NEAR(g(1, 1), 0.04, 1e-12);
}

static void test_manifold_direct_inverse_metric_accessor() {
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 1.0;
    auto m = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    const Eigen::MatrixXd& ginv = m->inverse_metric();
    SRFM_CHECK(ginv.rows() == 2);
    SRFM_CHECK(ginv.cols() == 2);
    // g = diag(-1, 1) → ginv = diag(-1, 1)
    SRFM_CHECK_NEAR(ginv(0, 0), -1.0, 1e-12);
    SRFM_CHECK_NEAR(ginv(1, 1),  1.0, 1e-12);
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    SRFM_SUITE("NAssetManifold Basic Construction",
        test_manifold_make_valid_1asset,
        test_manifold_make_valid_2asset,
        test_manifold_make_valid_3asset,
        test_manifold_make_valid_4asset,
        test_manifold_make_valid_10asset,
        test_manifold_make_valid_50asset
    );
    SRFM_SUITE("NAssetManifold Invalid Inputs",
        test_manifold_make_invalid_zero_n,
        test_manifold_make_invalid_negative_n,
        test_manifold_make_invalid_singular_cov,
        test_manifold_make_invalid_c_market_zero,
        test_manifold_make_invalid_c_market_negative,
        test_manifold_make_invalid_wrong_size_cov,
        test_manifold_make_invalid_asymmetric_cov
    );
    SRFM_SUITE("NAssetManifold Metric Signature",
        test_manifold_metric_diagonal_signature_n1,
        test_manifold_metric_diagonal_signature_n4,
        test_manifold_metric_time_space_cross_terms_zero_n2,
        test_manifold_metric_off_diagonal_correlation_n2
    );
    SRFM_SUITE("NAssetManifold Metric Symmetry",
        test_manifold_metric_symmetry_n1,
        test_manifold_metric_symmetry_n3,
        test_manifold_metric_symmetry_n10
    );
    SRFM_SUITE("NAssetManifold Metric Inverse",
        test_manifold_metric_inverse_identity_n1,
        test_manifold_metric_inverse_identity_n2,
        test_manifold_metric_inverse_identity_n4,
        test_manifold_metric_inverse_identity_n10,
        test_manifold_inverse_metric_nonsingular_n1,
        test_manifold_inverse_sweep_n_1_to_8
    );
    SRFM_SUITE("NAssetManifold Line Element",
        test_manifold_line_element_pure_time_negative_n1,
        test_manifold_line_element_pure_spatial_positive_n1,
        test_manifold_line_element_flat_minkowski_n1,
        test_manifold_line_element_spacelike_n2,
        test_manifold_line_element_timelike_n2,
        test_manifold_line_element_wrong_dim_returns_nullopt,
        test_manifold_line_element_analytic_n2
    );
    SRFM_SUITE("NAssetManifold Properties",
        test_manifold_is_flat_returns_true_n1,
        test_manifold_is_flat_returns_true_n10,
        test_manifold_dim_returns_n_plus_1_sweep,
        test_manifold_c_market_scales_time_component,
        test_manifold_c_market_sweeps
    );
    SRFM_SUITE("NAssetManifold Covariance",
        test_manifold_covariance_roundtrip_n1,
        test_manifold_covariance_roundtrip_n3,
        test_manifold_covariance_roundtrip_with_correlation
    );
    SRFM_SUITE("NAssetManifold 4D Compatibility",
        test_manifold_reduces_to_4d_compatibility_n3_n3,
        test_manifold_reduces_to_4d_compatibility_n5_n3,
        test_manifold_reduces_to_4d_incompatible_n1_n3,
        test_manifold_reduces_to_4d_other_n2_returns_false,
        test_manifold_n3_matches_diagonal_block
    );
    SRFM_SUITE("NAssetManifold Correlation Tests",
        test_manifold_zero_correlation_diagonal_n2,
        test_manifold_full_correlation_block_n2,
        test_manifold_correlation_sweep_2d
    );
    SRFM_SUITE("NAssetManifold Large N",
        test_manifold_large_n_50_constructs,
        test_manifold_large_n_100_constructs
    );
    SRFM_SUITE("NAssetManifold Validation",
        test_manifold_metric_at_wrong_dim_nullopt,
        test_manifold_inverse_metric_at_wrong_dim_nullopt,
        test_manifold_metric_constant_different_points
    );
    SRFM_SUITE("NAssetManifold Sweeps",
        test_manifold_n_sweep_metric_time_component,
        test_manifold_n_sweep_spatial_diagonal
    );
    SRFM_SUITE("NAssetManifold Accessors",
        test_manifold_direct_metric_accessor,
        test_manifold_direct_inverse_metric_accessor
    );
    return srfm_test::report();
}
