/**
 * @file  test_n_asset_interval.cpp
 * @brief Tests for NAssetEvent, NAssetInterval — Stage 4 N-Asset Manifold.
 *
 * Coverage:
 *   - NAssetEvent make() valid and invalid
 *   - to_coords() concatenation
 *   - Pure-time separation → TIMELIKE
 *   - Pure-spatial separation → SPACELIKE
 *   - Zero separation → LIGHTLIKE
 *   - ds² analytic values for N=1, N=2, N=4
 *   - Interval symmetry: compute(a,b) == compute(b,a) in |ds²|
 *   - magnitude = sqrt(|ds²|)
 *   - batch_from_reference size and consistency
 *   - Dimension mismatch → nullopt
 *   - c_market scaling
 *   - Classification threshold boundary
 */

#include "srfm_test_n_asset.hpp"
#include "../../include/srfm/manifold/n_asset_interval.hpp"

#include <cmath>
#include <vector>

using namespace srfm::manifold;
using srfm::tensor::NAssetManifold;

// ── Helpers ───────────────────────────────────────────────────────────────────

static NAssetManifold flat_manifold(int n, double var = 0.04, double c = 1.0) {
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(n, n) * var;
    return *NAssetManifold::make(n, cov, c);
}

static Eigen::VectorXd vec(std::initializer_list<double> vals) {
    Eigen::VectorXd v(static_cast<int>(vals.size()));
    int i = 0;
    for (double d : vals) { v(i++) = d; }
    return v;
}

// ── NAssetEvent make() tests ──────────────────────────────────────────────────

static void test_nassetevent_make_valid_n1() {
    auto ev = NAssetEvent::make(1.0, vec({100.0}));
    SRFM_HAS_VALUE(ev);
    SRFM_CHECK_NEAR(ev->t, 1.0, 1e-12);
    SRFM_CHECK_NEAR(ev->prices(0), 100.0, 1e-12);
}

static void test_nassetevent_make_valid_n4() {
    auto ev = NAssetEvent::make(5.0, vec({10.0, 20.0, 30.0, 40.0}));
    SRFM_HAS_VALUE(ev);
    SRFM_CHECK(ev->prices.size() == 4);
}

static void test_nAssetevent_make_invalid_empty_prices() {
    auto ev = NAssetEvent::make(1.0, Eigen::VectorXd{});
    SRFM_NO_VALUE(ev);
}

static void test_nassetevent_to_coords_n1() {
    auto ev = NAssetEvent::make(2.0, vec({50.0}));
    SRFM_HAS_VALUE(ev);
    Eigen::VectorXd c = ev->to_coords();
    SRFM_CHECK(c.size() == 2);
    SRFM_CHECK_NEAR(c(0), 2.0,  1e-12);
    SRFM_CHECK_NEAR(c(1), 50.0, 1e-12);
}

static void test_nassetevent_to_coords_n3() {
    auto ev = NAssetEvent::make(3.0, vec({10.0, 20.0, 30.0}));
    SRFM_HAS_VALUE(ev);
    Eigen::VectorXd c = ev->to_coords();
    SRFM_CHECK(c.size() == 4);
    SRFM_CHECK_NEAR(c(0), 3.0,  1e-12);
    SRFM_CHECK_NEAR(c(1), 10.0, 1e-12);
    SRFM_CHECK_NEAR(c(2), 20.0, 1e-12);
    SRFM_CHECK_NEAR(c(3), 30.0, 1e-12);
}

static void test_nassetevent_to_coords_zero_time() {
    auto ev = NAssetEvent::make(0.0, vec({1.0, 2.0}));
    SRFM_HAS_VALUE(ev);
    Eigen::VectorXd c = ev->to_coords();
    SRFM_CHECK_NEAR(c(0), 0.0, 1e-12);
    SRFM_CHECK_NEAR(c(1), 1.0, 1e-12);
    SRFM_CHECK_NEAR(c(2), 2.0, 1e-12);
}

// ── Interval classification: TIMELIKE ─────────────────────────────────────────

static void test_interval_pure_time_is_timelike_n1() {
    auto m  = flat_manifold(1, 0.04, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({100.0}));
    auto eb = NAssetEvent::make(1.0, vec({100.0})); // dt=1, dp=0
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto r = calc.compute(*ea, *eb, m);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->type == IntervalType::TIMELIKE);
    SRFM_CHECK(r->ds_sq < 0.0);
}

static void test_interval_pure_time_is_timelike_n4() {
    auto m  = flat_manifold(4, 0.04, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({10.0, 10.0, 10.0, 10.0}));
    auto eb = NAssetEvent::make(5.0, vec({10.0, 10.0, 10.0, 10.0}));
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto r = calc.compute(*ea, *eb, m);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->type == IntervalType::TIMELIKE);
}

// ── Interval classification: SPACELIKE ────────────────────────────────────────

static void test_interval_pure_spatial_is_spacelike_n1() {
    auto m  = flat_manifold(1, 0.04, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({100.0}));
    auto eb = NAssetEvent::make(0.0, vec({200.0})); // dt=0, dp=100
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto r = calc.compute(*ea, *eb, m);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->type == IntervalType::SPACELIKE);
    SRFM_CHECK(r->ds_sq > 0.0);
}

static void test_interval_pure_spatial_is_spacelike_n2() {
    auto m  = flat_manifold(2, 0.04, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({0.0, 0.0}));
    auto eb = NAssetEvent::make(0.0, vec({10.0, 10.0})); // dt=0, large dp
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto r = calc.compute(*ea, *eb, m);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->type == IntervalType::SPACELIKE);
}

// ── Interval classification: LIGHTLIKE ────────────────────────────────────────

static void test_interval_zero_separation_is_lightlike() {
    auto m  = flat_manifold(1, 0.04, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({100.0}));
    NAssetInterval calc;
    auto r = calc.compute(*ea, *ea, m); // same event
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->type == IntervalType::LIGHTLIKE);
    SRFM_CHECK_NEAR(r->ds_sq, 0.0, 1e-15);
    SRFM_CHECK_NEAR(r->magnitude, 0.0, 1e-12);
}

// ── Interval ds² analytic value N=1 ──────────────────────────────────────────

static void test_interval_n1_analytic_value() {
    // g = diag(-1, 0.04), dt=1, dp=2
    // ds² = -1*(1)² + 0.04*(2)² = -1 + 0.16 = -0.84
    auto m  = flat_manifold(1, 0.04, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({0.0}));
    auto eb = NAssetEvent::make(1.0, vec({2.0}));
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto r = calc.compute(*ea, *eb, m);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK_NEAR(r->ds_sq, -0.84, 1e-12);
    SRFM_CHECK(r->type == IntervalType::TIMELIKE);
    SRFM_CHECK_NEAR(r->magnitude, std::sqrt(0.84), 1e-10);
}

static void test_interval_n1_spacelike_analytic() {
    // g = diag(-1, 0.04), dt=0, dp=3
    // ds² = 0.04 * 9 = 0.36
    auto m  = flat_manifold(1, 0.04, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({0.0}));
    auto eb = NAssetEvent::make(0.0, vec({3.0}));
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto r = calc.compute(*ea, *eb, m);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK_NEAR(r->ds_sq, 0.36, 1e-12);
    SRFM_CHECK(r->type == IntervalType::SPACELIKE);
}

// ── Interval ds² analytic value N=2 ──────────────────────────────────────────

static void test_interval_n2_analytic_value() {
    // g = diag(-1, 0.04, 0.09), dt=1, dp1=2, dp2=1
    // ds² = -1 + 0.04*4 + 0.09*1 = -1 + 0.16 + 0.09 = -0.75
    Eigen::MatrixXd cov(2, 2);
    cov << 0.04, 0.0, 0.0, 0.09;
    auto m  = *NAssetManifold::make(2, cov, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({0.0, 0.0}));
    auto eb = NAssetEvent::make(1.0, vec({2.0, 1.0}));
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto r = calc.compute(*ea, *eb, m);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK_NEAR(r->ds_sq, -0.75, 1e-12);
}

// ── Analytic N=4 ─────────────────────────────────────────────────────────────

static void test_interval_n4_analytic_value() {
    // g = diag(-c², σ1², σ2², σ3², σ4²), c=2
    // dt=1, dp=(0,0,0,0) → ds² = -4
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4) * 0.04;
    auto m  = *NAssetManifold::make(4, cov, 2.0);
    auto ea = NAssetEvent::make(0.0, vec({0.0, 0.0, 0.0, 0.0}));
    auto eb = NAssetEvent::make(1.0, vec({0.0, 0.0, 0.0, 0.0}));
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto r = calc.compute(*ea, *eb, m);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK_NEAR(r->ds_sq, -4.0, 1e-12);
    SRFM_CHECK(r->type == IntervalType::TIMELIKE);
}

// ── Symmetry: compute(a,b) ds_sq == compute(b,a) ds_sq ───────────────────────

static void test_interval_symmetry_a_b_eq_b_a_n1() {
    auto m  = flat_manifold(1, 0.04, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({100.0}));
    auto eb = NAssetEvent::make(1.0, vec({105.0}));
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto rab = calc.compute(*ea, *eb, m);
    auto rba = calc.compute(*eb, *ea, m);
    SRFM_HAS_VALUE(rab); SRFM_HAS_VALUE(rba);
    // ds² is symmetric in displacement sign: ds²(Δx) = ds²(-Δx).
    SRFM_CHECK_NEAR(rab->ds_sq, rba->ds_sq, 1e-12);
}

static void test_interval_symmetry_a_b_eq_b_a_n4() {
    auto m  = flat_manifold(4, 0.04, 1.0);
    auto ea = NAssetEvent::make(1.0, vec({10.0, 20.0, 30.0, 40.0}));
    auto eb = NAssetEvent::make(2.0, vec({11.0, 21.0, 31.0, 41.0}));
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto rab = calc.compute(*ea, *eb, m);
    auto rba = calc.compute(*eb, *ea, m);
    SRFM_HAS_VALUE(rab); SRFM_HAS_VALUE(rba);
    SRFM_CHECK_NEAR(rab->ds_sq, rba->ds_sq, 1e-12);
}

// ── magnitude non-negative ────────────────────────────────────────────────────

static void test_interval_magnitude_nonneg_timelike() {
    auto m  = flat_manifold(1, 0.04, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({100.0}));
    auto eb = NAssetEvent::make(5.0, vec({100.0}));
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto r = calc.compute(*ea, *eb, m);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->magnitude >= 0.0);
}

static void test_interval_magnitude_nonneg_spacelike() {
    auto m  = flat_manifold(1, 0.04, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({100.0}));
    auto eb = NAssetEvent::make(0.0, vec({200.0}));
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto r = calc.compute(*ea, *eb, m);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->magnitude >= 0.0);
}

static void test_interval_magnitude_equals_sqrt_abs_ds_sq() {
    auto m  = flat_manifold(1, 0.04, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({0.0}));
    auto eb = NAssetEvent::make(2.0, vec({3.0}));
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto r = calc.compute(*ea, *eb, m);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK_NEAR(r->magnitude, std::sqrt(std::abs(r->ds_sq)), 1e-12);
}

// ── batch_from_reference ──────────────────────────────────────────────────────

static void test_interval_batch_from_reference_size() {
    auto m   = flat_manifold(2, 0.04, 1.0);
    auto ref = NAssetEvent::make(0.0, vec({0.0, 0.0}));
    SRFM_HAS_VALUE(ref);
    std::vector<NAssetEvent> events;
    for (int i = 1; i <= 10; ++i) {
        auto ev = NAssetEvent::make(static_cast<double>(i),
                                    vec({static_cast<double>(i),
                                         static_cast<double>(i * 2)}));
        SRFM_HAS_VALUE(ev);
        events.push_back(*ev);
    }
    NAssetInterval calc;
    auto results = calc.batch_from_reference(*ref, events, m);
    SRFM_HAS_VALUE(results);
    SRFM_CHECK(static_cast<int>(results->size()) == 10);
}

static void test_interval_batch_consistency_with_single() {
    auto m   = flat_manifold(1, 0.04, 1.0);
    auto ref = NAssetEvent::make(0.0, vec({100.0}));
    SRFM_HAS_VALUE(ref);

    std::vector<NAssetEvent> events;
    for (int i = 0; i < 5; ++i) {
        auto ev = NAssetEvent::make(static_cast<double>(i + 1),
                                    vec({100.0 + i * 5.0}));
        SRFM_HAS_VALUE(ev);
        events.push_back(*ev);
    }

    NAssetInterval calc;
    auto batch_results = calc.batch_from_reference(*ref, events, m);
    SRFM_HAS_VALUE(batch_results);

    for (int i = 0; i < 5; ++i) {
        auto single = calc.compute(*ref, events[static_cast<std::size_t>(i)], m);
        SRFM_HAS_VALUE(single);
        SRFM_CHECK_NEAR((*batch_results)[static_cast<std::size_t>(i)].ds_sq,
                        single->ds_sq, 1e-12);
    }
}

static void test_interval_batch_empty_events_ok() {
    auto m   = flat_manifold(1, 0.04, 1.0);
    auto ref = NAssetEvent::make(0.0, vec({100.0}));
    SRFM_HAS_VALUE(ref);
    NAssetInterval calc;
    std::vector<NAssetEvent> empty_events;
    auto results = calc.batch_from_reference(*ref, empty_events, m);
    SRFM_HAS_VALUE(results);
    SRFM_CHECK(results->empty());
}

// ── Dimension mismatch → nullopt ──────────────────────────────────────────────

static void test_interval_dimension_mismatch_nullopt() {
    auto m   = flat_manifold(2, 0.04, 1.0);
    auto ea  = NAssetEvent::make(0.0, vec({1.0}));      // N=1 event
    auto eb  = NAssetEvent::make(1.0, vec({2.0, 3.0})); // N=2 event
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
    NAssetInterval calc;
    auto r = calc.compute(*ea, *eb, m); // manifold N=2 but ea has N=1
    SRFM_NO_VALUE(r);
}

// ── c_market scaling ──────────────────────────────────────────────────────────

static void test_interval_c_market_scales_timelike() {
    // With larger c, a fixed dt contributes more negative ds².
    Eigen::MatrixXd cov(1, 1);
    cov(0, 0) = 0.04;
    double c1 = 1.0, c2 = 3.0;
    auto m1 = *NAssetManifold::make(1, cov, c1);
    auto m2 = *NAssetManifold::make(1, cov, c2);

    auto ea = NAssetEvent::make(0.0, vec({100.0}));
    auto eb = NAssetEvent::make(1.0, vec({100.0}));
    SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);

    NAssetInterval calc;
    auto r1 = calc.compute(*ea, *eb, m1);
    auto r2 = calc.compute(*ea, *eb, m2);
    SRFM_HAS_VALUE(r1); SRFM_HAS_VALUE(r2);

    // ds² for c=3 should be -9, for c=1 should be -1.
    SRFM_CHECK_NEAR(r1->ds_sq, -(c1 * c1), 1e-12);
    SRFM_CHECK_NEAR(r2->ds_sq, -(c2 * c2), 1e-12);
    SRFM_CHECK(r2->ds_sq < r1->ds_sq); // more negative with larger c
}

// ── Classification threshold boundary ────────────────────────────────────────

static void test_interval_classification_thresholds() {
    // ds² very near zero but positive → LIGHTLIKE.
    // ds² = 5e-15 (well below 1e-10) → LIGHTLIKE.
    // (Hard to achieve precisely; use known zero case.)
    auto m  = flat_manifold(1, 0.04, 1.0);
    auto ea = NAssetEvent::make(0.0, vec({100.0}));
    NAssetInterval calc;
    auto r = calc.compute(*ea, *ea, m);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->type == IntervalType::LIGHTLIKE);
}

// ── N sweep: all pure-time events are TIMELIKE ─────────────────────────────────

static void test_interval_pure_time_timelike_n_sweep() {
    for (int n = 1; n <= 8; ++n) {
        auto m = flat_manifold(n, 0.04, 1.0);
        std::vector<double> zeros(static_cast<std::size_t>(n), 0.0);
        Eigen::VectorXd p = Eigen::VectorXd::Zero(n);
        auto ea = NAssetEvent::make(0.0, p);
        auto eb = NAssetEvent::make(10.0, p);
        SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
        NAssetInterval calc;
        auto r = calc.compute(*ea, *eb, m);
        SRFM_HAS_VALUE(r);
        SRFM_CHECK(r->type == IntervalType::TIMELIKE);
        SRFM_CHECK(r->ds_sq < 0.0);
    }
}

// ── N sweep: all pure-spatial events are SPACELIKE ────────────────────────────

static void test_interval_pure_spatial_spacelike_n_sweep() {
    for (int n = 1; n <= 6; ++n) {
        auto m = flat_manifold(n, 0.04, 1.0);
        Eigen::VectorXd p1 = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd p2 = Eigen::VectorXd::Ones(n) * 5.0;
        auto ea = NAssetEvent::make(0.0, p1);
        auto eb = NAssetEvent::make(0.0, p2); // same time
        SRFM_HAS_VALUE(ea); SRFM_HAS_VALUE(eb);
        NAssetInterval calc;
        auto r = calc.compute(*ea, *eb, m);
        SRFM_HAS_VALUE(r);
        SRFM_CHECK(r->type == IntervalType::SPACELIKE);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    SRFM_SUITE("NAssetEvent Construction",
        test_nassetevent_make_valid_n1,
        test_nassetevent_make_valid_n4,
        test_nAssetevent_make_invalid_empty_prices,
        test_nassetevent_to_coords_n1,
        test_nassetevent_to_coords_n3,
        test_nassetevent_to_coords_zero_time
    );
    SRFM_SUITE("NAssetInterval TIMELIKE",
        test_interval_pure_time_is_timelike_n1,
        test_interval_pure_time_is_timelike_n4
    );
    SRFM_SUITE("NAssetInterval SPACELIKE",
        test_interval_pure_spatial_is_spacelike_n1,
        test_interval_pure_spatial_is_spacelike_n2
    );
    SRFM_SUITE("NAssetInterval LIGHTLIKE",
        test_interval_zero_separation_is_lightlike,
        test_interval_classification_thresholds
    );
    SRFM_SUITE("NAssetInterval Analytic Values",
        test_interval_n1_analytic_value,
        test_interval_n1_spacelike_analytic,
        test_interval_n2_analytic_value,
        test_interval_n4_analytic_value
    );
    SRFM_SUITE("NAssetInterval Symmetry",
        test_interval_symmetry_a_b_eq_b_a_n1,
        test_interval_symmetry_a_b_eq_b_a_n4
    );
    SRFM_SUITE("NAssetInterval Magnitude",
        test_interval_magnitude_nonneg_timelike,
        test_interval_magnitude_nonneg_spacelike,
        test_interval_magnitude_equals_sqrt_abs_ds_sq
    );
    SRFM_SUITE("NAssetInterval Batch",
        test_interval_batch_from_reference_size,
        test_interval_batch_consistency_with_single,
        test_interval_batch_empty_events_ok
    );
    SRFM_SUITE("NAssetInterval Validation",
        test_interval_dimension_mismatch_nullopt
    );
    SRFM_SUITE("NAssetInterval c_market",
        test_interval_c_market_scales_timelike
    );
    SRFM_SUITE("NAssetInterval N Sweeps",
        test_interval_pure_time_timelike_n_sweep,
        test_interval_pure_spatial_spacelike_n_sweep
    );
    return srfm_test::report();
}
