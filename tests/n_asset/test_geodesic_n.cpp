/**
 * @file  test_geodesic_n.cpp
 * @brief Tests for GeodesicSolverN — Stage 4 N-Asset Manifold.
 *
 * Coverage:
 *   - Flat manifold geodesics are straight lines
 *   - geodesic_deviation near zero for flat space
 *   - integrate() returns correct number of states
 *   - State dimensionality matches manifold
 *   - RK4 step: forward + backward ≈ original
 *   - Energy conservation (u·u) in flat space
 *   - Parameter sweeps over N = 1..10 and N = 50
 *   - Different dtau values
 *   - Step count verification
 */

#include "srfm_test_n_asset.hpp"
#include "../../include/srfm/tensor/geodesic_n.hpp"

#include <cmath>
#include <vector>

using namespace srfm::tensor;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Eigen::MatrixXd flat_cov(int n, double var = 0.04) {
    return Eigen::MatrixXd::Identity(n, n) * var;
}

static GeodesicState make_state(int D,
                                 double x_val = 0.0,
                                 double u_val = 0.1) {
    GeodesicState s;
    s.x = Eigen::VectorXd::Constant(D, x_val);
    s.u = Eigen::VectorXd::Constant(D, u_val);
    return s;
}

// ── Straight line in flat space ───────────────────────────────────────────────

static void test_geodesic_flat_straight_line_n1() {
    auto cov = flat_cov(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0;
    s0.x = Eigen::VectorXd::Zero(2);
    s0.u = Eigen::VectorXd::Constant(2, 1.0);

    double dtau  = 0.01;
    int n_steps  = 100;
    auto traj    = solver.integrate(s0, dtau, n_steps);
    SRFM_HAS_VALUE(traj);
    SRFM_CHECK(static_cast<int>(traj->size()) == n_steps);

    // Expect linear motion: x(τ) = x0 + u0 * τ.
    for (int i = 0; i < n_steps; ++i) {
        double tau = dtau * (i + 1);
        for (int d = 0; d < 2; ++d) {
            SRFM_CHECK_NEAR((*traj)[static_cast<std::size_t>(i)].x(d),
                            tau, 1e-6);
        }
    }
}

static void test_geodesic_flat_straight_line_n2() {
    auto cov = flat_cov(2);
    auto m   = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0;
    s0.x = Eigen::VectorXd::Zero(3);
    s0.u = Eigen::VectorXd::Zero(3);
    s0.u(0) = 1.0; s0.u(1) = 0.5; s0.u(2) = -0.3;

    auto traj = solver.integrate(s0, 0.01, 50);
    SRFM_HAS_VALUE(traj);

    for (int i = 0; i < 50; ++i) {
        double tau = 0.01 * (i + 1);
        SRFM_CHECK_NEAR((*traj)[static_cast<std::size_t>(i)].x(0), 1.0 * tau, 1e-6);
        SRFM_CHECK_NEAR((*traj)[static_cast<std::size_t>(i)].x(1), 0.5 * tau, 1e-6);
        SRFM_CHECK_NEAR((*traj)[static_cast<std::size_t>(i)].x(2), -0.3 * tau, 1e-6);
    }
}

// ── Deviation tests ───────────────────────────────────────────────────────────

static void test_geodesic_flat_deviation_below_1e8_n1() {
    auto cov = flat_cov(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0 = make_state(2);
    auto traj = solver.integrate(s0, 0.01, 100);
    SRFM_HAS_VALUE(traj);

    auto dev = solver.geodesic_deviation(*traj);
    SRFM_HAS_VALUE(dev);
    SRFM_CHECK(*dev < 1e-8);
}

static void test_geodesic_flat_deviation_below_1e8_n4() {
    auto cov = flat_cov(4);
    auto m   = NAssetManifold::make(4, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0 = make_state(5);
    auto traj = solver.integrate(s0, 0.01, 100);
    SRFM_HAS_VALUE(traj);

    auto dev = solver.geodesic_deviation(*traj);
    SRFM_HAS_VALUE(dev);
    SRFM_CHECK(*dev < 1e-8);
}

static void test_geodesic_flat_deviation_below_1e8_n10() {
    auto cov = flat_cov(10);
    auto m   = NAssetManifold::make(10, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0 = make_state(11, 0.0, 0.1);
    auto traj = solver.integrate(s0, 0.01, 100);
    SRFM_HAS_VALUE(traj);

    auto dev = solver.geodesic_deviation(*traj);
    SRFM_HAS_VALUE(dev);
    SRFM_CHECK(*dev < 1e-7);
}

static void test_geodesic_flat_n50_deviation() {
    int N = 50;
    auto cov = flat_cov(N, 0.01);
    auto m   = NAssetManifold::make(N, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0 = make_state(N + 1, 0.0, 0.01);
    auto traj = solver.integrate(s0, 0.1, 20);
    SRFM_HAS_VALUE(traj);

    auto dev = solver.geodesic_deviation(*traj);
    SRFM_HAS_VALUE(dev);
    SRFM_CHECK(*dev < 1e-6);
}

// ── integrate() returns correct count ─────────────────────────────────────────

static void test_geodesic_integrate_100_steps_returns_100_states() {
    auto cov = flat_cov(2);
    auto m   = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0 = make_state(3);
    auto traj = solver.integrate(s0, 0.01, 100);
    SRFM_HAS_VALUE(traj);
    SRFM_CHECK(static_cast<int>(traj->size()) == 100);
}

static void test_geodesic_integrate_0_steps_returns_empty() {
    auto cov = flat_cov(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0 = make_state(2);
    auto traj = solver.integrate(s0, 0.01, 0);
    SRFM_HAS_VALUE(traj);
    SRFM_CHECK(traj->empty());
}

static void test_geodesic_integrate_1_step_returns_1_state() {
    auto cov = flat_cov(2);
    auto m   = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0 = make_state(3);
    auto traj = solver.integrate(s0, 0.01, 1);
    SRFM_HAS_VALUE(traj);
    SRFM_CHECK(static_cast<int>(traj->size()) == 1);
}

static void test_geodesic_integrate_step_count_sweep() {
    auto cov = flat_cov(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0 = make_state(2);
    for (int n : {1, 5, 10, 50, 200}) {
        auto traj = solver.integrate(s0, 0.01, n);
        SRFM_HAS_VALUE(traj);
        SRFM_CHECK(static_cast<int>(traj->size()) == n);
    }
}

// ── State dimension matches manifold ─────────────────────────────────────────

static void test_geodesic_state_dimension_matches_manifold_n1() {
    auto cov = flat_cov(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0 = make_state(2);
    auto r = solver.step(s0, 0.01);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->x.size() == 2);
    SRFM_CHECK(r->u.size() == 2);
}

static void test_geodesic_state_dimension_matches_manifold_n5() {
    auto cov = flat_cov(5);
    auto m   = NAssetManifold::make(5, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0 = make_state(6);
    auto r = solver.step(s0, 0.01);
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->x.size() == 6);
    SRFM_CHECK(r->u.size() == 6);
}

// ── RK4 step reversibility ────────────────────────────────────────────────────

static void test_geodesic_rk4_step_reversible_n1() {
    auto cov = flat_cov(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0;
    s0.x = Eigen::VectorXd::Constant(2, 1.0);
    s0.u = Eigen::VectorXd::Constant(2, 0.5);

    double dtau = 0.001;
    auto s1 = solver.step(s0, dtau);
    SRFM_HAS_VALUE(s1);
    // Step backward.
    auto s_back = solver.step(*s1, -dtau);
    SRFM_HAS_VALUE(s_back);

    // Should approximately return to s0 (RK4 is time-reversible for geodesics).
    SRFM_CHECK_NEAR((s_back->x - s0.x).norm(), 0.0, 1e-8);
}

static void test_geodesic_rk4_step_reversible_n3() {
    auto cov = flat_cov(3);
    auto m   = NAssetManifold::make(3, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0;
    s0.x = Eigen::VectorXd::Zero(4);
    s0.u = Eigen::VectorXd::Ones(4) * 0.1;

    double dtau = 0.001;
    auto s1 = solver.step(s0, dtau);
    SRFM_HAS_VALUE(s1);
    auto s_back = solver.step(*s1, -dtau);
    SRFM_HAS_VALUE(s_back);

    SRFM_CHECK_NEAR((s_back->x - s0.x).norm(), 0.0, 1e-7);
    SRFM_CHECK_NEAR((s_back->u - s0.u).norm(), 0.0, 1e-7);
}

// ── Energy conservation in flat space ────────────────────────────────────────

static void test_geodesic_energy_conservation_flat_n1() {
    auto cov = flat_cov(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0;
    s0.x = Eigen::VectorXd::Zero(2);
    // u = (1, 0.5) → u·g·u = -(1)² + 0.04*(0.5)² = -1 + 0.01 = -0.99
    s0.u = Eigen::VectorXd::Zero(2);
    s0.u(0) = 1.0; s0.u(1) = 0.5;

    const Eigen::MatrixXd& g = m->metric();
    double e0 = s0.u.dot(g * s0.u);

    auto traj = solver.integrate(s0, 0.01, 100);
    SRFM_HAS_VALUE(traj);

    // Energy u·g·u should be conserved along the geodesic.
    for (const auto& state : *traj) {
        double e = state.u.dot(g * state.u);
        SRFM_CHECK_NEAR(e, e0, 1e-8);
    }
}

static void test_geodesic_energy_conservation_flat_n4() {
    auto cov = flat_cov(4);
    auto m   = NAssetManifold::make(4, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0;
    s0.x = Eigen::VectorXd::Zero(5);
    s0.u = Eigen::VectorXd::Constant(5, 0.1);
    s0.u(0) = 1.0; // time component dominates → timelike

    const Eigen::MatrixXd& g = m->metric();
    double e0 = s0.u.dot(g * s0.u);

    auto traj = solver.integrate(s0, 0.01, 50);
    SRFM_HAS_VALUE(traj);

    for (const auto& state : *traj) {
        double e = state.u.dot(g * state.u);
        SRFM_CHECK_NEAR(e, e0, 1e-7);
    }
}

// ── dtau half → smaller deviation ────────────────────────────────────────────

static void test_geodesic_dtau_half_more_accurate_n2() {
    auto cov = flat_cov(2);
    auto m   = NAssetManifold::make(2, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0 = make_state(3, 0.0, 0.1);

    // Coarse integration.
    auto traj1 = solver.integrate(s0, 0.1, 10);
    SRFM_HAS_VALUE(traj1);
    auto dev1 = solver.geodesic_deviation(*traj1);
    SRFM_HAS_VALUE(dev1);

    // Finer integration covering the same proper time.
    auto traj2 = solver.integrate(s0, 0.05, 20);
    SRFM_HAS_VALUE(traj2);
    auto dev2 = solver.geodesic_deviation(*traj2);
    SRFM_HAS_VALUE(dev2);

    // Both should be near zero for flat metric.
    SRFM_CHECK(*dev1 < 1e-8);
    SRFM_CHECK(*dev2 < 1e-8);
}

// ── geodesic_deviation with < 2 states ────────────────────────────────────────

static void test_geodesic_deviation_single_state_nullopt() {
    auto cov = flat_cov(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    std::vector<GeodesicState> single = {make_state(2)};
    auto dev = solver.geodesic_deviation(single);
    SRFM_NO_VALUE(dev);
}

static void test_geodesic_deviation_empty_nullopt() {
    auto cov = flat_cov(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    std::vector<GeodesicState> empty;
    auto dev = solver.geodesic_deviation(empty);
    SRFM_NO_VALUE(dev);
}

// ── dim() accessor ────────────────────────────────────────────────────────────

static void test_geodesic_dim_accessor_n3() {
    auto cov = flat_cov(3);
    auto m   = NAssetManifold::make(3, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);
    SRFM_CHECK(solver.dim() == 4);
}

// ── Flat geodesic direction test ──────────────────────────────────────────────

static void test_geodesic_flat_velocity_unchanged_n1() {
    // In flat space, the velocity should not change.
    auto cov = flat_cov(1);
    auto m   = NAssetManifold::make(1, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0;
    s0.x = Eigen::VectorXd::Zero(2);
    s0.u = Eigen::VectorXd::Constant(2, 0.3);

    auto traj = solver.integrate(s0, 0.01, 50);
    SRFM_HAS_VALUE(traj);
    for (const auto& state : *traj) {
        SRFM_CHECK_NEAR(state.u(0), 0.3, 1e-8);
        SRFM_CHECK_NEAR(state.u(1), 0.3, 1e-8);
    }
}

static void test_geodesic_flat_velocity_unchanged_n4() {
    auto cov = flat_cov(4);
    auto m   = NAssetManifold::make(4, cov, 1.0);
    SRFM_HAS_VALUE(m);
    ChristoffelN ch(*m);
    GeodesicSolverN solver(*m, ch);

    GeodesicState s0;
    s0.x = Eigen::VectorXd::Zero(5);
    s0.u = Eigen::VectorXd::Zero(5);
    s0.u(0) = 1.0; s0.u(1) = -0.2; s0.u(2) = 0.5;
    s0.u(3) = 0.0; s0.u(4) = -0.1;

    auto traj = solver.integrate(s0, 0.01, 30);
    SRFM_HAS_VALUE(traj);
    for (const auto& state : *traj) {
        SRFM_CHECK_NEAR(state.u(0),  1.0,  1e-7);
        SRFM_CHECK_NEAR(state.u(1), -0.2,  1e-7);
        SRFM_CHECK_NEAR(state.u(2),  0.5,  1e-7);
        SRFM_CHECK_NEAR(state.u(3),  0.0,  1e-7);
        SRFM_CHECK_NEAR(state.u(4), -0.1,  1e-7);
    }
}

// ── N sweep ───────────────────────────────────────────────────────────────────

static void test_geodesic_n_sweep_deviation_flat_1_to_8() {
    for (int n = 1; n <= 8; ++n) {
        auto cov = flat_cov(n, 0.01);
        auto m   = NAssetManifold::make(n, cov, 1.0);
        SRFM_HAS_VALUE(m);
        ChristoffelN ch(*m);
        GeodesicSolverN solver(*m, ch);

        GeodesicState s0 = make_state(n + 1, 0.0, 0.05);
        auto traj = solver.integrate(s0, 0.05, 20);
        SRFM_HAS_VALUE(traj);
        auto dev = solver.geodesic_deviation(*traj);
        SRFM_HAS_VALUE(dev);
        SRFM_CHECK(*dev < 1e-6);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    SRFM_SUITE("GeodesicSolverN Straight Lines",
        test_geodesic_flat_straight_line_n1,
        test_geodesic_flat_straight_line_n2
    );
    SRFM_SUITE("GeodesicSolverN Deviation",
        test_geodesic_flat_deviation_below_1e8_n1,
        test_geodesic_flat_deviation_below_1e8_n4,
        test_geodesic_flat_deviation_below_1e8_n10,
        test_geodesic_flat_n50_deviation,
        test_geodesic_dtau_half_more_accurate_n2
    );
    SRFM_SUITE("GeodesicSolverN Step Count",
        test_geodesic_integrate_100_steps_returns_100_states,
        test_geodesic_integrate_0_steps_returns_empty,
        test_geodesic_integrate_1_step_returns_1_state,
        test_geodesic_integrate_step_count_sweep
    );
    SRFM_SUITE("GeodesicSolverN State Dimension",
        test_geodesic_state_dimension_matches_manifold_n1,
        test_geodesic_state_dimension_matches_manifold_n5,
        test_geodesic_dim_accessor_n3
    );
    SRFM_SUITE("GeodesicSolverN Reversibility",
        test_geodesic_rk4_step_reversible_n1,
        test_geodesic_rk4_step_reversible_n3
    );
    SRFM_SUITE("GeodesicSolverN Energy Conservation",
        test_geodesic_energy_conservation_flat_n1,
        test_geodesic_energy_conservation_flat_n4
    );
    SRFM_SUITE("GeodesicSolverN Velocity",
        test_geodesic_flat_velocity_unchanged_n1,
        test_geodesic_flat_velocity_unchanged_n4
    );
    SRFM_SUITE("GeodesicSolverN Edge Cases",
        test_geodesic_deviation_single_state_nullopt,
        test_geodesic_deviation_empty_nullopt
    );
    SRFM_SUITE("GeodesicSolverN N Sweep",
        test_geodesic_n_sweep_deviation_flat_1_to_8
    );
    return srfm_test::report();
}
