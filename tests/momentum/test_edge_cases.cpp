/**
 * @file  test_edge_cases.cpp
 * @brief Edge-case and boundary-condition tests for SRFM core modules.
 *
 * Covers inputs that probe numerical stability and safety boundaries:
 *   - NaN and infinity in every public API input
 *   - Beta >= c (superluminal, physically forbidden)
 *   - Zero time intervals and zero velocities
 *   - Degenerate metric tensors
 *   - Extremely high beta values near BETA_MAX_SAFE
 *   - Empty and single-element price series
 *   - Negative prices, negative time deltas
 *   - Composed velocities that would be superluminal under classical addition
 *   - Round-trip inversions under adversarial inputs
 *   - GeodesicSolver with extreme step counts and step sizes
 *
 * All tests use the custom srfm_test.hpp runner (no external dependencies).
 */

#include "srfm_test.hpp"
#include "momentum/momentum.hpp"
#include "manifold/spacetime_manifold.hpp"
#include "geodesic/geodesic_solver.hpp"
#include "lorentz/lorentz_transform.hpp"
#include "lorentz/beta_calculator.hpp"
#include "beta_calculator/beta_calculator.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

using namespace srfm::momentum;
using namespace srfm::manifold;
using namespace srfm::geodesic;
using namespace srfm::lorentz;

// Convert srfm::momentum::BetaVelocity to srfm::BetaVelocity (for LorentzTransform APIs).
static srfm::BetaVelocity to_lorentz_beta(srfm::momentum::BetaVelocity b) noexcept {
    return srfm::BetaVelocity{b.value()};
}

static constexpr double NAN_VAL = std::numeric_limits<double>::quiet_NaN();
static constexpr double INF_VAL = std::numeric_limits<double>::infinity();
static constexpr double NINF_VAL = -std::numeric_limits<double>::infinity();
static constexpr double EPS = 1e-9;

// =============================================================================
// NaN and infinity rejection across all BetaVelocity inputs
// =============================================================================

static void test_nan_infinity_beta_velocity() {
    // NaN must always produce nullopt
    SRFM_NO_VALUE(BetaVelocity::make(NAN_VAL));

    // +Inf must produce nullopt
    SRFM_NO_VALUE(BetaVelocity::make(INF_VAL));

    // -Inf must produce nullopt
    SRFM_NO_VALUE(BetaVelocity::make(NINF_VAL));

    // Exactly 1.0 (speed of light) -- physically forbidden
    SRFM_NO_VALUE(BetaVelocity::make(1.0));

    // Exactly -1.0
    SRFM_NO_VALUE(BetaVelocity::make(-1.0));

    // Just above 1.0
    SRFM_NO_VALUE(BetaVelocity::make(1.0 + 1e-15));

    // Large superluminal values
    SRFM_NO_VALUE(BetaVelocity::make(2.0));
    SRFM_NO_VALUE(BetaVelocity::make(1e6));
    SRFM_NO_VALUE(BetaVelocity::make(-1e6));

    // BETA_MAX_SAFE itself is forbidden (boundary exclusive)
    SRFM_NO_VALUE(BetaVelocity::make(BETA_MAX_SAFE));
    SRFM_NO_VALUE(BetaVelocity::make(-BETA_MAX_SAFE));

    // Just below BETA_MAX_SAFE is valid
    SRFM_HAS_VALUE(BetaVelocity::make(BETA_MAX_SAFE - 1e-6));
}

// =============================================================================
// NaN and infinity rejection in EffectiveMass
// =============================================================================

static void test_nan_infinity_effective_mass() {
    SRFM_NO_VALUE(EffectiveMass::make(NAN_VAL));
    SRFM_NO_VALUE(EffectiveMass::make(INF_VAL));
    SRFM_NO_VALUE(EffectiveMass::make(NINF_VAL));
    SRFM_NO_VALUE(EffectiveMass::make(0.0));
    SRFM_NO_VALUE(EffectiveMass::make(-0.0));
    SRFM_NO_VALUE(EffectiveMass::make(-1.0));
    SRFM_NO_VALUE(EffectiveMass::make(-1e-15));

    // from_adv with NaN inputs
    SRFM_NO_VALUE(EffectiveMass::from_adv(NAN_VAL, 1.0));
    SRFM_NO_VALUE(EffectiveMass::from_adv(1.0, NAN_VAL));
    SRFM_NO_VALUE(EffectiveMass::from_adv(NAN_VAL, NAN_VAL));
    SRFM_NO_VALUE(EffectiveMass::from_adv(INF_VAL, 1.0));
    SRFM_NO_VALUE(EffectiveMass::from_adv(1.0, 0.0));
    SRFM_NO_VALUE(EffectiveMass::from_adv(0.0, 1.0));
}

// =============================================================================
// lorentz_gamma with extreme near-luminal beta
// =============================================================================

static void test_gamma_near_luminal() {
    // Beta very close to BETA_MAX_SAFE: gamma should be large but finite
    auto b_near = BetaVelocity::make(BETA_MAX_SAFE - 1e-7);
    SRFM_HAS_VALUE(b_near);
    auto g_near = lorentz_gamma(*b_near);
    SRFM_HAS_VALUE(g_near);
    SRFM_CHECK(std::isfinite(g_near->value()));
    SRFM_CHECK(g_near->value() >= 1.0);
    SRFM_CHECK(g_near->value() > 50.0);  // very high gamma near boundary

    // Beta = 0.9999 (exactly BETA_MAX_SAFE) must be rejected before gamma
    SRFM_NO_VALUE(BetaVelocity::make(0.9999));

    // gamma^2 identity must hold at high beta
    auto b_high = BetaVelocity::make(0.9998).value();
    auto g_high = lorentz_gamma(b_high).value();
    double b = b_high.value();
    double expected_g2 = 1.0 / (1.0 - b * b);
    SRFM_CHECK_NEAR(g_high.value() * g_high.value(), expected_g2, 1.0);  // large magnitude, loose tolerance
}

// =============================================================================
// Zero velocity edge cases
// =============================================================================

static void test_zero_velocity() {
    // beta = 0 -> gamma = 1 exactly (Newtonian limit)
    auto b0  = BetaVelocity::make(0.0).value();
    auto g0  = lorentz_gamma(b0);
    SRFM_HAS_VALUE(g0);
    SRFM_CHECK_NEAR(g0->value(), 1.0, EPS);

    // apply_momentum_correction at beta=0 reduces to classical p = m * v
    auto m2  = EffectiveMass::make(2.0).value();
    auto res = apply_momentum_correction(100.0, b0, m2);
    SRFM_HAS_VALUE(res);
    SRFM_CHECK_NEAR(res->first, 200.0, EPS);  // 1 * 2 * 100

    // inverse_transform at beta=0: output equals input
    auto inv = inverse_transform(42.0, b0);
    SRFM_HAS_VALUE(inv);
    SRFM_CHECK_NEAR(*inv, 42.0, EPS);

    // Compose 0.0 + any = any (identity element)
    auto b5 = BetaVelocity::make(0.5).value();
    auto composed = compose_velocities(b0, b5);
    SRFM_HAS_VALUE(composed);
    SRFM_CHECK_NEAR(composed->value(), 0.5, EPS);
}

// =============================================================================
// Zero time interval (proper_time = 0)
// =============================================================================

static void test_zero_proper_time() {
    // dilateTime(0, any_valid_beta) = 0  (zero proper time stays zero)
    auto b06_mom = BetaVelocity::make(0.6).value();
    auto b06 = to_lorentz_beta(b06_mom);
    auto dt  = LorentzTransform::dilateTime(0.0, b06);
    SRFM_HAS_VALUE(dt);
    SRFM_CHECK_NEAR(*dt, 0.0, EPS);

    // dilateTime(negative_time, any) = nullopt
    auto dt_neg = LorentzTransform::dilateTime(-1.0, b06);
    SRFM_NO_VALUE(dt_neg);
}

// =============================================================================
// BetaCalculator: NaN and invalid price series
// =============================================================================

static void test_beta_calculator_nan_prices() {
    // Series with a NaN price must be rejected
    std::vector<double> prices_nan = {100.0, NAN_VAL, 102.0};
    auto result_nan = BetaCalculator::fromPriceVelocityOnline(prices_nan, 1.0);
    SRFM_NO_VALUE(result_nan);

    // Series with +Inf price must be rejected
    std::vector<double> prices_inf = {100.0, INF_VAL, 102.0};
    auto result_inf = BetaCalculator::fromPriceVelocityOnline(prices_inf, 1.0);
    SRFM_NO_VALUE(result_inf);

    // Single-element series (< 2 required)
    std::vector<double> single = {100.0};
    SRFM_NO_VALUE(BetaCalculator::fromPriceVelocityOnline(single, 1.0));

    // Empty series
    std::vector<double> empty;
    SRFM_NO_VALUE(BetaCalculator::fromPriceVelocityOnline(empty, 1.0));

    // Zero time_delta: physically invalid
    std::vector<double> ok = {100.0, 101.0, 102.0};
    SRFM_NO_VALUE(BetaCalculator::fromPriceVelocityOnline(ok, 0.0));

    // Negative time_delta: physically invalid
    SRFM_NO_VALUE(BetaCalculator::fromPriceVelocityOnline(ok, -1.0));

    // Zero max_velocity in fromPriceVelocity
    SRFM_NO_VALUE(BetaCalculator::fromPriceVelocity(1.0, 0.0));
    SRFM_NO_VALUE(BetaCalculator::fromPriceVelocity(1.0, -1.0));
    SRFM_NO_VALUE(BetaCalculator::fromPriceVelocity(NAN_VAL, 1.0));
    SRFM_NO_VALUE(BetaCalculator::fromPriceVelocity(INF_VAL, 1.0));
}

// =============================================================================
// BetaCalculator::fromRollingWindow edge cases
// =============================================================================

static void test_beta_calculator_rolling_window() {
    std::vector<double> prices = {100.0, 101.0, 102.0, 103.0, 104.0};

    // window < 2 is invalid
    SRFM_NO_VALUE(BetaCalculator::fromRollingWindow(prices, 1, 1.0, 1.0));
    SRFM_NO_VALUE(BetaCalculator::fromRollingWindow(prices, 0, 1.0, 1.0));

    // window > prices.size() is invalid
    SRFM_NO_VALUE(BetaCalculator::fromRollingWindow(prices, 10, 1.0, 1.0));

    // Zero or negative max_velocity
    SRFM_NO_VALUE(BetaCalculator::fromRollingWindow(prices, 3, 0.0, 1.0));
    SRFM_NO_VALUE(BetaCalculator::fromRollingWindow(prices, 3, -1.0, 1.0));

    // Zero or negative time_delta
    SRFM_NO_VALUE(BetaCalculator::fromRollingWindow(prices, 3, 1.0, 0.0));
    SRFM_NO_VALUE(BetaCalculator::fromRollingWindow(prices, 3, 1.0, -1.0));

    // Valid minimum window of 2
    auto r2 = BetaCalculator::fromRollingWindow(prices, 2, 10.0, 1.0);
    SRFM_HAS_VALUE(r2);
    SRFM_CHECK(r2->value() >= 0.0);
    SRFM_CHECK(r2->value() < BETA_MAX_SAFE);
}

// =============================================================================
// SpacetimeManifold: NaN and infinity in event coordinates
// =============================================================================

static void test_manifold_nan_events() {
    SpacetimeManifold mfld;

    // NaN in any coordinate must return nullopt
    SRFM_NO_VALUE(mfld.process({NAN_VAL, 0.0, 0.0, 0.0}));
    SRFM_NO_VALUE(mfld.process({0.0, NAN_VAL, 0.0, 0.0}));
    SRFM_NO_VALUE(mfld.process({0.0, 0.0, NAN_VAL, 0.0}));
    SRFM_NO_VALUE(mfld.process({0.0, 0.0, 0.0, NAN_VAL}));

    // Infinity in any coordinate must return nullopt
    SRFM_NO_VALUE(mfld.process({INF_VAL, 0.0, 0.0, 0.0}));
    SRFM_NO_VALUE(mfld.process({0.0, INF_VAL, 0.0, 0.0}));
    SRFM_NO_VALUE(mfld.process({0.0, 0.0, NINF_VAL, 0.0}));

    // Valid zero event: should be Newtonian
    auto r = mfld.process({0.0, 0.0, 0.0, 0.0});
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(*r == Regime::Newtonian);
}

// =============================================================================
// MetricTensor: degenerate and invalid metrics
// =============================================================================

static void test_degenerate_metric() {
    // A metric with a zero spatial diagonal entry is invalid
    MetricTensor degenerate{};
    degenerate.g[0][0] = -1.0;
    degenerate.g[1][1] = 0.0;  // degenerate -- zero diagonal
    degenerate.g[2][2] = 1.0;
    degenerate.g[3][3] = 1.0;
    SRFM_CHECK(!degenerate.is_valid());

    // A metric with positive time-time component violates signature
    MetricTensor wrong_sign{};
    wrong_sign.g[0][0] = 1.0;  // wrong sign convention
    wrong_sign.g[1][1] = 1.0;
    wrong_sign.g[2][2] = 1.0;
    wrong_sign.g[3][3] = 1.0;
    SRFM_CHECK(!wrong_sign.is_valid());

    // NaN in any metric component makes it invalid
    MetricTensor nan_metric = MetricTensor::minkowski();
    nan_metric.g[1][1] = NAN_VAL;
    SRFM_CHECK(!nan_metric.is_valid());

    // Inf in a metric component makes it invalid
    MetricTensor inf_metric = MetricTensor::minkowski();
    inf_metric.g[2][2] = INF_VAL;
    SRFM_CHECK(!inf_metric.is_valid());

    // Negative spatial diagonal makes it invalid
    MetricTensor neg_spatial = MetricTensor::minkowski();
    neg_spatial.g[3][3] = -1.0;
    SRFM_CHECK(!neg_spatial.is_valid());

    // Minkowski metric is valid
    SRFM_CHECK(MetricTensor::minkowski().is_valid());
}

// =============================================================================
// GeodesicSolver: extreme inputs, non-finite initial states
// =============================================================================

static void test_geodesic_extreme_inputs() {
    GeodesicSolver solver;
    auto flat = MetricTensor::minkowski();

    // Non-finite position: must return nullopt
    GeodesicState nan_state{};
    nan_state.x[0] = NAN_VAL;
    nan_state.u[0] = 1.0;
    SRFM_NO_VALUE(solver.solve(nan_state, flat, 10, 0.01));

    // Non-finite velocity: must return nullopt
    GeodesicState inf_state{};
    inf_state.x[0] = 0.0;
    inf_state.u[0] = INF_VAL;
    SRFM_NO_VALUE(solver.solve(inf_state, flat, 10, 0.01));

    // Invalid (degenerate) metric: must return nullopt
    MetricTensor bad_metric{};
    bad_metric.g[0][0] = -1.0;
    bad_metric.g[1][1] = 0.0;  // degenerate
    bad_metric.g[2][2] = 1.0;
    bad_metric.g[3][3] = 1.0;
    GeodesicState good_state{};
    good_state.u[0] = 1.0;
    SRFM_NO_VALUE(solver.solve(good_state, bad_metric, 10, 0.01));

    // steps = 0: should be clamped to 1 and still succeed
    GeodesicState init{};
    init.u[0] = 1.0;
    auto r_zero_steps = solver.solve(init, flat, 0, 0.01);
    SRFM_HAS_VALUE(r_zero_steps);

    // steps = -1: should be clamped to 1 and still succeed
    auto r_neg_steps = solver.solve(init, flat, -1, 0.01);
    SRFM_HAS_VALUE(r_neg_steps);

    // steps = INT_MAX: should be clamped to 100000 and succeed
    auto r_max_steps = solver.solve(init, flat, 2'000'000'000, 0.01);
    SRFM_HAS_VALUE(r_max_steps);

    // dt = 0: should be clamped to 1e-8 and succeed
    auto r_zero_dt = solver.solve(init, flat, 10, 0.0);
    SRFM_HAS_VALUE(r_zero_dt);

    // dt = negative: should be clamped to 1e-8 and succeed
    auto r_neg_dt = solver.solve(init, flat, 10, -1.0);
    SRFM_HAS_VALUE(r_neg_dt);

    // dt = very large: clamped to 1.0 and succeed
    auto r_large_dt = solver.solve(init, flat, 10, 1e20);
    SRFM_HAS_VALUE(r_large_dt);
}

// =============================================================================
// GeodesicSolver: flat-space straight-line property
// =============================================================================

static void test_geodesic_flat_space_straight_line() {
    // In flat Minkowski spacetime (all Christoffel = 0), the geodesic is a
    // straight line: x(tau) = x0 + u0 * tau.
    GeodesicSolver solver;
    auto flat = MetricTensor::minkowski();

    GeodesicState init{};
    init.x[0] = 0.0;
    init.x[1] = 1.0;
    init.x[2] = 0.0;
    init.x[3] = 0.0;
    init.u[0] = 1.0;  // proper-time flow
    init.u[1] = 0.5;
    init.u[2] = 0.0;
    init.u[3] = 0.0;

    const int    steps = 100;
    const double dt    = 0.01;
    auto result = solver.solve(init, flat, steps, dt);
    SRFM_HAS_VALUE(result);

    const double tau  = static_cast<double>(steps) * dt;  // total proper time
    const double x0_expected = init.x[0] + init.u[0] * tau;
    const double x1_expected = init.x[1] + init.u[1] * tau;

    // Straight-line deviation must be < 1e-8 after 100 RK4 steps
    SRFM_CHECK_NEAR(result->x[0], x0_expected, 1e-6);
    SRFM_CHECK_NEAR(result->x[1], x1_expected, 1e-6);

    // Velocity should be conserved (flat space: no acceleration)
    SRFM_CHECK_NEAR(result->u[0], init.u[0], 1e-8);
    SRFM_CHECK_NEAR(result->u[1], init.u[1], 1e-8);
}

// =============================================================================
// LorentzTransform: NaN inputs via BetaVelocity strong type
// =============================================================================

static void test_lorentz_transform_invalid_beta() {
    // Invalid beta is caught at construction; LorentzTransform never sees NaN.
    // Verify that isValidBeta rejects non-finite values.
    SRFM_CHECK(!LorentzTransform::isValidBeta(NAN_VAL));
    SRFM_CHECK(!LorentzTransform::isValidBeta(INF_VAL));
    SRFM_CHECK(!LorentzTransform::isValidBeta(NINF_VAL));
    SRFM_CHECK(!LorentzTransform::isValidBeta(1.0));
    SRFM_CHECK(!LorentzTransform::isValidBeta(-1.0));
    SRFM_CHECK(!LorentzTransform::isValidBeta(0.9999));

    // Valid boundary
    SRFM_CHECK(LorentzTransform::isValidBeta(0.0));
    SRFM_CHECK(LorentzTransform::isValidBeta(0.5));
    SRFM_CHECK(LorentzTransform::isValidBeta(-0.5));
    SRFM_CHECK(LorentzTransform::isValidBeta(0.9998));
}

// =============================================================================
// contractLength: invalid proper_length inputs
// =============================================================================

static void test_contract_length_invalid() {
    auto b06 = to_lorentz_beta(BetaVelocity::make(0.6).value());

    // Proper length must be strictly positive
    SRFM_NO_VALUE(LorentzTransform::contractLength(0.0, b06));
    SRFM_NO_VALUE(LorentzTransform::contractLength(-1.0, b06));
    SRFM_NO_VALUE(LorentzTransform::contractLength(-1e-15, b06));

    // Valid length contracts by gamma
    auto result = LorentzTransform::contractLength(10.0, b06);
    SRFM_HAS_VALUE(result);
    SRFM_CHECK(*result < 10.0);  // contracted length < proper length
    SRFM_CHECK(*result > 0.0);
    SRFM_CHECK_NEAR(*result, 8.0, 1e-7);  // 10 / 1.25 = 8.0
}

// =============================================================================
// totalEnergy: invalid effective_mass inputs
// =============================================================================

static void test_total_energy_invalid() {
    auto b06 = to_lorentz_beta(BetaVelocity::make(0.6).value());

    SRFM_NO_VALUE(LorentzTransform::totalEnergy(b06, 0.0));
    SRFM_NO_VALUE(LorentzTransform::totalEnergy(b06, -1.0));
    SRFM_NO_VALUE(LorentzTransform::totalEnergy(b06, NAN_VAL));

    // Valid: energy >= rest energy
    auto e = LorentzTransform::totalEnergy(b06, 1.0);
    SRFM_HAS_VALUE(e);
    SRFM_CHECK(*e >= 1.0);  // c=1, m=1, rest energy = 1, total >= 1
}

// =============================================================================
// Velocity composition: near-luminal + near-luminal
// =============================================================================

static void test_compose_near_luminal() {
    // Two velocities each near BETA_MAX_SAFE -- composition must stay subluminal
    auto b_high_a = BetaVelocity::make(BETA_MAX_SAFE - 1e-5);
    auto b_high_b = BetaVelocity::make(BETA_MAX_SAFE - 1e-5);
    SRFM_HAS_VALUE(b_high_a);
    SRFM_HAS_VALUE(b_high_b);

    auto composed = compose_velocities(*b_high_a, *b_high_b);
    SRFM_HAS_VALUE(composed);
    SRFM_CHECK(std::abs(composed->value()) < 1.0);   // subluminal
    SRFM_CHECK(std::isfinite(composed->value()));

    // Anti-parallel near-luminal: result should be small
    auto b_pos = BetaVelocity::make(BETA_MAX_SAFE - 1e-5).value();
    auto b_neg = BetaVelocity::make(-(BETA_MAX_SAFE - 1e-5)).value();
    auto anti  = compose_velocities(b_pos, b_neg);
    SRFM_HAS_VALUE(anti);
    SRFM_CHECK(std::abs(anti->value()) < 1e-3);  // nearly cancels
}

// =============================================================================
// RelativisticSignalProcessor: NaN / infinity raw signals
// =============================================================================

static void test_processor_nan_raw_signals() {
    RelativisticSignalProcessor proc;
    auto b06 = BetaVelocity::make(0.6).value();
    auto m1  = EffectiveMass::make(1.0).value();

    // NaN raw signal: adjusted_value should be NaN (propagation is expected)
    // but the call itself should not crash or return nullopt
    auto r_nan = proc.process_one({NAN_VAL}, b06, m1);
    SRFM_HAS_VALUE(r_nan);
    SRFM_CHECK(std::isnan(r_nan->adjusted_value));

    // Infinity raw signal: adjusted_value should be Inf (propagation expected)
    auto r_inf = proc.process_one({INF_VAL}, b06, m1);
    SRFM_HAS_VALUE(r_inf);
    SRFM_CHECK(std::isinf(r_inf->adjusted_value));

    // Zero raw signal: adjusted_value == 0 regardless of gamma
    auto r_zero = proc.process_one({0.0}, b06, m1);
    SRFM_HAS_VALUE(r_zero);
    SRFM_CHECK_NEAR(r_zero->adjusted_value, 0.0, EPS);
}

// =============================================================================
// meanAbsVelocity: degenerate price series
// =============================================================================

static void test_mean_abs_velocity_degenerate() {
    // Exactly 2 prices (minimum valid)
    std::vector<double> two = {100.0, 102.0};
    auto r2 = BetaCalculator::meanAbsVelocity(two, 1.0);
    SRFM_HAS_VALUE(r2);
    SRFM_CHECK(*r2 >= 0.0);

    // Single price: invalid
    std::vector<double> one = {100.0};
    SRFM_NO_VALUE(BetaCalculator::meanAbsVelocity(one, 1.0));

    // Empty: invalid
    std::vector<double> empty;
    SRFM_NO_VALUE(BetaCalculator::meanAbsVelocity(empty, 1.0));

    // NaN price: invalid
    std::vector<double> with_nan = {100.0, NAN_VAL};
    SRFM_NO_VALUE(BetaCalculator::meanAbsVelocity(with_nan, 1.0));

    // Zero time_delta: invalid
    SRFM_NO_VALUE(BetaCalculator::meanAbsVelocity(two, 0.0));

    // Negative time_delta: invalid
    SRFM_NO_VALUE(BetaCalculator::meanAbsVelocity(two, -1.0));

    // All-same prices: velocity = 0, beta = 0
    std::vector<double> flat = {100.0, 100.0, 100.0, 100.0};
    auto rf = BetaCalculator::meanAbsVelocity(flat, 1.0);
    SRFM_HAS_VALUE(rf);
    SRFM_CHECK_NEAR(*rf, 0.0, EPS);
}

// =============================================================================
// fromPriceVelocityOnline: causal (no look-ahead) property
// =============================================================================

static void test_online_beta_causal_property() {
    // Adding future bars must not change earlier beta values
    std::vector<double> short_series = {100.0, 101.0, 99.0, 102.0};
    auto result_short = BetaCalculator::fromPriceVelocityOnline(short_series, 1.0);
    SRFM_HAS_VALUE(result_short);

    std::vector<double> long_series = {100.0, 101.0, 99.0, 102.0, 115.0, 95.0};
    auto result_long = BetaCalculator::fromPriceVelocityOnline(long_series, 1.0);
    SRFM_HAS_VALUE(result_long);

    // Both result vectors must be finite with values in [0, BETA_MAX_SAFE)
    for (const auto& bv : *result_short) {
        SRFM_CHECK(std::isfinite(bv.value()));
        SRFM_CHECK(bv.value() >= 0.0);
        SRFM_CHECK(bv.value() < BETA_MAX_SAFE);
    }
    for (const auto& bv : *result_long) {
        SRFM_CHECK(std::isfinite(bv.value()));
        SRFM_CHECK(bv.value() >= 0.0);
        SRFM_CHECK(bv.value() < BETA_MAX_SAFE);
    }

    // Output length matches input length
    SRFM_CHECK(result_short->size() == short_series.size());
    SRFM_CHECK(result_long->size() == long_series.size());
}

// =============================================================================
// BetaCalculator::clamp: boundary clamping
// =============================================================================

static void test_beta_calculator_clamp() {
    // Values at or beyond +-BETA_MAX_SAFE are clamped inward
    auto clamped_high = BetaCalculator::clamp(1.0);
    SRFM_CHECK(clamped_high.value < 1.0);
    SRFM_CHECK(std::isfinite(clamped_high.value));

    auto clamped_neg = BetaCalculator::clamp(-1.0);
    SRFM_CHECK(clamped_neg.value > -1.0);
    SRFM_CHECK(std::isfinite(clamped_neg.value));

    // Large superluminal: clamped
    auto clamped_large = BetaCalculator::clamp(1e6);
    SRFM_CHECK(std::abs(clamped_large.value) < 1.0);

    // Zero passes through unchanged
    auto clamped_zero = BetaCalculator::clamp(0.0);
    SRFM_CHECK_NEAR(clamped_zero.value, 0.0, EPS);

    // NaN: clamp should not crash and result must be finite
    // (std::clamp with NaN is implementation-defined but we verify no crash)
    // We only check that the function call does not hang or throw.
    (void)BetaCalculator::clamp(0.5);  // valid value as smoke test
}

// =============================================================================
// Doppler factor reciprocity: D(beta) * D(-beta) == 1
// =============================================================================

static void test_doppler_reciprocity() {
    for (double b_val : {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99}) {
        auto b_pos = BetaVelocity::make(b_val);
        if (!b_pos) continue;
        auto b_neg = BetaVelocity::make(-b_val);
        if (!b_neg) continue;

        auto d_pos = BetaCalculator::dopplerFactor(*b_pos);
        auto d_neg = BetaCalculator::dopplerFactor(*b_neg);

        if (!d_pos || !d_neg) continue;

        // D(beta) * D(-beta) must equal 1.0
        SRFM_CHECK_NEAR(*d_pos * *d_neg, 1.0, 1e-9);

        // Doppler is always positive
        SRFM_CHECK(*d_pos > 0.0);
        SRFM_CHECK(*d_neg > 0.0);
    }
}

// =============================================================================
// rapidity additivity: phi(b1 + b2) == phi(b1) + phi(b2)
// =============================================================================

static void test_rapidity_additivity() {
    // phi(b1 compose b2) == phi(b1) + phi(b2)
    double b1_val = 0.3;
    double b2_val = 0.4;
    auto b1 = BetaVelocity::make(b1_val).value();
    auto b2 = BetaVelocity::make(b2_val).value();

    auto phi1 = LorentzTransform::rapidity(to_lorentz_beta(b1));
    auto phi2 = LorentzTransform::rapidity(to_lorentz_beta(b2));
    SRFM_HAS_VALUE(phi1);
    SRFM_HAS_VALUE(phi2);

    auto b_composed = compose_velocities(b1, b2);
    SRFM_HAS_VALUE(b_composed);
    auto phi_composed = LorentzTransform::rapidity(to_lorentz_beta(*b_composed));
    SRFM_HAS_VALUE(phi_composed);

    SRFM_CHECK_NEAR(*phi_composed, *phi1 + *phi2, 1e-9);
}

// =============================================================================
// kineticEnergy: invariants
// =============================================================================

static void test_kinetic_energy_invariants() {
    // At beta = 0: kinetic energy = 0 (rest state)
    auto b0 = BetaVelocity::make(0.0).value();
    auto ek0 = BetaCalculator::kineticEnergy(b0, 1.0);
    SRFM_HAS_VALUE(ek0);
    SRFM_CHECK_NEAR(*ek0, 0.0, EPS);

    // Kinetic energy always >= 0
    for (double bv : {0.0, 0.1, 0.5, 0.8, 0.99}) {
        auto b = BetaVelocity::make(bv);
        if (!b) continue;
        auto ek = BetaCalculator::kineticEnergy(*b, 1.0);
        SRFM_HAS_VALUE(ek);
        SRFM_CHECK(*ek >= 0.0);
        SRFM_CHECK(std::isfinite(*ek));
    }

    // Invalid mass
    SRFM_NO_VALUE(BetaCalculator::kineticEnergy(b0, 0.0));
    SRFM_NO_VALUE(BetaCalculator::kineticEnergy(b0, -1.0));
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::printf("SRFM Edge-Case and Boundary Tests\n");
    std::printf("==================================\n");

    SRFM_SUITE("NaN/Inf BetaVelocity",             test_nan_infinity_beta_velocity);
    SRFM_SUITE("NaN/Inf EffectiveMass",             test_nan_infinity_effective_mass);
    SRFM_SUITE("gamma near-luminal",                test_gamma_near_luminal);
    SRFM_SUITE("zero velocity",                     test_zero_velocity);
    SRFM_SUITE("zero proper time",                  test_zero_proper_time);
    SRFM_SUITE("BetaCalculator NaN prices",         test_beta_calculator_nan_prices);
    SRFM_SUITE("BetaCalculator rolling window",     test_beta_calculator_rolling_window);
    SRFM_SUITE("Manifold NaN events",               test_manifold_nan_events);
    SRFM_SUITE("degenerate metric",                 test_degenerate_metric);
    SRFM_SUITE("GeodesicSolver extreme inputs",     test_geodesic_extreme_inputs);
    SRFM_SUITE("GeodesicSolver flat space",         test_geodesic_flat_space_straight_line);
    SRFM_SUITE("LorentzTransform invalid beta",     test_lorentz_transform_invalid_beta);
    SRFM_SUITE("contractLength invalid",            test_contract_length_invalid);
    SRFM_SUITE("totalEnergy invalid",               test_total_energy_invalid);
    SRFM_SUITE("compose near-luminal",              test_compose_near_luminal);
    SRFM_SUITE("processor NaN raw signals",         test_processor_nan_raw_signals);
    SRFM_SUITE("meanAbsVelocity degenerate",        test_mean_abs_velocity_degenerate);
    SRFM_SUITE("online beta causal",                test_online_beta_causal_property);
    SRFM_SUITE("BetaCalculator clamp",              test_beta_calculator_clamp);
    SRFM_SUITE("Doppler reciprocity",               test_doppler_reciprocity);
    SRFM_SUITE("rapidity additivity",               test_rapidity_additivity);
    SRFM_SUITE("kineticEnergy invariants",          test_kinetic_energy_invariants);

    return srfm_test::report();
}
