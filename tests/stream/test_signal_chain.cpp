// =============================================================================
// test_signal_chain.cpp — Signal chain integration + component unit tests
// Tests CoordinateNormalizer, BetaCalculator, LorentzTransform, SpacetimeManifold
// as integrated pipeline and as individual components under edge cases.
// =============================================================================

#include "srfm_stream_test.hpp"
#include "srfm/stream/coordinate_normalizer.hpp"
#include "srfm/stream/beta_calculator.hpp"
#include "srfm/stream/lorentz_transform.hpp"
#include "srfm/stream/spacetime_manifold.hpp"
#include "srfm/stream/tick.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <array>
#include <cstring>

using namespace srfm::stream;

// ---------------------------------------------------------------------------
// 4-stage pipeline helper
// ---------------------------------------------------------------------------
namespace {

struct PipelineResult {
    double normalised;
    double beta_val;
    double gamma_val;
    double t_prime;
    double x_prime;
    double ds2;
    double signal;
    Regime regime;
};

struct Pipeline4Stage {
    CoordinateNormalizer normalizer;
    BetaCalculatorFix3   beta_calc;
    LorentzTransform     lorentz;
    SpacetimeManifold    manifold;
    std::size_t          bar{0};

    Pipeline4Stage() : normalizer(20) {}

    PipelineResult step(double close) {
        normalizer.update(close);
        double norm = normalizer.normalise(close);
        beta_calc.update(close);
        double beta  = beta_calc.beta();
        double gamma = LorentzTransform::lorentz_gamma(beta);
        auto ev = lorentz.transform(static_cast<double>(bar), norm, beta);
        auto mr = manifold.update(ev.t_prime, ev.x_prime);
        ++bar;
        return { norm, beta, ev.gamma, ev.t_prime, ev.x_prime,
                 mr.ds2, gamma * mr.signal, mr.regime };
    }

    void reset() {
        normalizer.reset();
        beta_calc.reset();
        manifold.reset();
        bar = 0;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// CoordinateNormalizer
// ---------------------------------------------------------------------------

static void test_normalizer_window_size_2_basic() {
    CoordinateNormalizer n(2);
    n.update(10.0);
    STREAM_CHECK(!n.warmed_up());
    n.update(20.0);
    STREAM_CHECK(n.warmed_up());
    STREAM_CHECK_NEAR(n.mean(), 15.0, 1e-9);
    STREAM_CHECK_NEAR(n.sigma(), 5.0, 1e-6);
}

static void test_normalizer_window_size_50() {
    CoordinateNormalizer n(50);
    for (double i = 1.0; i <= 50.0; ++i) n.update(i);
    STREAM_CHECK(n.warmed_up());
    STREAM_CHECK_NEAR(n.mean(), 25.5, 1e-9);
}

static void test_normalizer_update_beyond_window() {
    CoordinateNormalizer n(5);
    for (double i = 1.0; i <= 10.0; ++i) n.update(i);
    // Newest 5: {6,7,8,9,10}, mean = 8.0
    STREAM_CHECK_NEAR(n.mean(), 8.0, 1e-9);
}

static void test_normalizer_normalise_at_mean_is_zero() {
    CoordinateNormalizer n(20);
    for (double i = 1.0; i <= 20.0; ++i) n.update(i);
    double mu = n.mean();
    STREAM_CHECK_NEAR(n.normalise(mu), 0.0, 1e-9);
}

static void test_normalizer_z_score_symmetric() {
    CoordinateNormalizer n(20);
    for (double i = 1.0; i <= 20.0; ++i) n.update(i);
    double mu = n.mean();
    double sig = n.sigma();
    STREAM_CHECK_NEAR(n.normalise(mu + sig),  1.0, 1e-9);
    STREAM_CHECK_NEAR(n.normalise(mu - sig), -1.0, 1e-9);
}

static void test_normalizer_monotone_z_scores() {
    CoordinateNormalizer n(20);
    for (double i = 1.0; i <= 20.0; ++i) n.update(i);
    double z1 = n.normalise(5.0);
    double z2 = n.normalise(10.0);
    double z3 = n.normalise(15.0);
    STREAM_CHECK(z1 < z2);
    STREAM_CHECK(z2 < z3);
}

static void test_normalizer_two_instances_scale_independently() {
    CoordinateNormalizer a(20), b(20);
    for (double i = 1.0; i <= 20.0; ++i) {
        a.update(i);
        b.update(i * 10.0);
    }
    STREAM_CHECK_NEAR(a.mean() * 10.0, b.mean(), 1e-9);
    STREAM_CHECK_NEAR(a.sigma() * 10.0, b.sigma(), 1e-6);
}

static void test_normalizer_reset_clears_warmup() {
    CoordinateNormalizer n(5);
    for (double i = 1.0; i <= 10.0; ++i) n.update(i);
    STREAM_CHECK(n.warmed_up());
    n.reset();
    STREAM_CHECK(!n.warmed_up());
}

static void test_normalizer_after_reset_rewarms() {
    CoordinateNormalizer n(5);
    for (double i = 1.0; i <= 5.0; ++i) n.update(i);
    n.reset();
    for (double i = 100.0; i <= 104.0; ++i) n.update(i);
    STREAM_CHECK(n.warmed_up());
    STREAM_CHECK_NEAR(n.mean(), 102.0, 1e-9);
}

static void test_normalizer_sigma_always_nonneg() {
    CoordinateNormalizer n(20);
    std::mt19937 rng(0xCAFE);
    std::uniform_real_distribution<double> dist(0.5, 500.0);
    for (int i = 0; i < 200; ++i) n.update(dist(rng));
    STREAM_CHECK_GE(n.sigma(), 0.0);
}

static void test_normalizer_welford_vs_naive() {
    const std::size_t W = 100;
    CoordinateNormalizer n(W);
    std::vector<double> vals;
    for (double i = 1.0; i <= static_cast<double>(W); ++i) {
        vals.push_back(i);
        n.update(i);
    }
    double mu = std::accumulate(vals.begin(), vals.end(), 0.0) / W;
    double var = 0.0;
    for (double v : vals) var += (v - mu) * (v - mu);
    var /= W;
    STREAM_CHECK_NEAR(n.mean(), mu, 1e-9);
    STREAM_CHECK_NEAR(n.sigma() * n.sigma(), var, 1e-6);
}

static void test_normalizer_count_accessor() {
    CoordinateNormalizer n(20);
    STREAM_CHECK(n.count() == std::size_t{0});
    n.update(1.0);
    STREAM_CHECK(n.count() == std::size_t{1});
    for (int i = 0; i < 19; ++i) n.update(2.0);
    STREAM_CHECK(n.count() == std::size_t{20});
}

static void test_normalizer_count_saturates_at_window() {
    CoordinateNormalizer n(5);
    for (int i = 0; i < 50; ++i) n.update(static_cast<double>(i));
    STREAM_CHECK(n.count() == std::size_t{5});
}

static void test_normalizer_high_frequency_stable() {
    CoordinateNormalizer n(50);
    for (int i = 0; i < 1000; ++i) n.update(100.0 + 0.001 * i);
    STREAM_CHECK(std::isfinite(n.mean()));
    STREAM_CHECK(std::isfinite(n.sigma()));
    STREAM_CHECK_GE(n.sigma(), 0.0);
}

// ---------------------------------------------------------------------------
// BetaCalculator
// ---------------------------------------------------------------------------

static void test_beta_warmup_exactly_n_plus_1() {
    BetaCalculatorFix3 bc;
    for (int i = 0; i < 3; ++i) {
        bc.update(100.0 + i);
        STREAM_CHECK(!bc.warmed_up());
    }
    bc.update(103.0);
    STREAM_CHECK(bc.warmed_up());
}

static void test_beta_zero_before_warmup() {
    BetaCalculatorFix3 bc;
    bc.update(100.0);
    STREAM_CHECK_NEAR(bc.beta(), 0.0, 1e-12);
    bc.update(110.0);
    STREAM_CHECK_NEAR(bc.beta(), 0.0, 1e-12);
}

static void test_beta_clamped_to_max_safe() {
    BetaCalculatorFix3 bc;
    bc.update(1.0); bc.update(1e9); bc.update(1.0); bc.update(1e9);
    STREAM_CHECK(std::abs(bc.beta()) <= BETA_MAX_SAFE);
}

static void test_beta_always_finite() {
    BetaCalculatorFix3 bc;
    std::mt19937 rng(0xDEAD);
    std::uniform_real_distribution<double> dist(0.5, 500.0);
    for (int i = 0; i < 200; ++i) {
        bc.update(dist(rng));
        STREAM_CHECK(std::isfinite(bc.beta()));
    }
}

static void test_beta_flat_series_near_zero() {
    BetaCalculatorFix3 bc;
    for (int i = 0; i < 10; ++i) bc.update(100.0);
    STREAM_CHECK_NEAR(bc.beta(), 0.0, 1e-9);
}

static void test_beta_rising_series_positive() {
    BetaCalculatorFix3 bc;
    for (int i = 0; i < 10; ++i) bc.update(100.0 + i * 10.0);
    STREAM_CHECK_GT(bc.beta(), 0.0);
}

static void test_beta_fix3_manual_check() {
    BetaCalculatorFix3 bc;
    double closes[] = {100.0, 110.0, 121.0, 133.1};
    for (double c : closes) bc.update(c);
    double r1 = std::log(110.0 / 100.0);
    double r2 = std::log(121.0 / 110.0);
    double r3 = std::log(133.1 / 121.0);
    double avg = (r1 + r2 + r3) / 3.0;
    double expected = std::clamp(avg / DEFAULT_C_MARKET,
                                  -BETA_MAX_SAFE,
                                   BETA_MAX_SAFE);
    STREAM_CHECK_NEAR(bc.beta(), expected, 1e-9);
}

static void test_beta_reset_returns_zero() {
    BetaCalculatorFix3 bc;
    for (int i = 0; i < 20; ++i) bc.update(100.0 + i);
    STREAM_CHECK(bc.warmed_up());
    bc.reset();
    STREAM_CHECK(!bc.warmed_up());
    STREAM_CHECK_NEAR(bc.beta(), 0.0, 1e-12);
}

static void test_beta_invalid_close_nan_ignored() {
    BetaCalculatorFix3 bc;
    bc.update(100.0);
    bc.update(std::numeric_limits<double>::quiet_NaN());
    STREAM_CHECK(!bc.warmed_up());
}

static void test_beta_invalid_close_neg_ignored() {
    BetaCalculatorFix3 bc;
    bc.update(100.0);
    bc.update(-50.0);
    STREAM_CHECK(!bc.warmed_up());
}

static void test_beta_two_instances_no_crosstalk() {
    BetaCalculatorFix3 a, b;
    for (int i = 0; i < 10; ++i) a.update(100.0 + i * 2.0);
    for (int i = 0; i < 10; ++i) b.update(100.0 - i * 2.0);
    STREAM_CHECK_GT(a.beta(), 0.0);
    STREAM_CHECK(b.beta() <= 0.0);
}

// ---------------------------------------------------------------------------
// LorentzTransform
// ---------------------------------------------------------------------------

static void test_lorentz_beta_zero_identity() {
    LorentzTransform lt;
    auto ev = lt.transform(3.0, 4.0, 0.0);
    STREAM_CHECK_NEAR(ev.t_prime, 3.0, 1e-12);
    STREAM_CHECK_NEAR(ev.x_prime, 4.0, 1e-12);
    STREAM_CHECK_NEAR(ev.gamma,   1.0, 1e-12);
}

static void test_lorentz_gamma_monotone() {
    std::vector<double> betas = {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99};
    double prev = 1.0;
    for (double b : betas) {
        double g = LorentzTransform::lorentz_gamma(b);
        STREAM_CHECK_GE(g, prev);
        prev = g;
    }
}

static void test_lorentz_gamma_always_ge_1() {
    double betas[] = {0.0, 0.1, 0.5, 0.8, 0.9, 0.99, -0.5, -0.9};
    for (double b : betas)
        STREAM_CHECK_GE(LorentzTransform::lorentz_gamma(b), 1.0);
}

static void test_lorentz_known_beta_0_6() {
    // gamma = 1/sqrt(1-0.36) = 1.25
    STREAM_CHECK_NEAR(LorentzTransform::lorentz_gamma(0.6), 1.25, 1e-9);
}

static void test_lorentz_known_beta_0_8() {
    // gamma = 1/sqrt(0.36) = 5/3
    STREAM_CHECK_NEAR(LorentzTransform::lorentz_gamma(0.8), 5.0/3.0, 1e-9);
}

static void test_lorentz_invalid_beta_identity() {
    LorentzTransform lt;
    auto ev1 = lt.transform(2.0, 3.0, 1.0);
    STREAM_CHECK_NEAR(ev1.t_prime, 2.0, 1e-12);
    STREAM_CHECK_NEAR(ev1.x_prime, 3.0, 1e-12);
    auto ev2 = lt.transform(2.0, 3.0, 1.5);
    STREAM_CHECK_NEAR(ev2.t_prime, 2.0, 1e-12);
}

static void test_lorentz_negative_beta_negates_x() {
    LorentzTransform lt;
    auto pos = lt.transform(1.0, 0.0,  0.5);
    auto neg = lt.transform(1.0, 0.0, -0.5);
    STREAM_CHECK_NEAR(pos.t_prime,  neg.t_prime, 1e-12);
    STREAM_CHECK_NEAR(pos.x_prime, -neg.x_prime, 1e-12);
}

static void test_lorentz_inverse_round_trip() {
    LorentzTransform lt;
    std::mt19937 rng(0xF00D);
    std::uniform_real_distribution<double> td(-100.0, 100.0);
    std::uniform_real_distribution<double> bd(-0.99, 0.99);
    for (int i = 0; i < 50; ++i) {
        double t = td(rng), x = td(rng), b = bd(rng);
        auto fwd = lt.transform(t, x, b);
        auto inv = lt.inverse(fwd.t_prime, fwd.x_prime, b);
        STREAM_CHECK_NEAR(inv.t_prime, t, 1e-9);
        STREAM_CHECK_NEAR(inv.x_prime, x, 1e-9);
    }
}

static void test_lorentz_result_always_finite() {
    LorentzTransform lt;
    std::mt19937 rng(0xBEEF);
    std::uniform_real_distribution<double> td(-1000.0, 1000.0);
    std::uniform_real_distribution<double> bd(-0.9999, 0.9999);
    for (int i = 0; i < 100; ++i) {
        auto ev = lt.transform(td(rng), td(rng), bd(rng));
        STREAM_CHECK(std::isfinite(ev.t_prime));
        STREAM_CHECK(std::isfinite(ev.x_prime));
        STREAM_CHECK(std::isfinite(ev.gamma));
    }
}

// ---------------------------------------------------------------------------
// SpacetimeManifold
// ---------------------------------------------------------------------------

static void test_manifold_first_update_lightlike() {
    SpacetimeManifold m;
    auto r = m.update(1.0, 2.0);
    STREAM_CHECK(r.regime == Regime::LIGHTLIKE);
    STREAM_CHECK_NEAR(r.signal, 0.0, 1e-12);
}

static void test_manifold_timelike() {
    SpacetimeManifold m;
    (void)m.update(0.0, 0.0);
    auto r = m.update(5.0, 1.0);  // dt=5,dx=1 → ds2=24
    STREAM_CHECK(r.regime == Regime::TIMELIKE);
    STREAM_CHECK_GT(r.ds2, 0.0);
    STREAM_CHECK_NEAR(r.signal, std::sqrt(24.0), 1e-9);
}

static void test_manifold_spacelike() {
    SpacetimeManifold m;
    (void)m.update(0.0, 0.0);
    auto r = m.update(1.0, 5.0);  // dt=1,dx=5 → ds2=-24
    STREAM_CHECK(r.regime == Regime::SPACELIKE);
    STREAM_CHECK_LT(r.ds2, 0.0);
    STREAM_CHECK_NEAR(r.signal, -std::sqrt(24.0), 1e-9);
}

static void test_manifold_lightlike_null() {
    SpacetimeManifold m;
    (void)m.update(0.0, 0.0);
    auto r = m.update(3.0, 3.0);  // dt=dx → ds2=0
    STREAM_CHECK(r.regime == Regime::LIGHTLIKE);
    STREAM_CHECK_NEAR(r.signal, 0.0, 1e-9);
}

static void test_manifold_reset_clears_state() {
    SpacetimeManifold m;
    (void)m.update(5.0, 5.0);
    m.reset();
    auto r = m.update(10.0, 10.0);
    STREAM_CHECK(r.regime == Regime::LIGHTLIKE);
}

static void test_manifold_has_previous() {
    SpacetimeManifold m;
    STREAM_CHECK(!m.has_previous());
    (void)m.update(1.0, 1.0);
    STREAM_CHECK(m.has_previous());
}

static void test_manifold_static_interval() {
    STREAM_CHECK_NEAR(SpacetimeManifold::interval(3.0, 4.0), 9.0 - 16.0, 1e-12);
    STREAM_CHECK_NEAR(SpacetimeManifold::interval(5.0, 3.0), 25.0 - 9.0, 1e-12);
}

static void test_manifold_classify_all_regimes() {
    STREAM_CHECK(SpacetimeManifold::classify( 1.0, 1e-9) == Regime::TIMELIKE);
    STREAM_CHECK(SpacetimeManifold::classify(-1.0, 1e-9) == Regime::SPACELIKE);
    STREAM_CHECK(SpacetimeManifold::classify( 0.0, 1e-9) == Regime::LIGHTLIKE);
}

static void test_manifold_signal_always_finite() {
    SpacetimeManifold m;
    std::mt19937 rng(0xABCD);
    std::uniform_real_distribution<double> d(-10.0, 10.0);
    for (int i = 0; i < 100; ++i) {
        auto r = m.update(d(rng), d(rng));
        STREAM_CHECK(std::isfinite(r.signal));
    }
}

static void test_manifold_all_regimes_reachable() {
    SpacetimeManifold m;
    bool tl = false, sl = false, ll = false;
    // first → LIGHTLIKE
    if (m.update(0.0,0.0).regime == Regime::LIGHTLIKE) ll = true;
    if (m.update(5.0,1.0).regime == Regime::TIMELIKE)  tl = true;
    if (m.update(6.0,11.0).regime == Regime::SPACELIKE) sl = true;
    STREAM_CHECK(tl); STREAM_CHECK(sl); STREAM_CHECK(ll);
}

// ---------------------------------------------------------------------------
// Full 4-Stage Pipeline
// ---------------------------------------------------------------------------

static void test_pipeline_signal_always_finite() {
    Pipeline4Stage p;
    for (double i = 1.0; i <= 100.0; ++i) {
        auto r = p.step(100.0 + i * 0.5);
        STREAM_CHECK(std::isfinite(r.signal));
    }
}

static void test_pipeline_gamma_always_ge_1() {
    Pipeline4Stage p;
    for (double i = 1.0; i <= 100.0; ++i) {
        auto r = p.step(100.0 + i * 0.3);
        STREAM_CHECK_GE(r.gamma_val, 1.0);
    }
}

static void test_pipeline_beta_in_bounds() {
    Pipeline4Stage p;
    std::mt19937 rng(0x1234);
    std::uniform_real_distribution<double> dist(1.0, 500.0);
    for (int i = 0; i < 200; ++i) {
        auto r = p.step(dist(rng));
        STREAM_CHECK(std::abs(r.beta_val) <= BETA_MAX_SAFE);
    }
}

static void test_pipeline_pre_warmup_beta_zero() {
    Pipeline4Stage p;
    for (int i = 0; i < 3; ++i) {
        auto r = p.step(100.0 + i);
        STREAM_CHECK_NEAR(r.beta_val, 0.0, 1e-12);
        STREAM_CHECK_NEAR(r.gamma_val, 1.0, 1e-12);
    }
}

static void test_pipeline_reset_restarts() {
    Pipeline4Stage p;
    for (int i = 0; i < 50; ++i) p.step(100.0 + i);
    p.reset();
    auto r = p.step(100.0);
    STREAM_CHECK_NEAR(r.beta_val, 0.0, 1e-12);
    STREAM_CHECK(!p.normalizer.warmed_up());
}

static void test_pipeline_first_event_lightlike() {
    Pipeline4Stage p;
    auto r = p.step(100.0);
    STREAM_CHECK(r.regime == Regime::LIGHTLIKE);
}

static void test_pipeline_deterministic() {
    Pipeline4Stage p1, p2;
    std::vector<double> closes = {100,102,101,103,104,102,105,103,106,108};
    std::vector<double> s1, s2;
    for (double c : closes) s1.push_back(p1.step(c).signal);
    for (double c : closes) s2.push_back(p2.step(c).signal);
    for (std::size_t i = 0; i < s1.size(); ++i)
        STREAM_CHECK_NEAR(s1[i], s2[i], 1e-12);
}

static void test_pipeline_bar_index_increments() {
    Pipeline4Stage p;
    for (std::size_t i = 0; i < 10; ++i) {
        STREAM_CHECK(p.bar == i);
        p.step(100.0 + i);
    }
}

static void test_pipeline_covers_regimes() {
    Pipeline4Stage p;
    std::mt19937 rng(0xCAFEBABE);
    std::uniform_real_distribution<double> dist(1.0, 500.0);
    bool ll = false;
    for (int i = 0; i < 200; ++i) {
        auto r = p.step(dist(rng));
        if (r.regime == Regime::LIGHTLIKE) ll = true;
    }
    STREAM_CHECK(ll);  // first tick is always LIGHTLIKE
}

static void test_pipeline_ds2_matches_formula() {
    Pipeline4Stage p;
    auto r0 = p.step(100.0);
    auto r1 = p.step(102.0);
    double dt = r1.t_prime - r0.t_prime;
    double dx = r1.x_prime - r0.x_prime;
    STREAM_CHECK_NEAR(r1.ds2, dt*dt - dx*dx, 1e-9);
}

static void test_pipeline_constant_price_finite() {
    Pipeline4Stage p;
    for (int i = 0; i < 30; ++i) {
        auto r = p.step(100.0);
        STREAM_CHECK(std::isfinite(r.signal));
    }
}

static void test_pipeline_two_instances_independent() {
    Pipeline4Stage a, b;
    std::vector<double> ca = {100,102,104,103,105,107,106,108,110,112};
    std::vector<double> cb = {200,198,196,197,195,193,194,192,190,188};
    std::vector<double> sa, sb;
    for (double c : ca) sa.push_back(a.step(c).signal);
    for (double c : cb) sb.push_back(b.step(c).signal);
    bool any = false;
    for (std::size_t i = 4; i < sa.size(); ++i)
        if (std::abs(sa[i] - sb[i]) > 1e-9) { any = true; break; }
    STREAM_CHECK(any);
}

// ---------------------------------------------------------------------------
// Extra normalizer
// ---------------------------------------------------------------------------

static void test_normalizer_window_size_2_warmsup_at_2() {
    // Minimum valid window = 2
    CoordinateNormalizer n(2);
    STREAM_CHECK(!n.warmed_up());
    n.update(10.0);
    STREAM_CHECK(!n.warmed_up());
    n.update(20.0);
    STREAM_CHECK(n.warmed_up());
    STREAM_CHECK(std::isfinite(n.normalise(15.0)));
}

static void test_normalizer_large_values() {
    CoordinateNormalizer n(20);
    for (double i = 1.0; i <= 20.0; ++i) n.update(i * 1e10);
    STREAM_CHECK(std::isfinite(n.mean()));
    STREAM_CHECK(std::isfinite(n.sigma()));
    STREAM_CHECK_GE(n.sigma(), 0.0);
}

static void test_normalizer_alternating_pos_neg() {
    CoordinateNormalizer n(4);
    n.update(1.0); n.update(-1.0); n.update(1.0); n.update(-1.0);
    STREAM_CHECK(n.warmed_up());
    STREAM_CHECK_NEAR(n.mean(), 0.0, 1e-9);
}

static void test_normalizer_z_at_minus_2sigma() {
    CoordinateNormalizer n(20);
    for (double i = 1.0; i <= 20.0; ++i) n.update(i);
    double mu = n.mean(), sig = n.sigma();
    STREAM_CHECK_NEAR(n.normalise(mu - 2.0 * sig), -2.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Extra lorentz
// ---------------------------------------------------------------------------

static void test_lorentz_nan_beta_identity() {
    LorentzTransform lt;
    double nan = std::numeric_limits<double>::quiet_NaN();
    auto ev = lt.transform(2.0, 3.0, nan);
    // Should fall back to identity
    STREAM_CHECK_NEAR(ev.t_prime, 2.0, 1e-12);
    STREAM_CHECK_NEAR(ev.x_prime, 3.0, 1e-12);
}

static void test_lorentz_x_zero_pure_time() {
    LorentzTransform lt;
    // x=0: x' = γ*(0 - β*t) = -γβt; t' = γ*t
    double beta = 0.6, gamma = 1.25, t = 4.0;
    auto ev = lt.transform(t, 0.0, beta);
    STREAM_CHECK_NEAR(ev.t_prime,  gamma * t,          1e-9);
    STREAM_CHECK_NEAR(ev.x_prime, -gamma * beta * t,   1e-9);
}

// ---------------------------------------------------------------------------
// Extra manifold
// ---------------------------------------------------------------------------

static void test_manifold_two_resets() {
    SpacetimeManifold m;
    (void)m.update(1.0, 2.0);
    m.reset();
    auto r1 = m.update(5.0, 1.0);  // TIMELIKE
    STREAM_CHECK(r1.regime == Regime::LIGHTLIKE);  // first after reset
    m.reset();
    auto r2 = m.update(3.0, 3.0);  // LIGHTLIKE again
    STREAM_CHECK(r2.regime == Regime::LIGHTLIKE);
}

static void test_manifold_large_coords() {
    SpacetimeManifold m;
    (void)m.update(0.0, 0.0);
    auto r = m.update(1e8, 1.0);
    STREAM_CHECK(std::isfinite(r.ds2));
    STREAM_CHECK(std::isfinite(r.signal));
}

// ---------------------------------------------------------------------------
// Extra pipeline
// ---------------------------------------------------------------------------

static void test_beta_clamped_neg() {
    BetaCalculatorFix3 bc;
    bc.update(1e9); bc.update(1.0); bc.update(1e9); bc.update(1.0);
    STREAM_CHECK(std::abs(bc.beta()) <= BETA_MAX_SAFE);
}

static void test_lorentz_beta_near_1() {
    LorentzTransform lt;
    auto ev = lt.transform(1.0, 0.5, 0.9999);
    STREAM_CHECK(std::isfinite(ev.t_prime));
    STREAM_CHECK(std::isfinite(ev.x_prime));
    STREAM_CHECK_GE(ev.gamma, 1.0);
}

static void test_beta_inf_close_ignored() {
    BetaCalculatorFix3 bc;
    bc.update(100.0);
    bc.update(std::numeric_limits<double>::infinity());
    STREAM_CHECK(!bc.warmed_up());
    STREAM_CHECK_NEAR(bc.beta(), 0.0, 1e-12);
}

static void test_beta_rising_20_ticks() {
    BetaCalculatorFix3 bc;
    for (int i = 0; i < 20; ++i) bc.update(50.0 + i * 5.0);
    STREAM_CHECK(bc.warmed_up());
    STREAM_CHECK_GT(bc.beta(), 0.0);
    STREAM_CHECK(std::abs(bc.beta()) <= BETA_MAX_SAFE);
}

static void test_normalizer_pre_warmup_mean() {
    CoordinateNormalizer n(20);
    // Before any update, count=0
    STREAM_CHECK(n.count() == std::size_t{0});
    STREAM_CHECK(!n.warmed_up());
    // normalise before warmup should return finite (0 or identity)
    STREAM_CHECK(std::isfinite(n.normalise(100.0)));
}

static void test_pipeline_spike_finite() {
    Pipeline4Stage p;
    for (int i = 0; i < 20; ++i) p.step(100.0 + i * 0.01);
    auto r = p.step(1e9);
    STREAM_CHECK(std::isfinite(r.signal));
    STREAM_CHECK(std::abs(r.beta_val) <= BETA_MAX_SAFE);
}

static void test_pipeline_200_ticks() {
    Pipeline4Stage p;
    std::mt19937 rng(0x9876);
    std::uniform_real_distribution<double> dist(10.0, 200.0);
    for (int i = 0; i < 200; ++i) {
        auto r = p.step(dist(rng));
        STREAM_CHECK(std::isfinite(r.signal));
        STREAM_CHECK(std::isfinite(r.ds2));
        STREAM_CHECK_GE(r.gamma_val, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Regime helpers
// ---------------------------------------------------------------------------

static void test_regime_strings() {
    STREAM_CHECK(std::strcmp(regime_to_str(Regime::TIMELIKE),  "TIMELIKE")  == 0);
    STREAM_CHECK(std::strcmp(regime_to_str(Regime::SPACELIKE), "SPACELIKE") == 0);
    STREAM_CHECK(std::strcmp(regime_to_str(Regime::LIGHTLIKE), "LIGHTLIKE") == 0);
}

static void test_regime_values_distinct() {
    STREAM_CHECK(static_cast<int>(Regime::TIMELIKE)  != static_cast<int>(Regime::SPACELIKE));
    STREAM_CHECK(static_cast<int>(Regime::SPACELIKE) != static_cast<int>(Regime::LIGHTLIKE));
    STREAM_CHECK(static_cast<int>(Regime::TIMELIKE)  != static_cast<int>(Regime::LIGHTLIKE));
}

// ---------------------------------------------------------------------------
// OHLCVTick — supplemental
// ---------------------------------------------------------------------------

static void test_tick_valid_standard() {
    OHLCVTick t;
    t.open=100.0; t.high=110.0; t.low=90.0; t.close=105.0;
    t.volume=1000.0; t.timestamp_ns=1;
    STREAM_CHECK(tick_is_valid(t));
}

static void test_tick_high_lt_low_invalid() {
    OHLCVTick t;
    t.open=100.0; t.high=80.0; t.low=90.0; t.close=100.0;
    t.volume=100.0; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_close_above_high_invalid() {
    OHLCVTick t;
    t.open=100.0; t.high=110.0; t.low=90.0; t.close=120.0;
    t.volume=100.0; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_close_below_low_invalid() {
    OHLCVTick t;
    t.open=100.0; t.high=110.0; t.low=90.0; t.close=80.0;
    t.volume=100.0; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_nan_volume_invalid() {
    OHLCVTick t;
    t.open=100.0; t.high=110.0; t.low=90.0; t.close=100.0;
    t.volume=std::numeric_limits<double>::quiet_NaN(); t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_inf_close_invalid() {
    OHLCVTick t;
    t.open=100.0; t.high=110.0; t.low=90.0;
    t.close=std::numeric_limits<double>::infinity();
    t.volume=100.0; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_zero_ts_invalid() {
    OHLCVTick t;
    t.open=100.0; t.high=110.0; t.low=90.0; t.close=100.0;
    t.volume=100.0; t.timestamp_ns=0;
    STREAM_CHECK(!tick_is_valid(t));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::printf("SRFM Stream — Signal Chain Tests\n");
    std::printf("=================================\n");

    // CoordinateNormalizer
    STREAM_SUITE("normalizer window=2 basic",        test_normalizer_window_size_2_basic);
    STREAM_SUITE("normalizer window=50",             test_normalizer_window_size_50);
    STREAM_SUITE("normalizer evicts oldest",         test_normalizer_update_beyond_window);
    STREAM_SUITE("normalizer z at mean = 0",         test_normalizer_normalise_at_mean_is_zero);
    STREAM_SUITE("normalizer symmetric z",           test_normalizer_z_score_symmetric);
    STREAM_SUITE("normalizer monotone z",            test_normalizer_monotone_z_scores);
    STREAM_SUITE("normalizer two instances",         test_normalizer_two_instances_scale_independently);
    STREAM_SUITE("normalizer reset clears warmup",   test_normalizer_reset_clears_warmup);
    STREAM_SUITE("normalizer rewarms after reset",   test_normalizer_after_reset_rewarms);
    STREAM_SUITE("normalizer sigma nonneg",          test_normalizer_sigma_always_nonneg);
    STREAM_SUITE("normalizer welford vs naive",      test_normalizer_welford_vs_naive);
    STREAM_SUITE("normalizer count accessor",        test_normalizer_count_accessor);
    STREAM_SUITE("normalizer count saturates",       test_normalizer_count_saturates_at_window);
    STREAM_SUITE("normalizer HF stable",             test_normalizer_high_frequency_stable);

    // BetaCalculator
    STREAM_SUITE("beta warmup n+1",                  test_beta_warmup_exactly_n_plus_1);
    STREAM_SUITE("beta zero before warmup",          test_beta_zero_before_warmup);
    STREAM_SUITE("beta clamped to max safe",         test_beta_clamped_to_max_safe);
    STREAM_SUITE("beta always finite",               test_beta_always_finite);
    STREAM_SUITE("beta flat series = 0",             test_beta_flat_series_near_zero);
    STREAM_SUITE("beta rising series positive",      test_beta_rising_series_positive);
    STREAM_SUITE("beta fix3 manual check",           test_beta_fix3_manual_check);
    STREAM_SUITE("beta reset returns 0",             test_beta_reset_returns_zero);
    STREAM_SUITE("beta nan ignored",                 test_beta_invalid_close_nan_ignored);
    STREAM_SUITE("beta neg close ignored",           test_beta_invalid_close_neg_ignored);
    STREAM_SUITE("beta two instances independent",   test_beta_two_instances_no_crosstalk);

    // LorentzTransform
    STREAM_SUITE("lorentz beta=0 identity",          test_lorentz_beta_zero_identity);
    STREAM_SUITE("lorentz gamma monotone",           test_lorentz_gamma_monotone);
    STREAM_SUITE("lorentz gamma >= 1",               test_lorentz_gamma_always_ge_1);
    STREAM_SUITE("lorentz gamma at beta=0.6",        test_lorentz_known_beta_0_6);
    STREAM_SUITE("lorentz gamma at beta=0.8",        test_lorentz_known_beta_0_8);
    STREAM_SUITE("lorentz invalid beta identity",    test_lorentz_invalid_beta_identity);
    STREAM_SUITE("lorentz neg beta negates x",       test_lorentz_negative_beta_negates_x);
    STREAM_SUITE("lorentz inverse round-trip",       test_lorentz_inverse_round_trip);
    STREAM_SUITE("lorentz result always finite",     test_lorentz_result_always_finite);

    // SpacetimeManifold
    STREAM_SUITE("manifold first = LIGHTLIKE",       test_manifold_first_update_lightlike);
    STREAM_SUITE("manifold TIMELIKE",                test_manifold_timelike);
    STREAM_SUITE("manifold SPACELIKE",               test_manifold_spacelike);
    STREAM_SUITE("manifold LIGHTLIKE null",          test_manifold_lightlike_null);
    STREAM_SUITE("manifold reset",                   test_manifold_reset_clears_state);
    STREAM_SUITE("manifold has_previous",            test_manifold_has_previous);
    STREAM_SUITE("manifold static interval",         test_manifold_static_interval);
    STREAM_SUITE("manifold classify all",            test_manifold_classify_all_regimes);
    STREAM_SUITE("manifold signal finite",           test_manifold_signal_always_finite);
    STREAM_SUITE("manifold all regimes",             test_manifold_all_regimes_reachable);

    // Pipeline integration
    STREAM_SUITE("pipeline signal finite",           test_pipeline_signal_always_finite);
    STREAM_SUITE("pipeline gamma >= 1",              test_pipeline_gamma_always_ge_1);
    STREAM_SUITE("pipeline beta in bounds",          test_pipeline_beta_in_bounds);
    STREAM_SUITE("pipeline pre-warmup beta=0",       test_pipeline_pre_warmup_beta_zero);
    STREAM_SUITE("pipeline reset restarts",          test_pipeline_reset_restarts);
    STREAM_SUITE("pipeline first LIGHTLIKE",         test_pipeline_first_event_lightlike);
    STREAM_SUITE("pipeline deterministic",           test_pipeline_deterministic);
    STREAM_SUITE("pipeline bar increments",          test_pipeline_bar_index_increments);
    STREAM_SUITE("pipeline covers regimes",          test_pipeline_covers_regimes);
    STREAM_SUITE("pipeline ds2 formula",             test_pipeline_ds2_matches_formula);
    STREAM_SUITE("pipeline constant price finite",   test_pipeline_constant_price_finite);
    STREAM_SUITE("pipeline two instances",           test_pipeline_two_instances_independent);

    // Helpers
    STREAM_SUITE("regime strings",                   test_regime_strings);
    STREAM_SUITE("regime values distinct",           test_regime_values_distinct);
    STREAM_SUITE("tick valid standard",              test_tick_valid_standard);
    STREAM_SUITE("tick high<low invalid",            test_tick_high_lt_low_invalid);
    STREAM_SUITE("tick close>high invalid",          test_tick_close_above_high_invalid);
    STREAM_SUITE("tick close<low invalid",           test_tick_close_below_low_invalid);
    STREAM_SUITE("tick nan volume invalid",          test_tick_nan_volume_invalid);
    STREAM_SUITE("tick inf close invalid",           test_tick_inf_close_invalid);
    STREAM_SUITE("tick zero ts invalid",             test_tick_zero_ts_invalid);

    // Extra normalizer suites
    STREAM_SUITE("normalizer window=2 warms at 2",   test_normalizer_window_size_2_warmsup_at_2);
    STREAM_SUITE("normalizer large values stable",   test_normalizer_large_values);
    STREAM_SUITE("normalizer alternating warm",      test_normalizer_alternating_pos_neg);
    STREAM_SUITE("normalizer z at -2sigma",          test_normalizer_z_at_minus_2sigma);

    // Extra lorentz suites
    STREAM_SUITE("lorentz nan beta identity",        test_lorentz_nan_beta_identity);
    STREAM_SUITE("lorentz x=0 zero x_prime",        test_lorentz_x_zero_pure_time);

    // Extra manifold suites
    STREAM_SUITE("manifold two resets",              test_manifold_two_resets);
    STREAM_SUITE("manifold large coords finite",     test_manifold_large_coords);

    // Extra pipeline
    STREAM_SUITE("pipeline large spike finite",      test_pipeline_spike_finite);
    STREAM_SUITE("pipeline 200 ticks all finite",    test_pipeline_200_ticks);

    // Extra beta
    STREAM_SUITE("beta inf close ignored",           test_beta_inf_close_ignored);
    STREAM_SUITE("beta rising 20 ticks positive",    test_beta_rising_20_ticks);
    STREAM_SUITE("normalizer pre-warmup mean zero",  test_normalizer_pre_warmup_mean);
    STREAM_SUITE("beta clamped neg direction",       test_beta_clamped_neg);
    STREAM_SUITE("lorentz beta near 1 valid",        test_lorentz_beta_near_1);

    return stream_test::report();
}
