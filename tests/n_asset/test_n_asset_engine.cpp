/**
 * @file  test_n_asset_engine.cpp
 * @brief Tests for NAssetEngine — Stage 4 N-Asset Manifold.
 *
 * Coverage:
 *   - ready() false before lookback filled
 *   - ready() true after lookback filled
 *   - ingest() wrong bar count → nullopt
 *   - process() before ready → nullopt
 *   - output size matches universe
 *   - relativistic momentum is positive
 *   - gamma >= 1 always
 *   - regime classification present
 *   - high-beta asset has larger gamma
 *   - uncorrelated assets have independent momenta
 *   - ingest_and_process() consistent with separate calls
 *   - N=1 single asset
 *   - N=10 ten assets
 *   - portfolio regime set
 *   - high volume increases m_eff
 *   - zero price change → beta=0, gamma=1
 *   - parameter sweeps
 */

#include "srfm_test_n_asset.hpp"
#include "../../include/srfm/engine/n_asset_engine.hpp"

#include <cmath>
#include <vector>

using namespace srfm::engine;

// ── Helpers ───────────────────────────────────────────────────────────────────

static OHLCVBar make_bar(double price,
                          double volume    = 1'000'000.0,
                          double timestamp = 1.0) {
    return OHLCVBar{price, price * 1.02, price * 0.98,
                    price, volume, timestamp};
}

/// Generate `n_bars` sequential bars for N assets, each bar's price = base + bar_idx * delta.
static void feed_engine(NAssetEngine& engine,
                         int            n_assets,
                         int            n_bars,
                         double         base_price = 100.0,
                         double         delta      = 0.5,
                         double         volume     = 1'000'000.0) {
    for (int t = 0; t < n_bars; ++t) {
        std::vector<OHLCVBar> bars;
        bars.reserve(static_cast<std::size_t>(n_assets));
        for (int i = 0; i < n_assets; ++i) {
            double p = base_price + delta * t + 0.1 * i;
            bars.push_back(make_bar(p, volume, static_cast<double>(t + 1)));
        }
        (void)engine.ingest(bars);
    }
}

static AssetUniverse make_universe(int n) {
    AssetUniverse u;
    for (int i = 0; i < n; ++i) {
        u.names.push_back("ASSET" + std::to_string(i));
    }
    return u;
}

// ── Ready state tests ─────────────────────────────────────────────────────────

static void test_engine_not_ready_before_lookback() {
    EngineConfig cfg;
    cfg.lookback_bars = 20;
    NAssetEngine engine(make_universe(2), cfg);
    SRFM_CHECK(engine.ready() == false);

    // Feed 19 bars — still not ready.
    feed_engine(engine, 2, 19);
    SRFM_CHECK(engine.ready() == false);
}

static void test_engine_ready_after_lookback_bars() {
    EngineConfig cfg;
    cfg.lookback_bars = 10;
    NAssetEngine engine(make_universe(2), cfg);
    feed_engine(engine, 2, 10);
    SRFM_CHECK(engine.ready() == true);
}

static void test_engine_ready_after_more_than_lookback() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    NAssetEngine engine(make_universe(2), cfg);
    feed_engine(engine, 2, 30);
    SRFM_CHECK(engine.ready() == true);
}

static void test_engine_not_ready_zero_bars() {
    NAssetEngine engine(make_universe(3));
    SRFM_CHECK(engine.ready() == false);
    SRFM_CHECK(engine.process() == std::nullopt);
}

// ── Ingest validation ─────────────────────────────────────────────────────────

static void test_engine_ingest_wrong_bar_count_returns_nullopt() {
    NAssetEngine engine(make_universe(3));
    std::vector<OHLCVBar> bars = {make_bar(100.0), make_bar(200.0)}; // only 2, expect 3
    auto r = engine.ingest(bars);
    SRFM_NO_VALUE(r);
}

static void test_engine_ingest_correct_count_returns_value() {
    NAssetEngine engine(make_universe(2));
    std::vector<OHLCVBar> bars = {make_bar(100.0), make_bar(200.0)};
    auto r = engine.ingest(bars);
    SRFM_HAS_VALUE(r);
}

static void test_engine_ingest_single_asset() {
    NAssetEngine engine(make_universe(1));
    std::vector<OHLCVBar> bars = {make_bar(100.0)};
    auto r = engine.ingest(bars);
    SRFM_HAS_VALUE(r);
}

// ── Process before ready ──────────────────────────────────────────────────────

static void test_engine_process_before_ready_returns_nullopt() {
    EngineConfig cfg;
    cfg.lookback_bars = 10;
    NAssetEngine engine(make_universe(2), cfg);
    feed_engine(engine, 2, 5); // only 5 < 10
    auto r = engine.process();
    SRFM_NO_VALUE(r);
}

// ── Output structure ──────────────────────────────────────────────────────────

static void test_engine_output_size_matches_universe_n2() {
    EngineConfig cfg;
    cfg.lookback_bars = 10;
    NAssetEngine engine(make_universe(2), cfg);
    feed_engine(engine, 2, 12);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(static_cast<int>(r->assets.size()) == 2);
}

static void test_engine_output_size_matches_universe_n5() {
    EngineConfig cfg;
    cfg.lookback_bars = 10;
    NAssetEngine engine(make_universe(5), cfg);
    feed_engine(engine, 5, 12);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(static_cast<int>(r->assets.size()) == 5);
}

static void test_engine_output_size_matches_universe_n10() {
    EngineConfig cfg;
    cfg.lookback_bars = 10;
    NAssetEngine engine(make_universe(10), cfg);
    feed_engine(engine, 10, 12);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(static_cast<int>(r->assets.size()) == 10);
}

static void test_engine_asset_names_match_universe() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    NAssetEngine engine(make_universe(3), cfg);
    feed_engine(engine, 3, 7);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->assets[0].asset_name == "ASSET0");
    SRFM_CHECK(r->assets[1].asset_name == "ASSET1");
    SRFM_CHECK(r->assets[2].asset_name == "ASSET2");
}

// ── Relativistic momentum properties ─────────────────────────────────────────

static void test_engine_relativistic_momentum_positive_n1() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    NAssetEngine engine(make_universe(1), cfg);
    feed_engine(engine, 1, 7);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->assets[0].relativistic_momentum > 0.0);
}

static void test_engine_relativistic_momentum_positive_n4() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    NAssetEngine engine(make_universe(4), cfg);
    feed_engine(engine, 4, 7);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    for (const auto& a : r->assets) {
        SRFM_CHECK(a.relativistic_momentum > 0.0);
    }
}

static void test_engine_gamma_at_least_one_n1() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    NAssetEngine engine(make_universe(1), cfg);
    feed_engine(engine, 1, 7);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->assets[0].gamma >= 1.0);
}

static void test_engine_gamma_at_least_one_all_assets() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    NAssetEngine engine(make_universe(6), cfg);
    feed_engine(engine, 6, 8);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    for (const auto& a : r->assets) {
        SRFM_CHECK(a.gamma >= 1.0);
    }
}

static void test_engine_beta_in_range() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    NAssetEngine engine(make_universe(3), cfg);
    feed_engine(engine, 3, 7);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    for (const auto& a : r->assets) {
        SRFM_CHECK(a.beta >= 0.0);
        SRFM_CHECK(a.beta < ENGINE_BETA_MAX_SAFE);
    }
}

// ── Regime classification ─────────────────────────────────────────────────────

static void test_engine_regime_classification_present_n1() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    NAssetEngine engine(make_universe(1), cfg);
    feed_engine(engine, 1, 7);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    // Regime is one of the three valid values (any is OK).
    bool valid = (r->assets[0].regime == IntervalType::TIMELIKE  ||
                  r->assets[0].regime == IntervalType::SPACELIKE ||
                  r->assets[0].regime == IntervalType::LIGHTLIKE);
    SRFM_CHECK(valid);
}

static void test_engine_portfolio_regime_set_n2() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    NAssetEngine engine(make_universe(2), cfg);
    feed_engine(engine, 2, 7);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    bool valid = (r->portfolio_regime == IntervalType::TIMELIKE  ||
                  r->portfolio_regime == IntervalType::SPACELIKE ||
                  r->portfolio_regime == IntervalType::LIGHTLIKE);
    SRFM_CHECK(valid);
}

// ── High beta → larger gamma ──────────────────────────────────────────────────

static void test_engine_high_beta_asset_larger_gamma() {
    EngineConfig cfg;
    cfg.lookback_bars   = 5;
    cfg.c_market        = 1.0;
    cfg.adv_baseline    = 1'000'000.0;

    NAssetEngine engine(make_universe(2), cfg);

    // Asset 0: tiny price movement (low beta).
    // Asset 1: large price movement (high beta).
    for (int t = 0; t < 7; ++t) {
        std::vector<OHLCVBar> bars;
        double p0 = 100.0 + 0.001 * t;  // tiny move
        double p1 = 100.0 + 5.0 * t;    // large move
        bars.push_back(make_bar(p0, 1'000'000.0, static_cast<double>(t + 1)));
        bars.push_back(make_bar(p1, 1'000'000.0, static_cast<double>(t + 1)));
        (void)engine.ingest(bars);
    }

    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(r->assets[1].beta  > r->assets[0].beta);
    SRFM_CHECK(r->assets[1].gamma > r->assets[0].gamma);
}

// ── Zero price change → beta=0, gamma=1 ──────────────────────────────────────

static void test_engine_zero_price_change_newtonian_gamma() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    cfg.c_market      = 1.0;

    NAssetEngine engine(make_universe(1), cfg);

    // All bars have identical price → zero log-returns and zero beta.
    for (int t = 0; t < 8; ++t) {
        std::vector<OHLCVBar> bars = {make_bar(100.0, 1'000'000.0,
                                                static_cast<double>(t + 1))};
        (void)engine.ingest(bars);
    }

    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    SRFM_CHECK_NEAR(r->assets[0].beta,  0.0, 1e-9);
    SRFM_CHECK_NEAR(r->assets[0].gamma, 1.0, 1e-9);
}

// ── High volume increases m_eff ───────────────────────────────────────────────

static void test_engine_high_volume_increases_m_eff() {
    EngineConfig cfg1, cfg2;
    cfg1.lookback_bars = 5; cfg1.adv_baseline = 1'000'000.0;
    cfg2.lookback_bars = 5; cfg2.adv_baseline = 1'000'000.0;

    NAssetEngine e1(make_universe(1), cfg1);
    NAssetEngine e2(make_universe(1), cfg2);

    double low_vol  = 500'000.0;
    double high_vol = 5'000'000.0;

    for (int t = 0; t < 7; ++t) {
        double p = 100.0 + 0.1 * t;
        std::vector<OHLCVBar> b1 = {make_bar(p, low_vol,  static_cast<double>(t + 1))};
        std::vector<OHLCVBar> b2 = {make_bar(p, high_vol, static_cast<double>(t + 1))};
        (void)e1.ingest(b1);
        (void)e2.ingest(b2);
    }

    auto r1 = e1.process();
    auto r2 = e2.process();
    SRFM_HAS_VALUE(r1);
    SRFM_HAS_VALUE(r2);
    SRFM_CHECK(r2->assets[0].m_eff > r1->assets[0].m_eff);
}

// ── ingest_and_process consistency ───────────────────────────────────────────

static void test_engine_ingest_and_process_consistent_n2() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;

    NAssetEngine e1(make_universe(2), cfg);
    NAssetEngine e2(make_universe(2), cfg);

    // Pre-fill both with 5 bars.
    feed_engine(e1, 2, 5);
    feed_engine(e2, 2, 5);

    // 6th bar: use ingest_and_process for e2, ingest+process for e1.
    std::vector<OHLCVBar> bars = {make_bar(105.0, 1'000'000.0, 6.0),
                                   make_bar(205.0, 2'000'000.0, 6.0)};

    (void)e1.ingest(bars);
    auto r1 = e1.process();
    auto r2 = e2.ingest_and_process(bars);

    SRFM_HAS_VALUE(r1);
    SRFM_HAS_VALUE(r2);
    SRFM_CHECK(static_cast<int>(r1->assets.size()) == static_cast<int>(r2->assets.size()));
    SRFM_CHECK_NEAR(r1->assets[0].gamma, r2->assets[0].gamma, 1e-12);
    SRFM_CHECK_NEAR(r1->assets[1].gamma, r2->assets[1].gamma, 1e-12);
}

// ── N=1 single asset ─────────────────────────────────────────────────────────

static void test_engine_n1_single_asset() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    NAssetEngine engine(make_universe(1), cfg);
    feed_engine(engine, 1, 7);
    SRFM_CHECK(engine.n_assets() == 1);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(static_cast<int>(r->assets.size()) == 1);
    SRFM_CHECK(r->assets[0].gamma >= 1.0);
    SRFM_CHECK(r->assets[0].relativistic_momentum > 0.0);
}

// ── N=10 ten assets ───────────────────────────────────────────────────────────

static void test_engine_n10_ten_assets() {
    EngineConfig cfg;
    cfg.lookback_bars = 10;
    NAssetEngine engine(make_universe(10), cfg);
    feed_engine(engine, 10, 15);
    SRFM_CHECK(engine.n_assets() == 10);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(static_cast<int>(r->assets.size()) == 10);
    for (const auto& a : r->assets) {
        SRFM_CHECK(a.gamma >= 1.0);
        SRFM_CHECK(a.relativistic_momentum > 0.0);
        SRFM_CHECK(a.beta >= 0.0);
    }
}

// ── m_eff proportional to volume ─────────────────────────────────────────────

static void test_engine_m_eff_proportional_to_volume() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    cfg.adv_baseline  = 1'000'000.0;
    NAssetEngine engine(make_universe(1), cfg);

    double vol = 2'000'000.0;
    for (int t = 0; t < 7; ++t) {
        double p = 100.0 + 0.1 * t;
        std::vector<OHLCVBar> b = {make_bar(p, vol, static_cast<double>(t + 1))};
        (void)engine.ingest(b);
    }
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    // m_eff = volume / adv_baseline = 2.0
    SRFM_CHECK_NEAR(r->assets[0].m_eff, 2.0, 1e-9);
}

// ── n_assets() accessor ───────────────────────────────────────────────────────

static void test_engine_n_assets_accessor() {
    for (int n = 1; n <= 5; ++n) {
        NAssetEngine engine(make_universe(n));
        SRFM_CHECK(engine.n_assets() == n);
    }
}

// ── timestamp in output ───────────────────────────────────────────────────────

static void test_engine_output_timestamp_matches_latest_bar() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    NAssetEngine engine(make_universe(1), cfg);
    for (int t = 0; t < 7; ++t) {
        std::vector<OHLCVBar> b = {make_bar(100.0 + t, 1'000'000.0,
                                             static_cast<double>(t + 1))};
        (void)engine.ingest(b);
    }
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    SRFM_CHECK_NEAR(r->timestamp, 7.0, 1e-9);
}

// ── portfolio_interval_sq finite ──────────────────────────────────────────────

static void test_engine_portfolio_interval_sq_finite() {
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    NAssetEngine engine(make_universe(2), cfg);
    feed_engine(engine, 2, 8);
    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    SRFM_CHECK(std::isfinite(r->portfolio_interval_sq));
}

// ── ingest_and_process before ready returns nullopt ───────────────────────────

static void test_engine_ingest_and_process_not_ready_nullopt() {
    EngineConfig cfg;
    cfg.lookback_bars = 10;
    NAssetEngine engine(make_universe(2), cfg);
    // Only 2 bars → not ready.
    for (int t = 0; t < 2; ++t) {
        std::vector<OHLCVBar> bars = {make_bar(100.0, 1e6, static_cast<double>(t + 1)),
                                       make_bar(200.0, 1e6, static_cast<double>(t + 1))};
        auto r = engine.ingest_and_process(bars);
        SRFM_NO_VALUE(r);
    }
}

// ── N sweep over 1..8 ────────────────────────────────────────────────────────

static void test_engine_n_sweep_output_sizes() {
    for (int n = 1; n <= 8; ++n) {
        EngineConfig cfg;
        cfg.lookback_bars = 5;
        NAssetEngine engine(make_universe(n), cfg);
        feed_engine(engine, n, 7);
        auto r = engine.process();
        SRFM_HAS_VALUE(r);
        SRFM_CHECK(static_cast<int>(r->assets.size()) == n);
    }
}

static void test_engine_n_sweep_momentum_positive() {
    for (int n = 1; n <= 6; ++n) {
        EngineConfig cfg;
        cfg.lookback_bars = 5;
        NAssetEngine engine(make_universe(n), cfg);
        feed_engine(engine, n, 7);
        auto r = engine.process();
        SRFM_HAS_VALUE(r);
        for (const auto& a : r->assets) {
            SRFM_CHECK(a.relativistic_momentum > 0.0);
        }
    }
}

// ── uncorrelated assets have independent momenta ──────────────────────────────

static void test_engine_uncorrelated_assets_independent_momenta() {
    // With uncorrelated assets (different price trajectories), each asset's
    // relativistic momentum should depend only on its own price history.
    EngineConfig cfg;
    cfg.lookback_bars = 5;
    cfg.adv_baseline  = 1'000'000.0;
    cfg.c_market      = 1.0;
    NAssetEngine engine(make_universe(2), cfg);

    // Asset 0: slow growth; Asset 1: fast growth.
    for (int t = 0; t < 7; ++t) {
        double p0 = 100.0 + 0.01 * t;
        double p1 = 100.0 + 2.0  * t;
        std::vector<OHLCVBar> bars = {
            make_bar(p0, 1'000'000.0, static_cast<double>(t + 1)),
            make_bar(p1, 1'000'000.0, static_cast<double>(t + 1))
        };
        (void)engine.ingest(bars);
    }

    auto r = engine.process();
    SRFM_HAS_VALUE(r);
    // Asset 1 (fast growth) should have higher beta and gamma.
    SRFM_CHECK(r->assets[1].beta  > r->assets[0].beta);
    SRFM_CHECK(r->assets[1].gamma > r->assets[0].gamma);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    SRFM_SUITE("NAssetEngine Ready State",
        test_engine_not_ready_before_lookback,
        test_engine_ready_after_lookback_bars,
        test_engine_ready_after_more_than_lookback,
        test_engine_not_ready_zero_bars
    );
    SRFM_SUITE("NAssetEngine Ingest Validation",
        test_engine_ingest_wrong_bar_count_returns_nullopt,
        test_engine_ingest_correct_count_returns_value,
        test_engine_ingest_single_asset
    );
    SRFM_SUITE("NAssetEngine Process Before Ready",
        test_engine_process_before_ready_returns_nullopt,
        test_engine_ingest_and_process_not_ready_nullopt
    );
    SRFM_SUITE("NAssetEngine Output Structure",
        test_engine_output_size_matches_universe_n2,
        test_engine_output_size_matches_universe_n5,
        test_engine_output_size_matches_universe_n10,
        test_engine_asset_names_match_universe,
        test_engine_output_timestamp_matches_latest_bar,
        test_engine_portfolio_interval_sq_finite
    );
    SRFM_SUITE("NAssetEngine Relativistic Properties",
        test_engine_relativistic_momentum_positive_n1,
        test_engine_relativistic_momentum_positive_n4,
        test_engine_gamma_at_least_one_n1,
        test_engine_gamma_at_least_one_all_assets,
        test_engine_beta_in_range
    );
    SRFM_SUITE("NAssetEngine Regime Classification",
        test_engine_regime_classification_present_n1,
        test_engine_portfolio_regime_set_n2
    );
    SRFM_SUITE("NAssetEngine High Beta",
        test_engine_high_beta_asset_larger_gamma,
        test_engine_uncorrelated_assets_independent_momenta
    );
    SRFM_SUITE("NAssetEngine Zero Motion",
        test_engine_zero_price_change_newtonian_gamma
    );
    SRFM_SUITE("NAssetEngine Volume",
        test_engine_high_volume_increases_m_eff,
        test_engine_m_eff_proportional_to_volume
    );
    SRFM_SUITE("NAssetEngine Consistency",
        test_engine_ingest_and_process_consistent_n2
    );
    SRFM_SUITE("NAssetEngine Single and Ten Asset",
        test_engine_n1_single_asset,
        test_engine_n10_ten_assets
    );
    SRFM_SUITE("NAssetEngine Accessors",
        test_engine_n_assets_accessor
    );
    SRFM_SUITE("NAssetEngine N Sweep",
        test_engine_n_sweep_output_sizes,
        test_engine_n_sweep_momentum_positive
    );
    return srfm_test::report();
}
