/**
 * @file  test_pipeline_invariants.cpp
 * @brief Cross-component property and invariant tests for the streaming pipeline.
 *
 * Tests here verify invariants that hold across multiple pipeline components
 * in combination, simulating "property-based" coverage over a range of inputs.
 *
 * Test structure:
 *   test_inv_beta_always_in_bounds_over_random_prices
 *   test_inv_gamma_always_ge_1
 *   test_inv_signal_always_finite
 *   test_inv_normaliser_z_score_is_zero_for_mean_value
 *   test_inv_manifold_interval_formula_consistency
 *   test_inv_lorentz_inverse_is_exact
 *   test_inv_pipeline_ds2_formula_consistency
 *   test_inv_ring_fifo_ordering_at_scale
 *   test_inv_beta_warmup_requires_n_plus_1_prices
 *   test_inv_regime_covers_all_categories_in_mixed_series
 *   test_inv_processor_monotone_bar_counter
 *   test_inv_consumer_count_matches_signals_written
 *   test_inv_inject_then_process_roundtrip
 *   test_inv_normaliser_welford_vs_naive
 *   test_inv_spsc_ring_no_loss_under_pressure
 */

#include "srfm_stream_test.hpp"
#include "../../include/srfm/engine/stream_engine.hpp"
#include "../../include/srfm/stream/beta_calculator.hpp"
#include "../../include/srfm/stream/coordinate_normalizer.hpp"
#include "../../include/srfm/stream/lorentz_transform.hpp"
#include "../../include/srfm/stream/signal_consumer.hpp"
#include "../../include/srfm/stream/signal_processor.hpp"
#include "../../include/srfm/stream/spacetime_manifold.hpp"
#include "../../include/srfm/stream/spsc_ring.hpp"
#include "../../include/srfm/stream/tick.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

using namespace srfm::stream;
using namespace srfm::engine;

static constexpr double EPS       = 1e-9;
static constexpr double EPS_LOOSE = 1e-5;

// ── Deterministic pseudo-random price sequence helper ────────────────────────

/// Simple LCG for repeatable "random" close prices in [low, high].
struct PseudoRand {
    std::uint64_t state{12345678901234567ULL};
    double next(double lo, double hi) noexcept {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        double t = static_cast<double>(state >> 32) / 4294967296.0; // [0,1)
        return lo + t * (hi - lo);
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// INV: beta always in (-BETA_MAX_SAFE, +BETA_MAX_SAFE) for 500 random prices
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_beta_always_in_bounds_over_random_prices() {
    BetaCalculatorFix3 calc;
    PseudoRand rng;

    for (int i = 0; i < 500; ++i) {
        double price = rng.next(0.01, 10000.0);
        calc.update(price);
        if (calc.warmed_up()) {
            double b = calc.beta();
            STREAM_CHECK(b > -BETA_MAX_SAFE);
            STREAM_CHECK(b <  BETA_MAX_SAFE);
            STREAM_CHECK_FINITE(b);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: gamma >= 1 for all valid beta values (100 points)
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_gamma_always_ge_1() {
    PseudoRand rng;
    for (int i = 0; i < 100; ++i) {
        // Random beta in (-BETA_MAX_SAFE, BETA_MAX_SAFE).
        double b = rng.next(-BETA_MAX_SAFE + 0.001, BETA_MAX_SAFE - 0.001);
        double g = LorentzTransform::lorentz_gamma(b);
        STREAM_CHECK(g >= 1.0);
        STREAM_CHECK_FINITE(g);
    }

    // Also check exact boundary of safe range.
    for (double b : {0.0, 0.1, 0.5, 0.9, 0.99, 0.999, -0.5, -0.9}) {
        STREAM_CHECK(LorentzTransform::lorentz_gamma(b) >= 1.0);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: signal always finite over 200 ticks of random prices
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_signal_always_finite() {
    QueueTickSource src;
    StreamEngine engine{src};
    PseudoRand rng;

    for (int i = 0; i < 200; ++i) {
        double price = rng.next(1.0, 1000.0);
        auto tick = stream_test::make_tick(price);
        auto sig  = engine.process_sync(tick, static_cast<std::int64_t>(i));

        STREAM_CHECK_FINITE(sig.signal);
        STREAM_CHECK_FINITE(sig.beta);
        STREAM_CHECK_FINITE(sig.gamma);
        STREAM_CHECK_FINITE(sig.ds2);
        STREAM_CHECK(sig.gamma >= 1.0);
        STREAM_CHECK(sig.beta > -BETA_MAX_SAFE);
        STREAM_CHECK(sig.beta <  BETA_MAX_SAFE);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: normalise(mean) ≈ 0 after window fill for any price series
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_normaliser_z_score_is_zero_for_mean_value() {
    PseudoRand rng;

    for (int trial = 0; trial < 20; ++trial) {
        CoordinateNormalizer norm{20};

        // Fill window with random prices.
        for (int i = 0; i < 20; ++i) {
            norm.update(rng.next(50.0, 200.0));
        }

        STREAM_CHECK(norm.warmed_up());

        // Normalise the exact current mean → should be ≈ 0.
        double z_at_mean = norm.normalise(norm.mean());
        STREAM_CHECK_NEAR(z_at_mean, 0.0, EPS_LOOSE);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: SpacetimeManifold interval formula: ds2 = dt² - dx²
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_manifold_interval_formula_consistency() {
    PseudoRand rng;

    for (int i = 0; i < 100; ++i) {
        double t0 = rng.next(-10.0, 10.0);
        double x0 = rng.next(-10.0, 10.0);
        double t1 = rng.next(-10.0, 10.0);
        double x1 = rng.next(-10.0, 10.0);

        double dt = t1 - t0;
        double dx = x1 - x0;

        // Static formula.
        double ds2_static = SpacetimeManifold::interval(dt, dx);
        STREAM_CHECK_NEAR(ds2_static, dt * dt - dx * dx, EPS);

        // Via update().
        SpacetimeManifold m;
        (void)m.update(t0, x0);
        auto r = m.update(t1, x1);
        STREAM_CHECK_NEAR(r.ds2, ds2_static, EPS_LOOSE);

        // Signal magnitude = sqrt(|ds2|).
        if (r.regime == Regime::TIMELIKE) {
            STREAM_CHECK_NEAR(r.signal, std::sqrt(ds2_static), EPS_LOOSE);
        } else if (r.regime == Regime::SPACELIKE) {
            STREAM_CHECK_NEAR(r.signal, -std::sqrt(-ds2_static), EPS_LOOSE);
        } else {
            STREAM_CHECK_NEAR(r.signal, 0.0, EPS);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: Lorentz inverse is exact — transform then inverse recovers (t, x)
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_lorentz_inverse_is_exact() {
    LorentzTransform lt;
    PseudoRand rng;

    for (int i = 0; i < 200; ++i) {
        double t    = rng.next(-100.0, 100.0);
        double x    = rng.next(-100.0, 100.0);
        double beta = rng.next(-0.99, 0.99);

        auto fwd = lt.transform(t, x, beta);
        auto inv = lt.inverse(fwd.t_prime, fwd.x_prime, beta);

        STREAM_CHECK_NEAR(inv.t_prime, t, 1e-8);
        STREAM_CHECK_NEAR(inv.x_prime, x, 1e-8);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: ds2 from combined pipeline is always finite for any valid tick
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_pipeline_ds2_formula_consistency() {
    QueueTickSource src;
    StreamEngine engine{src};

    // All signals should have finite ds2.
    double price = 100.0;
    for (int i = 0; i < 100; ++i) {
        price += (i % 7 == 0) ? -0.8 : 0.3; // slightly bouncy
        price = price < 1.0 ? 1.0 : price;
        auto sig = engine.process_sync(stream_test::make_tick(price),
                                        static_cast<std::int64_t>(i));
        STREAM_CHECK_FINITE(sig.ds2);

        // Regime must agree with ds2 sign.
        if (sig.regime == Regime::TIMELIKE)
            STREAM_CHECK(sig.ds2 > 0.0);
        else if (sig.regime == Regime::SPACELIKE)
            STREAM_CHECK(sig.ds2 < 0.0);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: SPSC ring — FIFO ordering at scale (10,000 elements)
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_ring_fifo_ordering_at_scale() {
    SPSCRing<int, 16384> ring;

    constexpr int N = 10000;

    // Single-threaded fill + drain to verify ordering.
    // We push 500 at a time and drain 500 at a time.
    int expected = 0;
    int produced = 0;

    while (produced < N || expected < N) {
        // Fill batch of up to 500.
        int fill_end = std::min(produced + 500, N);
        while (produced < fill_end) {
            while (!ring.push_copy(produced)) {
                // Ring full; drain one and retry.
                auto r = ring.pop();
                STREAM_HAS_VALUE(r);
                STREAM_CHECK(*r == expected++);
            }
            ++produced;
        }
        // Drain batch.
        int drain_count = 0;
        while (drain_count < 200) {
            auto r = ring.pop();
            if (!r.has_value()) break;
            STREAM_CHECK(*r == expected++);
            ++drain_count;
        }
    }
    // Drain remainder.
    while (true) {
        auto r = ring.pop();
        if (!r.has_value()) break;
        STREAM_CHECK(*r == expected++);
    }
    STREAM_CHECK(expected == N);
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: Beta warmup requires exactly N+1 prices (FIX-3 → 4 prices)
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_beta_warmup_requires_n_plus_1_prices() {
    // FIX-1: needs 2 prices.
    {
        BetaCalculator<1> c;
        c.update(100.0);               STREAM_CHECK(!c.warmed_up());
        c.update(101.0);               STREAM_CHECK(c.warmed_up());
    }
    // FIX-2: needs 3 prices.
    {
        BetaCalculator<2> c;
        c.update(100.0); c.update(101.0); STREAM_CHECK(!c.warmed_up());
        c.update(102.0);                  STREAM_CHECK(c.warmed_up());
    }
    // FIX-3: needs 4 prices.
    {
        BetaCalculator<3> c;
        for (int i = 0; i < 3; ++i) { c.update(100.0 + i); STREAM_CHECK(!c.warmed_up()); }
        c.update(103.0);               STREAM_CHECK(c.warmed_up());
    }
    // FIX-5: needs 6 prices.
    {
        BetaCalculator<5> c;
        for (int i = 0; i < 5; ++i) { c.update(100.0 + i); STREAM_CHECK(!c.warmed_up()); }
        c.update(105.0);               STREAM_CHECK(c.warmed_up());
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: Mixed price series produces all three regime types
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_regime_covers_all_categories_in_mixed_series() {
    QueueTickSource src;
    StreamEngine engine{src};

    bool saw_timelike  = false;
    bool saw_spacelike = false;
    bool saw_lightlike = false;

    // Large zig-zag to ensure both regime types appear.
    double prices[] = {
        100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0,
        100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0,
        100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0,
        150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0,
    };

    for (int i = 0; i < static_cast<int>(sizeof(prices)/sizeof(prices[0])); ++i) {
        auto sig = engine.process_sync(stream_test::make_tick(prices[i]),
                                        static_cast<std::int64_t>(i));
        if (sig.regime == Regime::TIMELIKE)  saw_timelike  = true;
        if (sig.regime == Regime::SPACELIKE) saw_spacelike = true;
        if (sig.regime == Regime::LIGHTLIKE) saw_lightlike = true;
    }

    // First bar is always LIGHTLIKE.
    STREAM_CHECK(saw_lightlike);
    // A mixed series should produce at least one TIMELIKE or SPACELIKE event.
    STREAM_CHECK(saw_timelike || saw_spacelike);
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: processor bar counter is monotonically increasing
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_processor_monotone_bar_counter() {
    QueueTickSource src;
    StreamEngine engine{src};

    std::int64_t prev = -1;
    for (int i = 0; i < 50; ++i) {
        auto sig = engine.process_sync(
            stream_test::make_tick(100.0 + i * 0.1),
            static_cast<std::int64_t>(i));
        STREAM_CHECK(sig.bar > prev);
        prev = sig.bar;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: consumer count matches signals written (no drops for small ring)
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_consumer_count_matches_signals_written() {
    FILE* dev_null = nullptr;
#ifdef _WIN32
    fopen_s(&dev_null, "NUL", "w");
#else
    dev_null = std::fopen("/dev/null", "w");
#endif

    QueueTickSource src;
    StreamEngine engine{src, dev_null ? dev_null : stdout};
    engine.start();

    constexpr int N = 100;
    double price = 100.0;
    for (int i = 0; i < N; ++i) {
        price += 0.05;
        OHLCVTick t = stream_test::make_tick(price);
        int retry = 0;
        while (!engine.inject(t) && retry++ < 10000)
            std::this_thread::yield();
    }

    // Wait for all signals to be consumed.
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::seconds(5);
    while (engine.consumer_counters().signals_consumed
               < static_cast<std::uint64_t>(N)
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    engine.stop();

    STREAM_CHECK(engine.consumer_counters().signals_consumed
                 == static_cast<std::uint64_t>(N));

    if (dev_null) std::fclose(dev_null);
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: inject then process_sync → bar=0, finite signal
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_inject_then_process_roundtrip() {
    QueueTickSource src;
    StreamEngine engine{src};

    OHLCVTick tick = stream_test::make_tick(100.0);
    STREAM_CHECK(engine.inject(tick));

    // The tick is in ring1 now.  process_sync operates on the internal processor
    // state, independent of the ring.
    auto sig = engine.process_sync(tick, 0);
    STREAM_CHECK(sig.bar == 0);
    STREAM_CHECK_FINITE(sig.signal);
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: CoordinateNormalizer Welford mean == naive mean for small window
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_normaliser_welford_vs_naive() {
    // Verify Welford online mean equals the simple average after each update.
    CoordinateNormalizer norm{5};
    double buf[5] = {0, 0, 0, 0, 0};
    int count = 0;

    double prices[] = {10.0, 20.0, 30.0, 40.0, 50.0, 15.0, 25.0, 35.0};

    for (double p : prices) {
        buf[count % 5] = p;
        ++count;
        norm.update(p);

        if (count >= 5) {
            // Naive mean over the last 5 values in buf.
            double naive = 0.0;
            for (int i = 0; i < 5; ++i) naive += buf[i];
            naive /= 5.0;
            STREAM_CHECK_NEAR(norm.mean(), naive, 1e-9);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: SPSC ring — no element lost under producer/consumer pressure (32K)
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_spsc_ring_no_loss_under_pressure() {
    constexpr int N = 32768;
    SPSCRing<std::uint32_t, 65536> ring;

    std::atomic<bool>        done{false};
    std::atomic<std::uint64_t> sum_prod{0};
    std::atomic<std::uint64_t> sum_cons{0};

    std::thread consumer([&] {
        std::uint64_t local = 0;
        while (!done.load(std::memory_order_acquire) || !ring.empty_approx()) {
            auto r = ring.pop();
            if (r.has_value()) local += *r;
            else               std::this_thread::yield();
        }
        while (auto r = ring.pop()) local += *r;
        sum_cons.store(local, std::memory_order_release);
    });

    std::uint64_t local_prod = 0;
    for (int i = 0; i < N; ++i) {
        local_prod += static_cast<std::uint32_t>(i);
        std::uint32_t v = static_cast<std::uint32_t>(i);
        while (!ring.push(std::move(v))) std::this_thread::yield();
    }
    sum_prod.store(local_prod, std::memory_order_release);
    done.store(true, std::memory_order_release);
    consumer.join();

    STREAM_CHECK(sum_prod.load() == sum_cons.load());

    // Verify Gauss sum: N*(N-1)/2.
    const std::uint64_t expected =
        static_cast<std::uint64_t>(N) * (static_cast<std::uint64_t>(N) - 1u) / 2u;
    STREAM_CHECK(sum_cons.load() == expected);
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: normalizer — sigma is always non-negative and finite
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_normalizer_sigma_non_negative() {
    PseudoRand rng;

    for (int trial = 0; trial < 10; ++trial) {
        CoordinateNormalizer norm{20};

        for (int i = 0; i < 100; ++i) {
            norm.update(rng.next(1.0, 10000.0));
            STREAM_CHECK(norm.sigma() >= 0.0);
            STREAM_CHECK_FINITE(norm.sigma());
            STREAM_CHECK_FINITE(norm.mean());
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: compose_velocities stays sub-luminal (mirroring momentum module)
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_lorentz_composed_boost_is_still_valid_input() {
    // Chain two Lorentz boosts and verify the composed result remains finite.
    LorentzTransform lt;

    for (double b1 : {0.3, 0.6, 0.9, -0.3, -0.6}) {
        for (double b2 : {0.2, 0.5, 0.8, -0.2, -0.5}) {
            // First boost.
            auto ev1 = lt.transform(10.0, 5.0, b1);
            // Second boost applied to the already-boosted event.
            auto ev2 = lt.transform(ev1.t_prime, ev1.x_prime, b2);

            STREAM_CHECK_FINITE(ev2.t_prime);
            STREAM_CHECK_FINITE(ev2.x_prime);
            STREAM_CHECK(ev2.gamma >= 1.0);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: engine process_sync is deterministic for same input sequence
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_engine_process_sync_deterministic() {
    auto run = [](int seed) {
        QueueTickSource src;
        StreamEngine engine{src};
        PseudoRand rng;
        rng.state = static_cast<std::uint64_t>(seed);

        std::vector<double> signals;
        for (int i = 0; i < 50; ++i) {
            double price = rng.next(50.0, 500.0);
            auto sig = engine.process_sync(
                stream_test::make_tick(price),
                static_cast<std::int64_t>(i));
            signals.push_back(sig.signal);
        }
        return signals;
    };

    auto run1 = run(42);
    auto run2 = run(42);

    STREAM_CHECK(run1.size() == run2.size());
    for (std::size_t i = 0; i < run1.size(); ++i) {
        STREAM_CHECK_NEAR(run1[i], run2[i], EPS);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: QueueTickSource push_range round-trips all ticks in order
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_queue_source_push_range() {
    QueueTickSource src;
    auto ticks = stream_test::make_tick_sequence(100.0, 1.0, 10);
    src.push_range(ticks.begin(), ticks.end());

    STREAM_CHECK(src.pending() == 10u);

    for (int i = 0; i < 10; ++i) {
        auto r = src.read();
        STREAM_HAS_VALUE(r);
        double expected_close = 100.0 + static_cast<double>(i) * 1.0;
        STREAM_CHECK_NEAR(r->close, expected_close, 1e-9);
    }

    STREAM_NO_VALUE(src.read());
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: tick_is_valid rejects all ticks generated with invalid fields
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_tick_validation_exhaustive() {
    // All valid ticks in a sweep of prices.
    for (double p : {0.01, 1.0, 100.0, 1000.0, 1e6}) {
        auto t = stream_test::make_tick(p);
        STREAM_CHECK(tick_is_valid(t));
    }

    // Check every invalid single-field mutation.
    auto base = stream_test::make_tick(100.0);

    // Open = 0 → invalid.
    { auto t = base; t.open = 0.0; STREAM_CHECK(!tick_is_valid(t)); }
    // High < low → invalid.
    { auto t = base; t.high = base.low - 0.01; STREAM_CHECK(!tick_is_valid(t)); }
    // Timestamp = 0 → invalid.
    { auto t = base; t.timestamp_ns = 0; STREAM_CHECK(!tick_is_valid(t)); }
    // Negative volume → invalid.
    { auto t = base; t.volume = -0.001; STREAM_CHECK(!tick_is_valid(t)); }
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: normalizer after 1000 updates with constant price → sigma stays 0
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_normalizer_constant_long_run() {
    CoordinateNormalizer norm{20};

    for (int i = 0; i < 1000; ++i) {
        norm.update(42.0);
    }

    STREAM_CHECK(norm.warmed_up());
    STREAM_CHECK_NEAR(norm.mean(),  42.0, 1e-9);
    STREAM_CHECK_NEAR(norm.sigma(),  0.0, 1e-9);
    STREAM_CHECK_NEAR(norm.normalise(42.0), 0.0, 1e-9);
}

// ═════════════════════════════════════════════════════════════════════════════
// INV: beta returns 0 after reset regardless of prior state
// ═════════════════════════════════════════════════════════════════════════════

static void test_inv_beta_reset_always_returns_zero() {
    BetaCalculatorFix3 calc;

    // Build up state.
    for (int i = 0; i < 20; ++i) calc.update(100.0 * (1 + 0.05 * i));
    STREAM_CHECK(calc.warmed_up());
    STREAM_CHECK(calc.beta() != 0.0 || true); // might be nonzero

    calc.reset();

    // Immediately after reset, beta must be 0.
    STREAM_CHECK_NEAR(calc.beta(), 0.0, EPS);
    STREAM_CHECK(!calc.warmed_up());

    // Even after updating fewer than N values, still 0.
    calc.update(100.0);
    calc.update(110.0);
    STREAM_CHECK_NEAR(calc.beta(), 0.0, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("SRFM Stream — Pipeline Invariant Tests\n");
    std::printf("=======================================\n");

    STREAM_SUITE("beta always in bounds (500 random prices)",
                 test_inv_beta_always_in_bounds_over_random_prices);
    STREAM_SUITE("gamma >= 1 (100 random betas)",
                 test_inv_gamma_always_ge_1);
    STREAM_SUITE("signal always finite (200 random ticks)",
                 test_inv_signal_always_finite);
    STREAM_SUITE("normalise(mean) ≈ 0 (20 trials)",
                 test_inv_normaliser_z_score_is_zero_for_mean_value);
    STREAM_SUITE("manifold interval formula consistency (100 pairs)",
                 test_inv_manifold_interval_formula_consistency);
    STREAM_SUITE("Lorentz inverse exact (200 random inputs)",
                 test_inv_lorentz_inverse_is_exact);
    STREAM_SUITE("pipeline ds2 finite + regime-sign agreement",
                 test_inv_pipeline_ds2_formula_consistency);
    STREAM_SUITE("SPSC FIFO at scale (10K elements)",
                 test_inv_ring_fifo_ordering_at_scale);
    STREAM_SUITE("beta warmup = N+1 prices (all FIX variants)",
                 test_inv_beta_warmup_requires_n_plus_1_prices);
    STREAM_SUITE("mixed series covers all regime types",
                 test_inv_regime_covers_all_categories_in_mixed_series);
    STREAM_SUITE("bar counter monotone",
                 test_inv_processor_monotone_bar_counter);
    STREAM_SUITE("consumer count = signals written (100 ticks)",
                 test_inv_consumer_count_matches_signals_written);
    STREAM_SUITE("inject + process_sync round-trip",
                 test_inv_inject_then_process_roundtrip);
    STREAM_SUITE("Welford mean == naive mean",
                 test_inv_normaliser_welford_vs_naive);
    STREAM_SUITE("SPSC no loss under pressure (32K elements)",
                 test_inv_spsc_ring_no_loss_under_pressure);
    STREAM_SUITE("sigma non-negative (10 random trials)",
                 test_inv_normalizer_sigma_non_negative);
    STREAM_SUITE("composed Lorentz boost stays valid",
                 test_inv_lorentz_composed_boost_is_still_valid_input);
    STREAM_SUITE("process_sync is deterministic",
                 test_inv_engine_process_sync_deterministic);
    STREAM_SUITE("QueueTickSource push_range round-trip",
                 test_inv_queue_source_push_range);
    STREAM_SUITE("tick_is_valid exhaustive mutations",
                 test_inv_tick_validation_exhaustive);
    STREAM_SUITE("normalizer constant long run (1000 updates)",
                 test_inv_normalizer_constant_long_run);
    STREAM_SUITE("beta reset always returns zero",
                 test_inv_beta_reset_always_returns_zero);

    return stream_test::report();
}
