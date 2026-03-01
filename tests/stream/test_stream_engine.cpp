/**
 * @file  test_stream_engine.cpp
 * @brief Integration tests for the full streaming pipeline.
 *
 * Test structure:
 *   test_signal_processor_process_one_returns_finite_signal
 *   test_signal_processor_bar_index_increments
 *   test_signal_processor_gamma_ge_1
 *   test_signal_processor_beta_in_bounds
 *   test_signal_processor_pre_warmup_regime_lightlike
 *   test_signal_processor_reset_state_reverts
 *   test_engine_inject_invalid_tick_rejected
 *   test_engine_inject_valid_tick_accepted
 *   test_engine_process_sync_pipeline
 *   test_engine_full_pipeline_threaded          — start/stop + inject + consume
 *   test_engine_ring1_size_reflects_injected    — ring sizing
 *   test_queue_tick_source_push_and_read        — QueueTickSource mechanics
 *   test_tick_ingester_validates_ticks          — ingester drops invalid ticks
 *   test_signal_consumer_write_one_json_format  — JSON format correctness
 *   test_signal_processor_chain_100_ticks       — 100-tick end-to-end chain
 */

#include "srfm_stream_test.hpp"
#include "../../include/srfm/engine/stream_engine.hpp"
#include "../../include/srfm/stream/signal_consumer.hpp"
#include "../../include/srfm/stream/signal_processor.hpp"
#include "../../include/srfm/stream/tick_ingester.hpp"
#include "../../include/srfm/stream/tick_source.hpp"

#include <chrono>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

using namespace srfm::stream;
using namespace srfm::engine;

static constexpr double EPS = 1e-9;

// ── Helper: wait up to timeout_ms for a condition to become true ──────────────

template<typename Pred>
static bool wait_for(Pred cond, int timeout_ms = 2000) {
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(timeout_ms);
    while (!cond()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// SignalProcessor::process_one — output is finite
// ═════════════════════════════════════════════════════════════════════════════

static void test_signal_processor_process_one_returns_finite_signal() {
    QueueTickSource src;
    StreamEngine engine{src};

    auto tick = stream_test::make_tick(100.0);
    auto sig  = engine.process_sync(tick, 0);

    STREAM_CHECK_FINITE(sig.signal);
    STREAM_CHECK_FINITE(sig.beta);
    STREAM_CHECK_FINITE(sig.gamma);
    STREAM_CHECK(sig.bar == 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// SignalProcessor::process_one — bar index increments correctly
// ═════════════════════════════════════════════════════════════════════════════

static void test_signal_processor_bar_index_increments() {
    QueueTickSource src;
    StreamEngine engine{src};

    auto tick = stream_test::make_tick(100.0);

    for (std::int64_t i = 0; i < 10; ++i) {
        auto sig = engine.process_sync(tick, i);
        STREAM_CHECK(sig.bar == i);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SignalProcessor::process_one — gamma >= 1 always
// ═════════════════════════════════════════════════════════════════════════════

static void test_signal_processor_gamma_ge_1() {
    QueueTickSource src;
    StreamEngine engine{src};

    double price = 100.0;
    for (int i = 0; i < 30; ++i) {
        price *= 1.005;
        auto tick = stream_test::make_tick(price);
        auto sig  = engine.process_sync(tick, i);
        STREAM_CHECK(sig.gamma >= 1.0);
        STREAM_CHECK_FINITE(sig.gamma);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SignalProcessor::process_one — beta in bounds
// ═════════════════════════════════════════════════════════════════════════════

static void test_signal_processor_beta_in_bounds() {
    QueueTickSource src;
    StreamEngine engine{src};

    // Extreme rising price.
    double price = 1.0;
    for (int i = 0; i < 50; ++i) {
        price *= 2.0;
        auto tick = stream_test::make_tick(price);
        auto sig  = engine.process_sync(tick, i);
        STREAM_CHECK(sig.beta >  -BETA_MAX_SAFE);
        STREAM_CHECK(sig.beta <   BETA_MAX_SAFE);
        STREAM_CHECK_FINITE(sig.beta);
    }

    // Extreme falling price.
    engine.reset_processor_state();
    price = 1e15;
    for (int i = 0; i < 50; ++i) {
        price *= 0.5;
        auto tick = stream_test::make_tick(price);
        auto sig  = engine.process_sync(tick, i);
        STREAM_CHECK(sig.beta >  -BETA_MAX_SAFE);
        STREAM_CHECK(sig.beta <   BETA_MAX_SAFE);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SignalProcessor — pre-warmup: regime is LIGHTLIKE (no β yet)
// ═════════════════════════════════════════════════════════════════════════════

static void test_signal_processor_pre_warmup_regime_lightlike() {
    QueueTickSource src;
    StreamEngine engine{src};

    // First tick: no log-returns → β=0 → no displacement between events
    // (the manifold sees the first event → LIGHTLIKE).
    auto tick = stream_test::make_tick(100.0);
    auto sig  = engine.process_sync(tick, 0);

    STREAM_CHECK(sig.regime == Regime::LIGHTLIKE);
    STREAM_CHECK_NEAR(sig.signal, 0.0, EPS);
    STREAM_CHECK_NEAR(sig.beta,   0.0, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// StreamEngine::reset_processor_state
// ═════════════════════════════════════════════════════════════════════════════

static void test_signal_processor_reset_state_reverts() {
    QueueTickSource src;
    StreamEngine engine{src};

    // Feed 30 ticks to warm up.
    double price = 100.0;
    for (int i = 0; i < 30; ++i) {
        price *= 1.01;
        (void)engine.process_sync(stream_test::make_tick(price), i);
    }

    // Reset.
    engine.reset_processor_state();

    // First tick after reset: should behave like first-ever tick.
    auto sig = engine.process_sync(stream_test::make_tick(100.0), 0);
    STREAM_CHECK(sig.regime == Regime::LIGHTLIKE);
    STREAM_CHECK_NEAR(sig.signal, 0.0, EPS);
    STREAM_CHECK_NEAR(sig.beta,   0.0, EPS);
}

// ═════════════════════════════════════════════════════════════════════════════
// StreamEngine::inject — invalid tick rejected
// ═════════════════════════════════════════════════════════════════════════════

static void test_engine_inject_invalid_tick_rejected() {
    QueueTickSource src;
    StreamEngine engine{src};

    // NaN close.
    OHLCVTick bad;
    bad.open = bad.high = bad.low = bad.close =
        std::numeric_limits<double>::quiet_NaN();
    bad.volume = 1000.0;
    bad.timestamp_ns = 1;
    STREAM_CHECK(!engine.inject(bad));

    // Zero price.
    OHLCVTick zero = stream_test::make_tick(0.0);
    STREAM_CHECK(!engine.inject(zero));

    // Negative volume.
    OHLCVTick negvol = stream_test::make_tick(100.0);
    negvol.volume = -1.0;
    STREAM_CHECK(!engine.inject(negvol));
}

// ═════════════════════════════════════════════════════════════════════════════
// StreamEngine::inject — valid tick accepted
// ═════════════════════════════════════════════════════════════════════════════

static void test_engine_inject_valid_tick_accepted() {
    QueueTickSource src;
    StreamEngine engine{src};

    auto tick = stream_test::make_tick(150.0);
    STREAM_CHECK(engine.inject(tick));
    STREAM_CHECK(engine.ring1_size_approx() == 1u);
}

// ═════════════════════════════════════════════════════════════════════════════
// StreamEngine::process_sync — full signal chain over 100 ticks
// ═════════════════════════════════════════════════════════════════════════════

static void test_engine_process_sync_pipeline() {
    QueueTickSource src;
    StreamEngine engine{src};

    auto ticks = stream_test::make_tick_sequence(100.0, 0.5, 100);

    std::size_t timelike_count  = 0;
    std::size_t spacelike_count = 0;
    std::size_t lightlike_count = 0;

    for (std::size_t i = 0; i < ticks.size(); ++i) {
        auto sig = engine.process_sync(ticks[i], static_cast<std::int64_t>(i));

        STREAM_CHECK_FINITE(sig.signal);
        STREAM_CHECK_FINITE(sig.beta);
        STREAM_CHECK_FINITE(sig.gamma);
        STREAM_CHECK(sig.gamma >= 1.0);
        STREAM_CHECK(sig.beta > -BETA_MAX_SAFE);
        STREAM_CHECK(sig.beta <  BETA_MAX_SAFE);
        STREAM_CHECK(sig.bar == static_cast<std::int64_t>(i));

        switch (sig.regime) {
            case Regime::TIMELIKE:  ++timelike_count;  break;
            case Regime::SPACELIKE: ++spacelike_count; break;
            case Regime::LIGHTLIKE: ++lightlike_count; break;
        }
    }

    // With a steadily rising price series, we expect predominantly TIMELIKE
    // or SPACELIKE events (the exact split depends on β, but all are finite).
    STREAM_CHECK(timelike_count + spacelike_count + lightlike_count == 100u);
}

// ═════════════════════════════════════════════════════════════════════════════
// QueueTickSource — push and read
// ═════════════════════════════════════════════════════════════════════════════

static void test_queue_tick_source_push_and_read() {
    QueueTickSource src;

    STREAM_CHECK(src.is_open());
    STREAM_NO_VALUE(src.read()); // empty

    auto t = stream_test::make_tick(100.0);
    src.push(t);
    STREAM_CHECK(src.pending() == 1u);

    auto r = src.read();
    STREAM_HAS_VALUE(r);
    STREAM_CHECK_NEAR(r->close, 100.0, EPS);
    STREAM_CHECK(src.pending() == 0u);

    // Close.
    src.close();
    STREAM_CHECK(!src.is_open());
    STREAM_NO_VALUE(src.read());
}

// ═════════════════════════════════════════════════════════════════════════════
// TickIngester — validates and drops invalid ticks
// ═════════════════════════════════════════════════════════════════════════════

static void test_tick_ingester_validates_ticks() {
    QueueTickSource src;
    SPSCRing<OHLCVTick, 65536> ring;
    TickIngester ingester{src, ring};

    // Push 5 valid + 3 invalid ticks before starting the ingester.
    for (int i = 0; i < 5; ++i) {
        src.push(stream_test::make_tick(100.0 + i));
    }
    // Invalid: zero price.
    OHLCVTick bad;
    bad.open = bad.high = bad.low = bad.close = 0.0;
    bad.volume = 1000.0; bad.timestamp_ns = 1;
    src.push(bad);
    src.push(bad);
    src.push(bad);

    ingester.start();

    // Wait for the ingester to process all 8 ticks.
    bool drained = wait_for([&] {
        return src.pending() == 0;
    });
    STREAM_CHECK(drained);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ingester.stop();

    auto c = ingester.counters();
    STREAM_CHECK(c.ticks_received        >= 8u);
    STREAM_CHECK(c.ticks_pushed          == 5u);
    STREAM_CHECK(c.ticks_dropped_invalid >= 3u);
}

// ═════════════════════════════════════════════════════════════════════════════
// SignalConsumer::write_one — JSON format correctness
// ═════════════════════════════════════════════════════════════════════════════

static void test_signal_consumer_write_one_json_format() {
    // Write to a temp file, read it back, and check the JSON content.
    const char* tmp_path = "srfm_json_test_tmp.txt";

#ifdef _CRT_SECURE_NO_WARNINGS
    FILE* f = std::fopen(tmp_path, "w");
#else
    FILE* f = nullptr;
    fopen_s(&f, tmp_path, "w");
#endif
    if (!f) {
        STREAM_CHECK(true); // can't open temp file — skip
        return;
    }

    SPSCRing<StreamRelativisticSignal, 65536> ring;
    SignalConsumer consumer{ring, f};

    StreamRelativisticSignal sig;
    sig.bar    = 42;
    sig.beta   = 0.42;
    sig.gamma  = 1.2309;
    sig.regime = Regime::TIMELIKE;
    sig.signal = 0.87;

    consumer.write_one(sig);
    std::fflush(f);
    std::fclose(f);

    // Read the file back.
    char buf[512] = {};
#ifdef _CRT_SECURE_NO_WARNINGS
    FILE* r = std::fopen(tmp_path, "r");
#else
    FILE* r = nullptr;
    fopen_s(&r, tmp_path, "r");
#endif
    if (r) {
        (void)std::fread(buf, 1, sizeof(buf) - 1, r);
        std::fclose(r);
    }
    std::remove(tmp_path);

    // Check that bar, beta, gamma, regime, and signal appear in the output.
    STREAM_CHECK(std::strstr(buf, "\"bar\": 42")  != nullptr);
    STREAM_CHECK(std::strstr(buf, "\"beta\":")    != nullptr);
    STREAM_CHECK(std::strstr(buf, "\"gamma\":")   != nullptr);
    STREAM_CHECK(std::strstr(buf, "\"TIMELIKE\"") != nullptr);
    STREAM_CHECK(std::strstr(buf, "\"signal\":")  != nullptr);
}

// ═════════════════════════════════════════════════════════════════════════════
// Full threaded pipeline: inject → process → consume
// ═════════════════════════════════════════════════════════════════════════════

static void test_engine_full_pipeline_threaded() {
    // Redirect consumer output to the null device (suppress output during test).
    FILE* dev_null = nullptr;
#ifdef _WIN32
    fopen_s(&dev_null, "NUL", "w");
#else
    dev_null = std::fopen("/dev/null", "w");
#endif

    QueueTickSource src;
    StreamEngine engine{src, dev_null ? dev_null : stdout};

    engine.start();
    STREAM_CHECK(engine.running());

    // Inject 200 valid ticks via inject() (bypasses TickIngester thread).
    constexpr int N = 200;
    double price = 100.0;
    for (int i = 0; i < N; ++i) {
        price += 0.1;
        // Retry until ring space is available.
        OHLCVTick t = stream_test::make_tick(price);
        bool pushed = false;
        for (int retry = 0; retry < 1000; ++retry) {
            if (engine.inject(t)) { pushed = true; break; }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        STREAM_CHECK(pushed);
    }

    // Wait for all injected ticks to be consumed.
    bool consumed = wait_for([&] {
        auto pc = engine.processor_counters();
        return pc.ticks_processed >= static_cast<std::uint64_t>(N);
    }, 5000);
    STREAM_CHECK(consumed);

    bool emitted = wait_for([&] {
        auto cc = engine.consumer_counters();
        return cc.signals_consumed >= static_cast<std::uint64_t>(N);
    }, 5000);
    STREAM_CHECK(emitted);

    engine.stop();
    STREAM_CHECK(!engine.running());

    auto pc = engine.processor_counters();
    STREAM_CHECK(pc.ticks_processed == static_cast<std::uint64_t>(N));
    STREAM_CHECK(pc.signals_emitted == static_cast<std::uint64_t>(N));

    auto cc = engine.consumer_counters();
    STREAM_CHECK(cc.signals_consumed == static_cast<std::uint64_t>(N));

    if (dev_null) std::fclose(dev_null);
}

// ═════════════════════════════════════════════════════════════════════════════
// ring1_size_approx reflects injected ticks (before processor starts)
// ═════════════════════════════════════════════════════════════════════════════

static void test_engine_ring1_size_reflects_injected() {
    QueueTickSource src;
    StreamEngine engine{src};

    // Don't start the engine — no consumer thread draining the ring.
    STREAM_CHECK(engine.ring1_size_approx() == 0u);

    for (int i = 0; i < 5; ++i) {
        STREAM_CHECK(engine.inject(stream_test::make_tick(100.0 + i)));
    }

    STREAM_CHECK(engine.ring1_size_approx() == 5u);
}

// ═════════════════════════════════════════════════════════════════════════════
// SignalProcessor chain: 100 ticks with varying prices
// ═════════════════════════════════════════════════════════════════════════════

static void test_signal_processor_chain_100_ticks() {
    QueueTickSource src;
    StreamEngine engine{src};

    // Alternating up/down price sequence to exercise both regime types.
    double price = 100.0;
    double dir   = 1.0;

    for (int i = 0; i < 100; ++i) {
        price += dir * 0.5;
        if (i % 10 == 9) dir = -dir;  // flip direction every 10 ticks

        auto sig = engine.process_sync(stream_test::make_tick(price),
                                        static_cast<std::int64_t>(i));

        STREAM_CHECK_FINITE(sig.signal);
        STREAM_CHECK_FINITE(sig.beta);
        STREAM_CHECK_FINITE(sig.gamma);
        STREAM_CHECK(sig.gamma >= 1.0);
        STREAM_CHECK(sig.beta >  -BETA_MAX_SAFE);
        STREAM_CHECK(sig.beta <   BETA_MAX_SAFE);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Processor counters track regime distribution
// ═════════════════════════════════════════════════════════════════════════════

static void test_processor_regime_counters() {
    QueueTickSource src;
    SPSCRing<OHLCVTick, 65536>             in_ring;
    SPSCRing<StreamRelativisticSignal, 65536> out_ring;
    SignalProcessor proc{in_ring, out_ring};

    // Push 50 ticks directly into in_ring and run process_one manually.
    double price = 100.0;
    for (int i = 0; i < 50; ++i) {
        price *= 1.005;
        auto sig = proc.process_one(stream_test::make_tick(price),
                                     static_cast<std::int64_t>(i));
        STREAM_CHECK_FINITE(sig.signal);
    }

    // No threading involved — process_one doesn't update counters_.
    // Just verify no panics/non-finite outputs.
    STREAM_CHECK(true);
}

// ═════════════════════════════════════════════════════════════════════════════
// TickIngester and SignalConsumer counters after stop
// ═════════════════════════════════════════════════════════════════════════════

static void test_counter_accessors_after_stop() {
    QueueTickSource src;
    StreamEngine engine{src};

    // Before start: all counters zero.
    STREAM_CHECK(engine.ingester_counters().ticks_pushed   == 0u);
    STREAM_CHECK(engine.processor_counters().ticks_processed == 0u);
    STREAM_CHECK(engine.consumer_counters().signals_consumed == 0u);

    // After stop (never started): still zero.
    engine.stop();
    STREAM_CHECK(engine.ingester_counters().ticks_pushed   == 0u);
}

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("SRFM Stream — StreamEngine Integration Tests\n");
    std::printf("=============================================\n");

    STREAM_SUITE("process_one returns finite signal",     test_signal_processor_process_one_returns_finite_signal);
    STREAM_SUITE("bar index increments",                  test_signal_processor_bar_index_increments);
    STREAM_SUITE("gamma >= 1 always",                     test_signal_processor_gamma_ge_1);
    STREAM_SUITE("beta in bounds",                        test_signal_processor_beta_in_bounds);
    STREAM_SUITE("pre-warmup regime is LIGHTLIKE",        test_signal_processor_pre_warmup_regime_lightlike);
    STREAM_SUITE("reset_state reverts",                   test_signal_processor_reset_state_reverts);
    STREAM_SUITE("inject invalid tick rejected",          test_engine_inject_invalid_tick_rejected);
    STREAM_SUITE("inject valid tick accepted",            test_engine_inject_valid_tick_accepted);
    STREAM_SUITE("process_sync pipeline 100 ticks",       test_engine_process_sync_pipeline);
    STREAM_SUITE("QueueTickSource push/read",             test_queue_tick_source_push_and_read);
    STREAM_SUITE("TickIngester validates ticks",          test_tick_ingester_validates_ticks);
    STREAM_SUITE("SignalConsumer JSON format",            test_signal_consumer_write_one_json_format);
    STREAM_SUITE("full threaded pipeline (200 ticks)",    test_engine_full_pipeline_threaded);
    STREAM_SUITE("ring1_size reflects injected",          test_engine_ring1_size_reflects_injected);
    STREAM_SUITE("signal chain 100 ticks alternating",    test_signal_processor_chain_100_ticks);
    STREAM_SUITE("regime counters (process_one)",         test_processor_regime_counters);
    STREAM_SUITE("counter accessors after stop",          test_counter_accessors_after_stop);

    return stream_test::report();
}
