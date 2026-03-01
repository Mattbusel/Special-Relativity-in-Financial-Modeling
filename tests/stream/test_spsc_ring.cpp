/**
 * @file  test_spsc_ring.cpp
 * @brief Unit + concurrency tests for SPSCRing<T, SIZE>.
 *
 * Test structure:
 *   test_spsc_ring_basic_push_pop           — single-element round-trip
 *   test_spsc_ring_empty_returns_nullopt    — pop on empty ring
 *   test_spsc_ring_full_returns_false       — push on full ring
 *   test_spsc_ring_capacity_is_size_minus_1 — capacity semantics
 *   test_spsc_ring_fifo_ordering            — FIFO property
 *   test_spsc_ring_fill_drain_refill        — fill → drain → fill again
 *   test_spsc_ring_power_of_two_mask        — index wraps correctly
 *   test_spsc_ring_move_semantics           — move-only element type
 *   test_spsc_ring_size_approx              — diagnostic size estimates
 *   test_spsc_ring_push_copy                — copy-push overload
 *   test_spsc_ring_concurrent_producer_consumer — threaded correctness
 *   test_spsc_ring_concurrent_throughput    — high-volume threaded run
 */

#include "srfm_stream_test.hpp"
#include "../../include/srfm/stream/spsc_ring.hpp"

#include <atomic>
#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

using namespace srfm::stream;

// ── Helpers ───────────────────────────────────────────────────────────────────

static constexpr double EPS = 1e-12;

// Small ring for testing full/empty transitions.
using SmallRing = SPSCRing<int, 4>;    // capacity = 3
using MedRing   = SPSCRing<int, 16>;   // capacity = 15
using TickRing  = SPSCRing<OHLCVTick, 65536>;

// ═════════════════════════════════════════════════════════════════════════════
// Basic push / pop
// ═════════════════════════════════════════════════════════════════════════════

static void test_spsc_ring_basic_push_pop() {
    SmallRing ring;

    // Push one element.
    int val = 42;
    STREAM_CHECK(ring.push(std::move(val)));

    // Pop and verify.
    auto r = ring.pop();
    STREAM_HAS_VALUE(r);
    STREAM_CHECK(*r == 42);

    // Ring should now be empty.
    STREAM_NO_VALUE(ring.pop());
}

// ═════════════════════════════════════════════════════════════════════════════
// Empty returns nullopt
// ═════════════════════════════════════════════════════════════════════════════

static void test_spsc_ring_empty_returns_nullopt() {
    SmallRing ring;

    // Freshly constructed ring is empty.
    STREAM_NO_VALUE(ring.pop());
    STREAM_NO_VALUE(ring.pop());
    STREAM_NO_VALUE(ring.pop());

    // After push + pop, should be empty again.
    int v = 1;
    STREAM_CHECK(ring.push(std::move(v)));
    auto r = ring.pop();
    STREAM_HAS_VALUE(r);
    STREAM_NO_VALUE(ring.pop());
}

// ═════════════════════════════════════════════════════════════════════════════
// Full returns false
// ═════════════════════════════════════════════════════════════════════════════

static void test_spsc_ring_full_returns_false() {
    SmallRing ring; // capacity = 3

    // Fill to capacity.
    int a = 1, b = 2, c = 3;
    STREAM_CHECK(ring.push(std::move(a)));
    STREAM_CHECK(ring.push(std::move(b)));
    STREAM_CHECK(ring.push(std::move(c)));

    // One more push must fail.
    int d = 4;
    STREAM_CHECK(!ring.push(std::move(d)));

    // Pop one and push again should succeed.
    auto r = ring.pop();
    STREAM_HAS_VALUE(r);
    STREAM_CHECK(*r == 1);
    int e = 5;
    STREAM_CHECK(ring.push(std::move(e)));
}

// ═════════════════════════════════════════════════════════════════════════════
// Capacity is SIZE - 1
// ═════════════════════════════════════════════════════════════════════════════

static void test_spsc_ring_capacity_is_size_minus_1() {
    // SIZE=4 → capacity=3
    STREAM_CHECK(SmallRing::capacity() == 3u);
    // SIZE=16 → capacity=15
    STREAM_CHECK(MedRing::capacity() == 15u);
    // SIZE=65536 → capacity=65535
    STREAM_CHECK(TickRing::capacity() == 65535u);
}

// ═════════════════════════════════════════════════════════════════════════════
// FIFO ordering
// ═════════════════════════════════════════════════════════════════════════════

static void test_spsc_ring_fifo_ordering() {
    MedRing ring;

    // Push 10 elements.
    for (int i = 0; i < 10; ++i) {
        int v = i;
        STREAM_CHECK(ring.push(std::move(v)));
    }

    // Pop all and verify order.
    for (int i = 0; i < 10; ++i) {
        auto r = ring.pop();
        STREAM_HAS_VALUE(r);
        STREAM_CHECK(*r == i);
    }

    STREAM_NO_VALUE(ring.pop());
}

// ═════════════════════════════════════════════════════════════════════════════
// Fill → drain → refill (wrapping index correctness)
// ═════════════════════════════════════════════════════════════════════════════

static void test_spsc_ring_fill_drain_refill() {
    MedRing ring; // capacity = 15

    // Cycle 3 times to exercise index wrap.
    for (int cycle = 0; cycle < 3; ++cycle) {
        // Fill.
        for (int i = 0; i < 15; ++i) {
            int v = cycle * 100 + i;
            STREAM_CHECK(ring.push(std::move(v)));
        }
        // Drain.
        for (int i = 0; i < 15; ++i) {
            auto r = ring.pop();
            STREAM_HAS_VALUE(r);
            STREAM_CHECK(*r == cycle * 100 + i);
        }
        STREAM_NO_VALUE(ring.pop());
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Power-of-two index mask — partial fill and wrap
// ═════════════════════════════════════════════════════════════════════════════

static void test_spsc_ring_power_of_two_mask() {
    // Use a size-8 ring (capacity=7) and push/pop in a stride pattern.
    SPSCRing<int, 8> ring;

    for (int round = 0; round < 20; ++round) {
        // Push 3, pop 3 — advances both indices by 3 each round.
        // After 3 rounds, indices wrap around (3*3=9 > 8).
        for (int i = 0; i < 3; ++i) {
            int v = round * 3 + i;
            STREAM_CHECK(ring.push(std::move(v)));
        }
        for (int i = 0; i < 3; ++i) {
            auto r = ring.pop();
            STREAM_HAS_VALUE(r);
            STREAM_CHECK(*r == round * 3 + i);
        }
    }

    STREAM_NO_VALUE(ring.pop());
}

// ═════════════════════════════════════════════════════════════════════════════
// Move semantics — move-only element type
// ═════════════════════════════════════════════════════════════════════════════

// A trivially noexcept-movable wrapper around double.
struct MovableDouble {
    double v{0.0};
    MovableDouble() noexcept = default;
    explicit MovableDouble(double x) noexcept : v{x} {}
    MovableDouble(const MovableDouble&) = delete;
    MovableDouble& operator=(const MovableDouble&) = delete;
    MovableDouble(MovableDouble&&) noexcept = default;
    MovableDouble& operator=(MovableDouble&&) noexcept = default;
};

static void test_spsc_ring_move_semantics() {
    SPSCRing<MovableDouble, 8> ring;

    STREAM_CHECK(ring.push(MovableDouble{3.14}));
    STREAM_CHECK(ring.push(MovableDouble{2.72}));

    auto r1 = ring.pop();
    STREAM_HAS_VALUE(r1);
    STREAM_CHECK_NEAR(r1->v, 3.14, EPS);

    auto r2 = ring.pop();
    STREAM_HAS_VALUE(r2);
    STREAM_CHECK_NEAR(r2->v, 2.72, EPS);

    STREAM_NO_VALUE(ring.pop());
}

// ═════════════════════════════════════════════════════════════════════════════
// size_approx / empty_approx / full_approx (single-threaded)
// ═════════════════════════════════════════════════════════════════════════════

static void test_spsc_ring_size_approx() {
    SmallRing ring; // capacity = 3

    STREAM_CHECK(ring.empty_approx());
    STREAM_CHECK(!ring.full_approx());
    STREAM_CHECK(ring.size_approx() == 0u);

    int a = 1;
    STREAM_CHECK(ring.push(std::move(a)));
    STREAM_CHECK(!ring.empty_approx());
    STREAM_CHECK(!ring.full_approx());
    STREAM_CHECK(ring.size_approx() == 1u);

    int b = 2, c = 3;
    STREAM_CHECK(ring.push(std::move(b)));
    STREAM_CHECK(ring.push(std::move(c)));
    STREAM_CHECK(!ring.empty_approx());
    STREAM_CHECK(ring.full_approx());
    STREAM_CHECK(ring.size_approx() == 3u);

    STREAM_HAS_VALUE(ring.pop());
    STREAM_CHECK(!ring.full_approx());
    STREAM_CHECK(ring.size_approx() == 2u);

    STREAM_HAS_VALUE(ring.pop());
    STREAM_HAS_VALUE(ring.pop());
    STREAM_CHECK(ring.empty_approx());
    STREAM_CHECK(ring.size_approx() == 0u);
}

// ═════════════════════════════════════════════════════════════════════════════
// push_copy overload
// ═════════════════════════════════════════════════════════════════════════════

static void test_spsc_ring_push_copy() {
    MedRing ring;

    const int val = 99;
    STREAM_CHECK(ring.push_copy(val));

    // Original val is unmodified.
    STREAM_CHECK(val == 99);

    auto r = ring.pop();
    STREAM_HAS_VALUE(r);
    STREAM_CHECK(*r == 99);
}

// ═════════════════════════════════════════════════════════════════════════════
// Concurrent producer/consumer correctness
// ═════════════════════════════════════════════════════════════════════════════

static void test_spsc_ring_concurrent_producer_consumer() {
    constexpr int N = 10000;
    SPSCRing<int, 16384> ring;

    std::vector<int> consumed;
    consumed.reserve(N);

    // Consumer thread.
    std::atomic<bool> done{false};
    std::thread consumer([&]() {
        while (!done.load(std::memory_order_acquire) || !ring.empty_approx()) {
            auto r = ring.pop();
            if (r.has_value()) {
                consumed.push_back(*r);
            } else {
                std::this_thread::yield();
            }
        }
        // Drain stragglers.
        while (true) {
            auto r = ring.pop();
            if (!r.has_value()) break;
            consumed.push_back(*r);
        }
    });

    // Producer: push 0..N-1.
    for (int i = 0; i < N; ++i) {
        int v = i;
        while (!ring.push(std::move(v))) {
            std::this_thread::yield();
        }
    }
    done.store(true, std::memory_order_release);
    consumer.join();

    // Verify all N values were received in FIFO order.
    STREAM_CHECK(consumed.size() == static_cast<std::size_t>(N));
    bool in_order = true;
    for (int i = 0; i < N; ++i) {
        if (consumed[static_cast<std::size_t>(i)] != i) { in_order = false; break; }
    }
    STREAM_CHECK(in_order);
}

// ═════════════════════════════════════════════════════════════════════════════
// Concurrent throughput — 1M elements
// ═════════════════════════════════════════════════════════════════════════════

static void test_spsc_ring_concurrent_throughput() {
    constexpr int N = 1'000'000;
    SPSCRing<std::uint64_t, 65536> ring;

    std::atomic<std::uint64_t> sum_produced{0};
    std::atomic<std::uint64_t> sum_consumed{0};
    std::atomic<bool>          done{false};

    std::thread consumer([&]() {
        std::uint64_t local_sum = 0;
        while (!done.load(std::memory_order_acquire) || !ring.empty_approx()) {
            auto r = ring.pop();
            if (r.has_value()) local_sum += *r;
            else               std::this_thread::yield();
        }
        while (true) {
            auto r = ring.pop();
            if (!r.has_value()) break;
            local_sum += *r;
        }
        sum_consumed.store(local_sum, std::memory_order_release);
    });

    std::uint64_t local_prod = 0;
    for (int i = 0; i < N; ++i) {
        std::uint64_t v = static_cast<std::uint64_t>(i);
        local_prod += v;
        while (!ring.push(std::move(v))) std::this_thread::yield();
    }
    sum_produced.store(local_prod, std::memory_order_release);
    done.store(true, std::memory_order_release);
    consumer.join();

    // Sum should match (Gauss: N*(N-1)/2).
    const std::uint64_t expected =
        static_cast<std::uint64_t>(N) * (static_cast<std::uint64_t>(N) - 1u) / 2u;
    STREAM_CHECK(sum_consumed.load() == expected);
    STREAM_CHECK(sum_produced.load() == expected);
}

// ═════════════════════════════════════════════════════════════════════════════
// OHLCVTick round-trip
// ═════════════════════════════════════════════════════════════════════════════

static void test_spsc_ring_ohlcv_round_trip() {
    TickRing ring;

    OHLCVTick t;
    t.open = 100.0; t.high = 105.0; t.low = 99.0; t.close = 102.0;
    t.volume = 5000.0; t.timestamp_ns = 1700000000000000000LL;

    STREAM_CHECK(ring.push(std::move(t)));
    auto r = ring.pop();
    STREAM_HAS_VALUE(r);
    STREAM_CHECK_NEAR(r->open,   100.0, EPS);
    STREAM_CHECK_NEAR(r->high,   105.0, EPS);
    STREAM_CHECK_NEAR(r->low,     99.0, EPS);
    STREAM_CHECK_NEAR(r->close,  102.0, EPS);
    STREAM_CHECK_NEAR(r->volume, 5000.0, EPS);
    STREAM_CHECK(r->timestamp_ns == 1700000000000000000LL);
}

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("SRFM Stream — SPSCRing Tests\n");
    std::printf("============================\n");

    STREAM_SUITE("basic push/pop",             test_spsc_ring_basic_push_pop);
    STREAM_SUITE("empty returns nullopt",       test_spsc_ring_empty_returns_nullopt);
    STREAM_SUITE("full returns false",          test_spsc_ring_full_returns_false);
    STREAM_SUITE("capacity = SIZE-1",           test_spsc_ring_capacity_is_size_minus_1);
    STREAM_SUITE("FIFO ordering",               test_spsc_ring_fifo_ordering);
    STREAM_SUITE("fill/drain/refill",           test_spsc_ring_fill_drain_refill);
    STREAM_SUITE("power-of-two mask",           test_spsc_ring_power_of_two_mask);
    STREAM_SUITE("move semantics",              test_spsc_ring_move_semantics);
    STREAM_SUITE("size_approx",                 test_spsc_ring_size_approx);
    STREAM_SUITE("push_copy",                   test_spsc_ring_push_copy);
    STREAM_SUITE("concurrent correctness",      test_spsc_ring_concurrent_producer_consumer);
    STREAM_SUITE("concurrent throughput (1M)",  test_spsc_ring_concurrent_throughput);
    STREAM_SUITE("OHLCVTick round-trip",        test_spsc_ring_ohlcv_round_trip);

    return stream_test::report();
}
