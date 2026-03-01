// =============================================================================
// test_spsc_stress.cpp — SPSC ring boundary, property, stress, and tick tests
// Complements test_spsc_ring.cpp with additional coverage.
// =============================================================================

#include "srfm_stream_test.hpp"
#include "srfm/stream/spsc_ring.hpp"
#include "srfm/stream/tick.hpp"

#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <random>
#include <limits>

using namespace srfm::stream;

// ---------------------------------------------------------------------------
// Ring capacity / boundary tests
// ---------------------------------------------------------------------------

static void test_ring_size_2_push_pop() {
    // SIZE=2 → capacity=1; only 1 element fits
    SPSCRing<int, 2> ring;
    STREAM_CHECK(ring.push(10));
    STREAM_CHECK(!ring.push(20));  // full
    auto v1 = ring.pop(); STREAM_HAS_VALUE(v1); STREAM_CHECK(*v1 == 10);
    STREAM_NO_VALUE(ring.pop());
}

static void test_ring_size_4_boundary() {
    // SIZE=4 → capacity=3; push 3, 4th fails
    SPSCRing<int, 4> ring;
    for (int i = 0; i < 3; ++i) STREAM_CHECK(ring.push_copy(i));
    STREAM_CHECK(!ring.push(99));
    for (int i = 0; i < 3; ++i) {
        auto v = ring.pop();
        STREAM_HAS_VALUE(v);
        STREAM_CHECK(*v == i);
    }
    STREAM_NO_VALUE(ring.pop());
}

static void test_ring_size_64_fill_and_drain() {
    // SIZE=64 → capacity=63
    SPSCRing<int, 64> ring;
    for (int i = 0; i < 63; ++i) STREAM_CHECK(ring.push_copy(i));
    STREAM_CHECK(ring.full_approx());
    for (int i = 0; i < 63; ++i) {
        auto v = ring.pop(); STREAM_HAS_VALUE(v); STREAM_CHECK(*v == i);
    }
    STREAM_CHECK(ring.empty_approx());
}

static void test_ring_size_128_fifo_order() {
    SPSCRing<int, 128> ring;
    for (int i = 0; i < 100; ++i) STREAM_CHECK(ring.push_copy(i));
    for (int i = 0; i < 100; ++i) {
        auto v = ring.pop(); STREAM_HAS_VALUE(v); STREAM_CHECK(*v == i);
    }
}

static void test_ring_capacity_is_size_minus_1() {
    SPSCRing<int, 8>   r8;   STREAM_CHECK(r8.capacity()  == std::size_t{7});
    SPSCRing<int, 16>  r16;  STREAM_CHECK(r16.capacity() == std::size_t{15});
    SPSCRing<int, 256> r256; STREAM_CHECK(r256.capacity()== std::size_t{255});
}

static void test_ring_alternating_push_pop() {
    SPSCRing<int, 8> ring;
    for (int i = 0; i < 100; ++i) {
        STREAM_CHECK(ring.push_copy(i));
        auto v = ring.pop();
        STREAM_HAS_VALUE(v);
        STREAM_CHECK(*v == i);
    }
}

static void test_ring_push_copy_semantics() {
    SPSCRing<std::vector<int>, 4> ring;
    std::vector<int> src = {1, 2, 3};
    STREAM_CHECK(ring.push_copy(src));
    auto v = ring.pop();
    STREAM_HAS_VALUE(v);
    STREAM_CHECK((*v)[0] == 1);
    STREAM_CHECK((*v)[2] == 3);
    STREAM_CHECK(src.size() == std::size_t{3});
}

static void test_ring_move_semantics_unique_ptr() {
    SPSCRing<std::unique_ptr<int>, 4> ring;
    STREAM_CHECK(ring.push(std::make_unique<int>(42)));
    auto v = ring.pop();
    STREAM_HAS_VALUE(v);
    STREAM_CHECK(**v == 42);
}

static void test_ring_empty_approx_initially() {
    SPSCRing<int, 8> ring;
    STREAM_CHECK(ring.empty_approx());
    STREAM_CHECK(!ring.full_approx());
    STREAM_CHECK(ring.size_approx() == std::size_t{0});
}

static void test_ring_size_approx_increments() {
    SPSCRing<int, 8> ring;
    for (int i = 1; i <= 4; ++i) {
        STREAM_CHECK(ring.push_copy(i));
        STREAM_CHECK(ring.size_approx() == std::size_t(i));
    }
}

static void test_ring_size_approx_decrements() {
    SPSCRing<int, 8> ring;
    for (int i = 0; i < 4; ++i) STREAM_CHECK(ring.push_copy(i));
    for (int i = 3; i >= 0; --i) {
        (void)ring.pop();
        STREAM_CHECK(ring.size_approx() == std::size_t(i));
    }
}

static void test_ring_pop_empty_nullopt() {
    SPSCRing<int, 8> ring;
    STREAM_NO_VALUE(ring.pop());
    STREAM_NO_VALUE(ring.pop());
}

static void test_ring_double_fill_drain_refill() {
    SPSCRing<int, 8> ring;
    for (int i = 0; i < 7; ++i) STREAM_CHECK(ring.push_copy(i));
    for (int i = 0; i < 7; ++i) (void)ring.pop();
    for (int i = 100; i < 107; ++i) STREAM_CHECK(ring.push_copy(i));
    for (int i = 100; i < 107; ++i) {
        auto v = ring.pop(); STREAM_HAS_VALUE(v); STREAM_CHECK(*v == i);
    }
}

static void test_ring_ohlcv_tick_round_trip() {
    SPSCRing<OHLCVTick, 16> ring;
    OHLCVTick t;
    t.open=100.5; t.high=105.0; t.low=99.0; t.close=102.3;
    t.volume=500000.0; t.timestamp_ns=1700000000000000000LL;
    STREAM_CHECK(ring.push(std::move(t)));
    auto v = ring.pop();
    STREAM_HAS_VALUE(v);
    STREAM_CHECK_NEAR(v->open,  100.5, 1e-12);
    STREAM_CHECK_NEAR(v->close, 102.3, 1e-12);
    STREAM_CHECK(v->timestamp_ns == std::int64_t{1700000000000000000LL});
}

static void test_ring_multiple_tick_round_trips() {
    SPSCRing<OHLCVTick, 32> ring;
    for (int i = 0; i < 20; ++i) {
        OHLCVTick t;
        t.open=100.0+i; t.high=110.0+i; t.low=90.0+i; t.close=105.0+i;
        t.volume=1000.0*i; t.timestamp_ns=1000LL+i;
        STREAM_CHECK(ring.push(std::move(t)));
    }
    for (int i = 0; i < 20; ++i) {
        auto v = ring.pop();
        STREAM_HAS_VALUE(v);
        STREAM_CHECK_NEAR(v->close, 105.0+i, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Concurrent SPSC correctness
// ---------------------------------------------------------------------------

static void test_ring_concurrent_1k_elements() {
    SPSCRing<int, 1024> ring;
    constexpr int N = 1000;
    std::atomic<int> sum_prod{0}, sum_cons{0};
    std::thread producer([&]{
        for (int i = 0; i < N; ++i) {
            while (!ring.push_copy(i)) {}
            sum_prod.fetch_add(i, std::memory_order_relaxed);
        }
    });
    std::thread consumer([&]{
        int count = 0;
        while (count < N) {
            auto v = ring.pop();
            if (v) { sum_cons.fetch_add(*v, std::memory_order_relaxed); ++count; }
        }
    });
    producer.join(); consumer.join();
    STREAM_CHECK(sum_prod.load() == sum_cons.load());
}

static void test_ring_concurrent_no_loss_5k() {
    SPSCRing<int, 512> ring;
    constexpr int N = 5000;
    std::vector<int> received;
    received.reserve(N);
    std::thread producer([&]{
        for (int i = 0; i < N; ++i) while (!ring.push_copy(i)) {}
    });
    std::thread consumer([&]{
        int count = 0;
        while (count < N) {
            auto v = ring.pop();
            if (v) { received.push_back(*v); ++count; }
        }
    });
    producer.join(); consumer.join();
    STREAM_CHECK(static_cast<int>(received.size()) == N);
    for (int i = 0; i < N; ++i) STREAM_CHECK(received[i] == i);
}

static void test_ring_concurrent_tick_pipeline() {
    SPSCRing<OHLCVTick, 256> ring;
    constexpr int N = 200;
    std::atomic<double> sum_in{0}, sum_out{0};
    std::thread producer([&]{
        for (int i = 0; i < N; ++i) {
            OHLCVTick t;
            t.open=100.0; t.high=110.0; t.low=90.0;
            t.close=100.0+i; t.volume=1000.0; t.timestamp_ns=1000LL+i;
            sum_in.fetch_add(t.close, std::memory_order_relaxed);
            while (!ring.push(std::move(t))) {}
        }
    });
    std::thread consumer([&]{
        int c = 0;
        while (c < N) {
            auto v = ring.pop();
            if (v) { sum_out.fetch_add(v->close, std::memory_order_relaxed); ++c; }
        }
    });
    producer.join(); consumer.join();
    STREAM_CHECK_NEAR(sum_in.load(), sum_out.load(), 1e-6);
}

// ---------------------------------------------------------------------------
// OHLCVTick validation
// ---------------------------------------------------------------------------

static void test_tick_minimum_valid() {
    OHLCVTick t;
    t.open=t.high=t.low=t.close=0.001; t.volume=0.001; t.timestamp_ns=1;
    STREAM_CHECK(tick_is_valid(t));
}

static void test_tick_large_valid() {
    OHLCVTick t;
    t.open=1e12; t.high=1.1e12; t.low=0.9e12; t.close=1e12;
    t.volume=1e15; t.timestamp_ns=std::numeric_limits<int64_t>::max();
    STREAM_CHECK(tick_is_valid(t));
}

static void test_tick_nan_open_invalid() {
    OHLCVTick t;
    t.open=std::numeric_limits<double>::quiet_NaN();
    t.high=110.0; t.low=90.0; t.close=100.0;
    t.volume=1000.0; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_nan_high_invalid() {
    OHLCVTick t;
    t.open=100.0; t.high=std::numeric_limits<double>::quiet_NaN();
    t.low=90.0; t.close=100.0; t.volume=1000.0; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_nan_low_invalid() {
    OHLCVTick t;
    t.open=100.0; t.high=110.0;
    t.low=std::numeric_limits<double>::quiet_NaN(); t.close=100.0;
    t.volume=1000.0; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_inf_open_invalid() {
    OHLCVTick t;
    t.open=std::numeric_limits<double>::infinity();
    t.high=110.0; t.low=90.0; t.close=100.0;
    t.volume=1000.0; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_negative_price_invalid() {
    OHLCVTick t;
    t.open=-1.0; t.high=110.0; t.low=-1.0; t.close=100.0;
    t.volume=1000.0; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_zero_close_invalid() {
    OHLCVTick t;
    t.open=t.high=t.low=t.close=0.0; t.volume=1000.0; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_zero_volume_valid() {
    // volume >= 0 is valid per tick.hpp invariants
    OHLCVTick t;
    t.open=100.0; t.high=110.0; t.low=90.0; t.close=100.0;
    t.volume=0.0; t.timestamp_ns=1;
    STREAM_CHECK(tick_is_valid(t));
}

static void test_tick_negative_volume_invalid() {
    OHLCVTick t;
    t.open=100.0; t.high=110.0; t.low=90.0; t.close=100.0;
    t.volume=-100.0; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_negative_timestamp_invalid() {
    OHLCVTick t;
    t.open=100.0; t.high=110.0; t.low=90.0; t.close=100.0;
    t.volume=1000.0; t.timestamp_ns=-1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_doji_valid() {
    OHLCVTick t;
    t.open=t.high=t.low=t.close=100.0; t.volume=500.0; t.timestamp_ns=1;
    STREAM_CHECK(tick_is_valid(t));
}

static void test_tick_random_valid_batch() {
    std::mt19937 rng(0xFACE);
    std::uniform_real_distribution<double> price(1.0, 1000.0);
    std::uniform_real_distribution<double> vol(1.0, 1e8);
    int count = 0;
    for (int i = 0; i < 100; ++i) {
        double lo = price(rng);
        double hi = lo + price(rng) * 0.01;
        double cl = lo + (hi - lo) * 0.5;
        OHLCVTick t;
        t.open=cl; t.high=hi; t.low=lo; t.close=cl;
        t.volume=vol(rng); t.timestamp_ns=1+i;
        if (tick_is_valid(t)) ++count;
    }
    STREAM_CHECK(count == 100);
}

static void test_tick_default_constructed_invalid() {
    OHLCVTick t;
    STREAM_CHECK(!tick_is_valid(t));
}

// ---------------------------------------------------------------------------
// Ring power-of-two
// ---------------------------------------------------------------------------

static void test_ring_sizes_power_of_two() {
    SPSCRing<int,2>    r2;    STREAM_CHECK(r2.capacity()   == std::size_t{1});
    SPSCRing<int,4>    r4;    STREAM_CHECK(r4.capacity()   == std::size_t{3});
    SPSCRing<int,8>    r8;    STREAM_CHECK(r8.capacity()   == std::size_t{7});
    SPSCRing<int,16>   r16;   STREAM_CHECK(r16.capacity()  == std::size_t{15});
    SPSCRing<int,32>   r32;   STREAM_CHECK(r32.capacity()  == std::size_t{31});
    SPSCRing<int,64>   r64;   STREAM_CHECK(r64.capacity()  == std::size_t{63});
    SPSCRing<int,128>  r128;  STREAM_CHECK(r128.capacity() == std::size_t{127});
    SPSCRing<int,256>  r256;  STREAM_CHECK(r256.capacity() == std::size_t{255});
    SPSCRing<int,512>  r512;  STREAM_CHECK(r512.capacity() == std::size_t{511});
    SPSCRing<int,1024> r1024; STREAM_CHECK(r1024.capacity()== std::size_t{1023});
    (void)r2;(void)r4;(void)r8;(void)r16;(void)r32;
    (void)r64;(void)r128;(void)r256;(void)r512;(void)r1024;
}

static void test_ring_mask_wrap() {
    SPSCRing<int, 8> ring;
    for (int i = 0; i < 7; ++i) STREAM_CHECK(ring.push_copy(i));
    for (int i = 0; i < 7; ++i) (void)ring.pop();
    for (int i = 10; i < 17; ++i) STREAM_CHECK(ring.push_copy(i));
    for (int i = 10; i < 17; ++i) {
        auto v = ring.pop(); STREAM_HAS_VALUE(v); STREAM_CHECK(*v == i);
    }
}

static void test_ring_repeated_wrap_cycles() {
    SPSCRing<int, 8> ring;
    for (int cycle = 0; cycle < 10; ++cycle) {
        for (int i = 0; i < 7; ++i) STREAM_CHECK(ring.push(cycle*10+i));
        for (int i = 0; i < 7; ++i) {
            auto v = ring.pop(); STREAM_HAS_VALUE(v); STREAM_CHECK(*v == cycle*10+i);
        }
    }
}

static void test_ring_full_after_capacity_pushes() {
    SPSCRing<int, 16> ring;
    for (int i = 0; i < 15; ++i) STREAM_CHECK(ring.push_copy(i));
    STREAM_CHECK(ring.full_approx());
    STREAM_CHECK(!ring.push(99));
}

static void test_ring_not_full_after_one_pop() {
    SPSCRing<int, 16> ring;
    for (int i = 0; i < 15; ++i) STREAM_CHECK(ring.push_copy(i));
    (void)ring.pop();
    STREAM_CHECK(!ring.full_approx());
    STREAM_CHECK(ring.push(999));
}

static void test_ring_push_after_partial_drain() {
    SPSCRing<int, 8> ring;
    for (int i = 0; i < 7; ++i) STREAM_CHECK(ring.push_copy(i));
    for (int i = 0; i < 3; ++i) (void)ring.pop();
    for (int i = 0; i < 3; ++i) STREAM_CHECK(ring.push(100+i));
    for (int i = 3; i < 7; ++i) {
        auto v = ring.pop(); STREAM_HAS_VALUE(v); STREAM_CHECK(*v == i);
    }
    for (int i = 0; i < 3; ++i) {
        auto v = ring.pop(); STREAM_HAS_VALUE(v); STREAM_CHECK(*v == 100+i);
    }
    STREAM_NO_VALUE(ring.pop());
}

static void test_ring_struct_payload_preserved() {
    struct Payload { int a; double b; char c; };
    SPSCRing<Payload, 8> ring;
    STREAM_CHECK(ring.push({42, 3.14, 'X'}));
    auto v = ring.pop();
    STREAM_HAS_VALUE(v);
    STREAM_CHECK(v->a == 42);
    STREAM_CHECK_NEAR(v->b, 3.14, 1e-12);
    STREAM_CHECK(v->c == 'X');
}

static void test_ring_concurrent_empty_after_drain() {
    SPSCRing<int, 256> ring;
    constexpr int N = 200;
    std::thread producer([&]{ for (int i=0;i<N;++i) while(!ring.push_copy(i)){} });
    std::thread consumer([&]{ int c=0; while(c<N){if(ring.pop()) ++c;} });
    producer.join(); consumer.join();
    STREAM_CHECK(ring.empty_approx());
    STREAM_CHECK(ring.size_approx() == std::size_t{0});
}

static void test_ring_concurrent_sum_2k() {
    SPSCRing<int, 512> ring;
    constexpr int N = 2000;
    std::atomic<long long> in{0}, out{0};
    std::thread prod([&]{
        for (int i=1;i<=N;++i) {
            in.fetch_add(i, std::memory_order_relaxed);
            while (!ring.push_copy(i)) {}
        }
    });
    std::thread cons([&]{
        int c=0;
        while (c<N) { auto v=ring.pop(); if(v){ out.fetch_add(*v,std::memory_order_relaxed); ++c; } }
    });
    prod.join(); cons.join();
    STREAM_CHECK(in.load() == out.load());
}

static void test_ring_full_flag_accurate() {
    SPSCRing<int, 4> ring;
    STREAM_CHECK(!ring.full_approx());
    STREAM_CHECK(ring.push(1)); STREAM_CHECK(ring.push(2)); STREAM_CHECK(ring.push(3));
    STREAM_CHECK(ring.full_approx());  // capacity = 3
    (void)ring.pop();
    STREAM_CHECK(!ring.full_approx());
}

static void test_tick_nan_all_fields() {
    double nan = std::numeric_limits<double>::quiet_NaN();
    OHLCVTick t;
    t.open=nan; t.high=nan; t.low=nan; t.close=nan;
    t.volume=nan; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_ring_pop_mid_fill() {
    // Fill halfway, pop all, fill again, verify FIFO intact
    SPSCRing<int, 8> ring;
    for (int i = 0; i < 3; ++i) STREAM_CHECK(ring.push_copy(i));
    for (int i = 0; i < 3; ++i) {
        auto v = ring.pop(); STREAM_HAS_VALUE(v); STREAM_CHECK(*v == i);
    }
    for (int i = 10; i < 17; ++i) STREAM_CHECK(ring.push_copy(i));
    for (int i = 10; i < 17; ++i) {
        auto v = ring.pop(); STREAM_HAS_VALUE(v); STREAM_CHECK(*v == i);
    }
}

static void test_ring_interleaved_200_ops() {
    // Push 2, pop 1 repeatedly — tests steady-state behaviour
    SPSCRing<int, 8> ring;
    int push_val = 0, pop_expected = 0;
    for (int i = 0; i < 200; ++i) {
        if (!ring.full_approx()) { int pv = push_val++; STREAM_CHECK(ring.push_copy(pv)); }
        if (!ring.empty_approx()) {
            auto v = ring.pop();
            if (v) { STREAM_CHECK(*v == pop_expected++); }
        }
    }
    // Drain remaining
    while (!ring.empty_approx()) {
        auto v = ring.pop();
        if (v) pop_expected++;
    }
    STREAM_CHECK(ring.empty_approx());
}

static void test_ring_double_capacity_size_32() {
    SPSCRing<int, 32> ring;
    STREAM_CHECK(ring.capacity() == std::size_t{31});
    for (int i = 0; i < 31; ++i) STREAM_CHECK(ring.push_copy(i));
    STREAM_CHECK(!ring.push(999));
    for (int i = 0; i < 31; ++i) {
        auto v = ring.pop(); STREAM_HAS_VALUE(v); STREAM_CHECK(*v == i);
    }
    STREAM_CHECK(ring.empty_approx());
}

static void test_ring_many_wrap_cycles_20() {
    SPSCRing<int, 8> ring;
    for (int cycle = 0; cycle < 20; ++cycle) {
        for (int i = 0; i < 7; ++i) STREAM_CHECK(ring.push(cycle * 100 + i));
        for (int i = 0; i < 7; ++i) {
            auto v = ring.pop();
            STREAM_HAS_VALUE(v);
            STREAM_CHECK(*v == cycle * 100 + i);
        }
    }
}

static void test_tick_all_ohlc_equal_valid() {
    OHLCVTick t;
    t.open = t.high = t.low = t.close = 50.0;
    t.volume = 1.0; t.timestamp_ns = 42;
    STREAM_CHECK(tick_is_valid(t));
}

static void test_tick_inf_volume_invalid() {
    OHLCVTick t;
    t.open=100.0; t.high=110.0; t.low=90.0; t.close=100.0;
    t.volume=std::numeric_limits<double>::infinity(); t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_tick_open_above_high_invalid() {
    OHLCVTick t;
    t.open=120.0; t.high=110.0; t.low=90.0; t.close=100.0;
    t.volume=1000.0; t.timestamp_ns=1;
    STREAM_CHECK(!tick_is_valid(t));
}

static void test_ring_size_1_after_push() {
    SPSCRing<int, 8> ring;
    STREAM_CHECK(ring.push(77));
    STREAM_CHECK(ring.size_approx() == std::size_t{1});
}

static void test_ring_size_0_after_pop_all() {
    SPSCRing<int, 8> ring;
    for (int i = 0; i < 5; ++i) STREAM_CHECK(ring.push_copy(i));
    for (int i = 0; i < 5; ++i) (void)ring.pop();
    STREAM_CHECK(ring.size_approx() == std::size_t{0});
}

static void test_ring_large_fill_verify_order() {
    SPSCRing<int, 512> ring;
    for (int i = 0; i < 511; ++i) { int v = i; STREAM_CHECK(ring.push(std::move(v))); }
    for (int i = 0; i < 511; ++i) {
        auto v = ring.pop(); STREAM_HAS_VALUE(v); STREAM_CHECK(*v == i);
    }
}

static void test_ring_push_copy_src_unchanged() {
    SPSCRing<std::vector<int>, 4> ring;
    std::vector<int> src = {10, 20, 30};
    STREAM_CHECK(ring.push_copy(src));
    // Modifying pop'd value should not affect src
    auto v = ring.pop();
    STREAM_HAS_VALUE(v);
    (*v)[0] = 999;
    STREAM_CHECK(src[0] == 10);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::printf("SRFM Stream — SPSC Stress Tests\n");
    std::printf("================================\n");

    STREAM_SUITE("ring size=2 push/pop",            test_ring_size_2_push_pop);
    STREAM_SUITE("ring size=4 boundary",             test_ring_size_4_boundary);
    STREAM_SUITE("ring size=64 fill/drain",          test_ring_size_64_fill_and_drain);
    STREAM_SUITE("ring size=128 FIFO",               test_ring_size_128_fifo_order);
    STREAM_SUITE("ring capacity = size-1",           test_ring_capacity_is_size_minus_1);
    STREAM_SUITE("ring alternating push/pop",        test_ring_alternating_push_pop);
    STREAM_SUITE("ring push_copy semantics",         test_ring_push_copy_semantics);
    STREAM_SUITE("ring move unique_ptr",             test_ring_move_semantics_unique_ptr);
    STREAM_SUITE("ring empty initially",             test_ring_empty_approx_initially);
    STREAM_SUITE("ring size_approx increments",      test_ring_size_approx_increments);
    STREAM_SUITE("ring size_approx decrements",      test_ring_size_approx_decrements);
    STREAM_SUITE("ring pop empty = nullopt",         test_ring_pop_empty_nullopt);
    STREAM_SUITE("ring fill/drain/refill",           test_ring_double_fill_drain_refill);
    STREAM_SUITE("ring OHLCVTick round-trip",        test_ring_ohlcv_tick_round_trip);
    STREAM_SUITE("ring 20 tick round-trips",         test_ring_multiple_tick_round_trips);
    STREAM_SUITE("ring concurrent 1K",               test_ring_concurrent_1k_elements);
    STREAM_SUITE("ring concurrent no-loss 5K",       test_ring_concurrent_no_loss_5k);
    STREAM_SUITE("ring concurrent tick pipeline",    test_ring_concurrent_tick_pipeline);
    STREAM_SUITE("tick minimum valid",               test_tick_minimum_valid);
    STREAM_SUITE("tick large valid",                 test_tick_large_valid);
    STREAM_SUITE("tick NaN open invalid",            test_tick_nan_open_invalid);
    STREAM_SUITE("tick NaN high invalid",            test_tick_nan_high_invalid);
    STREAM_SUITE("tick NaN low invalid",             test_tick_nan_low_invalid);
    STREAM_SUITE("tick Inf open invalid",            test_tick_inf_open_invalid);
    STREAM_SUITE("tick neg price invalid",           test_tick_negative_price_invalid);
    STREAM_SUITE("tick zero close invalid",          test_tick_zero_close_invalid);
    STREAM_SUITE("tick zero volume valid",           test_tick_zero_volume_valid);
    STREAM_SUITE("tick neg volume invalid",          test_tick_negative_volume_invalid);
    STREAM_SUITE("tick neg timestamp invalid",       test_tick_negative_timestamp_invalid);
    STREAM_SUITE("tick doji valid",                  test_tick_doji_valid);
    STREAM_SUITE("tick random 100 valid",            test_tick_random_valid_batch);
    STREAM_SUITE("tick default invalid",             test_tick_default_constructed_invalid);
    STREAM_SUITE("ring powers-of-two",               test_ring_sizes_power_of_two);
    STREAM_SUITE("ring mask wrap",                   test_ring_mask_wrap);
    STREAM_SUITE("ring 10 wrap cycles",              test_ring_repeated_wrap_cycles);
    STREAM_SUITE("ring full after capacity",         test_ring_full_after_capacity_pushes);
    STREAM_SUITE("ring not full after pop",          test_ring_not_full_after_one_pop);
    STREAM_SUITE("ring push after partial drain",    test_ring_push_after_partial_drain);
    STREAM_SUITE("ring struct payload",              test_ring_struct_payload_preserved);
    STREAM_SUITE("ring concurrent empty after",      test_ring_concurrent_empty_after_drain);
    STREAM_SUITE("ring concurrent sum 2K",           test_ring_concurrent_sum_2k);
    STREAM_SUITE("ring interleaved 200 ops",         test_ring_interleaved_200_ops);
    STREAM_SUITE("ring double capacity size",        test_ring_double_capacity_size_32);
    STREAM_SUITE("ring many wrap cycles 20",         test_ring_many_wrap_cycles_20);
    STREAM_SUITE("tick ohlc equal valid",            test_tick_all_ohlc_equal_valid);
    STREAM_SUITE("tick inf volume invalid",          test_tick_inf_volume_invalid);
    STREAM_SUITE("tick open above high invalid",     test_tick_open_above_high_invalid);
    STREAM_SUITE("ring size_approx after 1 push",    test_ring_size_1_after_push);
    STREAM_SUITE("ring size_approx after pop all",   test_ring_size_0_after_pop_all);
    STREAM_SUITE("ring large fill verify order",     test_ring_large_fill_verify_order);
    STREAM_SUITE("ring push_copy then modify src",   test_ring_push_copy_src_unchanged);
    STREAM_SUITE("ring full flag accurate",          test_ring_full_flag_accurate);
    STREAM_SUITE("tick nan all fields invalid",      test_tick_nan_all_fields);
    STREAM_SUITE("ring 3 cycles pop mid-fill",       test_ring_pop_mid_fill);

    return stream_test::report();
}
