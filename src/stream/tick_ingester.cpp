/**
 * @file  tick_ingester.cpp
 * @brief TickIngester implementation — ingestion thread body.
 *
 * See include/srfm/stream/tick_ingester.hpp for the full module contract.
 *
 * Hot-path design
 * ---------------
 * The inner loop in run_loop() is:
 *   1. Call source_.read()        — returns optional<OHLCVTick> or nullopt
 *   2. If nullopt and source closed → exit loop
 *   3. If nullopt and source open  → yield and continue (spin with backoff)
 *   4. Validate via tick_is_valid()
 *   5. Push to ring1_
 *   6. Update counters
 *
 * All operations are noexcept.  No heap allocation occurs in the loop.
 *
 * Backoff strategy
 * ----------------
 * When source_.read() returns nullopt (no data available), we call
 * std::this_thread::yield() to be cooperative with the OS scheduler rather
 * than burning a core in a tight spin.  A more aggressive variant could use
 * _mm_pause() on x86 or std::this_thread::sleep_for(0ns), but yield() gives
 * a reasonable balance between latency and CPU utilisation for this use-case.
 */

#include "../../include/srfm/stream/tick_ingester.hpp"

#include <thread>

namespace srfm::stream {

// ── TickIngester::start ───────────────────────────────────────────────────────

void TickIngester::start() noexcept {
    // Idempotent: do nothing if already running.
    if (running_.load(std::memory_order_acquire)) return;

    stop_requested_.store(false, std::memory_order_release);
    running_.store(true,  std::memory_order_release);

    thread_ = std::thread([this]() noexcept { run_loop(); });
}

// ── TickIngester::stop ────────────────────────────────────────────────────────

void TickIngester::stop() noexcept {
    // Signal the loop to exit.
    stop_requested_.store(true, std::memory_order_release);

    // Join if the thread is joinable.
    if (thread_.joinable()) {
        thread_.join();
    }

    running_.store(false, std::memory_order_release);
}

// ── TickIngester::run_loop ────────────────────────────────────────────────────

void TickIngester::run_loop() noexcept {
    while (!stop_requested_.load(std::memory_order_acquire)) {

        // ── Read one tick from the source ──────────────────────────────────────
        auto maybe_tick = source_.read();

        if (!maybe_tick.has_value()) {
            if (!source_.is_open()) {
                // Source closed permanently — exit the loop.
                break;
            }
            // Transient: no data yet.  Yield and retry.
            std::this_thread::yield();
            continue;
        }

        ++counters_.ticks_received;

        // ── Validate ──────────────────────────────────────────────────────────
        if (!tick_is_valid(*maybe_tick)) {
            ++counters_.ticks_dropped_invalid;
            continue;
        }

        // ── Push to ring ──────────────────────────────────────────────────────
        if (!ring_.push(std::move(*maybe_tick))) {
            ++counters_.ticks_dropped_ring_full;
            continue;
        }

        ++counters_.ticks_pushed;
    }

    running_.store(false, std::memory_order_release);
}

} // namespace srfm::stream
