/**
 * @file  stream_engine.cpp
 * @brief StreamEngine implementation — pipeline lifecycle and inject() path.
 *
 * See include/srfm/engine/stream_engine.hpp for the full module contract.
 *
 * Startup / shutdown order
 * ------------------------
 * start() launches threads in data-flow order:
 *   1. TickIngester   (produces to Ring1)
 *   2. SignalProcessor (consumes Ring1, produces Ring2)
 *   3. SignalConsumer  (consumes Ring2)
 *
 * stop() signals threads in the same order and joins them.  This ensures
 * in-flight ticks are processed before the downstream consumers exit.
 *
 * inject() path
 * -------------
 * inject() validates the tick (tick_is_valid) and pushes directly into Ring1.
 * It does not interact with the TickIngester's loop or its counters — it is
 * a separate producer path intended for tests.
 *
 * When the TickIngester is also running with a live TickSource, the caller
 * MUST NOT mix inject() calls with TickSource production (SPSC constraint:
 * only one producer thread at a time).
 */

#include "../../include/srfm/engine/stream_engine.hpp"

namespace srfm::engine {

// ── StreamEngine::start ───────────────────────────────────────────────────────

void StreamEngine::start() noexcept {
    if (running_.load(std::memory_order_acquire)) return;

    ingester_.start();
    processor_.start();
    consumer_.start();

    running_.store(true, std::memory_order_release);
}

// ── StreamEngine::stop ────────────────────────────────────────────────────────

void StreamEngine::stop() noexcept {
    if (!running_.load(std::memory_order_acquire)) return;

    // Stop in data-flow order: cut off input first, then let downstream drain.
    ingester_.stop();
    processor_.stop();
    consumer_.stop();

    running_.store(false, std::memory_order_release);
}

// ── StreamEngine::inject ──────────────────────────────────────────────────────

bool StreamEngine::inject(stream::OHLCVTick tick) noexcept {
    if (!stream::tick_is_valid(tick)) {
        ++inject_dropped_invalid_;
        return false;
    }
    if (!ring1_.push(std::move(tick))) {
        ++inject_dropped_ring_full_;
        return false;
    }
    ++inject_pushed_;
    return true;
}

// ── StreamEngine::counters ────────────────────────────────────────────────────

stream::TickIngesterCounters
StreamEngine::ingester_counters() const noexcept {
    return ingester_.counters();
}

stream::SignalProcessorCounters
StreamEngine::processor_counters() const noexcept {
    return processor_.counters();
}

stream::SignalConsumerCounters
StreamEngine::consumer_counters() const noexcept {
    return consumer_.counters();
}

// ── StreamEngine::process_sync ────────────────────────────────────────────────

stream::StreamRelativisticSignal
StreamEngine::process_sync(const stream::OHLCVTick& tick,
                            std::int64_t bar_index) noexcept {
    return processor_.process_one(tick, bar_index);
}

// ── StreamEngine::reset_processor_state ──────────────────────────────────────

void StreamEngine::reset_processor_state() noexcept {
    processor_.reset_state();
}

} // namespace srfm::engine
