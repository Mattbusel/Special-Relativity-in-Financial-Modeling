#pragma once
/**
 * @file  tick_ingester.hpp
 * @brief TickIngester — ingestion thread that reads, validates, and ring-pushes ticks.
 *
 * Module:  include/srfm/stream/
 * Owner:   AGT-10  (Builder)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Run the ingestion thread: repeatedly call source.read(), validate each tick
 * with tick_is_valid(), push valid ticks into SPSCRing<OHLCVTick, 65536>, and
 * count dropped ticks by rejection category.
 *
 * Guarantees
 * ----------
 *   • Never blocks the calling thread (start() launches a std::thread).
 *   • Never throws from the hot path (noexcept read/validate/push loop).
 *   • Counts all dropped ticks; drops are never silently lost.
 *   • Graceful shutdown: stop() signals the thread and join()s within the
 *     destructor if the caller forgets to call stop().
 *
 * NOT Responsible For
 * -------------------
 *   • Sourcing ticks     (TickSource)
 *   • Signal processing  (SignalProcessor)
 */

#include "spsc_ring.hpp"
#include "tick.hpp"
#include "tick_source.hpp"

#include <atomic>
#include <cstdint>
#include <thread>

namespace srfm::stream {

// ── TickIngesterCounters ──────────────────────────────────────────────────────

/**
 * @brief Diagnostic counters for the ingestion thread.
 *
 * All fields are updated by the ingestion thread only.  Reads from external
 * threads are approximate (no synchronisation guarantee beyond coherence).
 */
struct TickIngesterCounters {
    std::uint64_t ticks_received{0};  ///< Raw ticks read from source.
    std::uint64_t ticks_pushed{0};    ///< Valid ticks pushed to ring.
    std::uint64_t ticks_dropped_invalid{0}; ///< Dropped: failed tick_is_valid().
    std::uint64_t ticks_dropped_ring_full{0}; ///< Dropped: ring was full.
};

// ── TickIngester ──────────────────────────────────────────────────────────────

/**
 * @brief Ingestion thread owner.  Non-copyable, non-movable.
 *
 * @code
 *   SPSCRing<OHLCVTick, 65536> ring;
 *   QueueTickSource             src;
 *   TickIngester                ingester{src, ring};
 *   ingester.start();
 *   src.push(make_valid_tick());
 *   // ... let it run ...
 *   ingester.stop();
 * @endcode
 */
class TickIngester {
public:
    static constexpr std::size_t RING_SIZE = 65536;
    using Ring = SPSCRing<OHLCVTick, RING_SIZE>;

    /**
     * @brief Construct bound to a source and a ring buffer.
     *
     * Both @p source and @p ring must outlive this TickIngester.
     *
     * @param source  Tick source to read from (non-owning reference).
     * @param ring    Ring buffer to push valid ticks into (non-owning reference).
     */
    TickIngester(TickSource& source, Ring& ring) noexcept
        : source_{source}, ring_{ring}
    {}

    /// Destructor: ensures the thread is stopped and joined.
    ~TickIngester() noexcept { stop(); }

    // Not copyable or movable — owns a thread handle.
    TickIngester(const TickIngester&)            = delete;
    TickIngester& operator=(const TickIngester&) = delete;
    TickIngester(TickIngester&&)                 = delete;
    TickIngester& operator=(TickIngester&&)      = delete;

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    /**
     * @brief Launch the ingestion thread.
     *
     * Idempotent: a second call while running is a no-op.
     */
    void start() noexcept;

    /**
     * @brief Signal the ingestion thread to stop and wait for it to finish.
     *
     * Idempotent: safe to call multiple times.
     */
    void stop() noexcept;

    /// Whether the ingestion thread is currently running.
    [[nodiscard]] bool running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    // ── Diagnostic accessors ───────────────────────────────────────────────────

    /**
     * @brief Snapshot of ingestion counters.
     *
     * Non-atomic read — approximate under concurrency.  Safe for monitoring.
     */
    [[nodiscard]] TickIngesterCounters counters() const noexcept {
        return counters_;
    }

private:
    void run_loop() noexcept;

    TickSource& source_;
    Ring&       ring_;

    std::atomic<bool>       running_{false};
    std::atomic<bool>       stop_requested_{false};
    std::thread             thread_;
    TickIngesterCounters    counters_{};
};

} // namespace srfm::stream
