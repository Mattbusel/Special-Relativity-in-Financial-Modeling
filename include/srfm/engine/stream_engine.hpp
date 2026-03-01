#pragma once
/**
 * @file  stream_engine.hpp
 * @brief StreamEngine — top-level owner of the lock-free streaming pipeline.
 *
 * Module:  include/srfm/engine/
 * Owner:   AGT-10  (Builder)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Own and coordinate the three threads and two ring buffers that form the
 * streaming pipeline:
 *
 *   TickSource → [TickIngester] → Ring1<OHLCVTick>
 *             → [SignalProcessor] → Ring2<StreamRelativisticSignal>
 *             → [SignalConsumer] → stdout (or configured sink)
 *
 * Provide a test-injection path (inject()) that bypasses the TickSource and
 * pushes ticks directly into Ring1 from the calling thread.
 *
 * Guarantees
 * ----------
 *   • Owns Ring1 and Ring2 (inline storage — no heap allocation for rings).
 *   • start() / stop() are idempotent.
 *   • Graceful shutdown: stop() signals all threads and joins them in order.
 *   • inject() is safe to call from any thread while the engine is running,
 *     provided only one external thread calls inject() (SPSC constraint on Ring1).
 *
 * NOT Responsible For
 * -------------------
 *   • Tick validation     (TickIngester)
 *   • Signal computation  (SignalProcessor)
 *   • JSON formatting     (SignalConsumer)
 *
 * @code
 *   QueueTickSource src;
 *   StreamEngine engine{src};
 *   engine.start();
 *   engine.inject(make_valid_tick());
 *   // let it process...
 *   engine.stop();
 * @endcode
 */

#include "../stream/signal_consumer.hpp"
#include "../stream/signal_processor.hpp"
#include "../stream/spsc_ring.hpp"
#include "../stream/stream_signal.hpp"
#include "../stream/tick.hpp"
#include "../stream/tick_ingester.hpp"
#include "../stream/tick_source.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>

namespace srfm::engine {

// ── StreamEngine ──────────────────────────────────────────────────────────────

/**
 * @brief Top-level streaming pipeline owner.  Non-copyable, non-movable.
 */
class StreamEngine {
public:
    static constexpr std::size_t RING1_SIZE = 65536; ///< Tick ring capacity.
    static constexpr std::size_t RING2_SIZE = 65536; ///< Signal ring capacity.

    using Ring1 = stream::SPSCRing<stream::OHLCVTick, RING1_SIZE>;
    using Ring2 = stream::SPSCRing<stream::StreamRelativisticSignal, RING2_SIZE>;

    /**
     * @brief Construct the engine with a tick source and optional output sink.
     *
     * @param source  Source of OHLCVTick data (non-owning ref).  Callers may
     *                pass a QueueTickSource for testing or a PipeTickSource for
     *                live operation.
     * @param sink    Output sink for JSON lines (default: stdout).
     */
    explicit StreamEngine(stream::TickSource& source,
                          FILE* sink = stdout) noexcept
        : ingester_{source, ring1_}
        , processor_{ring1_, ring2_}
        , consumer_{ring2_, sink}
    {}

    ~StreamEngine() noexcept { stop(); }

    StreamEngine(const StreamEngine&)            = delete;
    StreamEngine& operator=(const StreamEngine&) = delete;
    StreamEngine(StreamEngine&&)                 = delete;
    StreamEngine& operator=(StreamEngine&&)      = delete;

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    /**
     * @brief Start all three pipeline threads.
     *
     * Idempotent.  Order: ingester → processor → consumer.
     */
    void start() noexcept;

    /**
     * @brief Stop all three pipeline threads.
     *
     * Idempotent.  Order: ingester → processor → consumer (drain order).
     * Blocks until all threads have joined.
     */
    void stop() noexcept;

    /// Whether the engine is currently running.
    [[nodiscard]] bool running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    // ── Test injection path ────────────────────────────────────────────────────

    /**
     * @brief Inject a tick directly into Ring1, bypassing the TickSource.
     *
     * The tick is validated by tick_is_valid() before pushing.  Invalid ticks
     * are silently dropped; the drop count is tracked in inject_counters_.
     *
     * @param tick  Tick to inject.
     * @return true  if the tick was pushed successfully.
     * @return false if the tick was invalid or Ring1 was full.
     *
     * @note Thread constraint: only one external thread may call inject()
     *       concurrently (SPSC constraint on Ring1 producer side).  When the
     *       TickIngester thread is also running and reading from a TickSource,
     *       do not mix inject() calls with that source producing ticks.
     */
    [[nodiscard]] bool inject(stream::OHLCVTick tick) noexcept;

    // ── Accessors ──────────────────────────────────────────────────────────────

    [[nodiscard]] stream::TickIngesterCounters    ingester_counters()  const noexcept;
    [[nodiscard]] stream::SignalProcessorCounters processor_counters() const noexcept;
    [[nodiscard]] stream::SignalConsumerCounters  consumer_counters()  const noexcept;

    /// Approximate number of ticks waiting in Ring1.
    [[nodiscard]] std::size_t ring1_size_approx() const noexcept {
        return ring1_.size_approx();
    }

    /// Approximate number of signals waiting in Ring2.
    [[nodiscard]] std::size_t ring2_size_approx() const noexcept {
        return ring2_.size_approx();
    }

    // ── Direct processor access (for synchronous test use) ────────────────────

    /**
     * @brief Process one tick synchronously (no threads, no ring buffers).
     *
     * Calls SignalProcessor::process_one() directly.  Useful for deterministic
     * unit tests that do not need the full thread machinery.
     *
     * @param tick       Validated tick to process.
     * @param bar_index  Sequence number assigned to this tick.
     */
    [[nodiscard]] stream::StreamRelativisticSignal
    process_sync(const stream::OHLCVTick& tick,
                 std::int64_t bar_index) noexcept;

    /**
     * @brief Reset all signal-processor state (for test reuse).
     *
     * Must only be called when the engine is stopped.
     */
    void reset_processor_state() noexcept;

private:
    Ring1 ring1_;   ///< Tick ring: TickIngester → SignalProcessor.
    Ring2 ring2_;   ///< Signal ring: SignalProcessor → SignalConsumer.

    stream::TickIngester    ingester_;
    stream::SignalProcessor processor_;
    stream::SignalConsumer  consumer_;

    std::atomic<bool> running_{false};

    // inject() diagnostic counters.
    std::uint64_t inject_dropped_invalid_{0};
    std::uint64_t inject_dropped_ring_full_{0};
    std::uint64_t inject_pushed_{0};
};

} // namespace srfm::engine
