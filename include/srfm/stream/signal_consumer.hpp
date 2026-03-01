#pragma once
/**
 * @file  signal_consumer.hpp
 * @brief SignalConsumer — JSON serialisation thread for relativistic signals.
 *
 * Module:  include/srfm/stream/
 * Owner:   AGT-10  (Builder)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Pop StreamRelativisticSignal objects from the output ring and write one
 * JSON line per signal to a configurable output stream (default: stdout).
 * Flush the output every FLUSH_INTERVAL signals to amortise syscall cost.
 *
 * Output format (one line per signal):
 * @code
 *   {"bar": 42, "beta": 0.4200, "gamma": 1.2309, "regime": "TIMELIKE", "signal": 0.8700}
 * @endcode
 *
 * Guarantees
 * ----------
 *   • Batched I/O: flushes every FLUSH_INTERVAL (default 100) signals, not per signal.
 *   • Non-throwing: the serialisation loop never throws.
 *   • Final flush: stop() performs a final flush before joining the thread.
 *   • Configurable sink: constructor accepts any FILE* for testability.
 *
 * NOT Responsible For
 * -------------------
 *   • Signal computation  (SignalProcessor)
 *   • Routing or fanout   (one sink only)
 */

#include "spsc_ring.hpp"
#include "stream_signal.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>

namespace srfm::stream {

// ── SignalConsumerCounters ────────────────────────────────────────────────────

/**
 * @brief Diagnostic counters for the signal consumer thread.
 */
struct SignalConsumerCounters {
    std::uint64_t signals_consumed{0};  ///< Total signals written to output.
    std::uint64_t flush_count{0};       ///< Number of explicit flushes performed.
};

// ── SignalConsumer ────────────────────────────────────────────────────────────

/**
 * @brief JSON output thread owner.  Non-copyable, non-movable.
 *
 * @code
 *   SPSCRing<StreamRelativisticSignal, 65536> out_ring;
 *   SignalConsumer consumer{out_ring};      // writes to stdout
 *   consumer.start();
 *   // ...
 *   consumer.stop();
 * @endcode
 */
class SignalConsumer {
public:
    static constexpr std::size_t RING_SIZE     = 65536;
    static constexpr std::size_t FLUSH_INTERVAL = 100;  ///< Flush every N signals.

    using Ring = SPSCRing<StreamRelativisticSignal, RING_SIZE>;

    /**
     * @brief Construct bound to the output ring and an I/O sink.
     *
     * @param ring  Ring populated by SignalProcessor (non-owning ref).
     * @param sink  Output stream (default: stdout).  Must remain valid for the
     *              lifetime of this SignalConsumer.
     */
    explicit SignalConsumer(Ring& ring, FILE* sink = stdout) noexcept
        : ring_{ring}, sink_{sink}
    {}

    ~SignalConsumer() noexcept { stop(); }

    SignalConsumer(const SignalConsumer&)            = delete;
    SignalConsumer& operator=(const SignalConsumer&) = delete;
    SignalConsumer(SignalConsumer&&)                 = delete;
    SignalConsumer& operator=(SignalConsumer&&)      = delete;

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    /// Launch the consumer thread.  Idempotent.
    void start() noexcept;

    /// Stop the consumer thread and perform a final flush.  Idempotent.
    void stop() noexcept;

    /// Whether the consumer thread is currently running.
    [[nodiscard]] bool running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    // ── Single-signal serialisation (usable from test without threading) ───────

    /**
     * @brief Serialise one signal to the sink immediately.
     *
     * Does not flush.  Useful in unit tests.
     *
     * @param sig  Signal to serialise.
     */
    void write_one(const StreamRelativisticSignal& sig) noexcept;

    /**
     * @brief Flush the output sink.
     *
     * Calls std::fflush(sink_).  noexcept.
     */
    void flush() noexcept { std::fflush(sink_); }

    // ── Accessors ──────────────────────────────────────────────────────────────

    [[nodiscard]] SignalConsumerCounters counters() const noexcept {
        return counters_;
    }

    [[nodiscard]] FILE* sink() const noexcept { return sink_; }

private:
    void run_loop() noexcept;

    Ring&                    ring_;
    FILE*                    sink_;

    std::atomic<bool>        running_{false};
    std::atomic<bool>        stop_requested_{false};
    std::thread              thread_;
    SignalConsumerCounters   counters_{};
};

} // namespace srfm::stream
