/**
 * @file  signal_consumer.cpp
 * @brief SignalConsumer implementation — JSON serialisation and batched I/O.
 *
 * See include/srfm/stream/signal_consumer.hpp for the full module contract.
 *
 * JSON format (one line per signal)
 * ----------------------------------
 * {"bar": 0, "beta": 0.0000, "gamma": 1.0000, "regime": "LIGHTLIKE", "signal": 0.0000}
 *
 * - All floating-point fields printed with 4 decimal places.
 * - No trailing comma.
 * - Fields always appear in the order: bar, beta, gamma, regime, signal.
 *
 * I/O batching
 * ------------
 * std::fflush() is expensive.  We accumulate FLUSH_INTERVAL signals in the
 * internal buffer (backed by the OS I/O buffer) before flushing.  On stop(),
 * we drain the ring and perform a final flush to ensure no signals are lost.
 *
 * Output correctness
 * ------------------
 * std::fprintf() is used rather than std::cout to avoid locale effects and
 * C++ stream synchronisation overhead.  The format string is compile-time
 * constant; no dynamic allocation occurs in write_one().
 */

#include "../../include/srfm/stream/signal_consumer.hpp"

#include <cstdio>
#include <thread>

namespace srfm::stream {

// ── SignalConsumer::start ─────────────────────────────────────────────────────

void SignalConsumer::start() noexcept {
    if (running_.load(std::memory_order_acquire)) return;

    stop_requested_.store(false, std::memory_order_release);
    running_.store(true,  std::memory_order_release);

    thread_ = std::thread([this]() noexcept { run_loop(); });
}

// ── SignalConsumer::stop ──────────────────────────────────────────────────────

void SignalConsumer::stop() noexcept {
    stop_requested_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
    // Final flush to ensure no buffered output is lost.
    flush();
    running_.store(false, std::memory_order_release);
}

// ── SignalConsumer::write_one ─────────────────────────────────────────────────

void SignalConsumer::write_one(const StreamRelativisticSignal& sig) noexcept {
    std::fprintf(sink_,
        "{\"bar\": %lld, \"beta\": %.4f, \"gamma\": %.4f,"
        " \"regime\": \"%s\", \"signal\": %.4f}\n",
        static_cast<long long>(sig.bar),
        sig.beta,
        sig.gamma,
        regime_to_str(sig.regime),
        sig.signal);
}

// ── SignalConsumer::run_loop ──────────────────────────────────────────────────

void SignalConsumer::run_loop() noexcept {
    std::size_t since_flush = 0;

    while (!stop_requested_.load(std::memory_order_acquire)) {

        auto maybe_sig = ring_.pop();

        if (!maybe_sig.has_value()) {
            // Flush pending output if we're idle and have buffered data.
            if (since_flush > 0) {
                flush();
                ++counters_.flush_count;
                since_flush = 0;
            }
            std::this_thread::yield();
            continue;
        }

        write_one(*maybe_sig);
        ++counters_.signals_consumed;
        ++since_flush;

        if (since_flush >= FLUSH_INTERVAL) {
            flush();
            ++counters_.flush_count;
            since_flush = 0;
        }
    }

    // Drain any remaining signals that arrived before stop was signalled.
    while (true) {
        auto maybe_sig = ring_.pop();
        if (!maybe_sig.has_value()) break;
        write_one(*maybe_sig);
        ++counters_.signals_consumed;
        ++since_flush;
    }

    // Final flush.
    if (since_flush > 0) {
        flush();
        ++counters_.flush_count;
    }

    running_.store(false, std::memory_order_release);
}

} // namespace srfm::stream
