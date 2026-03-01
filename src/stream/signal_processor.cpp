/**
 * @file  signal_processor.cpp
 * @brief SignalProcessor implementation — per-tick signal chain.
 *
 * See include/srfm/stream/signal_processor.hpp for the full module contract.
 *
 * Hot-path design
 * ---------------
 * The inner loop in run_loop() calls process_one() for each tick.
 * process_one() is the pure computation kernel — no ring I/O, no threading.
 * This separation allows unit tests to exercise the signal chain without
 * launching threads.
 *
 * Per-tick allocation budget: zero.
 *   - CoordinateNormalizer uses a pre-allocated std::vector (constructed once).
 *   - BetaCalculatorFix3 uses a std::array<double,3> (stack).
 *   - LorentzTransform is stateless.
 *   - SpacetimeManifold has two doubles + one bool (stack).
 *   - StreamRelativisticSignal is a plain struct (stack).
 *
 * Backoff
 * -------
 * When in_ring_ is empty, yields (same strategy as TickIngester).
 */

#include "../../include/srfm/stream/signal_processor.hpp"

#include <thread>

namespace srfm::stream {

// ── SignalProcessor::start ────────────────────────────────────────────────────

void SignalProcessor::start() noexcept {
    if (running_.load(std::memory_order_acquire)) return;

    stop_requested_.store(false, std::memory_order_release);
    running_.store(true,  std::memory_order_release);

    thread_ = std::thread([this]() noexcept { run_loop(); });
}

// ── SignalProcessor::stop ─────────────────────────────────────────────────────

void SignalProcessor::stop() noexcept {
    stop_requested_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
    running_.store(false, std::memory_order_release);
}

// ── SignalProcessor::process_one ──────────────────────────────────────────────

StreamRelativisticSignal
SignalProcessor::process_one(const OHLCVTick& tick,
                              std::int64_t     bar_index) noexcept {
    // ── Stage 1: CoordinateNormalizer ──────────────────────────────────────────
    // Update the rolling window with the close price first so that normalise()
    // reflects the current bar.
    normalizer_.update(tick.close);
    const double norm_close = normalizer_.normalise(tick.close);

    // ── Stage 2: BetaCalculator ────────────────────────────────────────────────
    beta_calc_.update(tick.close);
    const double beta = beta_calc_.beta(); // 0.0 until warm-up

    // ── Stage 3: LorentzTransform ──────────────────────────────────────────────
    const double t = static_cast<double>(bar_index);
    const double x = norm_close;
    const auto ev  = lorentz_.transform(t, x, beta);

    // ── Stage 4: SpacetimeManifold ─────────────────────────────────────────────
    const auto mr = manifold_.update(ev.t_prime, ev.x_prime);

    // ── Assemble output ────────────────────────────────────────────────────────
    // Final signal = γ * manifold_signal (Lorentz-scaled interval signal).
    const double final_signal = ev.gamma * mr.signal;

    StreamRelativisticSignal out;
    out.bar    = bar_index;
    out.beta   = beta;
    out.gamma  = ev.gamma;
    out.regime = mr.regime;
    out.signal = final_signal;
    out.ds2    = mr.ds2;

    return out;
}

// ── SignalProcessor::run_loop ─────────────────────────────────────────────────

void SignalProcessor::run_loop() noexcept {
    while (!stop_requested_.load(std::memory_order_acquire)) {

        auto maybe_tick = in_ring_.pop();

        if (!maybe_tick.has_value()) {
            std::this_thread::yield();
            continue;
        }

        ++counters_.ticks_processed;

        const auto sig = process_one(*maybe_tick, bar_counter_++);

        // Track regime distribution.
        switch (sig.regime) {
            case Regime::TIMELIKE:  ++counters_.timelike_count;  break;
            case Regime::LIGHTLIKE: ++counters_.lightlike_count; break;
            case Regime::SPACELIKE: ++counters_.spacelike_count; break;
        }

        // Push to output ring.
        StreamRelativisticSignal s = sig; // copy for move
        if (!out_ring_.push(std::move(s))) {
            ++counters_.signals_dropped_ring_full;
            continue;
        }

        ++counters_.signals_emitted;
    }

    running_.store(false, std::memory_order_release);
}

} // namespace srfm::stream
