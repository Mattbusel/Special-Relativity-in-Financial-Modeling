#pragma once
/**
 * @file  signal_processor.hpp
 * @brief SignalProcessor — per-tick relativistic signal computation thread.
 *
 * Module:  include/srfm/stream/
 * Owner:   AGT-10  (Builder)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Pop OHLCVTick objects from the ingestion ring, run each tick through the
 * four-stage signal chain (normalise → β → Lorentz → manifold), compute a
 * StreamRelativisticSignal, and push it into the output ring.
 *
 * Signal chain (per tick)
 * -----------------------
 *   1. CoordinateNormalizer (window=20): z-score(close) → norm_close
 *   2. BetaCalculatorFix3: log-returns(close) → β
 *   3. LorentzTransform: (bar_idx, norm_close, β) → (t', x', γ)
 *   4. SpacetimeManifold: (t', x') → Δs², regime, signal
 *
 * Output StreamRelativisticSignal:
 *   bar    = tick sequence number
 *   beta   = β from BetaCalculatorFix3
 *   gamma  = γ from LorentzTransform
 *   regime = TIMELIKE / LIGHTLIKE / SPACELIKE from SpacetimeManifold
 *   signal = γ * manifold_signal  (Lorentz-scaled)
 *   ds2    = raw Δs² from SpacetimeManifold
 *
 * Guarantees
 * ----------
 *   • No heap allocation in the hot path — all state pre-allocated in ctor.
 *   • Never throws from the processing loop.
 *   • Graceful shutdown via stop().
 *
 * NOT Responsible For
 * -------------------
 *   • Tick validation    (TickIngester)
 *   • JSON serialisation (SignalConsumer)
 */

#include "beta_calculator.hpp"
#include "coordinate_normalizer.hpp"
#include "lorentz_transform.hpp"
#include "spacetime_manifold.hpp"
#include "spsc_ring.hpp"
#include "stream_signal.hpp"
#include "tick.hpp"

#include <atomic>
#include <cstdint>
#include <thread>

namespace srfm::stream {

// ── SignalProcessorCounters ───────────────────────────────────────────────────

/**
 * @brief Diagnostic counters for the signal-processing thread.
 */
struct SignalProcessorCounters {
    std::uint64_t ticks_processed{0};      ///< Ticks popped from input ring.
    std::uint64_t signals_emitted{0};       ///< Signals pushed to output ring.
    std::uint64_t signals_dropped_ring_full{0}; ///< Dropped: output ring full.
    std::uint64_t timelike_count{0};        ///< TIMELIKE regime events.
    std::uint64_t lightlike_count{0};       ///< LIGHTLIKE regime events.
    std::uint64_t spacelike_count{0};       ///< SPACELIKE regime events.
};

// ── SignalProcessor ───────────────────────────────────────────────────────────

/**
 * @brief Signal processing thread owner.  Non-copyable, non-movable.
 *
 * @code
 *   SPSCRing<OHLCVTick, 65536>             in_ring;
 *   SPSCRing<StreamRelativisticSignal, 65536> out_ring;
 *   SignalProcessor proc{in_ring, out_ring};
 *   proc.start();
 *   // ...
 *   proc.stop();
 * @endcode
 */
class SignalProcessor {
public:
    static constexpr std::size_t IN_RING_SIZE  = 65536;
    static constexpr std::size_t OUT_RING_SIZE = 65536;

    using InRing  = SPSCRing<OHLCVTick, IN_RING_SIZE>;
    using OutRing = SPSCRing<StreamRelativisticSignal, OUT_RING_SIZE>;

    static constexpr std::size_t NORMALIZER_WINDOW = 20;

    /**
     * @brief Construct bound to the input and output ring buffers.
     *
     * All signal-chain objects are default-constructed here; no allocation
     * occurs in the processing loop.
     *
     * @param in_ring   Ring populated by TickIngester (non-owning ref).
     * @param out_ring  Ring consumed by SignalConsumer (non-owning ref).
     */
    SignalProcessor(InRing& in_ring, OutRing& out_ring) noexcept
        : in_ring_{in_ring}
        , out_ring_{out_ring}
        , normalizer_{NORMALIZER_WINDOW}
    {}

    ~SignalProcessor() noexcept { stop(); }

    SignalProcessor(const SignalProcessor&)            = delete;
    SignalProcessor& operator=(const SignalProcessor&) = delete;
    SignalProcessor(SignalProcessor&&)                 = delete;
    SignalProcessor& operator=(SignalProcessor&&)      = delete;

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    /// Launch the processing thread.  Idempotent.
    void start() noexcept;

    /// Stop and join the processing thread.  Idempotent.
    void stop() noexcept;

    /// Whether the processing thread is currently running.
    [[nodiscard]] bool running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    // ── Single-tick processing (usable from test without threading) ────────────

    /**
     * @brief Process one tick synchronously and return the resulting signal.
     *
     * Does not push to any ring.  Useful for unit tests.
     *
     * @param tick      Validated tick to process.
     * @param bar_index Sequence number for this tick.
     * @return The computed StreamRelativisticSignal.
     */
    [[nodiscard]] StreamRelativisticSignal process_one(const OHLCVTick& tick,
                                                        std::int64_t bar_index) noexcept;

    // ── Diagnostic accessors ───────────────────────────────────────────────────

    [[nodiscard]] SignalProcessorCounters counters() const noexcept {
        return counters_;
    }

    // ── State reset (for test reuse) ──────────────────────────────────────────

    /**
     * @brief Reset all signal-chain component state.
     *
     * Must only be called when the processing thread is not running.
     */
    void reset_state() noexcept {
        normalizer_.reset();
        beta_calc_.reset();
        manifold_.reset();
        bar_counter_ = 0;
        counters_    = {};
    }

private:
    void run_loop() noexcept;

    InRing&  in_ring_;
    OutRing& out_ring_;

    // Signal-chain components — pre-allocated, zero per-tick allocation.
    CoordinateNormalizer  normalizer_;
    BetaCalculatorFix3    beta_calc_{};
    LorentzTransform      lorentz_{};
    SpacetimeManifold     manifold_{};

    std::int64_t             bar_counter_{0};
    std::atomic<bool>        running_{false};
    std::atomic<bool>        stop_requested_{false};
    std::thread              thread_;
    SignalProcessorCounters  counters_{};
};

} // namespace srfm::stream
