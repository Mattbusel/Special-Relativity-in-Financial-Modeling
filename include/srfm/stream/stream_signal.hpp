#pragma once
/**
 * @file  stream_signal.hpp
 * @brief StreamRelativisticSignal — output unit of the signal-processing pipeline.
 *
 * Module:  include/srfm/stream/
 * Owner:   AGT-10  (Builder)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Define the canonical output type emitted by SignalProcessor and consumed by
 * SignalConsumer.  Each StreamRelativisticSignal corresponds to one processed
 * OHLCVTick and carries the full relativistic characterisation of that bar.
 *
 * Design Constraints
 * ------------------
 *   • Trivially movable — stored in SPSCRing<StreamRelativisticSignal, 65536>.
 *   • No dynamic allocation — fixed-size plain struct.
 *   • Regime is an enum (not std::string) — string formatting done at output.
 */

#include <cstdint>

namespace srfm::stream {

// ── Spacetime regime classification ───────────────────────────────────────────

/**
 * @brief Spacetime interval regime derived from the Lorentz-transformed
 *        coordinates of consecutive ticks.
 *
 * The invariant spacetime interval is Δs² = Δt'² - Δx'² (c = 1 in
 * normalised units).
 *
 *   TIMELIKE  : Δs² >  ε  — causal connection; signal is meaningful.
 *   LIGHTLIKE : |Δs²| ≤ ε — on the light cone boundary.
 *   SPACELIKE : Δs² < -ε  — acausal separation; signal strength is attenuated.
 */
enum class Regime : std::uint8_t {
    TIMELIKE  = 0,
    LIGHTLIKE = 1,
    SPACELIKE = 2,
};

/**
 * @brief Convert Regime to a null-terminated ASCII string.
 *
 * Returned pointer is to a string literal (static storage).
 */
[[nodiscard]] inline const char* regime_to_str(Regime r) noexcept {
    switch (r) {
        case Regime::TIMELIKE:  return "TIMELIKE";
        case Regime::LIGHTLIKE: return "LIGHTLIKE";
        case Regime::SPACELIKE: return "SPACELIKE";
    }
    return "UNKNOWN";
}

// ── StreamRelativisticSignal ──────────────────────────────────────────────────

/**
 * @brief Fully-characterised relativistic signal for one processed tick.
 *
 * Emitted by SignalProcessor into the output SPSCRing, consumed by
 * SignalConsumer for JSON serialisation.
 *
 * Fields
 * ------
 *   bar    — Monotonically increasing tick sequence number (0-based).
 *   beta   — Normalised market velocity β ∈ (-BETA_MAX_SAFE, BETA_MAX_SAFE).
 *   gamma  — Lorentz factor γ = 1/√(1−β²) ≥ 1.0.
 *   regime — Spacetime interval classification of this bar.
 *   signal — Relativistic-adjusted signal value: γ · m_eff · normalised_close.
 *   ds2    — Raw spacetime interval Δs² (diagnostic; not in JSON output).
 */
struct StreamRelativisticSignal {
    std::int64_t bar{0};           ///< Bar sequence number.
    double       beta{0.0};        ///< Market velocity β.
    double       gamma{1.0};       ///< Lorentz factor γ.
    Regime       regime{Regime::LIGHTLIKE}; ///< Spacetime interval regime.
    double       signal{0.0};      ///< Relativistic signal value.
    double       ds2{0.0};         ///< Spacetime interval Δs² (diagnostic).
};

} // namespace srfm::stream
