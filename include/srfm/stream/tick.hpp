#pragma once
/**
 * @file  tick.hpp
 * @brief OHLCVTick — atomic market data unit for the lock-free streaming pipeline.
 *
 * Module:  include/srfm/stream/
 * Owner:   AGT-10  (Builder)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Define the canonical OHLCVTick structure that flows through the streaming
 * pipeline from TickIngester → SignalProcessor.  Every field has a defined
 * valid range; validation is performed at the ingestion boundary.
 *
 * Design Constraints
 * ------------------
 *   • Trivially copyable — ring buffer moves raw bytes.
 *   • No dynamic allocation — fixed-size POD struct.
 *   • All validation is non-throwing (returns bool, never throws).
 *
 * NOT Responsible For
 * -------------------
 *   • Parsing from wire format        (TickIngester / PipeTickSource)
 *   • Storage or persistence          (stateless type definition)
 *   • Cross-asset normalisation       (CoordinateNormalizer)
 */

#include <cmath>
#include <cstdint>

namespace srfm::stream {

// ── OHLCVTick ─────────────────────────────────────────────────────────────────

/**
 * @brief Single OHLCV bar tick from the market data feed.
 *
 * All price fields are in quote-currency units (e.g., USD).
 * Volume is in base-currency units (e.g., contracts or shares).
 * timestamp_ns is nanoseconds since Unix epoch (UTC).
 *
 * Invariants (enforced by tick_is_valid()):
 *   - open, high, low, close > 0 and finite
 *   - high >= low
 *   - high >= open, high >= close
 *   - low  <= open, low  <= close
 *   - volume >= 0 and finite
 *   - timestamp_ns > 0
 */
struct OHLCVTick {
    double   open{0.0};          ///< Opening price of the bar.
    double   high{0.0};          ///< Highest price of the bar.
    double   low{0.0};           ///< Lowest price of the bar.
    double   close{0.0};         ///< Closing price of the bar.
    double   volume{0.0};        ///< Traded volume over the bar.
    std::int64_t timestamp_ns{0};///< Bar timestamp, nanoseconds since epoch.
};

// ── Validation ────────────────────────────────────────────────────────────────

/**
 * @brief Validate all fields of an OHLCVTick.
 *
 * Rejects ticks that would produce NaN or negative prices/volumes downstream.
 * This is the single authoritative validation gate; call it exactly once at
 * the ingestion boundary.
 *
 * @return true  iff the tick satisfies all field invariants.
 * @return false on any out-of-range, NaN, or Inf value.
 *
 * @note noexcept — safe to call in the hot path inside the ingestion thread.
 */
[[nodiscard]] inline bool tick_is_valid(const OHLCVTick& t) noexcept {
    // All price fields must be finite and positive.
    if (!std::isfinite(t.open)   || t.open   <= 0.0) return false;
    if (!std::isfinite(t.high)   || t.high   <= 0.0) return false;
    if (!std::isfinite(t.low)    || t.low    <= 0.0) return false;
    if (!std::isfinite(t.close)  || t.close  <= 0.0) return false;
    // Volume must be finite and non-negative.
    if (!std::isfinite(t.volume) || t.volume <  0.0) return false;
    // Timestamp must be positive.
    if (t.timestamp_ns <= 0) return false;
    // OHLC consistency.
    if (t.high < t.low)   return false;
    if (t.high < t.open)  return false;
    if (t.high < t.close) return false;
    if (t.low  > t.open)  return false;
    if (t.low  > t.close) return false;
    return true;
}

/**
 * @brief Reason codes for tick rejection (diagnostic, not returned on hot path).
 *
 * Used by TickIngester to categorise dropped ticks in its counters.
 */
enum class TickRejectReason : std::uint8_t {
    NanOrInfOpen     = 0,
    NanOrInfHigh     = 1,
    NanOrInfLow      = 2,
    NanOrInfClose    = 3,
    NanOrInfVolume   = 4,
    NonPositivePrice = 5,
    NegativeVolume   = 6,
    InvalidTimestamp = 7,
    OhlcInconsistent = 8,
    RingFull         = 9,  ///< Valid tick dropped because ring buffer was full.
};

} // namespace srfm::stream
