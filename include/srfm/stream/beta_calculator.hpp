#pragma once
/**
 * @file  beta_calculator.hpp
 * @brief Online market-velocity β calculator (FIX-N mode).
 *
 * Module:  include/srfm/stream/
 * Owner:   AGT-10  (Builder)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Compute the normalised market velocity β used by the Lorentz transform:
 *
 *   log_return[i]  = ln(close[i] / close[i-1])
 *   velocity_raw   = mean(log_return[last N bars])
 *   β              = clamp(velocity_raw / c_market, −BETA_MAX_SAFE, +BETA_MAX_SAFE)
 *
 * where N is the FIX parameter (default 3) and c_market is a calibration
 * constant representing the "speed of light" for this market (typical daily
 * log-return magnitude used as the upper bound).
 *
 * FIX-3 (N=3) means exactly 3 consecutive log-returns are averaged.  This
 * matches the original SRFM reference implementation's BetaCalculator.
 *
 * Guarantees
 * ----------
 *   • Online: O(1) per tick.
 *   • Bounded: stores exactly N+1 close prices.
 *   • Always valid: β ∈ (−BETA_MAX_SAFE, +BETA_MAX_SAFE) after warm-up.
 *   • Returns β = 0 until N+1 prices have been seen (conservative).
 *   • noexcept: update() and beta() are noexcept.
 *
 * NOT Responsible For
 * -------------------
 *   • Calibrating c_market    (caller provides it at construction)
 *   • Cross-asset composition (compose_velocities in srfm::momentum)
 */

#include <array>
#include <cmath>
#include <cstddef>

namespace srfm::stream {

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum safe |β| — mirrors srfm::momentum::BETA_MAX_SAFE.
inline constexpr double BETA_MAX_SAFE = 0.9999;

/// Default market "speed of light" calibration constant.
/// Represents a ~2 % per-tick log-return magnitude, which saturates β → 1.
inline constexpr double DEFAULT_C_MARKET = 0.02;

// ── BetaCalculator ────────────────────────────────────────────────────────────

/**
 * @brief Online β estimator with a sliding window of N log-returns.
 *
 * Template parameter N determines how many log-returns are averaged.
 * FIX-3 mode (default) uses N=3.
 *
 * @tparam N  Number of log-returns in the averaging window.  Must be ≥ 1.
 */
template<std::size_t N = 3u>
class BetaCalculator {
    static_assert(N >= 1u, "BetaCalculator: N must be at least 1");
    static_assert(N <= 64u, "BetaCalculator: N must be at most 64 (compile-time array limit)");

public:
    /**
     * @brief Construct with a market speed-of-light calibration constant.
     *
     * @param c_market  The denominator used to normalise raw log-returns into β.
     *                  Must be strictly positive and finite.  Defaults to 0.02.
     */
    explicit BetaCalculator(double c_market = DEFAULT_C_MARKET) noexcept
        : c_market_{c_market > 0.0 && std::isfinite(c_market) ? c_market : DEFAULT_C_MARKET}
    {}

    // ── State mutation ─────────────────────────────────────────────────────────

    /**
     * @brief Ingest one new close price and update the rolling log-return buffer.
     *
     * @param close  Current bar's close price.  Must be positive and finite;
     *               non-positive / non-finite values are silently ignored
     *               (no state change) to protect the pipeline from bad data.
     * @note noexcept.
     */
    void update(double close) noexcept {
        if (!std::isfinite(close) || close <= 0.0) return;

        if (prev_close_ > 0.0 && std::isfinite(prev_close_)) {
            // Compute log-return and slide into circular buffer.
            const double lr      = std::log(close / prev_close_);
            returns_[ret_pos_]   = lr;
            ret_pos_             = (ret_pos_ + 1u) % N;
            if (ret_count_ < N) ++ret_count_;
        }
        prev_close_    = close;
        ++close_count_;
    }

    // ── Query ──────────────────────────────────────────────────────────────────

    /**
     * @brief Current β estimate.
     *
     * Returns 0.0 until N log-returns have been accumulated.
     * Guaranteed: |result| < BETA_MAX_SAFE.
     *
     * @note noexcept.
     */
    [[nodiscard]] double beta() const noexcept {
        if (ret_count_ < N) return 0.0;

        // Mean of the N log-returns in the circular buffer.
        double sum = 0.0;
        for (std::size_t i = 0u; i < N; ++i) sum += returns_[i];
        const double velocity = sum / static_cast<double>(N);

        // Normalise by c_market and clamp to valid range.
        const double raw = velocity / c_market_;
        if (raw >=  BETA_MAX_SAFE) return  BETA_MAX_SAFE - 1e-9;
        if (raw <= -BETA_MAX_SAFE) return -BETA_MAX_SAFE + 1e-9;
        return raw;
    }

    /**
     * @brief Whether enough data has been seen to produce a reliable β.
     *
     * True once N+1 close prices have been ingested (N log-returns computed).
     */
    [[nodiscard]] bool warmed_up() const noexcept { return ret_count_ >= N; }

    /// Number of close prices ingested so far.
    [[nodiscard]] std::size_t close_count() const noexcept { return close_count_; }

    /// Number of log-returns accumulated (capped at N).
    [[nodiscard]] std::size_t ret_count() const noexcept { return ret_count_; }

    /// Configured market speed-of-light constant.
    [[nodiscard]] double c_market() const noexcept { return c_market_; }

    // ── Reset ──────────────────────────────────────────────────────────────────

    /**
     * @brief Reset all state as if no ticks have been seen.
     */
    void reset() noexcept {
        returns_.fill(0.0);
        ret_pos_     = 0;
        ret_count_   = 0;
        prev_close_  = 0.0;
        close_count_ = 0;
    }

private:
    double               c_market_;          ///< Speed of light calibration.
    std::array<double,N> returns_{};         ///< Circular buffer of log-returns.
    std::size_t          ret_pos_{0};        ///< Next write position in returns_.
    std::size_t          ret_count_{0};      ///< Returns accumulated (≤ N).
    double               prev_close_{0.0};   ///< Previous close for log-return.
    std::size_t          close_count_{0};    ///< Total close prices ingested.
};

/// Convenience alias for the default FIX-3 mode used throughout this pipeline.
using BetaCalculatorFix3 = BetaCalculator<3u>;

} // namespace srfm::stream
