#pragma once
/**
 * @file  coordinate_normalizer.hpp
 * @brief Rolling-window z-score normaliser for tick close prices.
 *
 * Module:  include/srfm/stream/
 * Owner:   AGT-10  (Builder)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Map raw close prices into a normalised coordinate space using a rolling
 * Welford online mean/variance estimator over a configurable window.
 *
 *   normalised = (close − μ_window) / σ_window
 *
 * This dimensionless coordinate feeds into BetaCalculator and LorentzTransform,
 * ensuring the relativistic machinery operates on unit-free quantities.
 *
 * Guarantees
 * ----------
 *   • Online: O(1) per tick, no recomputation over the full window.
 *   • Bounded: memory is exactly window * sizeof(double) + a few scalars.
 *   • Zero-allocation: circular buffer is pre-allocated at construction.
 *   • Non-throwing: update() and normalise() are noexcept.
 *   • Pre-warm: returns 0.0 until window is full (conservative).
 *
 * NOT Responsible For
 * -------------------
 *   • Cross-field normalisation (high/low/volume normalised separately if needed)
 *   • Persistence or serialisation of the rolling state
 *   • Thread safety — single-threaded use only
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace srfm::stream {

// ── CoordinateNormalizer ───────────────────────────────────────────────────────

/**
 * @brief Rolling z-score normaliser using Welford's online algorithm.
 *
 * Window size is fixed at construction time.  Call update() with each new
 * close price, then normalise() to obtain the current z-score.
 *
 * @code
 *   CoordinateNormalizer norm{20};
 *   for (const auto& tick : ticks) {
 *       norm.update(tick.close);
 *       double z = norm.normalise(tick.close);
 *       // z is 0.0 until the window is full
 *   }
 * @endcode
 */
class CoordinateNormalizer {
public:
    /**
     * @brief Construct with a given rolling window size.
     *
     * @param window  Number of ticks in the rolling window.  Must be ≥ 2.
     */
    explicit CoordinateNormalizer(std::size_t window)
        : window_{window}
        , buf_(window, 0.0)
    {
        assert(window >= 2u);
    }

    // ── State mutation ─────────────────────────────────────────────────────────

    /**
     * @brief Ingest one new close price into the rolling window.
     *
     * Updates the internal circular buffer and the running mean/M2 accumulators
     * using Welford's online algorithm.  O(1) per call.
     *
     * @param close  Raw close price.  Must be finite and positive.
     * @note noexcept — arithmetic only, no allocation.
     */
    void update(double close) noexcept {
        const double old_val = buf_[pos_];
        buf_[pos_] = close;
        pos_ = (pos_ + 1u) % window_;

        if (count_ < window_) {
            ++count_;
            // Welford add for growing window.
            const double delta  = close - mean_;
            mean_  += delta / static_cast<double>(count_);
            const double delta2 = close - mean_;
            m2_    += delta * delta2;
        } else {
            // Welford update for sliding window (remove old, add new).
            const double old_mean = mean_;
            mean_ += (close - old_val) / static_cast<double>(window_);
            m2_   += (close - old_val) * (close - mean_ + old_val - old_mean);
            // Clamp M2 to avoid floating-point drift into negatives.
            if (m2_ < 0.0) m2_ = 0.0;
        }
    }

    /**
     * @brief Return the z-score of @p value relative to the current window.
     *
     * Returns 0.0 if the window is not yet full or if the standard deviation
     * is effectively zero (constant price series).
     *
     * @param value  Value to normalise (typically the current close price).
     * @note noexcept — arithmetic only.
     */
    [[nodiscard]] double normalise(double close) const noexcept {
        if (count_ < window_) return 0.0;
        const double variance = m2_ / static_cast<double>(window_);
        const double sigma    = std::sqrt(variance);
        if (sigma < 1e-15) return 0.0;
        return (close - mean_) / sigma;
    }

    // ── Diagnostic accessors ───────────────────────────────────────────────────

    /// Rolling mean of the window.  0.0 if window not yet full.
    [[nodiscard]] double mean()  const noexcept { return mean_; }

    /// Rolling population standard deviation.  0.0 if window not yet full.
    [[nodiscard]] double sigma() const noexcept {
        if (count_ < window_) return 0.0;
        const double v = m2_ / static_cast<double>(window_);
        return v < 0.0 ? 0.0 : std::sqrt(v);
    }

    /// Number of ticks seen so far (saturates at window size).
    [[nodiscard]] std::size_t count() const noexcept { return count_; }

    /// Whether the window has been filled at least once.
    [[nodiscard]] bool warmed_up() const noexcept { return count_ >= window_; }

    /// Configured window size.
    [[nodiscard]] std::size_t window() const noexcept { return window_; }

    // ── Reset ──────────────────────────────────────────────────────────────────

    /**
     * @brief Reset all state as if no ticks have been seen.
     *
     * The window size is preserved.
     */
    void reset() noexcept {
        std::fill(buf_.begin(), buf_.end(), 0.0);
        pos_   = 0;
        count_ = 0;
        mean_  = 0.0;
        m2_    = 0.0;
    }

private:
    std::size_t         window_;       ///< Configured window length.
    std::vector<double> buf_;          ///< Circular buffer of close prices.
    std::size_t         pos_{0};       ///< Next write position in buf_.
    std::size_t         count_{0};     ///< Ticks ingested (capped at window_).
    double              mean_{0.0};    ///< Running rolling mean.
    double              m2_{0.0};      ///< Running sum of squared deviations.
};

} // namespace srfm::stream
