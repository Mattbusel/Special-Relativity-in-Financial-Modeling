#pragma once

/// @file include/srfm/normalizer.hpp
/// @brief CoordinateNormalizer — rolling z-score normalizer for SpacetimeEvent.
///
/// # Module: Coordinate Normalizer
///
/// ## Responsibility
/// Normalize the spatial coordinates of SpacetimeEvent instances using a
/// rolling window of observed values before they are passed to
/// SpacetimeInterval::compute.
///
/// ## Why This Matters
/// Raw market coordinates differ by many orders of magnitude:
///   - price    ~100
///   - volume   ~1e6
///   - momentum ~0.01
///
/// Without normalization, volume dominates ds² by ~10 orders of magnitude,
/// making interval classification (timelike/spacelike) meaningless. After
/// z-score normalization each coordinate contributes equally.
///
/// ## Formula
/// For each spatial coordinate x, rolling window of n observations:
///   x_norm = (x - rolling_mean(x)) / rolling_stddev(x)
///
/// Time coordinate passes through unchanged.
///
/// ## Warm-up Guard
/// Before at least `min_samples` data points have been accumulated (default =
/// window_size, minimum 20), the z-scores are potentially unreliable due to
/// insufficient statistical history.  warmed_up() returns false during this
/// period, and normalize() returns a zero-valued event (all spatial coordinates
/// set to 0.0 with time unchanged) instead of a potentially wild z-score.
/// Callers should check warmed_up() before using the output of normalize() for
/// trading decisions.
///
/// ## Edge Cases
/// - stddev < 1e-9 (flat / constant series): returns 0.0 (no variance to normalize)
/// - Window not yet full OR min_samples not yet reached: returns 0.0 for spatial coords
/// - Single-sample window: mean = x, stddev = 0 → returns 0.0
///
/// ## Guarantees
/// - Stateful: call in time order for correct rolling statistics
/// - Never divides by zero
/// - Time coordinate always unchanged
/// - No dynamic allocation on hot path (deque amortized)

#include "srfm/manifold.hpp"

#include <cstddef>
#include <deque>

namespace srfm {

/// Rolling z-score normalizer for SpacetimeEvent spatial coordinates.
///
/// Maintains independent rolling windows for price, volume, and momentum.
/// Each call to normalize() updates the windows and returns a z-scored event.
class CoordinateNormalizer {
public:
    /// Minimum number of samples required before warmed_up() returns true.
    static constexpr std::size_t DEFAULT_MIN_SAMPLES = 20;

    /// Construct with a given window size and optional min_samples warm-up guard.
    ///
    /// # Arguments
    /// * `window`      — Number of bars to use for rolling statistics.
    ///                   Minimum 1. Default: 20.
    /// * `min_samples` — Number of samples that must be seen before warmed_up()
    ///                   returns true and normalize() emits non-zero z-scores.
    ///                   Defaults to max(window, DEFAULT_MIN_SAMPLES).
    explicit CoordinateNormalizer(std::size_t window      = 20,
                                  std::size_t min_samples = 0) noexcept;

    /// Normalize the spatial coordinates of a SpacetimeEvent.
    ///
    /// Updates the rolling window with the current event's coordinates, then
    /// returns a new SpacetimeEvent where each spatial coordinate is z-scored
    /// using the updated window statistics.
    ///
    /// # Returns
    /// SpacetimeEvent with:
    ///   - time:     unchanged (pass-through, no normalization)
    ///   - price:    z-scored using current rolling window
    ///   - volume:   z-scored using current rolling window
    ///   - momentum: z-scored using current rolling window
    ///
    /// If stddev < 1e-9 for a coordinate (flat series, no variance),
    /// that coordinate is returned as 0.0.
    [[nodiscard]] manifold::SpacetimeEvent
    normalize(const manifold::SpacetimeEvent& raw) noexcept;

    /// Number of samples currently in the rolling window (≤ window_size()).
    [[nodiscard]] std::size_t size() const noexcept;

    /// Configured maximum window size.
    [[nodiscard]] std::size_t window_size() const noexcept;

    /// Total number of samples seen so far (does not saturate at window_size()).
    [[nodiscard]] std::size_t total_samples() const noexcept;

    /// Returns true once at least min_samples data points have been accumulated.
    ///
    /// Before warmed_up() is true, normalize() returns a zero-valued event to
    /// avoid producing wild z-scores from an insufficient statistical window.
    [[nodiscard]] bool warmed_up() const noexcept;

    /// Reset the normalizer, clearing all buffered observations.
    void reset() noexcept;

private:
    std::size_t        window_;
    std::size_t        min_samples_;  ///< Warm-up threshold.
    std::size_t        total_{0};     ///< Total samples ingested (unbounded).
    std::deque<double> price_buf_;
    std::deque<double> volume_buf_;
    std::deque<double> momentum_buf_;

    /// Push a value into a deque, evicting the oldest if at capacity.
    static void push(std::deque<double>& buf,
                     double value,
                     std::size_t max_size) noexcept;

    /// Mean of all values in buf. Precondition: buf is non-empty.
    static double rolling_mean(const std::deque<double>& buf) noexcept;

    /// Sample std-dev (Bessel-corrected). Returns 0.0 if buf.size() < 2.
    static double rolling_stddev(const std::deque<double>& buf,
                                 double mean) noexcept;

    /// Z-score `value` using the buffer's current statistics.
    /// Returns 0.0 if stddev < FLAT_STDDEV_THRESHOLD.
    static double zscore(double value,
                         const std::deque<double>& buf) noexcept;

    static constexpr double FLAT_STDDEV_THRESHOLD = 1e-9;
};

}  // namespace srfm
