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
/// ## Edge Cases
/// - stddev < 1e-9 (flat / constant series): returns 0.0 (no variance to normalize)
/// - Window not yet full: normalizes using all available samples (≥1)
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
    /// Construct with a given window size.
    ///
    /// # Arguments
    /// * `window` — Number of bars to use for rolling statistics.
    ///              Minimum 1. Default: 20.
    explicit CoordinateNormalizer(std::size_t window = 20) noexcept;

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

    /// Reset the normalizer, clearing all buffered observations.
    void reset() noexcept;

private:
    std::size_t        window_;
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
