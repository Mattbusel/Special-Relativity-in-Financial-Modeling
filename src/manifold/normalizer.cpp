/// @file src/manifold/normalizer.cpp
/// @brief CoordinateNormalizer — rolling z-score normalization for SpacetimeEvent.
///
/// Maintains three independent rolling deques (price, volume, momentum).
/// Each normalize() call:
///   1. Appends the new coordinate values to their respective deques
///   2. Trims each deque to the configured window size
///   3. Computes rolling mean and sample stddev for each coordinate
///   4. Returns the z-scored event (time passes through unchanged)

#include "srfm/normalizer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace srfm {

// ─── Constructor ──────────────────────────────────────────────────────────────

CoordinateNormalizer::CoordinateNormalizer(std::size_t window) noexcept
    : window_(window < 1 ? 1 : window) {}

// ─── push ─────────────────────────────────────────────────────────────────────

void CoordinateNormalizer::push(std::deque<double>& buf,
                                 double value,
                                 std::size_t max_size) noexcept {
    buf.push_back(value);
    if (buf.size() > max_size) {
        buf.pop_front();
    }
}

// ─── rolling_mean ─────────────────────────────────────────────────────────────

double CoordinateNormalizer::rolling_mean(const std::deque<double>& buf) noexcept {
    // Precondition: buf is non-empty.
    double sum = 0.0;
    for (double v : buf) {
        sum += v;
    }
    return sum / static_cast<double>(buf.size());
}

// ─── rolling_stddev ───────────────────────────────────────────────────────────

double CoordinateNormalizer::rolling_stddev(const std::deque<double>& buf,
                                             double mean) noexcept {
    if (buf.size() < 2) {
        // Cannot compute sample stddev from a single observation.
        return 0.0;
    }
    double sq_sum = 0.0;
    for (double v : buf) {
        const double d = v - mean;
        sq_sum += d * d;
    }
    // Bessel-corrected (n-1) sample standard deviation.
    return std::sqrt(sq_sum / static_cast<double>(buf.size() - 1));
}

// ─── zscore ───────────────────────────────────────────────────────────────────

double CoordinateNormalizer::zscore(double value,
                                     const std::deque<double>& buf) noexcept {
    if (buf.empty()) {
        return 0.0;
    }
    const double m = rolling_mean(buf);
    const double s = rolling_stddev(buf, m);
    if (s < FLAT_STDDEV_THRESHOLD) {
        // Flat series — no variance to normalize against.
        return 0.0;
    }
    return (value - m) / s;
}

// ─── normalize ────────────────────────────────────────────────────────────────

manifold::SpacetimeEvent
CoordinateNormalizer::normalize(const manifold::SpacetimeEvent& raw) noexcept {
    // Update each rolling buffer with the new bar's spatial coordinates.
    push(price_buf_,    raw.price,    window_);
    push(volume_buf_,   raw.volume,   window_);
    push(momentum_buf_, raw.momentum, window_);

    // Z-score each spatial coordinate using the updated window.
    // Time coordinate is always passed through unchanged.
    return manifold::SpacetimeEvent{
        .time     = raw.time,
        .price    = zscore(raw.price,    price_buf_),
        .volume   = zscore(raw.volume,   volume_buf_),
        .momentum = zscore(raw.momentum, momentum_buf_),
    };
}

// ─── size / window_size / reset ───────────────────────────────────────────────

std::size_t CoordinateNormalizer::size() const noexcept {
    // All three deques are always the same size; any one is representative.
    return price_buf_.size();
}

std::size_t CoordinateNormalizer::window_size() const noexcept {
    return window_;
}

void CoordinateNormalizer::reset() noexcept {
    price_buf_.clear();
    volume_buf_.clear();
    momentum_buf_.clear();
}

}  // namespace srfm
