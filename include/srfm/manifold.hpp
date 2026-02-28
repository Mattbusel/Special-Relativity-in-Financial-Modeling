#pragma once

/// @file include/srfm/manifold.hpp
/// @brief Spacetime Market Manifold — AGT-02 public API (implemented by AGT-06).
///
/// # Module: Market Manifold
///
/// ## Responsibility
/// Embed OHLCV bar data into the 4-dimensional financial spacetime manifold and
/// compute the invariant spacetime interval ds² between market events.
///
/// ## The Core Idea
/// Each market observation (bar) is treated as a spacetime event with four
/// coordinates: (time, price, volume, momentum).  The interval between two
/// events determines whether the market movement is:
///   - Timelike  (ds² < 0): causal, subluminal — the market "moved" at β < 1
///   - Lightlike (ds² = 0): signal propagation at the speed of information
///   - Spacelike (ds² > 0): stochastic, superluminal — no causal link possible
///
/// ## Interval Formula
///   ds² = −c²·Δt² + ΔP² + ΔV² + ΔM²
///
/// where c = SPEED_OF_INFORMATION (normalised to 1.0 in natural units).
///
/// ## Guarantees
/// - All functions are `noexcept` and return `std::optional` for fallible ops
/// - No dynamic allocation
/// - Thread-safe: all methods are stateless / const

#include "srfm/constants.hpp"

#include <optional>
#include <string>

// Forward declaration — avoids circular include with normalizer.hpp.
namespace srfm {
class CoordinateNormalizer;
}  // namespace srfm

namespace srfm::manifold {

// ─── SpacetimeEvent ───────────────────────────────────────────────────────────

/// A single market observation embedded as a 4-vector in financial spacetime.
///
/// Coordinates:
///   [0] time     — bar index or fractional seconds (market-time axis)
///   [1] price    — normalised mid-price (spatial axis 1)
///   [2] volume   — normalised traded volume (spatial axis 2)
///   [3] momentum — price momentum indicator, e.g. ROC (spatial axis 3)
struct SpacetimeEvent {
    double time;      ///< Market time coordinate (bar index or timestamp)
    double price;     ///< Price spatial coordinate
    double volume;    ///< Volume spatial coordinate
    double momentum;  ///< Momentum spatial coordinate
};

// ─── IntervalType ─────────────────────────────────────────────────────────────

/// Causal character of a spacetime interval.
enum class IntervalType {
    Timelike,   ///< ds² < 0  — causal market movement (β < c)
    Lightlike,  ///< ds² ≈ 0  — information propagation at c
    Spacelike,  ///< ds² > 0  — stochastic regime (no causal link)
};

/// Convert IntervalType to a human-readable string.
[[nodiscard]] const char* to_string(IntervalType t) noexcept;

// ─── SpacetimeInterval ────────────────────────────────────────────────────────

/// Computes the Minkowski-signature spacetime interval between two market events.
///
/// All methods are static and operate on value types — no heap allocation.
class SpacetimeInterval {
public:
    /// Compute ds² = −c²·Δt² + ΔP² + ΔV² + ΔM²
    ///
    /// # Arguments
    /// * `a`        — origin event
    /// * `b`        — destination event
    /// * `c_market` — speed of information (default: SPEED_OF_INFORMATION = 1.0)
    ///
    /// # Returns
    /// The raw signed interval squared.  Negative → timelike; zero → lightlike;
    /// positive → spacelike.  Returns `nullopt` if any coordinate is non-finite.
    [[nodiscard]] static std::optional<double>
    compute(const SpacetimeEvent& a,
            const SpacetimeEvent& b,
            double c_market = constants::SPEED_OF_INFORMATION) noexcept;

    /// Classify an already-computed interval squared value.
    ///
    /// Uses `FLOAT_EPSILON` as the lightlike tolerance band.
    [[nodiscard]] static IntervalType
    classify(double interval_squared) noexcept;
};

// ─── MarketManifold ───────────────────────────────────────────────────────────

/// High-level interface to the financial spacetime manifold.
///
/// Wraps SpacetimeInterval with convenience methods for common pipeline
/// queries: causal character, normalised velocity, and regime classification.
class MarketManifold {
public:
    /// Compute the spacetime interval between two events and classify it.
    ///
    /// # Returns
    /// `nullopt` if any coordinate is non-finite.
    [[nodiscard]] static std::optional<IntervalType>
    classify(const SpacetimeEvent& a,
             const SpacetimeEvent& b) noexcept;

    /// Compute the normalised 3-velocity β = |Δspace| / (c · |Δtime|).
    ///
    /// This is the financial analogue of the relativistic β = v/c used by
    /// BetaCalculator.  It measures how fast the "spatial" (price/vol/momentum)
    /// coordinates change relative to the market time axis.
    ///
    /// # Returns
    /// β in [0, BETA_MAX_SAFE), or `nullopt` if:
    ///   - Δt = 0 (simultaneous events — undefined velocity)
    ///   - Any coordinate is non-finite
    [[nodiscard]] static std::optional<double>
    beta(const SpacetimeEvent& a,
         const SpacetimeEvent& b,
         double c_market = constants::SPEED_OF_INFORMATION) noexcept;

    /// True if the trajectory from `a` to `b` is causal (timelike or lightlike).
    [[nodiscard]] static bool
    is_causal(const SpacetimeEvent& a,
              const SpacetimeEvent& b) noexcept;

    /// Normalize `curr_raw` via `normalizer`, then classify the interval
    /// between `prev_normalized` and the resulting normalized event.
    ///
    /// This is the canonical pipeline entry point. Every SpacetimeEvent is
    /// normalized before SpacetimeInterval::compute is called, preventing
    /// raw coordinate scale differences (price ~100, volume ~1e6) from
    /// dominating the interval computation.
    ///
    /// # Arguments
    /// * `normalizer`       — Rolling z-score normalizer (updated in place)
    /// * `prev_normalized`  — Previous event, already normalized
    /// * `curr_raw`         — Current event with raw market coordinates
    ///
    /// # Returns
    /// Interval type of the normalized trajectory, or `nullopt` if any
    /// coordinate is non-finite after normalization.
    [[nodiscard]] static std::optional<IntervalType>
    process(srfm::CoordinateNormalizer& normalizer,
            const SpacetimeEvent& prev_normalized,
            const SpacetimeEvent& curr_raw) noexcept;
};

}  // namespace srfm::manifold
