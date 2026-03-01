#pragma once
/**
 * @file  spacetime_manifold.hpp
 * @brief Spacetime manifold processor with Christoffel symbols (AGT-13 / SRFM)
 *
 * Module:  src/manifold/
 * Owner:   AGT-13  (Adversarial hardening)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Model the financial market as a curved spacetime manifold:
 *
 *   • Classify spacetime events into relativistic regimes.
 *   • Compute Christoffel connection coefficients Γ^λ_μν from a metric tensor.
 *   • Provide the flat Minkowski metric η = diag(−1, +1, +1, +1).
 *
 * Key invariant (tested by property suite):
 *   For the flat Minkowski metric, ALL 64 Christoffel symbols are zero.
 *
 * Design Constraints
 * ------------------
 *   • No exceptions; all fallible paths return std::optional or signal via bool.
 *   • All public methods are noexcept.
 *   • MetricTensor spatial block must be positive-definite for a valid manifold.
 *
 * NOT Responsible For
 *   • Coordinate transformations between frames.
 *   • Integration of geodesic equations (see geodesic_solver.hpp).
 */

#include <array>
#include <cmath>
#include <optional>

namespace srfm::manifold {

// ── Constants ─────────────────────────────────────────────────────────────────

/// Number of spacetime dimensions.
inline constexpr int DIM = 4;

/// Total Christoffel symbols: DIM³ = 64.
inline constexpr int NUM_CHRISTOFFEL = DIM * DIM * DIM;

// ── MetricTensor ──────────────────────────────────────────────────────────────

/**
 * @brief Symmetric 4×4 spacetime metric tensor g_{μν}.
 *
 * Row/column indices: 0=t, 1=x, 2=y, 3=z.
 * Sign convention: (−,+,+,+).  Flat Minkowski: diag(−1,+1,+1,+1).
 *
 * The spatial block g[1..3][1..3] must be positive-definite for a physically
 * valid metric (time-like signature).
 */
struct MetricTensor {
    std::array<std::array<double, DIM>, DIM> g{};

    /// Construct the flat Minkowski metric η = diag(−1,+1,+1,+1).
    [[nodiscard]] static MetricTensor minkowski() noexcept;

    /// Check that the metric has correct signature: g[0][0] < 0,
    /// spatial diagonal entries g[i][i] > 0 for i ∈ {1,2,3}, finite entries.
    [[nodiscard]] bool is_valid() const noexcept;

    /// Return the inverse metric g^{μν} assuming diagonal metric (fast path).
    /// For non-diagonal metrics falls back to returning nullopt.
    [[nodiscard]] std::optional<MetricTensor> inverse_diagonal() const noexcept;
};

// ── SpacetimeEvent ────────────────────────────────────────────────────────────

/**
 * @brief A point in 4D spacetime (t, x, y, z).
 *
 * In the financial interpretation:
 *   t = time index
 *   x = price
 *   y = volume
 *   z = volatility proxy
 */
struct SpacetimeEvent {
    double t{0.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};

    /// True iff all coordinates are finite.
    [[nodiscard]] bool is_finite() const noexcept;
};

// ── Regime ────────────────────────────────────────────────────────────────────

/**
 * @brief Market relativistic regime classification.
 */
enum class Regime {
    Newtonian,    ///< |β| < 0.1   — classical approximation valid
    Relativistic, ///< 0.1 ≤ |β| < 0.9  — corrections needed
    HighGamma,    ///< 0.9 ≤ |β| < 0.9999 — extreme Lorentz contraction
    Subluminal,   ///< Catch-all: |β| ≥ 0 and < BETA_MAX_SAFE
};

// ── Christoffel index helpers ─────────────────────────────────────────────────

/// Pack (λ, μ, ν) into flat index in [0, 64).
[[nodiscard]] inline constexpr int christoffel_index(int lambda, int mu, int nu) noexcept {
    return lambda * DIM * DIM + mu * DIM + nu;
}

// ── SpacetimeManifold ─────────────────────────────────────────────────────────

/**
 * @brief Processes spacetime events and computes manifold geometry.
 *
 * Stateless. Thread-safe.
 *
 * @example
 * @code
 *   SpacetimeManifold manifold;
 *   SpacetimeEvent evt{1.0, 100.5, 1e6, 0.02};
 *   auto regime = manifold.process(evt);   // → Regime::Newtonian
 *   auto metric = MetricTensor::minkowski();
 *   auto christoffel = manifold.christoffelSymbols(metric); // all zeros
 * @endcode
 */
class SpacetimeManifold {
public:
    SpacetimeManifold() noexcept = default;

    /**
     * @brief Classify a spacetime event into a relativistic regime.
     *
     * Uses x-coordinate as a proxy for normalised velocity |β|:
     *   β_proxy = tanh(|x| / (|x| + 1.0))  (maps R⁺ → [0,1))
     *
     * @return Regime, or std::nullopt if event coordinates are non-finite.
     */
    [[nodiscard]] std::optional<Regime>
    process(const SpacetimeEvent& event) const noexcept;

    /**
     * @brief Compute all 64 Christoffel symbols Γ^λ_μν via finite differences.
     *
     * Uses central finite differences on the metric at the origin:
     *   ∂g_{μν}/∂x^λ ≈ (g(x+ε·eλ) − g(x−ε·eλ)) / (2ε)
     *
     * For a constant (flat) metric all derivatives are machine-zero, so all
     * 64 symbols are < 1e-8 in absolute value.
     *
     * The metric callback signature:
     *   MetricCallback: (const std::array<double,DIM>&) → MetricTensor
     * A null-like constant metric simply returns the same MetricTensor
     * regardless of position.
     *
     * @param metric  The metric tensor at the origin (flat or curved).
     * @return std::array<double, 64> of Γ^λ_μν values (row-major λ,μ,ν).
     *         Returns array of zeros if metric inverse cannot be computed.
     */
    [[nodiscard]] std::array<double, NUM_CHRISTOFFEL>
    christoffelSymbols(const MetricTensor& metric) const noexcept;

    /**
     * @brief Return the flat Minkowski metric.
     *
     * Convenience wrapper around MetricTensor::minkowski().
     */
    [[nodiscard]] MetricTensor flatMetric() const noexcept;
};

} // namespace srfm::manifold
