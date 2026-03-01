#pragma once
/**
 * @file  simd_dispatch.hpp
 * @brief Public API for SIMD-accelerated β and γ batch computation.
 *
 * Module:  include/srfm/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Expose two vectorised batch functions and a stateful BetaCalculator that
 * wire into the hot path of the SRFM financial signal pipeline.
 *
 *   computeBetaBatch()  — batch |v_i| / running_max → vector<BetaVelocity>
 *   computeGammaBatch() — batch 1/√(1−β²)            → vector<LorentzFactor>
 *
 * At program start-up the module probes the CPU (via cpu_features.hpp) and
 * selects AVX-512F → AVX2 → scalar automatically.  Existing callers see no
 * interface change.
 *
 * Guarantees
 * ----------
 *   • All functions are noexcept.
 *   • computeBetaBatch()  returns one BetaVelocity per input velocity.
 *   • computeGammaBatch() returns one LorentzFactor per input BetaVelocity.
 *   • Results are numerically identical to the scalar reference path
 *     (verified by the test suite in tests/momentum/test_simd.cpp).
 *   • BetaCalculator is NOT thread-safe (it owns mutable running_max state).
 *     Use one BetaCalculator per thread, or protect with a mutex.
 *
 * NOT Responsible For
 * -------------------
 *   • Sourcing raw velocity data — caller provides the input vector.
 *   • Cross-session running_max persistence — call reset() between sessions.
 *   • Thread safety for BetaCalculator — caller's responsibility.
 */

#include "cpu_features.hpp"
#include "momentum/momentum.hpp"  // BetaVelocity, LorentzFactor  (src/ on include path)

#include <vector>

namespace srfm::simd {

// ── SimdGammaCompute ──────────────────────────────────────────────────────────

/**
 * @brief Internal factory helper for constructing LorentzFactor objects
 *        from pre-computed gamma scalars (friend of LorentzFactor).
 *
 * This struct is an implementation detail of the SIMD dispatch layer.
 * It is declared here (in the public header) solely because friend
 * declarations in LorentzFactor must refer to a fully-qualified name.
 * Do NOT use this struct in application code.
 */
struct SimdGammaCompute {
    /// Wrap a pre-validated gamma value in a LorentzFactor.
    /// Precondition: gamma_val >= 1.0 && std::isfinite(gamma_val).
    [[nodiscard]] static srfm::momentum::LorentzFactor
    make(double gamma_val) noexcept {
        return srfm::momentum::LorentzFactor{gamma_val};
    }
};

// ── Free-function batch API ───────────────────────────────────────────────────

/**
 * @brief Compute β_i = |velocities[i]| / running_max for every element.
 *
 * running_max is updated monotonically across successive calls: it equals
 * max(running_max_in, max(|velocities[i]|)).  Pass running_max=0.0 on the
 * first call.
 *
 * The selected SIMD kernel processes blocks of 8 (AVX-512) or 4 (AVX2)
 * doubles per cycle; tail elements use the scalar fallback.
 *
 * @param velocities  Raw price velocities (any finite double, arbitrary sign).
 * @param running_max Current session maximum |velocity|; updated in-place.
 * @return            One BetaVelocity per input velocity, in order.
 *                    Empty vector when velocities is empty.
 *
 * # Panics
 * This function never panics.
 *
 * @example
 * @code
 *   double rmax = 0.0;
 *   auto betas = srfm::simd::computeBetaBatch(velocities, rmax);
 * @endcode
 */
[[nodiscard]] std::vector<srfm::momentum::BetaVelocity>
computeBetaBatch(const std::vector<double>& velocities,
                 double&                    running_max) noexcept;

/**
 * @brief Compute γ_i = 1/√(1 − β_i²) for every element.
 *
 * Inputs are clamped to [0, BETA_MAX_SAFE) before the sqrt to prevent NaN.
 * SIMD kernel processes 8 (AVX-512) or 4 (AVX2) elements per cycle.
 *
 * @param betas  Validated BetaVelocity values from computeBetaBatch().
 * @return       One LorentzFactor per input beta, in order.
 *               Empty vector when betas is empty.
 *
 * # Panics
 * This function never panics.
 */
[[nodiscard]] std::vector<srfm::momentum::LorentzFactor>
computeGammaBatch(const std::vector<srfm::momentum::BetaVelocity>& betas) noexcept;

// ── BetaCalculator ────────────────────────────────────────────────────────────

/**
 * @brief Stateful wrapper around computeBetaBatch / computeGammaBatch.
 *
 * Maintains a session-scoped running_max so callers do not need to manage
 * it manually.  Existing call sites that use RelativisticSignalProcessor
 * can migrate by:
 *
 * @code
 *   // Before (scalar):
 *   double rmax = 0.0;
 *   auto betas  = scalar_beta_batch(velocities, rmax);
 *   auto gammas = scalar_gamma_batch(betas);
 *
 *   // After (SIMD-dispatched, same semantics):
 *   BetaCalculator calc;
 *   auto betas  = calc.computeBetaBatch(velocities);
 *   auto gammas = calc.computeGammaBatch(betas);
 * @endcode
 *
 * NOT thread-safe.  Create one BetaCalculator per thread.
 */
class BetaCalculator {
public:
    /// Construct a fresh calculator.  running_max starts at 0.0.
    BetaCalculator() noexcept;

    /**
     * @brief Compute β_i = |v_i| / running_max for a batch of velocities.
     *
     * Delegates to the runtime-selected SIMD kernel.
     * running_max is advanced to include this batch's maximum |velocity|.
     *
     * @param velocities  Raw price velocities (any finite double).
     * @return            One BetaVelocity per velocity, in order.
     */
    [[nodiscard]] std::vector<srfm::momentum::BetaVelocity>
    computeBetaBatch(const std::vector<double>& velocities) noexcept;

    /**
     * @brief Compute γ_i = 1/√(1 − β_i²) for a batch of betas.
     *
     * Delegates to the runtime-selected SIMD kernel.
     *
     * @param betas  Output from a previous call to computeBetaBatch().
     * @return       One LorentzFactor per beta, in order.
     */
    [[nodiscard]] std::vector<srfm::momentum::LorentzFactor>
    computeGammaBatch(
        const std::vector<srfm::momentum::BetaVelocity>& betas) noexcept;

    /**
     * @brief Reset the running maximum to 0.0.
     *
     * Call between trading sessions to prevent stale state from inflating
     * the denominator and artificially suppressing beta values.
     */
    void reset() noexcept;

    /// Returns the current running maximum (read-only).
    [[nodiscard]] double running_max() const noexcept;

    /// Returns the SIMD level selected for this process.
    [[nodiscard]] SimdLevel simd_level() const noexcept;

private:
    double     running_max_{ 0.0 };
    SimdLevel  simd_level_{ SimdLevel::SCALAR };
};

} // namespace srfm::simd
