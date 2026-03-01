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
 *   computeGammaBatch() — batch 1/√(1−β²)           → vector<LorentzFactor>
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
 *     (verified by the test suite in tests/simd/test_simd.cpp).
 *   • BetaCalculator is NOT thread-safe (mutable running_max state).
 *     Use one BetaCalculator per thread, or protect with a mutex.
 *
 * NOT Responsible For
 * -------------------
 *   • Sourcing raw velocity data — caller provides the input vector.
 *   • Cross-session running_max persistence — call reset() between sessions.
 */

#include "srfm/simd/cpu_features.hpp"
#include "srfm/types.hpp"       // BetaVelocity, LorentzFactor

#include <vector>

namespace srfm::simd {

// ── Free-function batch API ───────────────────────────────────────────────────

/**
 * @brief Compute β_i = |velocities[i]| / running_max for every element.
 *
 * Algorithm: batch-max.  running_max is advanced to max(running_max_in,
 * max(|velocities[i]|)) once per call, then used as the common denominator
 * for all elements in this call.  This guarantees bit-identical results
 * across scalar, AVX2, and AVX-512 paths.
 *
 * @param velocities  Raw price velocities (any finite double).
 * @param running_max Current session maximum |velocity|; updated in-place.
 * @return            One BetaVelocity per input velocity, in order.
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
[[nodiscard]] std::vector<BetaVelocity>
computeBetaBatch(const std::vector<double>& velocities,
                 double&                    running_max) noexcept;

/**
 * @brief Compute γ_i = 1/√(1 − β_i²) for every element.
 *
 * Input beta values are clamped to [0, BETA_MAX_SAFE) before the sqrt.
 *
 * @param betas  BetaVelocity values from computeBetaBatch().
 * @return       One LorentzFactor per input beta, in order.
 *
 * # Panics
 * This function never panics.
 */
[[nodiscard]] std::vector<LorentzFactor>
computeGammaBatch(const std::vector<BetaVelocity>& betas) noexcept;

// ── BetaCalculator ────────────────────────────────────────────────────────────

/**
 * @brief Stateful wrapper that maintains a session-scoped running_max.
 *
 * Drop-in acceleration for any loop that computes beta/gamma bar-by-bar:
 * @code
 *   // Before:
 *   for (auto& bar : bars) {
 *       bar.beta  = BetaVelocity{std::abs(bar.velocity) / running_max};
 *       bar.gamma = LorentzFactor{1.0 / std::sqrt(1.0 - bar.beta.value * bar.beta.value)};
 *   }
 *
 *   // After (SIMD-dispatched, same semantics):
 *   srfm::simd::BetaCalculator calc;
 *   auto betas  = calc.computeBetaBatch(velocities);
 *   auto gammas = calc.computeGammaBatch(betas);
 * @endcode
 *
 * NOT thread-safe.  Create one BetaCalculator per thread.
 */
class BetaCalculator {
public:
    /// Construct a fresh calculator with running_max = 0.0.
    BetaCalculator() noexcept;

    /**
     * @brief Compute β_i for a batch of raw price velocities.
     * running_max is advanced to include this batch's maximum |velocity|.
     */
    [[nodiscard]] std::vector<BetaVelocity>
    computeBetaBatch(const std::vector<double>& velocities) noexcept;

    /**
     * @brief Compute γ_i for a batch of beta values.
     */
    [[nodiscard]] std::vector<LorentzFactor>
    computeGammaBatch(const std::vector<BetaVelocity>& betas) noexcept;

    /// Reset running_max to 0.0 (call between trading sessions).
    void reset() noexcept;

    /// Current running maximum (read-only).
    [[nodiscard]] double running_max() const noexcept;

    /// SIMD level selected at construction (reflects current CPU).
    [[nodiscard]] SimdLevel simd_level() const noexcept;

private:
    double    running_max_{ 0.0 };
    SimdLevel simd_level_{ SimdLevel::SCALAR };
};

} // namespace srfm::simd
