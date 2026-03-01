/**
 * @file  gamma_scalar.cpp
 * @brief Scalar reference implementation of the gamma batch kernel.
 *
 * Module:  src/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Compute gamma_i = 1.0 / sqrt(1.0 - betas[i]^2) for all i in [0, n).
 * This is the authoritative reference implementation.  All SIMD variants
 * must produce results within 1 ULP of the scalar path for any valid input.
 *
 * Correctness Notes
 * -----------------
 *   • Input betas are expected to be in [0, BETA_MAX_SAFE) — the same range
 *     accepted by BetaVelocity::make().
 *   • A defensive clamp to BETA_CLAMP_LIMIT is applied before the sqrt to
 *     ensure the argument of sqrt is strictly positive even in edge cases.
 *   • gamma ≥ 1.0 for all valid betas; the result is always finite.
 */

#include "simd_batch_detail.hpp"
#include "momentum/momentum.hpp"  // BETA_MAX_SAFE

#include <cmath>

namespace srfm::simd::detail {

static constexpr double BETA_CLAMP_LIMIT =
    srfm::momentum::BETA_MAX_SAFE - 1.0e-10;

void compute_gamma_scalar(
    const double* SRFM_RESTRICT betas,
    std::size_t                n,
    double* SRFM_RESTRICT       out) noexcept
{
    for (std::size_t i = 0; i < n; ++i) {
        // Clamp to ensure sqrt argument > 0.
        const double b     = (betas[i] > BETA_CLAMP_LIMIT) ? BETA_CLAMP_LIMIT : betas[i];
        const double b2    = b * b;
        const double denom = 1.0 - b2;          // in (0, 1] for valid betas
        out[i]             = 1.0 / std::sqrt(denom);
    }
}

} // namespace srfm::simd::detail
