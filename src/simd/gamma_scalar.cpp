/**
 * @file  gamma_scalar.cpp
 * @brief Scalar reference implementation of the gamma batch kernel.
 *
 * Module:  src/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Computes gamma_i = 1.0 / sqrt(1.0 - betas[i]^2).
 * Clamps to BETA_CLAMP_LIMIT before sqrt to prevent NaN.
 */

#include "simd_batch_detail.hpp"
#include "srfm/constants.hpp"

#include <cmath>

namespace srfm::simd::detail {

static constexpr double BETA_CLAMP_LIMIT =
    srfm::constants::BETA_MAX_SAFE - 1.0e-10;

void compute_gamma_scalar(
    const double* SRFM_RESTRICT betas,
    std::size_t                 n,
    double* SRFM_RESTRICT       out) noexcept
{
    for (std::size_t i = 0; i < n; ++i) {
        const double b  = (betas[i] > BETA_CLAMP_LIMIT) ? BETA_CLAMP_LIMIT : betas[i];
        const double b2 = b * b;
        out[i]          = 1.0 / std::sqrt(1.0 - b2);
    }
}

} // namespace srfm::simd::detail
