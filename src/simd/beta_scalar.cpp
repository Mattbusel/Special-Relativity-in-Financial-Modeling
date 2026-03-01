/**
 * @file  beta_scalar.cpp
 * @brief Scalar reference implementation of the beta batch kernel.
 *
 * Module:  src/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Algorithm: Batch-max
 * --------------------
 * 1. Pass 1: batch_max = max(|velocities[i]|) over entire batch.
 * 2. running_max = max(running_max, batch_max).
 * 3. Pass 2: out[i] = |velocities[i]| / running_max, clamped to BETA_CLAMP_LIMIT.
 *
 * Using batch-max (not per-element) ensures bit-identical results with all
 * SIMD variants and allows the division pass to be fully vectorised.
 */

#include "simd_batch_detail.hpp"
#include "srfm/constants.hpp"

#include <cmath>

namespace srfm::simd::detail {

static constexpr double BETA_CLAMP_LIMIT =
    srfm::constants::BETA_MAX_SAFE - 1.0e-10;

void compute_beta_scalar(
    const double* SRFM_RESTRICT velocities,
    std::size_t                 n,
    double&                     running_max,
    double* SRFM_RESTRICT       out) noexcept
{
    if (n == 0) return;

    // Pass 1: batch max
    double batch_max = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = std::abs(velocities[i]);
        if (a > batch_max) batch_max = a;
    }
    if (batch_max > running_max) running_max = batch_max;

    // Pass 2: compute betas
    const double denom = (running_max > 0.0) ? running_max : 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        double beta = std::abs(velocities[i]) / denom;
        if (beta > BETA_CLAMP_LIMIT) beta = BETA_CLAMP_LIMIT;
        out[i] = beta;
    }
}

} // namespace srfm::simd::detail
