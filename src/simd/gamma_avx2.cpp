/**
 * @file  gamma_avx2.cpp
 * @brief AVX2 (256-bit, 4-wide) gamma batch kernel.
 *
 * Module:  src/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Computes gamma_i = 1.0 / sqrt(1.0 - betas[i]^2) for 4 lanes per cycle.
 * Clamp → square → subtract from 1 → sqrt → reciprocal.
 * Tail elements use the scalar fallback.
 *
 * Compiled with -mavx2 / /arch:AVX2.
 */

#include "simd_batch_detail.hpp"
#include "srfm/constants.hpp"

#include <immintrin.h>
#include <cmath>

namespace srfm::simd::detail {

static constexpr double BETA_CLAMP_LIMIT =
    srfm::constants::BETA_MAX_SAFE - 1.0e-10;

void compute_gamma_avx2(
    const double* SRFM_RESTRICT betas,
    std::size_t                 n,
    double* SRFM_RESTRICT       out) noexcept
{
    constexpr std::size_t LANE = 4;
    const __m256d ones    = _mm256_set1_pd(1.0);
    const __m256d clamp_v = _mm256_set1_pd(BETA_CLAMP_LIMIT);

    std::size_t i = 0;
    for (; i + LANE <= n; i += LANE) {
        __m256d b      = _mm256_loadu_pd(betas + i);
        b              = _mm256_min_pd(b, clamp_v);
        __m256d b2     = _mm256_mul_pd(b, b);
        __m256d denom  = _mm256_sub_pd(ones, b2);
        __m256d sqrt_d = _mm256_sqrt_pd(denom);
        __m256d gamma  = _mm256_div_pd(ones, sqrt_d);
        _mm256_storeu_pd(out + i, gamma);
    }
    for (; i < n; ++i) {
        const double b  = (betas[i] > BETA_CLAMP_LIMIT) ? BETA_CLAMP_LIMIT : betas[i];
        out[i]          = 1.0 / std::sqrt(1.0 - b * b);
    }
}

} // namespace srfm::simd::detail
