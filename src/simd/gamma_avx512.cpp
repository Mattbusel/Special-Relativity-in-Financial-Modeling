/**
 * @file  gamma_avx512.cpp
 * @brief AVX-512F (512-bit, 8-wide) gamma batch kernel.
 *
 * Module:  src/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Computes gamma_i = 1.0 / sqrt(1.0 - betas[i]^2) for 8 lanes per cycle.
 * Clamp → square → subtract from 1 → _mm512_sqrt_pd → _mm512_div_pd.
 * Tail elements use the scalar fallback.
 *
 * Compiled with -mavx512f / /arch:AVX512.
 */

#include "simd_batch_detail.hpp"
#include "srfm/constants.hpp"

#include <immintrin.h>
#include <cmath>

namespace srfm::simd::detail {

static constexpr double BETA_CLAMP_LIMIT =
    srfm::constants::BETA_MAX_SAFE - 1.0e-10;

void compute_gamma_avx512(
    const double* SRFM_RESTRICT betas,
    std::size_t                 n,
    double* SRFM_RESTRICT       out) noexcept
{
    constexpr std::size_t LANE = 8;
    const __m512d ones    = _mm512_set1_pd(1.0);
    const __m512d clamp_v = _mm512_set1_pd(BETA_CLAMP_LIMIT);

    std::size_t i = 0;
    for (; i + LANE <= n; i += LANE) {
        __m512d b      = _mm512_loadu_pd(betas + i);
        b              = _mm512_min_pd(b, clamp_v);
        __m512d b2     = _mm512_mul_pd(b, b);
        __m512d denom  = _mm512_sub_pd(ones, b2);
        __m512d sqrt_d = _mm512_sqrt_pd(denom);
        __m512d gamma  = _mm512_div_pd(ones, sqrt_d);
        _mm512_storeu_pd(out + i, gamma);
    }
    for (; i < n; ++i) {
        const double b = (betas[i] > BETA_CLAMP_LIMIT) ? BETA_CLAMP_LIMIT : betas[i];
        out[i]         = 1.0 / std::sqrt(1.0 - b * b);
    }
}

} // namespace srfm::simd::detail
