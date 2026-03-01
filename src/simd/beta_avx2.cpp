/**
 * @file  beta_avx2.cpp
 * @brief AVX2 (256-bit, 4-wide) beta batch kernel.
 *
 * Module:  src/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Algorithm: 2-pass batch-max (4-wide)
 *   Pass 1: SIMD abs + hmax → batch_max; update running_max.
 *   Pass 2: SIMD div + clamp → out[i] = |v_i| / running_max.
 *   Tail: scalar fallback for n % 4 != 0.
 *
 * Results are bit-identical to compute_beta_scalar().
 * Compiled with -mavx2 / /arch:AVX2.
 */

#include "simd_batch_detail.hpp"
#include "srfm/constants.hpp"

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace srfm::simd::detail {

static constexpr double BETA_CLAMP_LIMIT =
    srfm::constants::BETA_MAX_SAFE - 1.0e-10;

static constexpr std::uint64_t ABS_MASK_U64 = 0x7FFF'FFFF'FFFF'FFFFu;

[[nodiscard]] static inline double hmax_pd_avx2(__m256d v) noexcept {
    __m256d hi  = _mm256_permute2f128_pd(v, v, 0x01);
    __m256d mx1 = _mm256_max_pd(v, hi);
    __m256d swp = _mm256_permute_pd(mx1, 0x05);
    __m256d mx2 = _mm256_max_pd(mx1, swp);
    return _mm256_cvtsd_f64(mx2);
}

void compute_beta_avx2(
    const double* SRFM_RESTRICT velocities,
    std::size_t                 n,
    double&                     running_max,
    double* SRFM_RESTRICT       out) noexcept
{
    if (n == 0) return;

    constexpr std::size_t LANE = 4;
    const __m256d abs_mask = _mm256_castsi256_pd(
        _mm256_set1_epi64x(static_cast<long long>(ABS_MASK_U64)));

    // Pass 1: compute batch_max
    __m256d vmax = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + LANE <= n; i += LANE) {
        __m256d v     = _mm256_loadu_pd(velocities + i);
        __m256d abs_v = _mm256_and_pd(v, abs_mask);
        vmax = _mm256_max_pd(vmax, abs_v);
    }
    double batch_max = hmax_pd_avx2(vmax);
    for (; i < n; ++i) {
        const double a = std::abs(velocities[i]);
        if (a > batch_max) batch_max = a;
    }
    if (batch_max > running_max) running_max = batch_max;
    const double denom = (running_max > 0.0) ? running_max : 1.0;

    // Pass 2: compute betas
    const __m256d denom_v = _mm256_set1_pd(denom);
    const __m256d clamp_v = _mm256_set1_pd(BETA_CLAMP_LIMIT);
    i = 0;
    for (; i + LANE <= n; i += LANE) {
        __m256d v     = _mm256_loadu_pd(velocities + i);
        __m256d abs_v = _mm256_and_pd(v, abs_mask);
        __m256d beta  = _mm256_div_pd(abs_v, denom_v);
        beta          = _mm256_min_pd(beta, clamp_v);
        _mm256_storeu_pd(out + i, beta);
    }
    for (; i < n; ++i) {
        double beta = std::abs(velocities[i]) / denom;
        if (beta > BETA_CLAMP_LIMIT) beta = BETA_CLAMP_LIMIT;
        out[i] = beta;
    }
}

} // namespace srfm::simd::detail
