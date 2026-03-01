/**
 * @file  gamma_avx2.cpp
 * @brief AVX2 (256-bit, 4-wide) implementation of the gamma batch kernel.
 *
 * Module:  src/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Vectorised gamma_i = 1.0 / sqrt(1.0 - betas[i]^2) for machines with
 * AVX2 but without AVX-512F.  Processes 4 doubles per SIMD cycle.
 *
 * Algorithm
 * ---------
 *   For each 4-element chunk:
 *     1.  Load 4 betas: _mm256_loadu_pd
 *     2.  Clamp to BETA_CLAMP_LIMIT: _mm256_min_pd
 *     3.  Square: _mm256_mul_pd(b, b)
 *     4.  Subtract from 1.0: _mm256_sub_pd(ones, b2)      → denom ∈ (0,1]
 *     5.  sqrt: _mm256_sqrt_pd(denom)
 *     6.  Divide 1.0 by sqrt: _mm256_div_pd(ones, sqrt_d) → gamma
 *     7.  Store: _mm256_storeu_pd
 *   Tail (n % 4 != 0): scalar fallback.
 *
 * Note: This file must be compiled with -mavx2 / /arch:AVX2.
 */

#include "simd_batch_detail.hpp"
#include "momentum/momentum.hpp"

#include <immintrin.h>
#include <cmath>
#include <cstddef>

namespace srfm::simd::detail {

static constexpr double BETA_CLAMP_LIMIT =
    srfm::momentum::BETA_MAX_SAFE - 1.0e-10;

void compute_gamma_avx2(
    const double* SRFM_RESTRICT betas,
    std::size_t                n,
    double* SRFM_RESTRICT       out) noexcept
{
    constexpr std::size_t LANE = 4;

    const __m256d ones    = _mm256_set1_pd(1.0);
    const __m256d clamp_v = _mm256_set1_pd(BETA_CLAMP_LIMIT);

    std::size_t i = 0;

    // ── Vectorised main loop ───────────────────────────────────────────────────
    for (; i + LANE <= n; i += LANE) {
        // 1. Load 4 betas.
        __m256d b = _mm256_loadu_pd(betas + i);

        // 2. Clamp each beta to BETA_CLAMP_LIMIT.
        b = _mm256_min_pd(b, clamp_v);

        // 3. Compute b² = b * b.
        __m256d b2 = _mm256_mul_pd(b, b);

        // 4. Compute 1 - b².  Result is in (0, 1] for valid clamped betas.
        __m256d denom = _mm256_sub_pd(ones, b2);

        // 5. sqrt(1 - b²).
        __m256d sqrt_d = _mm256_sqrt_pd(denom);

        // 6. gamma = 1.0 / sqrt(1 - b²).
        __m256d gamma = _mm256_div_pd(ones, sqrt_d);

        // 7. Store.
        _mm256_storeu_pd(out + i, gamma);
    }

    // ── Scalar tail ───────────────────────────────────────────────────────────
    for (; i < n; ++i) {
        const double b  = (betas[i] > BETA_CLAMP_LIMIT) ? BETA_CLAMP_LIMIT : betas[i];
        const double b2 = b * b;
        out[i]          = 1.0 / std::sqrt(1.0 - b2);
    }
}

} // namespace srfm::simd::detail
