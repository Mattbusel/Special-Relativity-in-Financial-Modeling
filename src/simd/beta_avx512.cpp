/**
 * @file  beta_avx512.cpp
 * @brief AVX-512F (512-bit, 8-wide) implementation of the beta batch kernel.
 *
 * Module:  src/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Algorithm: Batch-max (8-wide SIMD)
 * ------------------------------------
 * 1. Pass 1 (vectorised): compute batch_max using 8-wide abs + _mm512_reduce_max_pd.
 * 2. Update scalar running_max.
 * 3. Pass 2 (vectorised): broadcast running_max, divide |v_i|/denom, clamp.
 * 4. Tail (scalar): handle n % 8 remaining elements.
 *
 * Using the batch maximum as the single denominator for all elements in one
 * call means the results are bit-identical to compute_beta_scalar() (which
 * uses the same batch-max algorithm) regardless of SIMD width.
 *
 * Note: Compiled with -mavx512f / /arch:AVX512.
 */

#include "simd_batch_detail.hpp"
#include "momentum/momentum.hpp"

#include <immintrin.h>
#include <cmath>

namespace srfm::simd::detail {

static constexpr double BETA_CLAMP_LIMIT =
    srfm::momentum::BETA_MAX_SAFE - 1.0e-10;

void compute_beta_avx512(
    const double* SRFM_RESTRICT velocities,
    std::size_t                 n,
    double&                     running_max,
    double* SRFM_RESTRICT       out) noexcept
{
    if (n == 0) return;

    constexpr std::size_t LANE = 8;

    // ── Pass 1: compute batch_max ────────────────────────────────────────────
    __m512d vmax = _mm512_setzero_pd();
    std::size_t i = 0;
    for (; i + LANE <= n; i += LANE) {
        __m512d v     = _mm512_loadu_pd(velocities + i);
        __m512d abs_v = _mm512_abs_pd(v);
        vmax = _mm512_max_pd(vmax, abs_v);
    }
    // Reduce the 8-lane vector to a single scalar max.
    double batch_max = _mm512_reduce_max_pd(vmax);
    // Scalar tail for pass 1
    for (; i < n; ++i) {
        const double a = std::abs(velocities[i]);
        if (a > batch_max) batch_max = a;
    }

    // Update running max
    if (batch_max > running_max) running_max = batch_max;
    const double denom = (running_max > 0.0) ? running_max : 1.0;

    // ── Pass 2: compute betas ────────────────────────────────────────────────
    const __m512d denom_v = _mm512_set1_pd(denom);
    const __m512d clamp_v = _mm512_set1_pd(BETA_CLAMP_LIMIT);

    i = 0;
    for (; i + LANE <= n; i += LANE) {
        __m512d v     = _mm512_loadu_pd(velocities + i);
        __m512d abs_v = _mm512_abs_pd(v);
        __m512d beta  = _mm512_div_pd(abs_v, denom_v);
        beta          = _mm512_min_pd(beta, clamp_v);
        _mm512_storeu_pd(out + i, beta);
    }
    // Scalar tail for pass 2
    for (; i < n; ++i) {
        const double abs_v = std::abs(velocities[i]);
        double beta = abs_v / denom;
        if (beta > BETA_CLAMP_LIMIT) beta = BETA_CLAMP_LIMIT;
        out[i] = beta;
    }
}

} // namespace srfm::simd::detail
