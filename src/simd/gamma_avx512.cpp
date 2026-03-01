/**
 * @file  gamma_avx512.cpp
 * @brief AVX-512F (512-bit, 8-wide) implementation of the gamma batch kernel.
 *
 * Module:  src/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Vectorised gamma_i = 1.0 / sqrt(1.0 - betas[i]^2) for machines with
 * AVX-512F.  Processes 8 doubles per SIMD cycle using 512-bit ZMM registers.
 *
 * Algorithm
 * ---------
 *   For each 8-element chunk:
 *     1.  Load 8 betas:      _mm512_loadu_pd
 *     2.  Clamp to limit:    _mm512_min_pd(b, clamp_v)
 *     3.  Square:            _mm512_mul_pd(b, b)           → b²
 *     4.  Subtract from 1:   _mm512_sub_pd(ones, b2)       → 1 - b² ∈ (0,1]
 *     5.  sqrt:              _mm512_sqrt_pd(denom)
 *     6.  Reciprocal exact:  _mm512_div_pd(ones, sqrt_d)   → γ
 *     7.  Store:             _mm512_storeu_pd
 *   Tail (n % 8 != 0): scalar fallback.
 *
 * Performance Characteristics (Skylake-X / Ice Lake)
 * ---------------------------------------------------
 *   _mm512_sqrt_pd   — throughput 1/cycle (reciprocal), latency 18 cycles
 *   _mm512_div_pd    — throughput 1/cycle (reciprocal), latency 15 cycles
 *   Overall throughput goal: ~8 gammas per ~35 cycles ≈ 229M gammas/sec
 *   at 3.5 GHz with 2 AVX-512 FMA units.
 *
 * Clamp-before-sqrt rationale
 * ---------------------------
 * beta_i is pre-validated to [0, BETA_MAX_SAFE) by compute_beta_avx512.
 * A second clamp here guards against any stale or externally constructed
 * BetaVelocity value whose internal double was set to exactly BETA_MAX_SAFE
 * via future refactoring.  The overhead is one _mm512_min_pd per chunk.
 *
 * Note: This file must be compiled with -mavx512f (GCC/Clang) or
 *       /arch:AVX512 (MSVC) so that the intrinsics are recognised.
 */

#include "simd_batch_detail.hpp"
#include "momentum/momentum.hpp"

#include <immintrin.h>
#include <cmath>
#include <cstddef>

namespace srfm::simd::detail {

static constexpr double BETA_CLAMP_LIMIT =
    srfm::momentum::BETA_MAX_SAFE - 1.0e-10;

void compute_gamma_avx512(
    const double* SRFM_RESTRICT betas,
    std::size_t                n,
    double* SRFM_RESTRICT       out) noexcept
{
    constexpr std::size_t LANE = 8;

    const __m512d ones    = _mm512_set1_pd(1.0);
    const __m512d clamp_v = _mm512_set1_pd(BETA_CLAMP_LIMIT);

    std::size_t i = 0;

    // ── Vectorised main loop (8-wide) ─────────────────────────────────────────
    for (; i + LANE <= n; i += LANE) {
        // 1. Load 8 betas (unaligned).
        __m512d b = _mm512_loadu_pd(betas + i);

        // 2. Clamp betas to BETA_CLAMP_LIMIT to ensure sqrt argument > 0.
        b = _mm512_min_pd(b, clamp_v);

        // 3. b² = b * b.
        __m512d b2 = _mm512_mul_pd(b, b);

        // 4. denom = 1.0 - b²  (strictly positive after clamping).
        __m512d denom = _mm512_sub_pd(ones, b2);

        // 5. sqrt(1 - b²).
        __m512d sqrt_d = _mm512_sqrt_pd(denom);

        // 6. gamma = 1.0 / sqrt(1 - b²).
        __m512d gamma = _mm512_div_pd(ones, sqrt_d);

        // 7. Store 8 results.
        _mm512_storeu_pd(out + i, gamma);
    }

    // ── Scalar tail: n % 8 remaining elements ─────────────────────────────────
    for (; i < n; ++i) {
        const double b  = (betas[i] > BETA_CLAMP_LIMIT) ? BETA_CLAMP_LIMIT : betas[i];
        const double b2 = b * b;
        out[i]          = 1.0 / std::sqrt(1.0 - b2);
    }
}

} // namespace srfm::simd::detail
