#pragma once
/**
 * @file  simd_batch_detail.hpp
 * @brief Internal raw-double compute signatures for SIMD batch kernels.
 *
 * Module:  src/simd/  (internal — do NOT include from public headers)
 * Owner:   AGT-08  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Declare the low-level SIMD and scalar kernel signatures that operate on
 * raw double pointers.  These are implementation details; callers should
 * use the public API in include/srfm/simd/simd_dispatch.hpp instead.
 *
 * Kernels
 * -------
 *
 * Beta batch (batch-max algorithm):
 *   1. Pass 1: batch_max = max(|velocities[i]|) over all i.
 *   2. running_max = max(running_max, batch_max).
 *   3. Pass 2: out[i] = |velocities[i]| / running_max, clamped.
 *   This guarantees bit-identical results across scalar, AVX2, AVX-512.
 *
 * Gamma batch:
 *   Computes gamma_i = 1.0 / sqrt(1.0 - betas[i]^2) for all i in [0, n).
 *   Clamps betas[i] to BETA_CLAMP_LIMIT before sqrt to prevent NaN.
 *
 * Guarantees
 * ----------
 *   • All functions are noexcept.
 *   • Tail elements (n % LANE != 0) are handled with a scalar fallback.
 *   • Unaligned loads/stores — no alignment requirement on pointers.
 *   • Input and output pointers must not alias (SRFM_RESTRICT).
 */

#include <cstddef>

// ── Portability: restrict keyword ─────────────────────────────────────────────
// GCC/Clang use __restrict__; MSVC uses __restrict (no trailing underscores).
#if defined(_MSC_VER)
#  define SRFM_RESTRICT __restrict
#else
#  define SRFM_RESTRICT __restrict__
#endif

namespace srfm::simd::detail {

// ── Beta kernels ───────────────────────────────────────────────────────────────

void compute_beta_scalar(
    const double* SRFM_RESTRICT velocities,
    std::size_t                 n,
    double&                     running_max,
    double* SRFM_RESTRICT       out) noexcept;

void compute_beta_avx2(
    const double* SRFM_RESTRICT velocities,
    std::size_t                 n,
    double&                     running_max,
    double* SRFM_RESTRICT       out) noexcept;

void compute_beta_avx512(
    const double* SRFM_RESTRICT velocities,
    std::size_t                 n,
    double&                     running_max,
    double* SRFM_RESTRICT       out) noexcept;

// ── Gamma kernels ──────────────────────────────────────────────────────────────

void compute_gamma_scalar(
    const double* SRFM_RESTRICT betas,
    std::size_t                 n,
    double* SRFM_RESTRICT       out) noexcept;

void compute_gamma_avx2(
    const double* SRFM_RESTRICT betas,
    std::size_t                 n,
    double* SRFM_RESTRICT       out) noexcept;

void compute_gamma_avx512(
    const double* SRFM_RESTRICT betas,
    std::size_t                 n,
    double* SRFM_RESTRICT       out) noexcept;

} // namespace srfm::simd::detail
