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
 * Beta batch:
 *   Computes beta_i = |velocities[i]| / running_max for all i in [0, n).
 *   running_max is updated in-place: at each chunk boundary it is set to
 *   max(running_max, max(|velocities[chunk]|)).  This guarantees:
 *     •  beta_i ∈ [0.0, BETA_MAX_SAFE]  always
 *     •  running_max is monotonically non-decreasing across calls
 *
 * Gamma batch:
 *   Computes gamma_i = 1.0 / sqrt(1.0 − betas[i]²) for all i in [0, n).
 *   Input betas must be pre-validated (0 ≤ betas[i] ≤ BETA_MAX_SAFE).
 *   Clamping is applied before sqrt to prevent NaN.
 *
 * Guarantees
 * ----------
 *   • All functions are noexcept.
 *   • Tail elements (n % LANE != 0) are handled with a scalar fallback.
 *   • Output pointers must be writable for n doubles; they need not be
 *     aligned (all loads/stores use unaligned intrinsics).
 *   • Input and output pointers must not alias (SRFM_RESTRICT).
 *
 * NOT Responsible For
 *   • Wrapping outputs into BetaVelocity / LorentzFactor (dispatch layer).
 *   • Selecting the correct kernel at runtime (simd_dispatch.cpp).
 */

#include <cstddef>

// ── Portability: restrict keyword ─────────────────────────────────────────────
// GCC/Clang use SRFM_RESTRICT; MSVC uses __restrict (no trailing underscores).
#if defined(_MSC_VER)
#  define SRFM_RESTRICT __restrict
#else
#  define SRFM_RESTRICT SRFM_RESTRICT
#endif

namespace srfm::simd::detail {

// ── Beta kernels ───────────────────────────────────────────────────────────────

/**
 * @brief Scalar reference implementation of the beta batch kernel.
 *
 * Always available regardless of hardware.  Used as the authoritative
 * correctness reference and as the tail handler for wider kernels.
 *
 * @param velocities  Input price velocities (any finite double).
 * @param n           Number of elements.
 * @param running_max Current running maximum of |velocity|; updated in-place.
 * @param out         Output beta values, one per input velocity.
 */
void compute_beta_scalar(
    const double* SRFM_RESTRICT velocities,
    std::size_t                n,
    double&                    running_max,
    double* SRFM_RESTRICT       out) noexcept;

/**
 * @brief AVX2 (256-bit, 4-wide) beta batch kernel.
 *
 * Requires AVX2 support (detected at runtime).  Falls back to scalar for
 * tail elements when n % 4 != 0.
 */
void compute_beta_avx2(
    const double* SRFM_RESTRICT velocities,
    std::size_t                n,
    double&                    running_max,
    double* SRFM_RESTRICT       out) noexcept;

/**
 * @brief AVX-512F (512-bit, 8-wide) beta batch kernel.
 *
 * Requires AVX-512F support (detected at runtime).  Falls back to scalar
 * for tail elements when n % 8 != 0.
 */
void compute_beta_avx512(
    const double* SRFM_RESTRICT velocities,
    std::size_t                n,
    double&                    running_max,
    double* SRFM_RESTRICT       out) noexcept;

// ── Gamma kernels ──────────────────────────────────────────────────────────────

/**
 * @brief Scalar reference implementation of the gamma batch kernel.
 *
 * Computes gamma_i = 1.0 / sqrt(1.0 - betas[i]^2).
 * Clamps betas[i] to BETA_CLAMP_SAFE before the sqrt to prevent NaN.
 *
 * @param betas   Input beta values in [0, BETA_MAX_SAFE].
 * @param n       Number of elements.
 * @param out     Output gamma values, one per input beta.
 */
void compute_gamma_scalar(
    const double* SRFM_RESTRICT betas,
    std::size_t                n,
    double* SRFM_RESTRICT       out) noexcept;

/**
 * @brief AVX2 (256-bit, 4-wide) gamma batch kernel.
 */
void compute_gamma_avx2(
    const double* SRFM_RESTRICT betas,
    std::size_t                n,
    double* SRFM_RESTRICT       out) noexcept;

/**
 * @brief AVX-512F (512-bit, 8-wide) gamma batch kernel.
 */
void compute_gamma_avx512(
    const double* SRFM_RESTRICT betas,
    std::size_t                n,
    double* SRFM_RESTRICT       out) noexcept;

} // namespace srfm::simd::detail
