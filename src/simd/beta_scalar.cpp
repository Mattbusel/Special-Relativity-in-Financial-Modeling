/**
 * @file  beta_scalar.cpp
 * @brief Scalar reference implementation of the beta batch kernel.
 *
 * Module:  src/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Algorithm: Batch-max
 * --------------------
 * 1. Compute batch_max = max(|velocities[i]|) over all i in [0, n).
 * 2. Update running_max = max(running_max, batch_max).
 * 3. For each i: beta_i = |velocities[i]| / running_max.
 * 4. Clamp each beta_i to BETA_CLAMP_LIMIT.
 *
 * Why batch-max (not element-wise max)?
 * --------------------------------------
 * Using the maximum of the *entire batch* as the denominator allows all SIMD
 * variants to produce bit-identical results: AVX-512, AVX2, and scalar all
 * compute the same running_max and perform identical divisions.
 * Element-wise running_max updates introduce a serial dependency that breaks
 * SIMD parallelism AND causes different betas depending on vector width.
 *
 * Correctness: beta_i = |v_i| / running_max ≤ 1.0 because running_max ≥ batch_max ≥ |v_i|.
 * The invariant beta_i ∈ [0, BETA_MAX_SAFE) is always preserved.
 */

#include "simd_batch_detail.hpp"
#include "momentum/momentum.hpp"

#include <cmath>
#include <algorithm>

namespace srfm::simd::detail {

static constexpr double BETA_CLAMP_LIMIT =
    srfm::momentum::BETA_MAX_SAFE - 1.0e-10;

void compute_beta_scalar(
    const double* SRFM_RESTRICT velocities,
    std::size_t                 n,
    double&                     running_max,
    double* SRFM_RESTRICT       out) noexcept
{
    if (n == 0) return;

    // ── Pass 1: find batch max ───────────────────────────────────────────────
    double batch_max = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double abs_v = std::abs(velocities[i]);
        if (abs_v > batch_max) batch_max = abs_v;
    }

    // ── Update running max ───────────────────────────────────────────────────
    if (batch_max > running_max) running_max = batch_max;

    // ── Pass 2: compute betas ────────────────────────────────────────────────
    // If running_max is 0, all velocities are 0, so all betas are 0.
    const double denom = (running_max > 0.0) ? running_max : 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double abs_v = std::abs(velocities[i]);
        double beta = abs_v / denom;
        if (beta > BETA_CLAMP_LIMIT) beta = BETA_CLAMP_LIMIT;
        out[i] = beta;
    }
}

} // namespace srfm::simd::detail
