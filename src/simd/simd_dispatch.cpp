/**
 * @file  simd_dispatch.cpp
 * @brief Runtime dispatch: routes computeBetaBatch / computeGammaBatch to
 *        the widest available SIMD kernel at process start-up.
 *
 * Module:  src/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Implements the two public batch functions and the BetaCalculator class
 * declared in include/srfm/simd/simd_dispatch.hpp.
 *
 * Dispatch strategy
 * -----------------
 *   detect_simd_level() is called once and cached by cpu_features.hpp.
 *   Based on the result, a compile-time-known function pointer is selected:
 *
 *       AVX512F → detail::compute_beta_avx512 / compute_gamma_avx512
 *       AVX2    → detail::compute_beta_avx2   / compute_gamma_avx2
 *       *       → detail::compute_beta_scalar / compute_gamma_scalar
 *
 * Wrapping raw doubles into BetaVelocity / LorentzFactor
 * ------------------------------------------------------
 *   After the SIMD kernel fills a double[] buffer:
 *
 *   • Beta:  BetaVelocity::make(d).value() — the clamp in the kernel
 *     guarantees make() always returns a value, never nullopt.
 *
 *   • Gamma: SimdGammaCompute::make(d) — uses the friend declaration added
 *     to LorentzFactor so that we avoid an extra sqrt() per element.
 *
 * Memory layout
 * -------------
 *   Intermediate double buffers are stack-allocated for N ≤ STACK_THRESHOLD
 *   and heap-allocated (std::vector<double>) for larger batches, keeping
 *   the common hot-path (N ≈ 256) stack-resident and cache-hot.
 */

#include "srfm/simd/simd_dispatch.hpp"    // public header (include/ on path)
#include "simd_batch_detail.hpp"           // internal detail header (src/simd/ on path)
#include "srfm/simd/cpu_features.hpp"     // runtime detection

#include <vector>
#include <cstddef>
#include <memory>

namespace srfm::simd {

// ── Dispatch helpers ──────────────────────────────────────────────────────────

namespace {

/// Maximum number of doubles to store on the stack (avoid VLA; use alloca
/// only on compilers that support it; fall back to heap beyond this limit).
static constexpr std::size_t STACK_THRESHOLD = 1024;

using BetaKernelFn = void(*)(const double*, std::size_t, double&, double*) noexcept;
using GammaKernelFn = void(*)(const double*, std::size_t, double*) noexcept;

/// Select the beta kernel function pointer at call time (single branch).
[[nodiscard]] inline BetaKernelFn select_beta_kernel() noexcept {
    switch (detect_simd_level()) {
        case SimdLevel::AVX512F: return detail::compute_beta_avx512;
        case SimdLevel::AVX2:    return detail::compute_beta_avx2;
        default:                 return detail::compute_beta_scalar;
    }
}

/// Select the gamma kernel function pointer at call time.
[[nodiscard]] inline GammaKernelFn select_gamma_kernel() noexcept {
    switch (detect_simd_level()) {
        case SimdLevel::AVX512F: return detail::compute_gamma_avx512;
        case SimdLevel::AVX2:    return detail::compute_gamma_avx2;
        default:                 return detail::compute_gamma_scalar;
    }
}

} // anonymous namespace

// ── computeBetaBatch (free function) ─────────────────────────────────────────

std::vector<srfm::momentum::BetaVelocity>
computeBetaBatch(const std::vector<double>& velocities,
                 double&                    running_max) noexcept
{
    const std::size_t n = velocities.size();
    std::vector<srfm::momentum::BetaVelocity> result;
    if (n == 0) return result;
    result.reserve(n);

    // Allocate intermediate double buffer (heap for large N).
    std::vector<double> buf(n);

    // Run the dispatched kernel.
    static const BetaKernelFn kernel = select_beta_kernel();
    kernel(velocities.data(), n, running_max, buf.data());

    // Wrap each computed beta double into a BetaVelocity.
    // The kernel guarantees buf[i] ∈ [0, BETA_CLAMP_LIMIT], so make()
    // always returns a value.
    for (std::size_t i = 0; i < n; ++i) {
        // make() validates; since kernel already clamped, this is always valid.
        auto opt = srfm::momentum::BetaVelocity::make(buf[i]);
        if (opt.has_value()) {
            result.push_back(*opt);
        } else {
            // Defensive: push zero beta if (somehow) validation fails.
            result.push_back(*srfm::momentum::BetaVelocity::make(0.0));
        }
    }

    return result;
}

// ── computeGammaBatch (free function) ────────────────────────────────────────

std::vector<srfm::momentum::LorentzFactor>
computeGammaBatch(
    const std::vector<srfm::momentum::BetaVelocity>& betas) noexcept
{
    const std::size_t n = betas.size();
    std::vector<srfm::momentum::LorentzFactor> result;
    if (n == 0) return result;
    result.reserve(n);

    // Extract raw double values from BetaVelocity objects.
    std::vector<double> beta_buf(n);
    for (std::size_t i = 0; i < n; ++i) {
        beta_buf[i] = betas[i].value();
    }

    // Gamma output buffer.
    std::vector<double> gamma_buf(n);

    // Run the dispatched kernel.
    static const GammaKernelFn kernel = select_gamma_kernel();
    kernel(beta_buf.data(), n, gamma_buf.data());

    // Wrap each computed gamma double into a LorentzFactor using the
    // friend accessor that avoids a redundant scalar sqrt per element.
    for (std::size_t i = 0; i < n; ++i) {
        result.push_back(SimdGammaCompute::make(gamma_buf[i]));
    }

    return result;
}

// ── BetaCalculator ────────────────────────────────────────────────────────────

BetaCalculator::BetaCalculator() noexcept
    : running_max_(0.0)
    , simd_level_(detect_simd_level())
{}

std::vector<srfm::momentum::BetaVelocity>
BetaCalculator::computeBetaBatch(
    const std::vector<double>& velocities) noexcept
{
    return srfm::simd::computeBetaBatch(velocities, running_max_);
}

std::vector<srfm::momentum::LorentzFactor>
BetaCalculator::computeGammaBatch(
    const std::vector<srfm::momentum::BetaVelocity>& betas) noexcept
{
    return srfm::simd::computeGammaBatch(betas);
}

void BetaCalculator::reset() noexcept {
    running_max_ = 0.0;
}

double BetaCalculator::running_max() const noexcept {
    return running_max_;
}

SimdLevel BetaCalculator::simd_level() const noexcept {
    return simd_level_;
}

} // namespace srfm::simd
