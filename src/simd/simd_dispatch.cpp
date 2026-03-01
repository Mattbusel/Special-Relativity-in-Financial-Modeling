/**
 * @file  simd_dispatch.cpp
 * @brief Runtime dispatch: routes computeBetaBatch / computeGammaBatch to
 *        the widest available SIMD kernel at process start-up.
 *
 * Module:  src/simd/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Dispatch strategy
 * -----------------
 *   detect_simd_level() is called once (cached by cpu_features.hpp) and a
 *   function pointer is selected:
 *
 *       AVX512F → detail::compute_beta_avx512 / compute_gamma_avx512
 *       AVX2    → detail::compute_beta_avx2   / compute_gamma_avx2
 *       *       → detail::compute_beta_scalar / compute_gamma_scalar
 *
 * Wrapping raw doubles into BetaVelocity / LorentzFactor
 * ------------------------------------------------------
 *   srfm::BetaVelocity and srfm::LorentzFactor are plain structs with a
 *   public `value` field (defined in srfm/types.hpp), so no friend access
 *   or factory functions are required — direct aggregate initialisation
 *   suffices: BetaVelocity{buf[i]}, LorentzFactor{buf[i]}.
 */

#include "srfm/simd/simd_dispatch.hpp"
#include "simd_batch_detail.hpp"
#include "srfm/simd/cpu_features.hpp"

#include <vector>
#include <cstddef>

namespace srfm::simd {

// ── Kernel selection ──────────────────────────────────────────────────────────

namespace {

using BetaKernelFn  = void(*)(const double*, std::size_t, double&, double*) noexcept;
using GammaKernelFn = void(*)(const double*, std::size_t, double*) noexcept;

[[nodiscard]] inline BetaKernelFn select_beta_kernel() noexcept {
    switch (detect_simd_level()) {
        case SimdLevel::AVX512F: return detail::compute_beta_avx512;
        case SimdLevel::AVX2:    return detail::compute_beta_avx2;
        default:                 return detail::compute_beta_scalar;
    }
}

[[nodiscard]] inline GammaKernelFn select_gamma_kernel() noexcept {
    switch (detect_simd_level()) {
        case SimdLevel::AVX512F: return detail::compute_gamma_avx512;
        case SimdLevel::AVX2:    return detail::compute_gamma_avx2;
        default:                 return detail::compute_gamma_scalar;
    }
}

} // anonymous namespace

// ── computeBetaBatch ─────────────────────────────────────────────────────────

std::vector<BetaVelocity>
computeBetaBatch(const std::vector<double>& velocities,
                 double&                    running_max) noexcept
{
    const std::size_t n = velocities.size();
    std::vector<BetaVelocity> result;
    if (n == 0) return result;
    result.reserve(n);

    std::vector<double> buf(n);

    static const BetaKernelFn kernel = select_beta_kernel();
    kernel(velocities.data(), n, running_max, buf.data());

    // BetaVelocity is a plain struct: BetaVelocity{val} — no validation needed.
    // The kernel guarantees buf[i] ∈ [0, BETA_CLAMP_LIMIT] ⊂ [0, BETA_MAX_SAFE).
    for (std::size_t i = 0; i < n; ++i) {
        result.push_back(BetaVelocity{buf[i]});
    }
    return result;
}

// ── computeGammaBatch ─────────────────────────────────────────────────────────

std::vector<LorentzFactor>
computeGammaBatch(const std::vector<BetaVelocity>& betas) noexcept
{
    const std::size_t n = betas.size();
    std::vector<LorentzFactor> result;
    if (n == 0) return result;
    result.reserve(n);

    // Extract raw doubles.
    std::vector<double> beta_buf(n);
    for (std::size_t i = 0; i < n; ++i) {
        beta_buf[i] = betas[i].value;
    }

    std::vector<double> gamma_buf(n);
    static const GammaKernelFn kernel = select_gamma_kernel();
    kernel(beta_buf.data(), n, gamma_buf.data());

    // LorentzFactor is a plain struct — direct aggregate initialisation.
    for (std::size_t i = 0; i < n; ++i) {
        result.push_back(LorentzFactor{gamma_buf[i]});
    }
    return result;
}

// ── BetaCalculator ────────────────────────────────────────────────────────────

BetaCalculator::BetaCalculator() noexcept
    : running_max_(0.0)
    , simd_level_(detect_simd_level())
{}

std::vector<BetaVelocity>
BetaCalculator::computeBetaBatch(const std::vector<double>& velocities) noexcept {
    return srfm::simd::computeBetaBatch(velocities, running_max_);
}

std::vector<LorentzFactor>
BetaCalculator::computeGammaBatch(const std::vector<BetaVelocity>& betas) noexcept {
    return srfm::simd::computeGammaBatch(betas);
}

void BetaCalculator::reset() noexcept { running_max_ = 0.0; }

double BetaCalculator::running_max() const noexcept { return running_max_; }

SimdLevel BetaCalculator::simd_level() const noexcept { return simd_level_; }

} // namespace srfm::simd
