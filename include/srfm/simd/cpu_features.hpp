#pragma once
/**
 * @file  cpu_features.hpp
 * @brief Runtime SIMD capability detection for the SRFM acceleration layer.
 *
 * Module:  include/srfm/simd/
 * Owner:   AGT-08  (AVX-512 SIMD Acceleration)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Probe the executing CPU at start-up for the widest SIMD instruction set
 * available, and expose the result as a strongly-typed enum so that the
 * dispatch layer can select the correct kernel at zero per-call overhead.
 *
 * Supported levels (highest wins):
 *   SCALAR  — always available; pure C++ reference path
 *   SSE42   — SSE 4.2 (128-bit, 2 doubles per register)
 *   AVX2    — AVX2   (256-bit, 4 doubles per register)
 *   AVX512F — AVX-512F (512-bit, 8 doubles per register)
 *
 * Guarantees
 * ----------
 *   • detect_simd_level() is thread-safe: result is computed once.
 *   • No dynamic allocation, no exceptions, noexcept throughout.
 *   • Portable: supports GCC/Clang (__builtin_cpu_supports) and
 *     MSVC (__cpuid / __cpuidex).
 *
 * NOT Responsible For
 * -------------------
 *   • Verifying OS support (XSAVE/XGETBV) — assumed present if CPUID
 *     advertises the feature and the OS is modern (≥ Win10 / Linux ≥ 4.x).
 *   • AVX-512 sub-features beyond F (VL, DQ, BW) — only F is required.
 */

#include <cstdint>

// ── Compiler-specific CPUID helpers ───────────────────────────────────────────

#if defined(_MSC_VER)
#  include <intrin.h>      // __cpuid, __cpuidex
#  include <immintrin.h>   // _xgetbv
#elif defined(__GNUC__) || defined(__clang__)
#  include <cpuid.h>       // __get_cpuid, __get_cpuid_count
#  include <immintrin.h>
#endif

namespace srfm::simd {

// ── SimdLevel ─────────────────────────────────────────────────────────────────

/**
 * @brief Ordered enumeration of SIMD capability tiers.
 *
 * Compare with < / <= to check "at least this level":
 * @code
 *   if (detect_simd_level() >= SimdLevel::AVX2) { ... }
 * @endcode
 */
enum class SimdLevel : std::uint8_t {
    SCALAR  = 0, ///< No SIMD; pure scalar C++ path.
    SSE42   = 1, ///< SSE 4.2 — 128-bit; 2 doubles per YMM half.
    AVX2    = 2, ///< AVX2   — 256-bit; 4 doubles per YMM register.
    AVX512F = 3, ///< AVX-512F — 512-bit; 8 doubles per ZMM register.
};

// ── Internal CPUID utilities ───────────────────────────────────────────────────

namespace detail {

/// Thin portable wrapper around the CPUID instruction.
/// @param leaf       EAX input (function).
/// @param subleaf    ECX input (sub-function).  0 for most leaves.
/// @param out        Output array: [EAX, EBX, ECX, EDX].
inline void cpuid(std::uint32_t leaf, std::uint32_t subleaf,
                  std::uint32_t out[4]) noexcept {
#if defined(_MSC_VER)
    // MSVC intrinsic signature: void __cpuidex(int[4], int, int)
    int regs[4] = {0};
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    for (int i = 0; i < 4; ++i)
        out[i] = static_cast<std::uint32_t>(regs[i]);
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]);
#else
    // Fallback: assume no SIMD
    out[0] = out[1] = out[2] = out[3] = 0u;
#endif
}

/// Read XCR0 (used to verify OS XSAVE support for YMM/ZMM state).
inline std::uint64_t xgetbv(std::uint32_t xcr) noexcept {
#if defined(_MSC_VER)
    return _xgetbv(xcr);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__XSAVE__)
    std::uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(xcr));
    return (static_cast<std::uint64_t>(hi) << 32u) | lo;
#else
    (void)xcr;
    return 0u;  // Cannot read XCR0; assume OS doesn't save YMM/ZMM
#endif
}

/// Probe OS support for YMM (AVX) register save/restore.
inline bool os_saves_ymm() noexcept {
    std::uint32_t regs[4] = {};
    cpuid(0x1u, 0u, regs);
    // ECX bit 27: OSXSAVE
    const bool osxsave = (regs[2] >> 27u) & 1u;
    if (!osxsave) return false;
    // XCR0 bits 1 (SSE) and 2 (AVX/YMM) must both be set
    const std::uint64_t xcr0 = xgetbv(0u);
    return (xcr0 & 0x6u) == 0x6u;
}

/// Probe OS support for ZMM (AVX-512) register save/restore.
inline bool os_saves_zmm() noexcept {
    if (!os_saves_ymm()) return false;
    // XCR0 bits 5 (opmask), 6 (ZMM hi256), 7 (ZMM hi16) must all be set
    const std::uint64_t xcr0 = xgetbv(0u);
    return (xcr0 & 0xE0u) == 0xE0u;
}

} // namespace detail

// ── detect_simd_level ─────────────────────────────────────────────────────────

/**
 * @brief Detect the widest SIMD level available on the executing CPU.
 *
 * The result is computed once on first call and cached in a function-local
 * static, so subsequent calls are effectively free (a single load).
 *
 * Probes (in order, highest wins):
 *   1. AVX-512F   — CPUID leaf 7 EBX bit 16, plus OS ZMM save.
 *   2. AVX2       — CPUID leaf 7 EBX bit 5,  plus OS YMM save.
 *   3. SSE 4.2    — CPUID leaf 1 ECX bit 20.
 *   4. SCALAR     — unconditional fallback.
 *
 * @return The highest SimdLevel the current CPU and OS support.
 *
 * # Panics
 * This function never panics.
 *
 * @example
 * @code
 *   auto level = srfm::simd::detect_simd_level();
 *   if (level >= srfm::simd::SimdLevel::AVX512F) {
 *       // use AVX-512 path
 *   }
 * @endcode
 */
[[nodiscard]] inline SimdLevel detect_simd_level() noexcept {
    // Function-local static — initialized exactly once, thread-safe per C++11.
    static const SimdLevel kLevel = []() noexcept -> SimdLevel {
        // ── Leaf 0: get max supported leaf ───────────────────────────────────
        std::uint32_t max_leaf[4] = {};
        detail::cpuid(0u, 0u, max_leaf);
        const std::uint32_t max_func = max_leaf[0];

        // ── Leaf 1: SSE4.2 ───────────────────────────────────────────────────
        if (max_func < 1u) return SimdLevel::SCALAR;
        std::uint32_t leaf1[4] = {};
        detail::cpuid(1u, 0u, leaf1);

        const bool sse42 = (leaf1[2] >> 20u) & 1u;   // ECX bit 20

        // ── Leaf 7: AVX2, AVX-512F ───────────────────────────────────────────
        if (max_func < 7u) return sse42 ? SimdLevel::SSE42 : SimdLevel::SCALAR;
        std::uint32_t leaf7[4] = {};
        detail::cpuid(7u, 0u, leaf7);

        const bool avx2    = (leaf7[1] >> 5u)  & 1u;  // EBX bit 5
        const bool avx512f = (leaf7[1] >> 16u) & 1u;  // EBX bit 16

        // ── Check OS register-save support ───────────────────────────────────
        if (avx512f && detail::os_saves_zmm()) return SimdLevel::AVX512F;
        if (avx2    && detail::os_saves_ymm()) return SimdLevel::AVX2;
        if (sse42)                              return SimdLevel::SSE42;
        return SimdLevel::SCALAR;
    }();

    return kLevel;
}

// ── Convenience predicates ────────────────────────────────────────────────────

/// Returns true when AVX-512F is available on this CPU/OS.
[[nodiscard]] inline bool has_avx512f() noexcept {
    return detect_simd_level() >= SimdLevel::AVX512F;
}

/// Returns true when AVX2 is available on this CPU/OS.
[[nodiscard]] inline bool has_avx2() noexcept {
    return detect_simd_level() >= SimdLevel::AVX2;
}

/// Returns true when SSE 4.2 is available on this CPU/OS.
[[nodiscard]] inline bool has_sse42() noexcept {
    return detect_simd_level() >= SimdLevel::SSE42;
}

/// Human-readable name for a SimdLevel value.
[[nodiscard]] inline const char* simd_level_name(SimdLevel level) noexcept {
    switch (level) {
        case SimdLevel::SCALAR:  return "SCALAR";
        case SimdLevel::SSE42:   return "SSE42";
        case SimdLevel::AVX2:    return "AVX2";
        case SimdLevel::AVX512F: return "AVX512F";
        default:                 return "UNKNOWN";
    }
}

} // namespace srfm::simd
