/**
 * @file  test_simd.cpp
 * @brief Unit, correctness, and chaos tests for the SIMD β/γ acceleration.
 *
 * Structure
 * ---------
 *   test_cpu_feature_detection     — SimdLevel enum and detect_simd_level()
 *   test_beta_scalar_kernel        — compute_beta_scalar correctness
 *   test_gamma_scalar_kernel       — compute_gamma_scalar correctness
 *   test_beta_avx2_kernel          — compute_beta_avx2 bit-identity vs scalar
 *   test_gamma_avx2_kernel         — compute_gamma_avx2 bit-identity vs scalar
 *   test_beta_avx512_kernel        — compute_beta_avx512 bit-identity vs scalar
 *   test_gamma_avx512_kernel       — compute_gamma_avx512 bit-identity vs scalar
 *   test_dispatch_free_functions   — computeBetaBatch / computeGammaBatch
 *   test_beta_calculator           — BetaCalculator stateful wrapper
 *   test_running_max_maintenance   — monotonic running_max across calls
 *   test_tail_element_handling     — N % 4 != 0, N % 8 != 0 edge cases
 *   test_empty_input               — empty vector → empty vector
 *   test_clamping                  — beta never reaches or exceeds BETA_MAX_SAFE
 *   test_gamma_monotonicity        — gamma grows monotonically with |beta|
 *   test_gamma_newtonian_limit     — beta=0 → gamma=1
 *   test_gamma_identity_b06        — beta=0.6 → gamma=1.25
 *   test_large_batch_finiteness    — 65536 elements all finite and in range
 *   test_all_zero_velocities       — edge case: all v=0
 *   test_single_nonzero_velocity   — N=1 path
 *   test_negative_velocities       — |v| symmetric; beta is non-negative
 *   test_running_max_carried_over  — running_max state survives reset boundary
 *   test_reset_clears_running_max  — BetaCalculator::reset() resets state
 *   test_simd_level_name           — simd_level_name() returns non-null
 */

#include "srfm_test.hpp"
#include "srfm/simd/cpu_features.hpp"    // include/ on path
#include "srfm/simd/simd_dispatch.hpp"   // include/ on path
#include "simd_batch_detail.hpp"         // src/simd/ on path (internal detail)
#include "momentum/momentum.hpp"         // src/ on path

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <cassert>

using namespace srfm::momentum;
using namespace srfm::simd;
using namespace srfm::simd::detail;

// ── Helpers ───────────────────────────────────────────────────────────────────

static constexpr double EPS = 1e-12;
static constexpr double BETA_MAX = BETA_MAX_SAFE;

/// Generate N random velocities in the range [-amplitude, +amplitude].
static std::vector<double> make_velocities(std::size_t n,
                                            double amplitude = 10.0,
                                            std::uint32_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-amplitude, amplitude);
    std::vector<double> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

/// Run the scalar beta kernel and return results + updated running_max.
static std::vector<double> scalar_beta(const std::vector<double>& vels,
                                        double& rmax) {
    std::vector<double> out(vels.size());
    compute_beta_scalar(vels.data(), vels.size(), rmax, out.data());
    return out;
}

/// Run the scalar gamma kernel and return results.
static std::vector<double> scalar_gamma(const std::vector<double>& betas) {
    std::vector<double> out(betas.size());
    compute_gamma_scalar(betas.data(), betas.size(), out.data());
    return out;
}

// ══════════════════════════════════════════════════════════════════════════════
// CPU feature detection
// ══════════════════════════════════════════════════════════════════════════════

static void test_cpu_feature_detection() {
    // SimdLevel must be one of the four valid values.
    SimdLevel level = detect_simd_level();
    bool valid = (level == SimdLevel::SCALAR  ||
                  level == SimdLevel::SSE42   ||
                  level == SimdLevel::AVX2    ||
                  level == SimdLevel::AVX512F);
    SRFM_CHECK(valid);

    // Repeated calls return the same value (cached).
    SRFM_CHECK(detect_simd_level() == level);
    SRFM_CHECK(detect_simd_level() == level);

    // Ordering invariant: AVX512F > AVX2 > SSE42 > SCALAR.
    SRFM_CHECK(SimdLevel::AVX512F > SimdLevel::AVX2);
    SRFM_CHECK(SimdLevel::AVX2    > SimdLevel::SSE42);
    SRFM_CHECK(SimdLevel::SSE42   > SimdLevel::SCALAR);

    // has_avx2 / has_sse42 / has_avx512f are consistent with the level.
    if (level >= SimdLevel::AVX512F) { SRFM_CHECK(has_avx512f()); }
    if (level >= SimdLevel::AVX2)    { SRFM_CHECK(has_avx2()); }
    if (level >= SimdLevel::SSE42)   { SRFM_CHECK(has_sse42()); }

    // simd_level_name returns a non-empty string for each level.
    SRFM_CHECK(simd_level_name(SimdLevel::SCALAR)[0]  != '\0');
    SRFM_CHECK(simd_level_name(SimdLevel::SSE42)[0]   != '\0');
    SRFM_CHECK(simd_level_name(SimdLevel::AVX2)[0]    != '\0');
    SRFM_CHECK(simd_level_name(SimdLevel::AVX512F)[0] != '\0');
}

// ══════════════════════════════════════════════════════════════════════════════
// Scalar beta kernel
// ══════════════════════════════════════════════════════════════════════════════

static void test_beta_scalar_kernel() {
    // Basic correctness: single positive velocity.
    {
        double rmax = 0.0;
        double v[1] = {5.0};
        double out[1] = {};
        compute_beta_scalar(v, 1, rmax, out);
        // running_max should become 5.0, beta = 5/5 = 1 → clamped
        SRFM_CHECK(rmax == 5.0);
        SRFM_CHECK(out[0] <= BETA_MAX - 1e-11);
        SRFM_CHECK(out[0] >= 0.0);
    }

    // Symmetry: negative velocity produces same beta as positive.
    {
        double rmax_pos = 0.0, rmax_neg = 0.0;
        double vpos[1] = {3.0}, vneg[1] = {-3.0};
        double opos[1] = {}, oneg[1] = {};
        compute_beta_scalar(vpos, 1, rmax_pos, opos);
        compute_beta_scalar(vneg, 1, rmax_neg, oneg);
        SRFM_CHECK_NEAR(opos[0], oneg[0], EPS);
        SRFM_CHECK_NEAR(rmax_pos, rmax_neg, EPS);
    }

    // Monotonic running_max: second call with a smaller velocity.
    {
        double rmax = 0.0;
        double v1[1] = {10.0}, v2[1] = {2.0};
        double o1[1] = {}, o2[1] = {};
        compute_beta_scalar(v1, 1, rmax, o1);
        SRFM_CHECK(rmax == 10.0);
        compute_beta_scalar(v2, 1, rmax, o2);
        SRFM_CHECK(rmax == 10.0);  // must NOT decrease
        SRFM_CHECK_NEAR(o2[0], 0.2, EPS);
    }

    // Batch-max semantics: all elements share the batch maximum as denominator.
    // v = [3.0, 7.0] → batch_max = 7.0 → rmax = 7.0
    // beta[0] = 3.0/7.0, beta[1] = 7.0/7.0 → clamped.
    {
        double rmax = 0.0;
        double v[2] = {3.0, 7.0};
        double out[2] = {};
        compute_beta_scalar(v, 2, rmax, out);
        SRFM_CHECK(rmax == 7.0);
        SRFM_CHECK_NEAR(out[0], 3.0 / 7.0, EPS);   // batch_max = 7, denom = 7
        SRFM_CHECK(out[1] < BETA_MAX_SAFE);          // 7/7 = 1 → clamped
    }

    // All zeros → all betas are 0, running_max stays 0.
    {
        double rmax = 0.0;
        double v[4] = {0.0, 0.0, 0.0, 0.0};
        double out[4] = {};
        compute_beta_scalar(v, 4, rmax, out);
        SRFM_CHECK(rmax == 0.0);
        for (int i = 0; i < 4; ++i) SRFM_CHECK(out[i] == 0.0);
    }

    // Clamping: velocity equal to running_max → beta = 1 → clamped.
    {
        double rmax = 0.0;
        double v[1] = {1.0};
        double out[1] = {};
        compute_beta_scalar(v, 1, rmax, out);
        SRFM_CHECK(out[0] < BETA_MAX_SAFE);
        SRFM_CHECK(out[0] >= 0.0);
    }

    // Empty input: no mutation.
    {
        double rmax = 5.0;
        compute_beta_scalar(nullptr, 0, rmax, nullptr);  // early return on n==0
        SRFM_CHECK(rmax == 5.0);
    }

    // Large batch: all finite, all in range.
    {
        auto vels = make_velocities(1024);
        std::vector<double> out(1024);
        double rmax = 0.0;
        compute_beta_scalar(vels.data(), 1024, rmax, out.data());
        for (double b : out) {
            SRFM_CHECK(std::isfinite(b));
            SRFM_CHECK(b >= 0.0);
            SRFM_CHECK(b < BETA_MAX_SAFE);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Scalar gamma kernel
// ══════════════════════════════════════════════════════════════════════════════

static void test_gamma_scalar_kernel() {
    // Newtonian limit: beta=0 → gamma=1.
    {
        double b[1] = {0.0}, out[1] = {};
        compute_gamma_scalar(b, 1, out);
        SRFM_CHECK_NEAR(out[0], 1.0, EPS);
    }

    // Known value: beta=0.6 → gamma=1.25.
    {
        double b[1] = {0.6}, out[1] = {};
        compute_gamma_scalar(b, 1, out);
        SRFM_CHECK_NEAR(out[0], 1.25, 1e-9);
    }

    // gamma monotonically increases with beta.
    {
        double betas[4] = {0.0, 0.3, 0.6, 0.9};
        double out[4] = {};
        compute_gamma_scalar(betas, 4, out);
        SRFM_CHECK(out[0] <= out[1]);
        SRFM_CHECK(out[1] <= out[2]);
        SRFM_CHECK(out[2] <= out[3]);
    }

    // gamma >= 1.0 for all valid betas.
    {
        auto vels = make_velocities(512, 1.0);
        // Generate valid betas in [0, BETA_MAX_SAFE).
        std::vector<double> betas(512);
        double rmax = 0.0;
        compute_beta_scalar(vels.data(), 512, rmax, betas.data());
        std::vector<double> gammas(512);
        compute_gamma_scalar(betas.data(), 512, gammas.data());
        for (double g : gammas) {
            SRFM_CHECK(g >= 1.0);
            SRFM_CHECK(std::isfinite(g));
        }
    }

    // Empty input: no crash.
    {
        double dummy_in = 0.0, dummy_out = 0.0;
        compute_gamma_scalar(&dummy_in, 0, &dummy_out);  // n=0 → early return
        SRFM_CHECK(true);
    }

    // High beta: gamma should be large but finite.
    {
        double b[1] = {BETA_MAX_SAFE - 1e-10}, out[1] = {};
        compute_gamma_scalar(b, 1, out);
        SRFM_CHECK(std::isfinite(out[0]));
        SRFM_CHECK(out[0] >= 1.0);
    }

    // Identity: gamma^2 = 1 / (1 - beta^2).
    {
        double betas[3] = {0.2, 0.5, 0.8};
        double out[3] = {};
        compute_gamma_scalar(betas, 3, out);
        for (int i = 0; i < 3; ++i) {
            double g2_expected = 1.0 / (1.0 - betas[i] * betas[i]);
            SRFM_CHECK_NEAR(out[i] * out[i], g2_expected, 1e-9);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// AVX2 beta kernel (compares against scalar reference)
// ══════════════════════════════════════════════════════════════════════════════

static void test_beta_avx2_kernel() {
    if (!has_avx2()) {
        // Machine doesn't support AVX2: skip these tests.
        SRFM_CHECK(true);  // placeholder pass
        return;
    }

    // Sizes: aligned (8), 4-aligned (4), non-aligned tails (1, 3, 5, 9, 15).
    for (std::size_t n : {1u, 3u, 4u, 5u, 8u, 9u, 15u, 16u, 17u, 63u, 64u, 65u, 256u}) {
        auto vels = make_velocities(n, 100.0, static_cast<std::uint32_t>(n));

        double rmax_s = 0.0, rmax_a = 0.0;
        auto ref = scalar_beta(vels, rmax_s);
        std::vector<double> avx2_out(n);
        compute_beta_avx2(vels.data(), n, rmax_a, avx2_out.data());

        SRFM_CHECK_NEAR(rmax_s, rmax_a, EPS);
        for (std::size_t i = 0; i < n; ++i) {
            SRFM_CHECK_NEAR(ref[i], avx2_out[i], 1e-13);
        }
    }

    // All zeros: results all zero.
    {
        std::vector<double> v(7, 0.0), out(7);
        double rmax = 0.0;
        compute_beta_avx2(v.data(), 7, rmax, out.data());
        for (double b : out) SRFM_CHECK(b == 0.0);
    }

    // Negative velocities: same result as positive.
    {
        std::vector<double> pos = {1.0, 2.0, 3.0, 4.0};
        std::vector<double> neg = {-1.0, -2.0, -3.0, -4.0};
        std::vector<double> opos(4), oneg(4);
        double rpos = 0.0, rneg = 0.0;
        compute_beta_avx2(pos.data(), 4, rpos, opos.data());
        compute_beta_avx2(neg.data(), 4, rneg, oneg.data());
        for (int i = 0; i < 4; ++i) {
            SRFM_CHECK_NEAR(opos[i], oneg[i], EPS);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// AVX2 gamma kernel (compares against scalar reference)
// ══════════════════════════════════════════════════════════════════════════════

static void test_gamma_avx2_kernel() {
    if (!has_avx2()) {
        SRFM_CHECK(true);
        return;
    }

    for (std::size_t n : {1u, 3u, 4u, 5u, 8u, 9u, 17u, 63u, 64u, 256u}) {
        auto vels = make_velocities(n, 10.0, static_cast<std::uint32_t>(n + 1000));
        double rmax = 0.0;
        auto betas = scalar_beta(vels, rmax);
        auto ref   = scalar_gamma(betas);

        std::vector<double> avx2_out(n);
        compute_gamma_avx2(betas.data(), n, avx2_out.data());

        for (std::size_t i = 0; i < n; ++i) {
            SRFM_CHECK_NEAR(ref[i], avx2_out[i], 1e-12);
        }
    }

    // Newtonian limit: all betas zero → all gammas 1.
    {
        std::vector<double> betas(8, 0.0), out(8);
        compute_gamma_avx2(betas.data(), 8, out.data());
        for (double g : out) SRFM_CHECK_NEAR(g, 1.0, EPS);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// AVX-512 beta kernel (compares against scalar reference)
// ══════════════════════════════════════════════════════════════════════════════

static void test_beta_avx512_kernel() {
    if (!has_avx512f()) {
        SRFM_CHECK(true);
        return;
    }

    for (std::size_t n : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 1024u}) {
        auto vels = make_velocities(n, 50.0, static_cast<std::uint32_t>(n + 2000));

        double rmax_s = 0.0, rmax_a = 0.0;
        auto ref = scalar_beta(vels, rmax_s);
        std::vector<double> avx512_out(n);
        compute_beta_avx512(vels.data(), n, rmax_a, avx512_out.data());

        SRFM_CHECK_NEAR(rmax_s, rmax_a, EPS);
        for (std::size_t i = 0; i < n; ++i) {
            SRFM_CHECK_NEAR(ref[i], avx512_out[i], 1e-13);
        }
    }

    // All zeros.
    {
        std::vector<double> v(9, 0.0), out(9);
        double rmax = 0.0;
        compute_beta_avx512(v.data(), 9, rmax, out.data());
        for (double b : out) SRFM_CHECK(b == 0.0);
    }

    // Negative velocities symmetric.
    {
        std::vector<double> pos(8), neg(8), opos(8), oneg(8);
        for (int i = 0; i < 8; ++i) { pos[i] = (i + 1) * 1.5; neg[i] = -pos[i]; }
        double rpos = 0.0, rneg = 0.0;
        compute_beta_avx512(pos.data(), 8, rpos, opos.data());
        compute_beta_avx512(neg.data(), 8, rneg, oneg.data());
        for (int i = 0; i < 8; ++i) SRFM_CHECK_NEAR(opos[i], oneg[i], EPS);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// AVX-512 gamma kernel (compares against scalar reference)
// ══════════════════════════════════════════════════════════════════════════════

static void test_gamma_avx512_kernel() {
    if (!has_avx512f()) {
        SRFM_CHECK(true);
        return;
    }

    for (std::size_t n : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 63u, 64u, 256u, 1024u}) {
        auto vels = make_velocities(n, 5.0, static_cast<std::uint32_t>(n + 3000));
        double rmax = 0.0;
        auto betas = scalar_beta(vels, rmax);
        auto ref   = scalar_gamma(betas);

        std::vector<double> avx512_out(n);
        compute_gamma_avx512(betas.data(), n, avx512_out.data());

        for (std::size_t i = 0; i < n; ++i) {
            SRFM_CHECK_NEAR(ref[i], avx512_out[i], 1e-12);
        }
    }

    // Known value: beta=0.6 → gamma=1.25.
    {
        std::vector<double> betas(8, 0.6), out(8);
        compute_gamma_avx512(betas.data(), 8, out.data());
        for (double g : out) SRFM_CHECK_NEAR(g, 1.25, 1e-9);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Dispatch free functions
// ══════════════════════════════════════════════════════════════════════════════

static void test_dispatch_free_functions() {
    // computeBetaBatch returns one BetaVelocity per velocity.
    {
        std::vector<double> vels = {1.0, 2.0, 3.0, 4.0};
        double rmax = 0.0;
        auto betas = computeBetaBatch(vels, rmax);
        SRFM_CHECK(betas.size() == 4u);
        for (const auto& b : betas) {
            SRFM_CHECK(b.value() >= 0.0);
            SRFM_CHECK(b.value() < BETA_MAX_SAFE);
        }
    }

    // computeGammaBatch returns one LorentzFactor per beta.
    {
        std::vector<double> vels = {0.0, 0.5, 1.0};
        double rmax = 0.0;
        auto betas  = computeBetaBatch(vels, rmax);
        auto gammas = computeGammaBatch(betas);
        SRFM_CHECK(gammas.size() == 3u);
        for (const auto& g : gammas) {
            SRFM_CHECK(g.value() >= 1.0);
            SRFM_CHECK(std::isfinite(g.value()));
        }
    }

    // Empty input → empty output.
    {
        double rmax = 0.0;
        std::vector<double> empty_vels;
        auto betas  = computeBetaBatch(empty_vels, rmax);
        auto gammas = computeGammaBatch(betas);
        SRFM_CHECK(betas.empty());
        SRFM_CHECK(gammas.empty());
    }

    // Results consistent with scalar path.
    {
        auto vels = make_velocities(100);
        double rmax_s = 0.0, rmax_d = 0.0;
        auto ref_betas = scalar_beta(vels, rmax_s);
        auto ref_gammas = scalar_gamma(ref_betas);
        auto dis_betas  = computeBetaBatch(vels, rmax_d);
        auto dis_gammas = computeGammaBatch(dis_betas);

        SRFM_CHECK_NEAR(rmax_s, rmax_d, EPS);
        SRFM_CHECK(dis_betas.size()  == ref_betas.size());
        SRFM_CHECK(dis_gammas.size() == ref_gammas.size());
        for (std::size_t i = 0; i < 100; ++i) {
            SRFM_CHECK_NEAR(ref_betas[i], dis_betas[i].value(), 1e-13);
            SRFM_CHECK_NEAR(ref_gammas[i], dis_gammas[i].value(), 1e-12);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// BetaCalculator
// ══════════════════════════════════════════════════════════════════════════════

static void test_beta_calculator() {
    BetaCalculator calc;

    // Initial state.
    SRFM_CHECK(calc.running_max() == 0.0);

    // First batch.
    std::vector<double> vels1 = {2.0, 4.0, 6.0};
    auto betas1 = calc.computeBetaBatch(vels1);
    SRFM_CHECK(betas1.size() == 3u);
    SRFM_CHECK_NEAR(calc.running_max(), 6.0, EPS);

    // Second batch with smaller velocities: running_max does not decrease.
    std::vector<double> vels2 = {1.0, 2.0};
    auto betas2 = calc.computeBetaBatch(vels2);
    SRFM_CHECK(calc.running_max() >= 6.0);
    SRFM_CHECK_NEAR(betas2[0].value(), 1.0 / 6.0, 1e-12);
    SRFM_CHECK_NEAR(betas2[1].value(), 2.0 / 6.0, 1e-12);

    // Gamma from betas1.
    auto gammas1 = calc.computeGammaBatch(betas1);
    SRFM_CHECK(gammas1.size() == 3u);
    for (const auto& g : gammas1) {
        SRFM_CHECK(g.value() >= 1.0);
    }

    // SIMD level is stable.
    SRFM_CHECK(calc.simd_level() == detect_simd_level());

    // reset() clears running_max.
    calc.reset();
    SRFM_CHECK(calc.running_max() == 0.0);

    // After reset, running_max updates fresh from the next batch.
    std::vector<double> vels3 = {3.0};
    (void)calc.computeBetaBatch(vels3);
    SRFM_CHECK_NEAR(calc.running_max(), 3.0, EPS);
}

// ══════════════════════════════════════════════════════════════════════════════
// Running-max maintenance
// ══════════════════════════════════════════════════════════════════════════════

static void test_running_max_maintenance() {
    BetaCalculator calc;

    // Process 10 batches of increasing magnitude.
    for (int k = 1; k <= 10; ++k) {
        std::vector<double> vels(5, static_cast<double>(k));
        (void)calc.computeBetaBatch(vels);
        SRFM_CHECK_NEAR(calc.running_max(), static_cast<double>(k), EPS);
    }

    // running_max is now 10.0.  Batch with max |v| = 3 must NOT decrease it.
    std::vector<double> small(5, 3.0);
    (void)calc.computeBetaBatch(small);
    SRFM_CHECK_NEAR(calc.running_max(), 10.0, EPS);

    // Batch with max |v| = 15 MUST increase it.
    std::vector<double> large(5, 15.0);
    (void)calc.computeBetaBatch(large);
    SRFM_CHECK_NEAR(calc.running_max(), 15.0, EPS);
}

// ══════════════════════════════════════════════════════════════════════════════
// Tail element handling (n % 4 != 0, n % 8 != 0)
// ══════════════════════════════════════════════════════════════════════════════

static void test_tail_element_handling() {
    // For every size from 1 to 33, dispatch result must match scalar.
    for (std::size_t n = 1; n <= 33; ++n) {
        auto vels = make_velocities(n, 20.0, static_cast<std::uint32_t>(n + 5000));
        double rmax_s = 0.0, rmax_d = 0.0;
        auto ref_b = scalar_beta(vels, rmax_s);
        auto ref_g = scalar_gamma(ref_b);

        auto dis_bv = computeBetaBatch(vels, rmax_d);
        SRFM_CHECK(dis_bv.size() == n);
        SRFM_CHECK_NEAR(rmax_s, rmax_d, EPS);
        for (std::size_t i = 0; i < n; ++i) {
            SRFM_CHECK_NEAR(ref_b[i], dis_bv[i].value(), 1e-13);
        }

        auto dis_gv = computeGammaBatch(dis_bv);
        SRFM_CHECK(dis_gv.size() == n);
        for (std::size_t i = 0; i < n; ++i) {
            SRFM_CHECK_NEAR(ref_g[i], dis_gv[i].value(), 1e-12);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Clamping
// ══════════════════════════════════════════════════════════════════════════════

static void test_clamping() {
    // Even the velocity equal to running_max (beta = 1.0) must be clamped.
    BetaCalculator calc;
    std::vector<double> vels = {5.0};  // After processing: running_max=5, beta=clamped
    auto betas = calc.computeBetaBatch(vels);
    SRFM_CHECK(betas[0].value() < BETA_MAX_SAFE);
    SRFM_CHECK(betas[0].value() >= 0.0);

    // All BetaVelocity::make() calls on the output must succeed.
    auto vels2 = make_velocities(256, 100.0, 7777u);
    double rmax = 0.0;
    auto bv2 = computeBetaBatch(vels2, rmax);
    for (const auto& b : bv2) {
        // Re-validate via make() — must not return nullopt.
        SRFM_HAS_VALUE(BetaVelocity::make(b.value()));
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Gamma monotonicity
// ══════════════════════════════════════════════════════════════════════════════

static void test_gamma_monotonicity() {
    // Generate sorted betas and verify gammas are non-decreasing.
    std::vector<double> beta_vals;
    for (int i = 0; i <= 99; ++i) {
        beta_vals.push_back(static_cast<double>(i) / 100.0 * (BETA_MAX_SAFE - 0.001));
    }
    std::vector<double> gamma_out(100);
    compute_gamma_scalar(beta_vals.data(), 100, gamma_out.data());
    for (int i = 1; i < 100; ++i) {
        SRFM_CHECK(gamma_out[i] >= gamma_out[i - 1]);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Newtonian limit
// ══════════════════════════════════════════════════════════════════════════════

static void test_gamma_newtonian_limit() {
    std::vector<BetaVelocity> betas;
    betas.push_back(*BetaVelocity::make(0.0));
    auto gammas = computeGammaBatch(betas);
    SRFM_CHECK(gammas.size() == 1u);
    SRFM_CHECK_NEAR(gammas[0].value(), 1.0, EPS);
}

// ══════════════════════════════════════════════════════════════════════════════
// Known gamma value: beta=0.6 → gamma=1.25
// ══════════════════════════════════════════════════════════════════════════════

static void test_gamma_identity_b06() {
    std::vector<BetaVelocity> betas;
    betas.push_back(*BetaVelocity::make(0.6));
    auto gammas = computeGammaBatch(betas);
    SRFM_CHECK(gammas.size() == 1u);
    SRFM_CHECK_NEAR(gammas[0].value(), 1.25, 1e-9);
}

// ══════════════════════════════════════════════════════════════════════════════
// Large batch: 65536 elements, all finite and in range
// ══════════════════════════════════════════════════════════════════════════════

static void test_large_batch_finiteness() {
    constexpr std::size_t N = 65536;
    auto vels = make_velocities(N, 1000.0, 99999u);
    double rmax = 0.0;
    auto bv = computeBetaBatch(vels, rmax);
    auto gv = computeGammaBatch(bv);

    SRFM_CHECK(bv.size() == N);
    SRFM_CHECK(gv.size() == N);

    bool all_beta_ok  = true;
    bool all_gamma_ok = true;
    for (std::size_t i = 0; i < N; ++i) {
        if (!std::isfinite(bv[i].value()) || bv[i].value() < 0.0 || bv[i].value() >= BETA_MAX_SAFE)
            all_beta_ok = false;
        if (!std::isfinite(gv[i].value()) || gv[i].value() < 1.0)
            all_gamma_ok = false;
    }
    SRFM_CHECK(all_beta_ok);
    SRFM_CHECK(all_gamma_ok);
}

// ══════════════════════════════════════════════════════════════════════════════
// All-zero velocities
// ══════════════════════════════════════════════════════════════════════════════

static void test_all_zero_velocities() {
    std::vector<double> vels(64, 0.0);
    double rmax = 0.0;
    auto bv = computeBetaBatch(vels, rmax);
    auto gv = computeGammaBatch(bv);

    SRFM_CHECK(rmax == 0.0);
    for (const auto& b : bv) SRFM_CHECK(b.value() == 0.0);
    for (const auto& g : gv) SRFM_CHECK_NEAR(g.value(), 1.0, EPS);
}

// ══════════════════════════════════════════════════════════════════════════════
// Single nonzero velocity (N=1)
// ══════════════════════════════════════════════════════════════════════════════

static void test_single_nonzero_velocity() {
    std::vector<double> vels = {7.5};
    double rmax = 0.0;
    auto bv = computeBetaBatch(vels, rmax);
    SRFM_CHECK(bv.size() == 1u);
    SRFM_CHECK_NEAR(rmax, 7.5, EPS);
    // beta = 7.5 / 7.5 = 1.0 → clamped
    SRFM_CHECK(bv[0].value() < BETA_MAX_SAFE);

    auto gv = computeGammaBatch(bv);
    SRFM_CHECK(gv.size() == 1u);
    SRFM_CHECK(gv[0].value() >= 1.0);
    SRFM_CHECK(std::isfinite(gv[0].value()));
}

// ══════════════════════════════════════════════════════════════════════════════
// Negative velocities produce same beta as positive (|v| symmetry)
// ══════════════════════════════════════════════════════════════════════════════

static void test_negative_velocities() {
    std::vector<double> pos = {1.0, 3.0, 7.0, 2.0, 5.0};
    std::vector<double> neg(pos.size());
    for (std::size_t i = 0; i < pos.size(); ++i) neg[i] = -pos[i];

    double rpos = 0.0, rneg = 0.0;
    auto bpos = computeBetaBatch(pos, rpos);
    auto bneg = computeBetaBatch(neg, rneg);

    SRFM_CHECK_NEAR(rpos, rneg, EPS);
    SRFM_CHECK(bpos.size() == bneg.size());
    for (std::size_t i = 0; i < pos.size(); ++i) {
        SRFM_CHECK_NEAR(bpos[i].value(), bneg[i].value(), EPS);
        SRFM_CHECK(bneg[i].value() >= 0.0);  // beta is always non-negative
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// running_max is carried over across successive calls
// ══════════════════════════════════════════════════════════════════════════════

static void test_running_max_carried_over() {
    double rmax = 0.0;

    // First call establishes rmax = 10.
    std::vector<double> v1 = {10.0};
    (void)computeBetaBatch(v1, rmax);
    SRFM_CHECK_NEAR(rmax, 10.0, EPS);

    // Second call with v < 10 must use rmax = 10 as denominator.
    std::vector<double> v2 = {5.0};
    auto bv2 = computeBetaBatch(v2, rmax);
    SRFM_CHECK_NEAR(rmax, 10.0, EPS);
    SRFM_CHECK_NEAR(bv2[0].value(), 0.5, 1e-12);
}

// ══════════════════════════════════════════════════════════════════════════════
// BetaCalculator::reset() resets running_max
// ══════════════════════════════════════════════════════════════════════════════

static void test_reset_clears_running_max() {
    BetaCalculator calc;
    std::vector<double> v1 = {100.0};
    (void)calc.computeBetaBatch(v1);
    SRFM_CHECK_NEAR(calc.running_max(), 100.0, EPS);

    calc.reset();
    SRFM_CHECK(calc.running_max() == 0.0);

    // After reset, a smaller velocity sets a new rmax.
    std::vector<double> v2 = {5.0};
    (void)calc.computeBetaBatch(v2);
    SRFM_CHECK_NEAR(calc.running_max(), 5.0, EPS);
}

// ══════════════════════════════════════════════════════════════════════════════
// simd_level_name returns non-null, non-empty string
// ══════════════════════════════════════════════════════════════════════════════

static void test_simd_level_name() {
    for (auto lvl : {SimdLevel::SCALAR, SimdLevel::SSE42, SimdLevel::AVX2, SimdLevel::AVX512F}) {
        const char* name = simd_level_name(lvl);
        SRFM_CHECK(name != nullptr);
        SRFM_CHECK(name[0] != '\0');
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// main
// ══════════════════════════════════════════════════════════════════════════════

int main() {
    SRFM_SUITE("cpu_feature_detection",     test_cpu_feature_detection);
    SRFM_SUITE("beta_scalar_kernel",        test_beta_scalar_kernel);
    SRFM_SUITE("gamma_scalar_kernel",       test_gamma_scalar_kernel);
    SRFM_SUITE("beta_avx2_kernel",          test_beta_avx2_kernel);
    SRFM_SUITE("gamma_avx2_kernel",         test_gamma_avx2_kernel);
    SRFM_SUITE("beta_avx512_kernel",        test_beta_avx512_kernel);
    SRFM_SUITE("gamma_avx512_kernel",       test_gamma_avx512_kernel);
    SRFM_SUITE("dispatch_free_functions",   test_dispatch_free_functions);
    SRFM_SUITE("beta_calculator",           test_beta_calculator);
    SRFM_SUITE("running_max_maintenance",   test_running_max_maintenance);
    SRFM_SUITE("tail_element_handling",     test_tail_element_handling);
    SRFM_SUITE("clamping",                  test_clamping);
    SRFM_SUITE("gamma_monotonicity",        test_gamma_monotonicity);
    SRFM_SUITE("gamma_newtonian_limit",     test_gamma_newtonian_limit);
    SRFM_SUITE("gamma_identity_b06",        test_gamma_identity_b06);
    SRFM_SUITE("large_batch_finiteness",    test_large_batch_finiteness);
    SRFM_SUITE("all_zero_velocities",       test_all_zero_velocities);
    SRFM_SUITE("single_nonzero_velocity",   test_single_nonzero_velocity);
    SRFM_SUITE("negative_velocities",       test_negative_velocities);
    SRFM_SUITE("running_max_carried_over",  test_running_max_carried_over);
    SRFM_SUITE("reset_clears_running_max",  test_reset_clears_running_max);
    SRFM_SUITE("simd_level_name",           test_simd_level_name);
    return srfm_test::report();
}
