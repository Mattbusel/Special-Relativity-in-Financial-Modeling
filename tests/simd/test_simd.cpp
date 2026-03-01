/// @file tests/simd/test_simd.cpp
/// @brief Unit + correctness tests for the AGT-08 SIMD β/γ acceleration.
///
/// Test suites
/// -----------
///   CpuFeatureDetection   — SimdLevel enum and detect_simd_level()
///   BetaScalarKernel      — compute_beta_scalar correctness
///   GammaScalarKernel     — compute_gamma_scalar correctness
///   BetaAvx2Kernel        — compute_beta_avx2 bit-identity vs scalar
///   GammaAvx2Kernel       — compute_gamma_avx2 bit-identity vs scalar
///   BetaAvx512Kernel      — compute_beta_avx512 bit-identity vs scalar
///   GammaAvx512Kernel     — compute_gamma_avx512 bit-identity vs scalar
///   DispatchFreeFunctions — computeBetaBatch / computeGammaBatch
///   BetaCalculatorClass   — BetaCalculator stateful wrapper
///   RunningMaxMaintenance — monotonic running_max across calls
///   TailElementHandling   — n % 4 != 0 and n % 8 != 0 sizes
///   Clamping              — beta never reaches BETA_MAX_SAFE
///   GammaMonotonicity     — gamma grows with |beta|
///   KnownValues           — beta=0 → gamma=1, beta=0.6 → gamma=1.25
///   LargeBatchFiniteness  — 65536 elements all finite and in range
///   EdgeCases             — empty, all-zero, single element, negative

#include "srfm/simd/cpu_features.hpp"
#include "srfm/simd/simd_dispatch.hpp"
#include "srfm/types.hpp"
#include "srfm/constants.hpp"
// Internal kernel header (src/simd/ on include path):
#include "simd_batch_detail.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

using namespace srfm;
using namespace srfm::simd;
using namespace srfm::simd::detail;

// ── Test helpers ──────────────────────────────────────────────────────────────

static constexpr double EPS      = 1e-12;
static constexpr double BETA_MAX = constants::BETA_MAX_SAFE;

static std::vector<double> make_velocities(std::size_t n,
                                            double amplitude = 10.0,
                                            std::uint32_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-amplitude, amplitude);
    std::vector<double> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

/// Run scalar beta kernel, return raw doubles.
static std::vector<double> run_scalar_beta(const std::vector<double>& vels,
                                            double& rmax) {
    std::vector<double> out(vels.size());
    compute_beta_scalar(vels.data(), vels.size(), rmax, out.data());
    return out;
}

/// Run scalar gamma kernel, return raw doubles.
static std::vector<double> run_scalar_gamma(const std::vector<double>& betas) {
    std::vector<double> out(betas.size());
    compute_gamma_scalar(betas.data(), betas.size(), out.data());
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
// CpuFeatureDetection
// ════════════════════════════════════════════════════════════════════════════

TEST(CpuFeatureDetection, LevelIsValid) {
    SimdLevel level = detect_simd_level();
    bool valid = (level == SimdLevel::SCALAR  ||
                  level == SimdLevel::SSE42   ||
                  level == SimdLevel::AVX2    ||
                  level == SimdLevel::AVX512F);
    EXPECT_TRUE(valid);
}

TEST(CpuFeatureDetection, ResultIsCached) {
    EXPECT_EQ(detect_simd_level(), detect_simd_level());
    EXPECT_EQ(detect_simd_level(), detect_simd_level());
}

TEST(CpuFeatureDetection, OrderingInvariant) {
    EXPECT_GT(SimdLevel::AVX512F, SimdLevel::AVX2);
    EXPECT_GT(SimdLevel::AVX2,    SimdLevel::SSE42);
    EXPECT_GT(SimdLevel::SSE42,   SimdLevel::SCALAR);
}

TEST(CpuFeatureDetection, PredicatesConsistentWithLevel) {
    SimdLevel level = detect_simd_level();
    if (level >= SimdLevel::AVX512F) EXPECT_TRUE(has_avx512f());
    if (level >= SimdLevel::AVX2)    EXPECT_TRUE(has_avx2());
    if (level >= SimdLevel::SSE42)   EXPECT_TRUE(has_sse42());
}

TEST(CpuFeatureDetection, LevelNameNonEmpty) {
    for (auto lvl : {SimdLevel::SCALAR, SimdLevel::SSE42,
                     SimdLevel::AVX2,   SimdLevel::AVX512F}) {
        const char* name = simd_level_name(lvl);
        ASSERT_NE(name, nullptr);
        EXPECT_NE(name[0], '\0');
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BetaScalarKernel
// ════════════════════════════════════════════════════════════════════════════

TEST(BetaScalarKernel, SinglePositiveVelocity) {
    double rmax = 0.0;
    double v[1] = {5.0}, out[1] = {};
    compute_beta_scalar(v, 1, rmax, out);
    EXPECT_DOUBLE_EQ(rmax, 5.0);
    // beta = 5/5 = 1.0 → clamped
    EXPECT_LT(out[0], BETA_MAX);
    EXPECT_GE(out[0], 0.0);
}

TEST(BetaScalarKernel, NegativeVelocitySymmetry) {
    double rpos = 0.0, rneg = 0.0;
    double vpos[1] = {3.0}, vneg[1] = {-3.0}, opos[1] = {}, oneg[1] = {};
    compute_beta_scalar(vpos, 1, rpos, opos);
    compute_beta_scalar(vneg, 1, rneg, oneg);
    EXPECT_NEAR(opos[0], oneg[0], EPS);
    EXPECT_NEAR(rpos, rneg, EPS);
}

TEST(BetaScalarKernel, RunningMaxMonotonic_SmallSecondBatch) {
    double rmax = 0.0;
    double v1[1] = {10.0}, v2[1] = {2.0}, o1[1] = {}, o2[1] = {};
    compute_beta_scalar(v1, 1, rmax, o1);
    EXPECT_DOUBLE_EQ(rmax, 10.0);
    compute_beta_scalar(v2, 1, rmax, o2);
    EXPECT_DOUBLE_EQ(rmax, 10.0);  // must NOT decrease
    EXPECT_NEAR(o2[0], 0.2, EPS);
}

TEST(BetaScalarKernel, BatchMaxSemantics) {
    // v = [3, 7] → batch_max = 7 → rmax = 7
    // beta[0] = 3/7, beta[1] = 7/7 → clamped
    double rmax = 0.0;
    double v[2] = {3.0, 7.0}, out[2] = {};
    compute_beta_scalar(v, 2, rmax, out);
    EXPECT_DOUBLE_EQ(rmax, 7.0);
    EXPECT_NEAR(out[0], 3.0 / 7.0, EPS);
    EXPECT_LT(out[1], BETA_MAX);
}

TEST(BetaScalarKernel, AllZeros) {
    double rmax = 0.0;
    double v[4] = {0.0, 0.0, 0.0, 0.0}, out[4] = {};
    compute_beta_scalar(v, 4, rmax, out);
    EXPECT_DOUBLE_EQ(rmax, 0.0);
    for (int i = 0; i < 4; ++i) EXPECT_DOUBLE_EQ(out[i], 0.0);
}

TEST(BetaScalarKernel, EmptyInputNoMutation) {
    double rmax = 5.0;
    compute_beta_scalar(nullptr, 0, rmax, nullptr);
    EXPECT_DOUBLE_EQ(rmax, 5.0);
}

TEST(BetaScalarKernel, LargeBatchAllFiniteInRange) {
    auto vels = make_velocities(1024);
    std::vector<double> out(1024);
    double rmax = 0.0;
    compute_beta_scalar(vels.data(), 1024, rmax, out.data());
    for (double b : out) {
        EXPECT_TRUE(std::isfinite(b));
        EXPECT_GE(b, 0.0);
        EXPECT_LT(b, BETA_MAX);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// GammaScalarKernel
// ════════════════════════════════════════════════════════════════════════════

TEST(GammaScalarKernel, NewtonianLimit) {
    double b[1] = {0.0}, out[1] = {};
    compute_gamma_scalar(b, 1, out);
    EXPECT_NEAR(out[0], 1.0, EPS);
}

TEST(GammaScalarKernel, KnownValueBeta06) {
    double b[1] = {0.6}, out[1] = {};
    compute_gamma_scalar(b, 1, out);
    EXPECT_NEAR(out[0], 1.25, 1e-9);
}

TEST(GammaScalarKernel, GammaAlwaysGeOne) {
    auto vels = make_velocities(512, 1.0);
    std::vector<double> betas(512);
    double rmax = 0.0;
    compute_beta_scalar(vels.data(), 512, rmax, betas.data());
    std::vector<double> gammas(512);
    compute_gamma_scalar(betas.data(), 512, gammas.data());
    for (double g : gammas) {
        EXPECT_GE(g, 1.0);
        EXPECT_TRUE(std::isfinite(g));
    }
}

TEST(GammaScalarKernel, Monotonicity) {
    double betas[4] = {0.0, 0.3, 0.6, 0.9}, out[4] = {};
    compute_gamma_scalar(betas, 4, out);
    EXPECT_LE(out[0], out[1]);
    EXPECT_LE(out[1], out[2]);
    EXPECT_LE(out[2], out[3]);
}

TEST(GammaScalarKernel, GammaSquaredIdentity) {
    double betas[3] = {0.2, 0.5, 0.8}, out[3] = {};
    compute_gamma_scalar(betas, 3, out);
    for (int i = 0; i < 3; ++i) {
        const double expected = 1.0 / (1.0 - betas[i] * betas[i]);
        EXPECT_NEAR(out[i] * out[i], expected, 1e-9);
    }
}

TEST(GammaScalarKernel, HighBetaStillFinite) {
    double b[1] = {BETA_MAX - 1e-10}, out[1] = {};
    compute_gamma_scalar(b, 1, out);
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_GE(out[0], 1.0);
}

// ════════════════════════════════════════════════════════════════════════════
// BetaAvx2Kernel  (skipped on non-AVX2 machines)
// ════════════════════════════════════════════════════════════════════════════

TEST(BetaAvx2Kernel, BitIdenticalToScalar) {
    if (!has_avx2()) GTEST_SKIP() << "AVX2 not available";
    for (std::size_t n : {1u, 3u, 4u, 5u, 8u, 9u, 15u, 16u, 17u, 63u, 64u, 256u}) {
        auto vels = make_velocities(n, 100.0, static_cast<std::uint32_t>(n));
        double rmax_s = 0.0, rmax_a = 0.0;
        auto ref = run_scalar_beta(vels, rmax_s);
        std::vector<double> avx2_out(n);
        compute_beta_avx2(vels.data(), n, rmax_a, avx2_out.data());
        EXPECT_NEAR(rmax_s, rmax_a, EPS) << "n=" << n;
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(ref[i], avx2_out[i], 1e-13) << "n=" << n << " i=" << i;
    }
}

TEST(BetaAvx2Kernel, AllZeros) {
    if (!has_avx2()) GTEST_SKIP() << "AVX2 not available";
    std::vector<double> v(7, 0.0), out(7);
    double rmax = 0.0;
    compute_beta_avx2(v.data(), 7, rmax, out.data());
    for (double b : out) EXPECT_DOUBLE_EQ(b, 0.0);
}

TEST(BetaAvx2Kernel, NegativeSymmetry) {
    if (!has_avx2()) GTEST_SKIP() << "AVX2 not available";
    std::vector<double> pos = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> neg = {-1.0, -2.0, -3.0, -4.0};
    std::vector<double> opos(4), oneg(4);
    double rpos = 0.0, rneg = 0.0;
    compute_beta_avx2(pos.data(), 4, rpos, opos.data());
    compute_beta_avx2(neg.data(), 4, rneg, oneg.data());
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(opos[i], oneg[i], EPS) << "i=" << i;
}

// ════════════════════════════════════════════════════════════════════════════
// GammaAvx2Kernel
// ════════════════════════════════════════════════════════════════════════════

TEST(GammaAvx2Kernel, BitIdenticalToScalar) {
    if (!has_avx2()) GTEST_SKIP() << "AVX2 not available";
    for (std::size_t n : {1u, 3u, 4u, 5u, 8u, 9u, 17u, 63u, 64u, 256u}) {
        auto vels = make_velocities(n, 10.0, static_cast<std::uint32_t>(n + 1000));
        double rmax = 0.0;
        auto betas = run_scalar_beta(vels, rmax);
        auto ref   = run_scalar_gamma(betas);
        std::vector<double> avx2_out(n);
        compute_gamma_avx2(betas.data(), n, avx2_out.data());
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(ref[i], avx2_out[i], 1e-12) << "n=" << n << " i=" << i;
    }
}

TEST(GammaAvx2Kernel, NewtonianLimit) {
    if (!has_avx2()) GTEST_SKIP() << "AVX2 not available";
    std::vector<double> betas(8, 0.0), out(8);
    compute_gamma_avx2(betas.data(), 8, out.data());
    for (double g : out) EXPECT_NEAR(g, 1.0, EPS);
}

// ════════════════════════════════════════════════════════════════════════════
// BetaAvx512Kernel
// ════════════════════════════════════════════════════════════════════════════

TEST(BetaAvx512Kernel, BitIdenticalToScalar) {
    if (!has_avx512f()) GTEST_SKIP() << "AVX-512F not available";
    for (std::size_t n : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 63u, 64u, 65u, 256u, 1024u}) {
        auto vels = make_velocities(n, 50.0, static_cast<std::uint32_t>(n + 2000));
        double rmax_s = 0.0, rmax_a = 0.0;
        auto ref = run_scalar_beta(vels, rmax_s);
        std::vector<double> avx512_out(n);
        compute_beta_avx512(vels.data(), n, rmax_a, avx512_out.data());
        EXPECT_NEAR(rmax_s, rmax_a, EPS) << "n=" << n;
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(ref[i], avx512_out[i], 1e-13) << "n=" << n << " i=" << i;
    }
}

TEST(BetaAvx512Kernel, AllZeros) {
    if (!has_avx512f()) GTEST_SKIP() << "AVX-512F not available";
    std::vector<double> v(9, 0.0), out(9);
    double rmax = 0.0;
    compute_beta_avx512(v.data(), 9, rmax, out.data());
    for (double b : out) EXPECT_DOUBLE_EQ(b, 0.0);
}

TEST(BetaAvx512Kernel, NegativeSymmetry) {
    if (!has_avx512f()) GTEST_SKIP() << "AVX-512F not available";
    std::vector<double> pos(8), neg(8), opos(8), oneg(8);
    for (int i = 0; i < 8; ++i) { pos[i] = (i + 1) * 1.5; neg[i] = -pos[i]; }
    double rpos = 0.0, rneg = 0.0;
    compute_beta_avx512(pos.data(), 8, rpos, opos.data());
    compute_beta_avx512(neg.data(), 8, rneg, oneg.data());
    for (int i = 0; i < 8; ++i) EXPECT_NEAR(opos[i], oneg[i], EPS);
}

// ════════════════════════════════════════════════════════════════════════════
// GammaAvx512Kernel
// ════════════════════════════════════════════════════════════════════════════

TEST(GammaAvx512Kernel, BitIdenticalToScalar) {
    if (!has_avx512f()) GTEST_SKIP() << "AVX-512F not available";
    for (std::size_t n : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 63u, 64u, 256u, 1024u}) {
        auto vels = make_velocities(n, 5.0, static_cast<std::uint32_t>(n + 3000));
        double rmax = 0.0;
        auto betas = run_scalar_beta(vels, rmax);
        auto ref   = run_scalar_gamma(betas);
        std::vector<double> avx512_out(n);
        compute_gamma_avx512(betas.data(), n, avx512_out.data());
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(ref[i], avx512_out[i], 1e-12) << "n=" << n << " i=" << i;
    }
}

TEST(GammaAvx512Kernel, KnownValueBeta06) {
    if (!has_avx512f()) GTEST_SKIP() << "AVX-512F not available";
    std::vector<double> betas(8, 0.6), out(8);
    compute_gamma_avx512(betas.data(), 8, out.data());
    for (double g : out) EXPECT_NEAR(g, 1.25, 1e-9);
}

// ════════════════════════════════════════════════════════════════════════════
// DispatchFreeFunctions
// ════════════════════════════════════════════════════════════════════════════

TEST(DispatchFreeFunctions, BetaBatchSizeMatches) {
    std::vector<double> vels = {1.0, 2.0, 3.0, 4.0};
    double rmax = 0.0;
    auto betas = computeBetaBatch(vels, rmax);
    EXPECT_EQ(betas.size(), 4u);
    for (const auto& b : betas) {
        EXPECT_GE(b.value, 0.0);
        EXPECT_LT(b.value, BETA_MAX);
    }
}

TEST(DispatchFreeFunctions, GammaBatchSizeMatches) {
    std::vector<double> vels = {0.0, 0.5, 1.0};
    double rmax = 0.0;
    auto betas  = computeBetaBatch(vels, rmax);
    auto gammas = computeGammaBatch(betas);
    EXPECT_EQ(gammas.size(), 3u);
    for (const auto& g : gammas) {
        EXPECT_GE(g.value, 1.0);
        EXPECT_TRUE(std::isfinite(g.value));
    }
}

TEST(DispatchFreeFunctions, EmptyInputReturnsEmpty) {
    double rmax = 0.0;
    std::vector<double> empty;
    auto betas  = computeBetaBatch(empty, rmax);
    auto gammas = computeGammaBatch(betas);
    EXPECT_TRUE(betas.empty());
    EXPECT_TRUE(gammas.empty());
}

TEST(DispatchFreeFunctions, ConsistentWithScalarReference) {
    auto vels = make_velocities(100);
    double rmax_s = 0.0, rmax_d = 0.0;
    auto ref_betas  = run_scalar_beta(vels, rmax_s);
    auto ref_gammas = run_scalar_gamma(ref_betas);
    auto dis_betas  = computeBetaBatch(vels, rmax_d);
    auto dis_gammas = computeGammaBatch(dis_betas);

    EXPECT_NEAR(rmax_s, rmax_d, EPS);
    ASSERT_EQ(dis_betas.size(),  ref_betas.size());
    ASSERT_EQ(dis_gammas.size(), ref_gammas.size());
    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_NEAR(ref_betas[i],  dis_betas[i].value,  1e-13) << "i=" << i;
        EXPECT_NEAR(ref_gammas[i], dis_gammas[i].value, 1e-12) << "i=" << i;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BetaCalculatorClass
// ════════════════════════════════════════════════════════════════════════════

TEST(BetaCalculatorClass, InitialStateIsZero) {
    BetaCalculator calc;
    EXPECT_DOUBLE_EQ(calc.running_max(), 0.0);
}

TEST(BetaCalculatorClass, FirstBatchSetsRunningMax) {
    BetaCalculator calc;
    (void)calc.computeBetaBatch({2.0, 4.0, 6.0});
    EXPECT_NEAR(calc.running_max(), 6.0, EPS);
}

TEST(BetaCalculatorClass, SecondBatchSmaller_RunningMaxUnchanged) {
    BetaCalculator calc;
    (void)calc.computeBetaBatch({6.0});
    auto betas2 = calc.computeBetaBatch({1.0, 2.0});
    EXPECT_GE(calc.running_max(), 6.0);
    EXPECT_NEAR(betas2[0].value, 1.0 / 6.0, 1e-12);
    EXPECT_NEAR(betas2[1].value, 2.0 / 6.0, 1e-12);
}

TEST(BetaCalculatorClass, GammaBatchSizeMatches) {
    BetaCalculator calc;
    auto betas  = calc.computeBetaBatch({2.0, 4.0, 6.0});
    auto gammas = calc.computeGammaBatch(betas);
    ASSERT_EQ(gammas.size(), 3u);
    for (const auto& g : gammas) EXPECT_GE(g.value, 1.0);
}

TEST(BetaCalculatorClass, SimdLevelStable) {
    BetaCalculator calc;
    EXPECT_EQ(calc.simd_level(), detect_simd_level());
}

TEST(BetaCalculatorClass, ResetClearsRunningMax) {
    BetaCalculator calc;
    (void)calc.computeBetaBatch({100.0});
    EXPECT_NEAR(calc.running_max(), 100.0, EPS);
    calc.reset();
    EXPECT_DOUBLE_EQ(calc.running_max(), 0.0);
    (void)calc.computeBetaBatch({5.0});
    EXPECT_NEAR(calc.running_max(), 5.0, EPS);
}

// ════════════════════════════════════════════════════════════════════════════
// RunningMaxMaintenance
// ════════════════════════════════════════════════════════════════════════════

TEST(RunningMaxMaintenance, MonotonicAcrossIncreasingBatches) {
    BetaCalculator calc;
    for (int k = 1; k <= 10; ++k) {
        (void)calc.computeBetaBatch(std::vector<double>(5, static_cast<double>(k)));
        EXPECT_NEAR(calc.running_max(), static_cast<double>(k), EPS);
    }
}

TEST(RunningMaxMaintenance, DoesNotDecreaseOnSmallerBatch) {
    BetaCalculator calc;
    (void)calc.computeBetaBatch({10.0});
    (void)calc.computeBetaBatch({3.0, 3.0, 3.0});
    EXPECT_NEAR(calc.running_max(), 10.0, EPS);
}

TEST(RunningMaxMaintenance, IncreasesOnLargerBatch) {
    BetaCalculator calc;
    (void)calc.computeBetaBatch({10.0});
    (void)calc.computeBetaBatch({15.0});
    EXPECT_NEAR(calc.running_max(), 15.0, EPS);
}

TEST(RunningMaxMaintenance, CarriedOverFreeFunctionInterface) {
    double rmax = 0.0;
    (void)computeBetaBatch({10.0}, rmax);
    EXPECT_NEAR(rmax, 10.0, EPS);
    auto bv2 = computeBetaBatch({5.0}, rmax);
    EXPECT_NEAR(rmax, 10.0, EPS);
    EXPECT_NEAR(bv2[0].value, 0.5, 1e-12);
}

// ════════════════════════════════════════════════════════════════════════════
// TailElementHandling
// ════════════════════════════════════════════════════════════════════════════

TEST(TailElementHandling, AllSizesMatchScalar) {
    for (std::size_t n = 1; n <= 33; ++n) {
        auto vels = make_velocities(n, 20.0, static_cast<std::uint32_t>(n + 5000));
        double rmax_s = 0.0, rmax_d = 0.0;
        auto ref_b = run_scalar_beta(vels, rmax_s);
        auto ref_g = run_scalar_gamma(ref_b);

        auto dis_bv = computeBetaBatch(vels, rmax_d);
        ASSERT_EQ(dis_bv.size(), n) << "n=" << n;
        EXPECT_NEAR(rmax_s, rmax_d, EPS) << "n=" << n;
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(ref_b[i], dis_bv[i].value, 1e-13) << "n=" << n << " i=" << i;

        auto dis_gv = computeGammaBatch(dis_bv);
        ASSERT_EQ(dis_gv.size(), n) << "n=" << n;
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(ref_g[i], dis_gv[i].value, 1e-12) << "n=" << n << " i=" << i;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Clamping
// ════════════════════════════════════════════════════════════════════════════

TEST(Clamping, BetaNeverReachesBetaMaxSafe) {
    BetaCalculator calc;
    // velocity == running_max → beta would be 1.0 → must be clamped
    auto betas = calc.computeBetaBatch({5.0});
    EXPECT_LT(betas[0].value, BETA_MAX);
    EXPECT_GE(betas[0].value, 0.0);
}

TEST(Clamping, LargeBatchAllClamped) {
    auto vels = make_velocities(256, 100.0, 7777u);
    double rmax = 0.0;
    auto bv = computeBetaBatch(vels, rmax);
    for (const auto& b : bv) {
        EXPECT_GE(b.value, 0.0);
        EXPECT_LT(b.value, BETA_MAX);
        EXPECT_TRUE(std::isfinite(b.value));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// KnownValues
// ════════════════════════════════════════════════════════════════════════════

TEST(KnownValues, NewtonianLimit_Beta0_Gamma1) {
    std::vector<BetaVelocity> betas = {BetaVelocity{0.0}};
    auto gammas = computeGammaBatch(betas);
    ASSERT_EQ(gammas.size(), 1u);
    EXPECT_NEAR(gammas[0].value, 1.0, EPS);
}

TEST(KnownValues, Beta06_Gamma125) {
    std::vector<BetaVelocity> betas = {BetaVelocity{0.6}};
    auto gammas = computeGammaBatch(betas);
    ASSERT_EQ(gammas.size(), 1u);
    EXPECT_NEAR(gammas[0].value, 1.25, 1e-9);
}

TEST(KnownValues, GammaMonotonicalltyIncreasesWithBeta) {
    std::vector<double> beta_vals;
    for (int i = 0; i <= 99; ++i)
        beta_vals.push_back(static_cast<double>(i) / 100.0 * (BETA_MAX - 0.001));
    std::vector<double> gamma_out(100);
    compute_gamma_scalar(beta_vals.data(), 100, gamma_out.data());
    for (int i = 1; i < 100; ++i)
        EXPECT_GE(gamma_out[i], gamma_out[i - 1]) << "i=" << i;
}

// ════════════════════════════════════════════════════════════════════════════
// LargeBatchFiniteness
// ════════════════════════════════════════════════════════════════════════════

TEST(LargeBatchFiniteness, N65536AllFiniteInRange) {
    constexpr std::size_t N = 65536;
    auto vels = make_velocities(N, 1000.0, 99999u);
    double rmax = 0.0;
    auto bv = computeBetaBatch(vels, rmax);
    auto gv = computeGammaBatch(bv);
    ASSERT_EQ(bv.size(), N);
    ASSERT_EQ(gv.size(), N);
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_TRUE(std::isfinite(bv[i].value)) << "i=" << i;
        EXPECT_GE(bv[i].value, 0.0)             << "i=" << i;
        EXPECT_LT(bv[i].value, BETA_MAX)         << "i=" << i;
        EXPECT_TRUE(std::isfinite(gv[i].value)) << "i=" << i;
        EXPECT_GE(gv[i].value, 1.0)             << "i=" << i;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// EdgeCases
// ════════════════════════════════════════════════════════════════════════════

TEST(EdgeCases, AllZeroVelocities) {
    std::vector<double> vels(64, 0.0);
    double rmax = 0.0;
    auto bv = computeBetaBatch(vels, rmax);
    auto gv = computeGammaBatch(bv);
    EXPECT_DOUBLE_EQ(rmax, 0.0);
    for (const auto& b : bv) EXPECT_DOUBLE_EQ(b.value, 0.0);
    for (const auto& g : gv) EXPECT_NEAR(g.value, 1.0, EPS);
}

TEST(EdgeCases, SingleNonzeroVelocity) {
    double rmax = 0.0;
    auto bv = computeBetaBatch({7.5}, rmax);
    ASSERT_EQ(bv.size(), 1u);
    EXPECT_NEAR(rmax, 7.5, EPS);
    EXPECT_LT(bv[0].value, BETA_MAX);
    auto gv = computeGammaBatch(bv);
    ASSERT_EQ(gv.size(), 1u);
    EXPECT_GE(gv[0].value, 1.0);
    EXPECT_TRUE(std::isfinite(gv[0].value));
}

TEST(EdgeCases, NegativeVelocitiesProduceSameBeta) {
    std::vector<double> pos = {1.0, 3.0, 7.0, 2.0, 5.0};
    std::vector<double> neg(pos.size());
    for (std::size_t i = 0; i < pos.size(); ++i) neg[i] = -pos[i];
    double rpos = 0.0, rneg = 0.0;
    auto bpos = computeBetaBatch(pos, rpos);
    auto bneg = computeBetaBatch(neg, rneg);
    EXPECT_NEAR(rpos, rneg, EPS);
    for (std::size_t i = 0; i < pos.size(); ++i) {
        EXPECT_NEAR(bpos[i].value, bneg[i].value, EPS) << "i=" << i;
        EXPECT_GE(bneg[i].value, 0.0) << "i=" << i;
    }
}
