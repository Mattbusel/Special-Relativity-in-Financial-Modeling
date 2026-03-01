/**
 * @file  bench/bench_beta_gamma.cpp
 * @brief Google Benchmark suite for SRFM AGT-08 SIMD β/γ batch computation.
 *
 * Module:  bench/
 * Owner:   AGT-08  —  2026-03-01
 *
 * Benchmarks
 * ----------
 *   BM_Beta_Scalar / AVX2 / AVX512 / Dispatch
 *   BM_Gamma_Scalar / AVX2 / AVX512 / Dispatch
 *   BM_BetaCalculator_BothBatches
 *   BM_BetaVelocity_Aggregate     — aggregate-init overhead
 *
 * Build (CMake):
 *   cmake --build build --target bench_beta_gamma
 *   ./build/bench_beta_gamma --benchmark_format=json
 *
 * Build (manual, MSVC):
 *   see scripts/build_simd.bat
 *
 * Throughput units: items/second (doubles processed).
 * Custom counter "Mbars_per_sec" = throughput / 1e6.
 */

#include "benchmark/benchmark.h"

// Kernel detail header (internal, needs src/simd on include path)
#include "simd_batch_detail.hpp"

// Public dispatch header
#include "srfm/simd/simd_dispatch.hpp"
#include "srfm/simd/cpu_features.hpp"
#include "srfm/types.hpp"

#include <vector>
#include <numeric>
#include <cmath>
#include <cstddef>

// ── Fixture helpers ────────────────────────────────────────────────────────────

/// Generate N synthetic price velocities spread evenly across [-1, 1].
static std::vector<double> make_velocities(std::size_t n) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n - 1 > 0 ? n - 1 : 1);
    }
    return v;
}

/// Generate N BetaVelocity values in [0, 0.9999].
static std::vector<srfm::BetaVelocity> make_betas(std::size_t n) {
    std::vector<srfm::BetaVelocity> b(n);
    for (std::size_t i = 0; i < n; ++i) {
        double val = 0.9999 * static_cast<double>(i) / static_cast<double>(n - 1 > 0 ? n - 1 : 1);
        b[i] = srfm::BetaVelocity{val};
    }
    return b;
}

// ── β batch benchmarks ─────────────────────────────────────────────────────────

static void BM_Beta_Scalar(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto vels = make_velocities(n);
    std::vector<double> out(n);
    double running_max = 0.0;
    for (auto _ : state) {
        srfm::simd::detail::compute_beta_scalar(vels.data(), n, running_max, out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
        running_max = 0.0;  // reset to keep identical work each iter
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Beta_Scalar)->RangeMultiplier(4)->Range(256, 65536)->Unit(benchmark::kMicrosecond);

static void BM_Beta_Avx2(benchmark::State& state) {
    if (srfm::simd::detect_simd_level() < srfm::simd::SimdLevel::AVX2) {
        state.SkipWithError("AVX2 not available on this CPU");
        return;
    }
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto vels = make_velocities(n);
    std::vector<double> out(n);
    double running_max = 0.0;
    for (auto _ : state) {
        srfm::simd::detail::compute_beta_avx2(vels.data(), n, running_max, out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
        running_max = 0.0;
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Beta_Avx2)->RangeMultiplier(4)->Range(256, 65536)->Unit(benchmark::kMicrosecond);

static void BM_Beta_Avx512(benchmark::State& state) {
    if (srfm::simd::detect_simd_level() < srfm::simd::SimdLevel::AVX512F) {
        state.SkipWithError("AVX-512F not available on this CPU");
        return;
    }
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto vels = make_velocities(n);
    std::vector<double> out(n);
    double running_max = 0.0;
    for (auto _ : state) {
        srfm::simd::detail::compute_beta_avx512(vels.data(), n, running_max, out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
        running_max = 0.0;
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Beta_Avx512)->RangeMultiplier(4)->Range(256, 65536)->Unit(benchmark::kMicrosecond);

static void BM_Beta_Dispatch(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto vels = make_velocities(n);
    double running_max = 0.0;
    for (auto _ : state) {
        auto result = srfm::simd::computeBetaBatch(vels, running_max);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
        running_max = 0.0;
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Beta_Dispatch)->RangeMultiplier(4)->Range(256, 65536)->Unit(benchmark::kMicrosecond);

// ── γ batch benchmarks ─────────────────────────────────────────────────────────

static void BM_Gamma_Scalar(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto betas_raw = [&]{
        std::vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i)
            b[i] = 0.9999 * static_cast<double>(i) / static_cast<double>(n - 1 > 0 ? n - 1 : 1);
        return b;
    }();
    std::vector<double> out(n);
    for (auto _ : state) {
        srfm::simd::detail::compute_gamma_scalar(betas_raw.data(), n, out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Gamma_Scalar)->RangeMultiplier(4)->Range(256, 65536)->Unit(benchmark::kMicrosecond);

static void BM_Gamma_Avx2(benchmark::State& state) {
    if (srfm::simd::detect_simd_level() < srfm::simd::SimdLevel::AVX2) {
        state.SkipWithError("AVX2 not available on this CPU");
        return;
    }
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto betas_raw = [&]{
        std::vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i)
            b[i] = 0.9999 * static_cast<double>(i) / static_cast<double>(n - 1 > 0 ? n - 1 : 1);
        return b;
    }();
    std::vector<double> out(n);
    for (auto _ : state) {
        srfm::simd::detail::compute_gamma_avx2(betas_raw.data(), n, out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Gamma_Avx2)->RangeMultiplier(4)->Range(256, 65536)->Unit(benchmark::kMicrosecond);

static void BM_Gamma_Avx512(benchmark::State& state) {
    if (srfm::simd::detect_simd_level() < srfm::simd::SimdLevel::AVX512F) {
        state.SkipWithError("AVX-512F not available on this CPU");
        return;
    }
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto betas_raw = [&]{
        std::vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i)
            b[i] = 0.9999 * static_cast<double>(i) / static_cast<double>(n - 1 > 0 ? n - 1 : 1);
        return b;
    }();
    std::vector<double> out(n);
    for (auto _ : state) {
        srfm::simd::detail::compute_gamma_avx512(betas_raw.data(), n, out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Gamma_Avx512)->RangeMultiplier(4)->Range(256, 65536)->Unit(benchmark::kMicrosecond);

static void BM_Gamma_Dispatch(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto betas = make_betas(n);
    for (auto _ : state) {
        auto result = srfm::simd::computeGammaBatch(betas);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Gamma_Dispatch)->RangeMultiplier(4)->Range(256, 65536)->Unit(benchmark::kMicrosecond);

// ── End-to-end BetaCalculator ──────────────────────────────────────────────────

static void BM_BetaCalculator_BothBatches(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto vels = make_velocities(n);
    for (auto _ : state) {
        srfm::simd::BetaCalculator calc;
        auto betas  = calc.computeBetaBatch(vels);
        auto gammas = calc.computeGammaBatch(betas);
        benchmark::DoNotOptimize(gammas.data());
        benchmark::ClobberMemory();
    }
    // Each iteration processes n elements through both β and γ pipelines.
    const int64_t items = static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * 2;
    state.SetItemsProcessed(items);
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(items) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_BetaCalculator_BothBatches)->RangeMultiplier(4)->Range(256, 65536)->Unit(benchmark::kMicrosecond);

// ── Micro: BetaVelocity aggregate initialisation overhead ─────────────────────

static void BM_BetaVelocity_Aggregate(benchmark::State& state) {
    volatile double val = 0.5;
    for (auto _ : state) {
        srfm::BetaVelocity bv{val};
        benchmark::DoNotOptimize(bv.value);
    }
}
BENCHMARK(BM_BetaVelocity_Aggregate);

// ── Micro: scalar gamma for a single value ─────────────────────────────────────

static void BM_LorentzGamma_Scalar(benchmark::State& state) {
    volatile double beta = 0.6;
    for (auto _ : state) {
        const double b     = beta;
        const double gamma = 1.0 / std::sqrt(1.0 - b * b);
        benchmark::DoNotOptimize(gamma);
    }
}
BENCHMARK(BM_LorentzGamma_Scalar);

BENCHMARK_MAIN();
