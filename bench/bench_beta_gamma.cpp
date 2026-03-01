/**
 * @file  bench_beta_gamma.cpp
 * @brief Google Benchmark suite for β and γ batch computation.
 *
 * Module:  bench/
 * Owner:   AGT-08  —  2026-03-01
 *
 * What is benchmarked
 * -------------------
 *   For N ∈ {256, 1024, 4096, 16384, 65536} elements:
 *
 *   Beta computation:
 *     BM_Beta_Scalar_N    — compute_beta_scalar kernel
 *     BM_Beta_AVX2_N      — compute_beta_avx2   kernel  (skipped if !AVX2)
 *     BM_Beta_AVX512_N    — compute_beta_avx512 kernel  (skipped if !AVX-512)
 *     BM_Beta_Dispatch_N  — full computeBetaBatch (runtime dispatch)
 *
 *   Gamma computation:
 *     BM_Gamma_Scalar_N   — compute_gamma_scalar kernel
 *     BM_Gamma_AVX2_N     — compute_gamma_avx2   kernel  (skipped if !AVX2)
 *     BM_Gamma_AVX512_N   — compute_gamma_avx512 kernel  (skipped if !AVX-512)
 *     BM_Gamma_Dispatch_N — full computeGammaBatch (runtime dispatch)
 *
 *   End-to-end:
 *     BM_BetaCalculator_BothBatches_N — BetaCalculator::computeBetaBatch
 *                                       + computeGammaBatch in sequence
 *
 * Throughput metric
 * -----------------
 *   benchmark::Counter::kIsRate is set so that Google Benchmark reports
 *   operations/second automatically.  Each "item" is one bar (one double
 *   processed), so the reported rate is "bars/second".  We then also set
 *   a custom counter "MB_per_sec" = bytes_processed / elapsed for memory
 *   bandwidth context.
 *
 * Building
 * --------
 *   cmake --build build --target bench
 *   ./build/bench/bench_beta_gamma --benchmark_format=json
 *
 * Interpreting results
 * --------------------
 *   The benchmark drives the hot loop in a state_range loop so the compiler
 *   cannot eliminate it.  black_box() (DoNotOptimize) is applied to both
 *   inputs and outputs to defeat dead-code elimination.
 */

#include <benchmark/benchmark.h>

#include "srfm/simd/cpu_features.hpp"    // include/ on path
#include "srfm/simd/simd_dispatch.hpp"   // include/ on path
#include "simd_batch_detail.hpp"         // src/simd/ on path (internal)

#include <vector>
#include <random>
#include <cmath>
#include <numeric>

using namespace srfm::simd;
using namespace srfm::simd::detail;

// ── Shared test data ──────────────────────────────────────────────────────────

namespace {

/// Generate N random velocities in [−100, +100].
std::vector<double> make_velocities(std::size_t n, std::uint32_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    std::vector<double> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

/// Generate N beta values in [0, BETA_MAX_SAFE - 0.001] (uniform spacing).
std::vector<double> make_betas(std::size_t n) {
    const double limit = srfm::momentum::BETA_MAX_SAFE - 0.001;
    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) {
        b[i] = limit * static_cast<double>(i) / static_cast<double>(n - 1 ? n - 1 : 1);
    }
    return b;
}

} // anonymous namespace

// ══════════════════════════════════════════════════════════════════════════════
// Beta benchmarks — scalar
// ══════════════════════════════════════════════════════════════════════════════

static void BM_Beta_Scalar(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    auto vels = make_velocities(n);
    std::vector<double> out(n);

    for (auto _ : state) {
        double rmax = 0.0;
        compute_beta_scalar(vels.data(), n, rmax, out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::DoNotOptimize(rmax);
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = n * sizeof(double) * 2;  // read + write
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(bytes));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Beta_Scalar)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)
                          ->Unit(benchmark::kMicrosecond);

// ══════════════════════════════════════════════════════════════════════════════
// Beta benchmarks — AVX2
// ══════════════════════════════════════════════════════════════════════════════

static void BM_Beta_AVX2(benchmark::State& state) {
    if (!has_avx2()) {
        state.SkipWithError("AVX2 not available on this CPU");
        return;
    }

    const std::size_t n = static_cast<std::size_t>(state.range(0));
    auto vels = make_velocities(n);
    std::vector<double> out(n);

    for (auto _ : state) {
        double rmax = 0.0;
        compute_beta_avx2(vels.data(), n, rmax, out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::DoNotOptimize(rmax);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n * sizeof(double) * 2));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Beta_AVX2)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)
                         ->Unit(benchmark::kMicrosecond);

// ══════════════════════════════════════════════════════════════════════════════
// Beta benchmarks — AVX-512
// ══════════════════════════════════════════════════════════════════════════════

static void BM_Beta_AVX512(benchmark::State& state) {
    if (!has_avx512f()) {
        state.SkipWithError("AVX-512F not available on this CPU");
        return;
    }

    const std::size_t n = static_cast<std::size_t>(state.range(0));
    auto vels = make_velocities(n);
    std::vector<double> out(n);

    for (auto _ : state) {
        double rmax = 0.0;
        compute_beta_avx512(vels.data(), n, rmax, out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::DoNotOptimize(rmax);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n * sizeof(double) * 2));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Beta_AVX512)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)
                           ->Unit(benchmark::kMicrosecond);

// ══════════════════════════════════════════════════════════════════════════════
// Beta benchmarks — runtime dispatch (includes wrapping into BetaVelocity)
// ══════════════════════════════════════════════════════════════════════════════

static void BM_Beta_Dispatch(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    auto vels = make_velocities(n);

    for (auto _ : state) {
        double rmax = 0.0;
        auto result = computeBetaBatch(vels, rmax);
        benchmark::DoNotOptimize(result.data());
        benchmark::DoNotOptimize(rmax);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Beta_Dispatch)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)
                             ->Unit(benchmark::kMicrosecond);

// ══════════════════════════════════════════════════════════════════════════════
// Gamma benchmarks — scalar
// ══════════════════════════════════════════════════════════════════════════════

static void BM_Gamma_Scalar(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    auto betas = make_betas(n);
    std::vector<double> out(n);

    for (auto _ : state) {
        compute_gamma_scalar(betas.data(), n, out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n * sizeof(double) * 2));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Gamma_Scalar)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)
                            ->Unit(benchmark::kMicrosecond);

// ══════════════════════════════════════════════════════════════════════════════
// Gamma benchmarks — AVX2
// ══════════════════════════════════════════════════════════════════════════════

static void BM_Gamma_AVX2(benchmark::State& state) {
    if (!has_avx2()) {
        state.SkipWithError("AVX2 not available on this CPU");
        return;
    }

    const std::size_t n = static_cast<std::size_t>(state.range(0));
    auto betas = make_betas(n);
    std::vector<double> out(n);

    for (auto _ : state) {
        compute_gamma_avx2(betas.data(), n, out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n * sizeof(double) * 2));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Gamma_AVX2)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)
                          ->Unit(benchmark::kMicrosecond);

// ══════════════════════════════════════════════════════════════════════════════
// Gamma benchmarks — AVX-512
// ══════════════════════════════════════════════════════════════════════════════

static void BM_Gamma_AVX512(benchmark::State& state) {
    if (!has_avx512f()) {
        state.SkipWithError("AVX-512F not available on this CPU");
        return;
    }

    const std::size_t n = static_cast<std::size_t>(state.range(0));
    auto betas = make_betas(n);
    std::vector<double> out(n);

    for (auto _ : state) {
        compute_gamma_avx512(betas.data(), n, out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n * sizeof(double) * 2));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Gamma_AVX512)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)
                            ->Unit(benchmark::kMicrosecond);

// ══════════════════════════════════════════════════════════════════════════════
// Gamma benchmarks — runtime dispatch (includes LorentzFactor wrapping)
// ══════════════════════════════════════════════════════════════════════════════

static void BM_Gamma_Dispatch(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    auto vels = make_velocities(n);
    double rmax = 0.0;
    auto bv = computeBetaBatch(vels, rmax);  // Pre-computed; not part of benchmark

    for (auto _ : state) {
        auto gammas = computeGammaBatch(bv);
        benchmark::DoNotOptimize(gammas.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Gamma_Dispatch)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)
                              ->Unit(benchmark::kMicrosecond);

// ══════════════════════════════════════════════════════════════════════════════
// End-to-end: BetaCalculator — both batches in sequence
// ══════════════════════════════════════════════════════════════════════════════

static void BM_BetaCalculator_BothBatches(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    auto vels = make_velocities(n);

    BetaCalculator calc;

    for (auto _ : state) {
        calc.reset();
        auto betas  = calc.computeBetaBatch(vels);
        auto gammas = calc.computeGammaBatch(betas);
        benchmark::DoNotOptimize(gammas.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(n));
    state.counters["Mbars_per_sec"] = benchmark::Counter(
        static_cast<double>(n) / 1e6,
        benchmark::Counter::kIsRate);
    state.counters["SIMD_level"] = static_cast<double>(
        static_cast<int>(calc.simd_level()));
}
BENCHMARK(BM_BetaCalculator_BothBatches)
    ->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)
    ->Unit(benchmark::kMicrosecond);

// ══════════════════════════════════════════════════════════════════════════════
// Micro: cost of a single BetaVelocity::make() validation call
// ══════════════════════════════════════════════════════════════════════════════

static void BM_BetaVelocity_Make(benchmark::State& state) {
    double beta = 0.6;
    for (auto _ : state) {
        auto bv = srfm::momentum::BetaVelocity::make(beta);
        benchmark::DoNotOptimize(bv);
        beta = (beta < 0.9) ? beta + 1e-7 : 0.1;  // avoid dead-code
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_BetaVelocity_Make)->Unit(benchmark::kNanosecond);

// ══════════════════════════════════════════════════════════════════════════════
// Micro: cost of lorentz_gamma() (single scalar sqrt)
// ══════════════════════════════════════════════════════════════════════════════

static void BM_LorentzGamma_Scalar(benchmark::State& state) {
    auto bv = *srfm::momentum::BetaVelocity::make(0.6);
    for (auto _ : state) {
        auto g = srfm::momentum::lorentz_gamma(bv);
        benchmark::DoNotOptimize(g);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_LorentzGamma_Scalar)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
