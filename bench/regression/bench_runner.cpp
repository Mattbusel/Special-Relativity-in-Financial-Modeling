/**
 * @file  bench_runner.cpp
 * @brief Standalone micro-benchmark runner for performance regression suite.
 *
 * Usage:
 *   ./bench_runner <benchmark_name>
 *
 * Outputs a single double: nanoseconds per operation, to stdout.
 * Returns 0 on success, 1 on unknown benchmark name.
 *
 * Each benchmark runs for a wall-clock duration of at least 500ms to get
 * stable measurements, then divides total time by iteration count.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "beta_calculator/beta_calculator.hpp"
#include "geodesic/geodesic_solver.hpp"
#include "manifold/spacetime_manifold.hpp"
#include "momentum/momentum.hpp"
#include "engine/engine.hpp"

using namespace srfm;
using namespace std::chrono;

// ── Timing harness ────────────────────────────────────────────────────────────

template<typename Fn>
double measure_ns_per_op(Fn&& fn, long min_iters = 100) {
    // Warmup
    for (long i = 0; i < std::min(min_iters / 10L, 1000L); ++i) fn();

    // Measure until we have at least 500ms of wall time
    long iters      = 0;
    double total_ns = 0.0;

    const auto deadline = steady_clock::now() + milliseconds(500);
    do {
        const auto t0 = steady_clock::now();
        fn();
        fn(); fn(); fn(); fn(); fn();  // batch 5 to reduce timer overhead
        const auto t1 = steady_clock::now();
        total_ns += static_cast<double>(duration_cast<nanoseconds>(t1 - t0).count());
        iters += 5;
    } while (steady_clock::now() < deadline || iters < min_iters);

    return total_ns / static_cast<double>(iters);
}

// ── Benchmark implementations ─────────────────────────────────────────────────

double bench_gamma_compute_1M() {
    using namespace srfm::momentum;
    auto bv = BetaVelocity::make(0.6).value();
    volatile double sink = 0.0;
    return measure_ns_per_op([&]() {
        auto g = lorentz_gamma(bv);
        sink += g ? g->value() : 0.0;
    }, 1'000'000);
}

double bench_beta_compute_1M() {
    using namespace srfm::beta_calculator;
    const std::vector<double> prices{100.0, 100.5, 101.0, 100.8, 101.5};
    BetaCalculator calc;
    volatile double sink = 0.0;
    return measure_ns_per_op([&]() {
        auto r = calc.fromPriceVelocityOnline(prices, 1.0);
        sink += r ? r->beta : 0.0;
    }, 1'000'000);
}

double bench_full_pipeline_1M() {
    using namespace srfm::engine;
    Engine eng;
    const std::string csv = "100.0,101.5,102.0,101.8,103.0";
    volatile double sink = 0.0;
    return measure_ns_per_op([&]() {
        auto r = eng.process(csv);
        sink += r ? r->beta : 0.0;
    }, 1'000'000);
}

double bench_christoffel_compute() {
    using namespace srfm::manifold;
    SpacetimeManifold mfld;
    const MetricTensor flat = MetricTensor::minkowski();
    volatile double sink = 0.0;
    return measure_ns_per_op([&]() {
        auto ch = mfld.christoffelSymbols(flat);
        sink += ch[0];
    });
}

double bench_rk4_geodesic_100steps() {
    using namespace srfm::geodesic;
    using namespace srfm::manifold;
    GeodesicSolver solver;
    GeodesicState init{};
    init.x = {0.0, 0.0, 0.0, 0.0};
    init.u = {1.0, 0.3, 0.1, 0.0};
    const MetricTensor flat = MetricTensor::minkowski();
    volatile double sink = 0.0;
    return measure_ns_per_op([&]() {
        auto r = solver.solve(init, flat, 100, 0.001);
        sink += r ? r->x[0] : 0.0;
    });
}

double bench_doppler_factor_1M() {
    using namespace srfm::momentum;
    using namespace srfm::beta_calculator;
    auto bv = BetaVelocity::make(0.7).value();
    volatile double sink = 0.0;
    return measure_ns_per_op([&]() {
        auto d = doppler_factor(bv);
        sink += d ? *d : 0.0;
    }, 1'000'000);
}

double bench_rapidity_1M() {
    using namespace srfm::momentum;
    using namespace srfm::beta_calculator;
    auto bv = BetaVelocity::make(0.5).value();
    volatile double sink = 0.0;
    return measure_ns_per_op([&]() {
        auto phi = rapidity(bv);
        sink += phi ? *phi : 0.0;
    }, 1'000'000);
}

double bench_compose_velocities_1M() {
    using namespace srfm::momentum;
    auto b1 = BetaVelocity::make(0.3).value();
    auto b2 = BetaVelocity::make(0.4).value();
    volatile double sink = 0.0;
    return measure_ns_per_op([&]() {
        auto r = compose_velocities(b1, b2);
        sink += r ? r->value() : 0.0;
    }, 1'000'000);
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <benchmark_name>\n", argv[0]);
        return 1;
    }

    const std::string name = argv[1];
    double result = -1.0;

    if (name == "gamma_compute_1M")          result = bench_gamma_compute_1M();
    else if (name == "beta_compute_1M")      result = bench_beta_compute_1M();
    else if (name == "full_pipeline_1M")     result = bench_full_pipeline_1M();
    else if (name == "christoffel_compute")  result = bench_christoffel_compute();
    else if (name == "rk4_geodesic_100steps") result = bench_rk4_geodesic_100steps();
    else if (name == "doppler_factor_1M")    result = bench_doppler_factor_1M();
    else if (name == "rapidity_1M")          result = bench_rapidity_1M();
    else if (name == "compose_velocities_1M") result = bench_compose_velocities_1M();
    else {
        std::fprintf(stderr, "Unknown benchmark: %s\n", name.c_str());
        return 1;
    }

    std::printf("%.2f\n", result);
    return 0;
}
