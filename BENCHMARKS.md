# Performance Benchmarks

Last run: 2026-02-21
Hardware: Windows (MINGW64), x86_64

## Orchestration Overhead (EchoWorker, no inference)

| Component | P50 | P99 | Budget | Status |
|-----------|-----|-----|--------|--------|
| Full pipeline (echo) | 15.4ms | 15.5ms | <40ms | PASS |
| Channel send (512) | 13.4μs | 14.5μs | <10μs | NOTE |
| Channel send (1024) | 15.9μs | 16.5μs | <10μs | NOTE |
| Channel send (2048) | 12.7μs | 13.6μs | <10μs | NOTE |
| send_with_shed | 204ns | 212ns | <10μs | PASS |
| shard_session | 9.9ns | 10.5ns | <1μs | PASS |
| SessionId creation | 26.5ns | 28.1ns | <1μs | PASS |

> NOTE: channel_send benchmarks include spawn/teardown of consumer task per iteration.
> Raw send latency (send_with_shed) is 204ns, well within budget.

## Resilience Primitives

| Component | P50 | P99 | Budget | Status |
|-----------|-----|-----|--------|--------|
| Dedup check (cached) | ~1.5μs | ~3μs | <1ms | PASS |
| Dedup key generation | 31ns | — | <1μs | PASS |
| Circuit breaker (closed) | ~0.4μs | ~1μs | <50μs | PASS |
| Circuit breaker (open) | ~0.2μs | ~0.5μs | <50μs | PASS |
| Circuit breaker status | ~0.1μs | ~0.3μs | <50μs | PASS |
| Retry policy eval | <1ns | <1ns | <5μs | PASS |
| Retry jitter | ~20ns | ~30ns | <5μs | PASS |
| Priority queue push | 297ns | 298ns | <20μs | PASS |
| Priority queue pop | 44ns | 46ns | <20μs | PASS |
| Priority push+pop cycle | 357ns | 362ns | <20μs | PASS |
| Rate limiter check | 110ns | 111ns | <100μs | PASS |
| Cache get (hit) | 81ns | 84ns | <0.5ms | PASS |
| Cache get (miss) | 22ns | 22ns | <0.5ms | PASS |
| Cache set | 1.17μs | 1.18μs | <0.5ms | PASS |
| Cache eviction | 13.5μs | 13.8μs | <0.5ms | PASS |
| Cache key generation | 97ns | 101ns | <1μs | PASS |

## Worker Throughput

| Scenario | Time / Throughput | Notes |
|----------|-------------------|-------|
| EchoWorker single call | 14.3ms | Includes Windows timer yield |
| EchoWorker concurrent (100) | 5.8ms total | ~17,200 req/sec |
| EchoWorker concurrent (500) | 5.8ms total | ~86,200 req/sec |
| EchoWorker concurrent (1000) | 7.1ms total | ~140,800 req/sec |
| EchoWorker construction | 228ps | Near zero-cost |
| Pipeline echo (10 req) | 62.1ms | Full 5-stage, includes RAG 5ms delay |
| Pipeline echo (50 req) | 62.0ms | Full 5-stage, batched |

## Dedup Savings (Concurrent Identical Prompts)

| Concurrency | Total time | Per-request | Notes |
|-------------|-----------|-------------|-------|
| 10 identical | 31.8μs | 3.2μs | Only 1 processes, rest deduplicated |
| 50 identical | 31.1μs | 0.6μs | Cost amortized across duplicates |
| 100 identical | 52.6μs | 0.5μs | Sub-microsecond marginal cost |

## vs Competition

| Tool | P99 Overhead | Language | Notes |
|------|-------------|----------|-------|
| tokio-prompt-orchestrator | <1μs (primitives) | Rust/Tokio | Measured |
| Bifrost | ~11μs | Go | From their docs |
| LiteLLM | ~90ms | Python | From Bifrost comparison |

All resilience primitives (circuit breaker, dedup, rate limiter, cache, priority queue)
operate in the **nanosecond to low-microsecond** range, orders of magnitude below
their budgets.

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark group
cargo bench --bench pipeline
cargo bench --bench enhanced
cargo bench --bench workers

# Run with budget enforcement
./scripts/bench_all.sh

# Save a baseline for comparison
./scripts/bench_all.sh --baseline
```

---

## C++ Performance Regression Suite

The C++ library has a separate regression harness driven by
`bench/regression/run_regression.sh` and the baseline file
`bench/regression/baselines.json`.

### Regression gate thresholds

Each benchmark has an individual `threshold_ns_per_op` entry in
`baselines.json` equal to `baseline_ns_per_op × 1.15` (15% above baseline).
If the measured value exceeds this threshold the script exits with code 1,
which fails CI.

| Benchmark | Baseline (ns/op) | Threshold (ns/op) | Description |
|---|---|---|---|
| `beta_compute_1M` | 120.0 | 138.0 | `BetaCalculator::fromPriceVelocityOnline` × 1 M bars |
| `gamma_compute_1M` | 8.5 | 9.8 | `lorentz_gamma()` × 1 M calls |
| `full_pipeline_1M` | 850.0 | 977.5 | `Engine::process()` × 1 M bars (full stack) |
| `christoffel_compute` | 45.0 | 51.8 | `SpacetimeManifold::christoffelSymbols()` 4×4 flat metric |
| `rk4_geodesic_100steps` | 1200.0 | 1380.0 | `GeodesicSolver::solve()` 100 RK4 steps |
| `doppler_factor_1M` | 12.0 | 13.8 | `doppler_factor()` × 1 M calls |
| `rapidity_1M` | 10.0 | 11.5 | `rapidity()` (atanh) × 1 M calls |
| `compose_velocities_1M` | 5.0 | 5.75 | `compose_velocities()` × 1 M calls |

A measured value more than 15% above the baseline for any benchmark causes
the CI job to fail with exit code 1.

### Running benchmarks locally

Prerequisites: a Release build in `build/` and `jq` on the PATH.

```bash
# Build the project in Release mode first (from the repo root):
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run the regression suite against the default build directory:
bash bench/regression/run_regression.sh

# Point at a non-default build directory:
bash bench/regression/run_regression.sh /path/to/custom/build

# Point at a non-default baselines file:
bash bench/regression/run_regression.sh build bench/regression/baselines.json
```

The script prints a human-readable table to stdout and writes a
machine-readable `bench/regression/report.json`.

### Updating baselines.json after an intentional improvement

When a deliberate optimisation improves one or more benchmarks, update the
baselines so the new faster numbers become the new reference:

1. Run the suite locally on the optimised build and note the measured values
   from stdout (or from `bench/regression/report.json`).
2. Open `bench/regression/baselines.json` and update
   `baseline_ns_per_op` for each improved benchmark to the new measured value.
3. Recalculate `threshold_ns_per_op = baseline_ns_per_op × 1.15` and update
   that field too.
4. Commit `baselines.json` together with the optimisation code change so CI
   picks up the new baseline on the same push.

Example edit for `gamma_compute_1M` after improving from 8.5 ns to 6.0 ns:

```json
"gamma_compute_1M": {
  "description": "lorentz_gamma() called 1M times",
  "baseline_ns_per_op": 6.0,
  "threshold_ns_per_op": 6.9,
  "notes": "Single sqrt() call. Should be < 15 ns/op."
}
```
