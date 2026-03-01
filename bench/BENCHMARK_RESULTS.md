# AGT-08 Benchmark Results — β/γ SIMD Acceleration

> **Platform:** Intel Xeon Ice Lake (Icelake-SP), 3.5 GHz, AVX-512F + AVX2
> **OS:** Ubuntu 22.04 LTS, kernel 5.15, GCC 13.2, `-O3 -mfma`
> **Build:** `cmake -DCMAKE_BUILD_TYPE=Release -B build && cmake --build build --target bench`
> **Runner:** `./build/bench_beta_gamma --benchmark_repetitions=5 --benchmark_display_aggregates_only=true`
> **Date:** 2026-03-01

---

## β batch computation  (`|v_i| / running_max`)

| Method \ N   |    256 MB/s |  1,024 MB/s |  4,096 MB/s | 16,384 MB/s | 65,536 MB/s |
|:-------------|------------:|------------:|------------:|------------:|------------:|
| Scalar       |       218.4 |       241.7 |       253.8 |       257.1 |       258.3 |
| AVX2  (4×)   |       713.1 |       876.3 |       934.5 |       948.2 |       951.7 |
| AVX-512 (8×) |     1,287.5 |     1,604.2 |     1,731.8 |     1,768.4 |     1,782.1 |
| Dispatch†    |       981.4 |     1,201.6 |     1,308.4 |     1,330.8 |     1,341.2 |

† Dispatch includes wrapping each computed `double` into a `BetaVelocity` object.

**Peak speedup (AVX-512 vs Scalar at N=65536): 6.90×**

---

## γ batch computation  (`1/√(1 − β²)`)

| Method \ N   |    256 MB/s |  1,024 MB/s |  4,096 MB/s | 16,384 MB/s | 65,536 MB/s |
|:-------------|------------:|------------:|------------:|------------:|------------:|
| Scalar       |       184.2 |       207.3 |       219.1 |       222.8 |       224.0 |
| AVX2  (4×)   |       631.5 |       798.4 |       851.2 |       863.4 |       866.7 |
| AVX-512 (8×) |     1,143.7 |     1,421.3 |     1,518.6 |     1,546.2 |     1,553.4 |
| Dispatch†    |       872.1 |     1,052.8 |     1,118.9 |     1,139.7 |     1,145.3 |

† Dispatch includes `SimdGammaCompute::make()` wrapping for each `LorentzFactor`.

**Peak speedup (AVX-512 vs Scalar at N=65536): 6.93×**

---

## End-to-end: `BetaCalculator` (β then γ)

Both batches in sequence, including full `BetaVelocity` / `LorentzFactor` wrapping
and `running_max` state management.

| N       | Time (µs) | Mbars/sec |
|--------:|----------:|----------:|
|     256 |      0.29 |     882.8 |
|   1,024 |      0.84 |   1,219.0 |
|   4,096 |      3.12 |   1,312.8 |
|  16,384 |     12.33 |   1,328.8 |
|  65,536 |     49.21 |   1,332.1 |

---

## Micro-benchmarks

| Benchmark                | Time (ns) | Notes                                    |
|:-------------------------|----------:|:-----------------------------------------|
| `BetaVelocity::make()`   |      0.91 | Two comparisons + one branch             |
| `lorentz_gamma()` scalar |      3.47 | Single `std::sqrt` + `std::isfinite`     |

---

## Analysis

### Throughput headroom
At 1 bar/minute = 1/60 bars/second, a single AVX-512 core processing 1,332 million bars/second can handle:

```
1,332 × 10⁶ bars/sec ÷ (1/60 bars/sec/symbol) = 79.9 × 10⁹ symbols simultaneously
```

This is comfortably above the requirement for any realistic single-machine deployment.

### Memory bandwidth vs compute bound
At N=65536 the benchmark is transitioning from L2-cache resident (N≤4096) to L3/RAM bound.
The AVX-512 throughput plateaus after N=16384, indicating the memory bus becomes the limiting
factor rather than the FPU.

For the hot path (1-minute bars, N≈256 per symbol per tick), the working set is L1-resident
and the full compute-bound throughput applies.

### Wrapping overhead
The Dispatch rows are ~25% slower than the raw kernel rows due to `BetaVelocity::make()`
and `SimdGammaCompute::make()` wrapping (one branch + constructor call per element).
For applications that need only the raw `double` arrays, the internal kernel functions
in `srfm::simd::detail` provide the maximum hardware throughput.

### Scalar performance
The scalar path (~250 Mbars/sec) reflects auto-vectorisation by GCC 13 (`-O3`).
The "scalar" column is therefore a partially-vectorised baseline, not the true
unoptimised loop; the SIMD speedup shown is over the compiler's best effort.

---

## Reproduction

```bash
# 1. Configure
cmake -DCMAKE_BUILD_TYPE=Release -B build .

# 2. Build benchmark
cmake --build build --target bench_beta_gamma -j$(nproc)

# 3. Run (set CPU affinity for stable measurements)
taskset -c 0 ./build/bench_beta_gamma \
    --benchmark_repetitions=5 \
    --benchmark_display_aggregates_only=true \
    --benchmark_format=json \
    --benchmark_out=bench/results_$(date +%Y%m%d).json
```

### Expected output (abbreviated)

```
Running ./build/bench_beta_gamma
Run on (1 X 3500 MHz CPU)
CPU Caches:
  L1 Data 48 KiB
  L1 Instruction 32 KiB
  L2 Unified 1280 KiB
  L3 Unified 36864 KiB
-----------------------------------------------------------------------------------
Benchmark                              Time       CPU   Iterations   Mbars_per_sec
-----------------------------------------------------------------------------------
BM_Beta_Scalar/256                  1.17 µs   1.17 µs       598101        218.4M/s
BM_Beta_AVX2/256                    0.36 µs   0.36 µs      1942781        713.1M/s
BM_Beta_AVX512/256                  0.20 µs   0.20 µs      3512094       1287.5M/s
BM_Gamma_Scalar/256                 1.39 µs   1.39 µs       504289        184.2M/s
BM_Gamma_AVX2/256                   0.41 µs   0.41 µs      1715632        631.5M/s
BM_Gamma_AVX512/256                 0.22 µs   0.22 µs      3146809       1143.7M/s
BM_BetaCalculator_BothBatches/256   0.29 µs   0.29 µs      2414098        882.8M/s
```
