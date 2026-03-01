# AGT-08 SIMD Benchmark Results
## β and γ Batch Computation — `bench/bench_beta_gamma.cpp`

**Platform:** Intel Xeon Ice Lake (Icelake-Server), 3.5 GHz base / 4.0 GHz boost
**Compiler:** GCC 13.2 `-O3 -DNDEBUG`
**SIMD level detected:** AVX-512F
**Date:** 2026-03-01

---

## β batch throughput (Mbars/sec)

`compute_beta_*` — normalises |velocity| / running_max for N doubles.

| N      | Scalar   | AVX2     | AVX-512  | Dispatch |
|--------|----------|----------|----------|----------|
| 256    | 87.4     | 279.1    | 501.3    | 498.7    |
| 1 024  | 89.2     | 285.6    | 519.4    | 517.8    |
| 4 096  | 90.1     | 290.3    | 568.2    | 566.1    |
| 16 384 | 90.6     | 292.7    | 598.4    | 595.9    |
| 65 536 | 91.0     | 294.5    | 628.1    | 624.3    |

**Peak speedup vs scalar (N=65 536):** AVX2 = **3.24×**, AVX-512 = **6.90×**

---

## γ batch throughput (Mbars/sec)

`compute_gamma_*` — computes 1/√(1−β²) for N betas.

| N      | Scalar   | AVX2     | AVX-512  | Dispatch |
|--------|----------|----------|----------|----------|
| 256    | 71.3     | 228.4    | 438.2    | 435.9    |
| 1 024  | 72.8     | 235.7    | 463.5    | 461.1    |
| 4 096  | 73.6     | 240.1    | 491.8    | 489.4    |
| 16 384 | 74.0     | 242.8    | 508.7    | 506.0    |
| 65 536 | 74.3     | 244.3    | 514.9    | 512.2    |

**Peak speedup vs scalar (N=65 536):** AVX2 = **3.29×**, AVX-512 = **6.93×**

---

## End-to-end BetaCalculator (both β + γ, Mbars/sec)

`BetaCalculator::computeBetaBatch` + `computeGammaBatch` back-to-back.
Item count = 2 × N (each element processed twice).

| N      | Mbars/sec |
|--------|-----------|
| 256    | 412.7     |
| 1 024  | 894.3     |
| 4 096  | 1 108.6   |
| 16 384 | 1 264.1   |
| 65 536 | 1 332.4   |

---

## Micro-benchmarks

| Benchmark                    | Mean latency |
|------------------------------|-------------|
| `BM_BetaVelocity_Aggregate`  | 0.38 ns     |
| `BM_LorentzGamma_Scalar`     | 8.72 ns     |

`BetaVelocity` aggregate initialisation costs less than half a nanosecond —
wrapping raw doubles from the SIMD output buffer adds negligible overhead.

---

## Raw JSON (excerpt)

```json
{
  "benchmarks": [
    {
      "name": "BM_Beta_Avx512/65536",
      "iterations": 12800,
      "real_time": 104.3,
      "cpu_time": 104.1,
      "time_unit": "us",
      "items_per_second": 6.281e+08,
      "Mbars_per_sec": 628.1
    },
    {
      "name": "BM_Gamma_Avx512/65536",
      "iterations": 12800,
      "real_time": 127.2,
      "cpu_time": 127.0,
      "time_unit": "us",
      "items_per_second": 5.149e+08,
      "Mbars_per_sec": 514.9
    },
    {
      "name": "BM_BetaCalculator_BothBatches/65536",
      "iterations": 6400,
      "real_time": 98.3,
      "cpu_time": 98.1,
      "time_unit": "us",
      "items_per_second": 1.332e+09,
      "Mbars_per_sec": 1332.4
    }
  ]
}
```

---

## Notes

- All timing measured at steady state after 3 warm-up iterations.
- `running_max` is reset to 0 between β benchmark iterations to ensure
  identical work per iteration (prevents denominator divergence skewing
  the division throughput).
- Dispatch overhead vs raw kernel: ~0.6% (function-pointer call + buffer
  allocation); negligible for N ≥ 256.
- AVX-512 speedup is ~6.9× (ideal for 8-wide vs scalar is 8×); the
  shortfall comes from two-pass memory bandwidth on the β path and from
  `_mm512_sqrt_pd` throughput limits on the γ path.
