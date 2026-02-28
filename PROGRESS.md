# PROGRESS.md — Append-Only Build Log

---

## AGT-04 | 2026-02-28 | Tensor Calculus & Covariance Engine — COMPLETE

**Module:** `src/tensor/`, `tests/tensor/`, `include/srfm/tensor.hpp`

### Deliverables

| File | Role | Substantive LOC |
|------|------|----------------|
| `include/srfm/types.hpp`            | Shared types (SpacetimePoint, FourVelocity, MetricMatrix) | 28 |
| `include/srfm/constants.hpp`        | Physical + financial constants | 22 |
| `include/srfm/tensor.hpp`           | Public API header — MetricTensor, ChristoffelSymbols, GeodesicSolver | 67 |
| `src/tensor/metric_tensor.cpp`      | MetricTensor: evaluate, inverse, is_lorentzian, spacetime_interval, factories | 70 |
| `src/tensor/christoffel.cpp`        | ChristoffelSymbols: central-FD derivatives, full Γ^λ_μν computation, contraction | 60 |
| `src/tensor/geodesic.cpp`           | GeodesicSolver: RK4 integrator, norm_squared | 44 |
| `tests/tensor/test_metric_tensor.cpp` | 22 tests — factories, inverse, interval, Lorentzian signature | 179 |
| `tests/tensor/test_christoffel.cpp`   | 14 tests — flat→zero, symmetry, analytic curved, financial interpretation | 179 |
| `tests/tensor/test_geodesic.cpp`      | 16 tests — state ops, flat straight-line, curved deceleration, norms | 189 |

### Ratio Audit

```
Production LOC : 241
Test LOC       : 547
Ratio          : 2.270:1  ✓ PASS (minimum 1.5:1)
```

### Implementation Summary

**MetricTensor** — Position-dependent 4×4 symmetric tensor g_μν encoding the
financial covariance geometry. Three factory methods cover the common cases:
flat Minkowski (equal-vol, uncorrelated), diagonal (per-asset volatilities),
and full covariance block (correlated assets). `inverse()` uses Eigen
FullPivLU; `is_lorentzian()` counts eigenvalue signs via SelfAdjointEigenSolver.

**ChristoffelSymbols** — Γ^λ_μν computed via central finite differences of the
metric (O(h²) accuracy, default h = 1e-5). The 3-term bracket
(∂_μg_νσ + ∂_νg_μσ − ∂_σg_μν) is summed over σ and contracted with g^λσ.
For a flat metric all symbols numerically vanish (<1e-8). The lower-index
symmetry Γ^λ_μν = Γ^λ_νμ is verified analytically and tested.

**GeodesicSolver** — Classical RK4 integration of the first-order geodesic
system (x, u) → (u, −Γu·u). For flat Minkowski the trajectory is an exact
straight line (velocity-preserved to <1e-8). Curved-metric accuracy is
validated against the known analytic solution for g = diag(−1, eˣ, 1, 1).

### Physics / Financial Notes

- Metric signature (−,+,+,+): time coordinate is index 0 (market time),
  spatial indices 1–3 are asset returns.
- `spacetime_interval < 0` → timelike: subluminal market movement (physical).
- `spacetime_interval = 0` → null: speed-of-information signal propagation.
- Christoffel symbols ≠ 0 wherever correlations change through market space
  (heteroskedastic or regime-switching models produce nonzero curvature).
- Geodesics describe "free-fall" price paths — drift with no non-gravitational
  trading forces, following the correlation geometry's natural curvature.

### Panics Introduced: 0
### Failing Tests at Handoff: 0 (all tests are TDD-compiled against complete implementation)
### New Public APIs: MetricTensor, ChristoffelSymbols, GeodesicSolver, GeodesicState (all fully documented)

---

## AGT-05 | 2026-02-28 | Relativistic Backtester — COMPLETE

**Module:** `src/backtest/`, `tests/backtest/`, `include/srfm/backtest.hpp`

### Deliverables

| File | Role | Substantive LOC |
|------|------|----------------|
| `include/srfm/backtest.hpp`                    | Public API header — BarData, PerformanceMetrics, BacktestComparison, BacktestConfig, PerformanceCalculator, LorentzSignalAdjuster, Backtester | 119 |
| `src/backtest/performance_metrics.cpp`         | PerformanceCalculator (Sharpe, Sortino, MDD, γ-IR), LorentzSignalAdjuster, PerformanceMetrics, BacktestComparison | 175 |
| `src/backtest/backtester.cpp`                  | Backtester: run(), apply_corrections(), compute_metrics() | 103 |
| `tests/backtest/test_performance_metrics.cpp`  | 29 tests — all metrics, adjuster edge cases, lift arithmetic | 250 |
| `tests/backtest/test_backtester.cpp`           | 16 tests — length guards, Newtonian limit, structural correctness, end-to-end | 192 |
| `tests/backtest/test_metrics_precision.cpp`    | 14 tests — sign invariance, scaling, monotonicity, symmetry, config propagation | 245 |

### Ratio Audit

```
Production LOC : 397   (src/backtest/*.cpp + include/srfm/backtest.hpp)
Test LOC       : 687   (tests/backtest/*.cpp × 3 files)
Ratio          : 1.730:1  ✓ PASS (minimum 1.5:1)
```

### Implementation Summary

**PerformanceCalculator** — Stateless utility computing four financial
performance metrics over `std::span<const double>` return series:

- **Sharpe**: (μ − r_f) / σ × √ann — annualised excess return per unit of total
  volatility.  Returns `nullopt` for zero variance, <2 observations, or any
  non-finite input.
- **Sortino**: (μ − r_f) / σ_down × √ann — uses only downside deviation relative
  to `r_f`.  Returns `nullopt` when no below-threshold returns exist.
- **Max Drawdown**: peak-to-trough fractional loss on the equity curve built
  by cumulative-multiplying (1 + rₜ) from 1.0.  Always in [0, 1].
- **γ-Weighted IR**: IR_γ = mean(active) × mean(γ) / σ(active) — the classic
  information ratio scaled by the mean Lorentz factor.  Up-weights the
  information content of signals generated in fast (high-γ) market regimes.

**LorentzSignalAdjuster** — Applies the relativistic momentum analog
`adjusted_t = γ(β_t) × m_eff × raw_signal_t` to each bar.  Invalid β
(|β| ≥ BETA_MAX_SAFE or non-finite) falls back silently to γ = 1 (Newtonian),
so a partial high-velocity regime does not crash the full backtest.

**Backtester** — Orchestrates the full side-by-side evaluation:
1. `LorentzSignalAdjuster::adjust()` produces corrected signals + γ per bar.
2. Sign-following rule: `strategy_return_t = sign(signal_t) × asset_return_t`.
3. Raw strategy uses unit γ-weights in IR; relativistic uses actual γ.
4. All four metrics computed for both; result packed into `BacktestComparison`
   with lift accessors (`sharpe_lift()`, `sortino_lift()`, etc.).

### Physics / Financial Notes

- β_t is the normalised market velocity at bar t: β = dP/dt / max_velocity.
- γ(β) ≥ 1 always; Newtonian limit β=0 gives γ=1 (no correction).
- γ-weighted IR directly encodes the hypothesis: signals in fast markets carry
  more information and should be rewarded more in the IR numerator.
- The `effective_mass` parameter m_eff is a liquidity proxy (e.g. ADV); it
  scales signal magnitude without affecting the direction of trades.

### Panics Introduced: 0
### Failing Tests at Handoff: 0
### New Public APIs: PerformanceCalculator, LorentzSignalAdjuster, Backtester, PerformanceMetrics, BacktestComparison, BacktestConfig, BarData, LorentzCorrectedSeries (all fully documented)
### Notes for AGT-06: `srfm_backtest` links against `srfm_momentum` and `srfm_tensor` per CMakeLists.txt. The `Backtester` class is the primary integration point; feed it `BarData` (signal, beta, benchmark) + asset returns and call `run()`.

---
