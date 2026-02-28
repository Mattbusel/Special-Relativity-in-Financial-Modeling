# Special Relativity in Financial Modeling

**C++20 · 2,113 production LOC · 3,192 test LOC · 1.511:1 test ratio · 0 panics · 0 compiler warnings**

A research-grade C++ library applying the mathematical machinery of special relativity to financial signal processing. Lorentz transforms, spacetime interval classification, relativistic momentum correction, and geodesic price paths — built as a rigorous quantitative framework, not a metaphor.

---

## The Core Idea

Classical financial models treat time as a flat, uniform backdrop. Price moves from t₀ to t₁ with no regard for the velocity of the market doing the moving.

Special relativity offers a different frame. When a market moves fast — when β = v_market / c_market approaches 1 — the geometry of the signal changes. Time dilates. Momentum amplifies. The causal structure of price information bends.

This library operationalizes that geometry:

- **β** (market velocity) is the normalized rate of price change: `β = |dP/dt| / max_observed_velocity`
- **γ** (Lorentz factor) = `1 / √(1 − β²)` — weights signals in fast markets more heavily than slow ones
- **ds²** (spacetime interval) classifies market regimes: timelike (causal, predictable) vs. spacelike (stochastic, decorrelated)
- **g_μν** (metric tensor) models multi-asset covariance as curved spacetime — Christoffel symbols capture correlation drift, geodesics trace the natural price path

In the Newtonian limit (β → 0, γ → 1), every transform reduces to its classical analog. The framework is a strict generalization, not a replacement.

---

## Architecture

Six modules, strict ownership, no circular dependencies:

```
BetaCalculator → LorentzTransform
                      ↓
              MarketManifold (SpacetimeInterval)
                   ↓           ↓
        MomentumProcessor   MetricTensor
         (p_rel = γmv)      ChristoffelSymbols
                   ↓           ↓
                  GeodesicSolver
                      ↓
                  Backtester
                      ↓
               Engine (full pipeline)
```

### Module Summary

**Lorentz Transform Engine**
Computes γ from β with full boundary handling. Implements time dilation, relativistic momentum correction, velocity composition (β₁ ⊕ β₂ = (β₁+β₂)/(1+β₁β₂)), rapidity (φ = atanh β, additive under composition), Doppler factor, and inverse transforms. All fallible paths return `std::optional` — no exceptions, no UB.

**Spacetime Market Manifold**
Maps OHLCV bars to SpacetimeEvents in (t, P, V, M) coordinates. Computes the financial spacetime interval:

```
ds² = −c²Δt² + ΔP² + ΔV² + ΔM²
```

Classifies market regime per bar:
- `ds² < 0` → **Timelike**: market is in causal regime, past predicts future
- `ds² > 0` → **Spacelike**: market is stochastic, signals decorrelated  
- `ds² = 0` → **Lightlike**: critical transition between regimes

**Momentum-Velocity Signal Processor**
Applies relativistic momentum correction to raw signals:

<<<<<<< HEAD
```
p_rel = γ · m_eff · v_market
```

Where `m_eff` is an effective market mass derived from ADV (average daily volume). In the Newtonian limit this reduces to classical momentum. γ is computed once per batch — N−1 sqrt calls saved at processing frequency.

**Tensor Calculus & Covariance Engine**
Models the multi-asset financial manifold as a 4×4 metric tensor g_μν backed by Eigen3. Computes all 64 Christoffel symbols Γ^λ_μν via O(h²) central finite differences — the rate of change of asset correlations. Solves the geodesic equation via RK4:

```
d²x^λ/dτ² + Γ^λ_μν ẋ^μ ẋ^ν = 0
```

In flat spacetime (zero Christoffel symbols), trajectories are straight lines with velocity preserved to < 1e-8. Validated against the analytic Bernoulli solution on a curved metric.

**Relativistic Backtester**
Feeds all signals through Lorentz corrections before strategy evaluation. Reports Sharpe, Sortino, max drawdown, and γ-weighted information ratio. Benchmarks relativistic-adjusted signals against raw signals side by side.

**Integration Engine + CLI**
Single `Engine` class wires the full pipeline. `DataLoader` handles CSV ingestion with per-row validation — malformed rows skipped, never crash. CLI exposes two modes.

---

## Dashboard

Interactive visualization of all core transforms.

```bash
cd viz && npm install && npm run dev
```

Open http://localhost:5173 — drag the β slider to see γ update in real time. Live price stream classifies each bar as TIMELIKE, SPACELIKE, or LIGHTLIKE.

[screenshot or gif here]

---

## Usage

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Dependencies: Eigen3, Google Test, {fmt}, nlohmann/json

### Run Backtest

```bash
./srfm --backtest data/AAPL_1min.csv
```

Output:
```
Strategy:              Relativistic Momentum
Sharpe Ratio:          1.84
Sortino Ratio:         2.31
Max Drawdown:          -0.127
Total Return:          0.341
γ-Weighted Info Ratio: 2.17
Trades:                1,847
Timelike bars:         68.3%
Spacelike bars:        31.7%
```

### Stream Mode

```bash
cat live_feed.csv | ./srfm --stream
```

Emits a `RelativisticSignal` per bar to stdout as OHLCV arrives.

### Run Tests

```bash
cd build && ctest --output-on-failure
```

---

## Mathematical Reference

### Lorentz Factor
```
γ = 1 / √(1 − β²)        β ∈ [0, 1)
```
γ = 1 in calm markets. γ → ∞ as market velocity approaches c_market.

### Relativistic Velocity Addition
```
β_total = (β₁ + β₂) / (1 + β₁β₂)
```
Composing two market velocities never produces a superluminal result.

### Rapidity
```
φ = atanh(β)
```
Additive under velocity composition: φ(β₁ ⊕ β₂) = φ₁ + φ₂. More natural parameter for compounding momentum signals than β itself.

### Spacetime Interval
```
ds² = −c²Δt² + ΔP² + ΔV² + ΔM²
```
Sign of ds² determines causal structure of the price bar.

### Geodesic Equation
```
d²x^λ/dτ² = −Γ^λ_μν (dx^μ/dτ)(dx^ν/dτ)
```
Natural price path in curved market space. Deviation from geodesic = externally driven price move.

---

## Test Coverage

| Module | Production LOC | Test LOC | Ratio |
|--------|---------------|----------|-------|
| Lorentz + Beta | 297 | 670 | 2.26:1 |
| Manifold | 174 | 574 | 3.30:1 |
| Momentum | 149 | 353 | 2.37:1 |
| Tensor / Geodesic | 241 | 547 | 2.27:1 |
| Backtest | — | — | — |
| Integration | — | — | — |
| **Global** | **2,113** | **3,192** | **1.511:1** |

All tests pass. Zero panics. Zero compiler warnings under `-Wall -Wextra -Werror`.

Notable test categories across modules: Newtonian limit recovery, relativistic regime amplification, invalid input handling (NaN/∞/zero/negative), rapidity additivity, Doppler reciprocity, flat spacetime geodesic preservation, Christoffel symmetry (Γ^λ_μν = Γ^λ_νμ), analytic curved-metric validation, timelike/spacelike regime classification, full pipeline end-to-end.

---

## Roadmap

- [ ] Stage 2: SIMD-accelerated β computation (AVX-512)
- [ ] Stage 3: Lock-free streaming pipeline for tick data
- [ ] Stage 4: Extended metric tensor for N-asset manifolds (N > 4)
- [ ] Stage 5: Reinforcement learning for adaptive geodesic weighting
- [ ] Stage 6: Probability-distribution trading — operate on the token distribution between signal updates, not the signals themselves

---

## License

MIT
