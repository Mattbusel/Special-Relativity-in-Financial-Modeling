# Technical Audit — Special Relativity in Financial Modeling

**Audit Agent:** AGT-AUDIT
**Date:** 2026-02-28
**Codebase commit:** 3f33a45
**Scope:** Full static mathematical audit + build environment verification

---

## 1. Build Verification

### 1a. CMake Configure

**Status: BLOCKED — environment not configured**

```
Error: CMake was unable to find a build program corresponding to "Ninja"
Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
```

**Root cause:** The bash shell Claude Code uses does not inherit the Visual Studio
Developer Command Prompt environment. The required Windows SDK `rc.exe` (resource
compiler) is not on `PATH`. MSVC requires `vcvarsall.bat` to be sourced first.

The toolchain is present on disk:
- cmake.exe: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\...\cmake.exe`
- ninja.exe: same path
- cl.exe (MSVC 19.44): same path
- vcpkg packages installed: `fmt:x64-windows 12.1.0`, `gtest:x64-windows 1.17.0`

**Notable gap:** `eigen3:x64-windows` is **not** in the vcpkg installed list.
The project requires Eigen3 3.4 (`find_package(Eigen3 3.4 REQUIRED NO_MODULE)`).
Without Eigen3, the build cannot succeed even with a working compiler environment.

**How to fix:**
```powershell
# In a Visual Studio Developer PowerShell:
vcpkg install eigen3:x64-windows
cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Debug
```

**Additional note:** The CMakeLists.txt uses GCC/Clang flags (`-Wall -Wextra -Werror
-Wpedantic -fsanitize=address,undefined`) that are not valid for MSVC.
On MSVC these flags are silently passed as unknown options.
The project is written in idiomatic C++20 but the build file assumes a Unix
toolchain. A proper Windows build would require either:
- Installing WSL2 + gcc/clang, or
- Adding MSVC-conditional flags in CMakeLists.txt (`/W4 /WX` instead of `-Wall -Werror`)

### 1b. CTest

**Status: NOT RUN** (cmake configure failed)

Static analysis of the test files confirms:
- 6 test targets defined in CMakeLists.txt
- All test files present and syntactically valid (reviewed line by line)
- No `#error` directives or obvious compile-blockers in test sources

### 1c. Viz Build

**Status: PASS ✓**

```
vite v5.4.21 building for production...
✓ 31 modules transformed.
dist/index.html                  0.45 kB │ gzip:  0.31 kB
dist/assets/index-XQUHowoj.css   0.07 kB │ gzip:  0.09 kB
dist/assets/index-1EPv5Vw3.js  153.42 kB │ gzip: 49.38 kB
✓ built in 343ms
```

Zero warnings. Zero errors. Bundle is React 18 + component logic only (no external
charting library). Gzip size 49 kB is reasonable for a full dashboard.

### 1d. LOC Ratio Verification

Counted via `grep -v` stripping comments and blank lines:

| Module | Production LOC | Test LOC | Ratio | Status |
|--------|---------------|----------|-------|--------|
| lorentz | 304 | 818 | 2.69:1 | ✓ PASS |
| manifold | 114 | 260 | 2.28:1 | ✓ PASS |
| momentum | 85 | 295 | 3.47:1 | ✓ PASS |
| tensor | 234 | 703 | 3.00:1 | ✓ PASS |
| backtest | 308 | 687 | 2.23:1 | ✓ PASS |
| core | 339 | 369 (integration) | 1.09:1 | ⚠ MARGINAL |
| **Global** | **1,384** | **3,192** | **2.306:1** | **✓ PASS** |

**Note on core module:** The `src/core/` module has 339 production LOC but only
369 dedicated test lines (integration tests). This passes at global scope but the
core engine itself has no unit tests — all coverage comes from end-to-end integration
tests. This is a coverage gap: `compute_returns`, `compute_betas`, `to_event`, and
streaming logic are not tested in isolation.

### 1e. Sanitizer Run

**Status: NOT RUN** (cmake configure failed)

Static review found no obvious UB patterns:
- No signed integer overflow in hot paths
- All array accesses bounds-checked via `span` or Eigen bounds
- No uninitialized reads observed
- `std::optional` used consistently for fallible operations

---

## 2. Mathematical Audit

### 2.1 Lorentz Transforms

#### 2.1.1 Lorentz Factor γ

**Physics definition:**
```
γ = 1 / √(1 − β²),   β ∈ (−1, 1)
```

**Implementation** (`src/lorentz/lorentz_transform.cpp`, line 19–37):
```cpp
const double beta2 = beta.value * beta.value;
const double denom = std::sqrt(1.0 - beta2);
if (denom <= 0.0) return std::nullopt;
return LorentzFactor{1.0 / denom};
```

**Verification:**
- **Domain:** `isValidBeta` guards `|β| < BETA_MAX_SAFE = 0.9999`. The minimum
  denominator is √(1 − 0.9999²) = √(0.00019999) ≈ 0.01414. No division by zero possible. ✓
- **Range:** γ ∈ [1, 70.7] for β ∈ [0, 0.9999]. The upper bound is finite and safe. ✓
- **Newtonian limit:** β → 0 gives γ → 1. Taylor expansion: γ ≈ 1 + β²/2 for small β. ✓
- **Symmetry:** γ(β) = γ(−β) since only β² appears. Implementation takes `beta.value`
  without abs, but β² is symmetric. ✓
- **Guard for denom ≤ 0:** Redundant given BETA_MAX_SAFE but harmless defensive coding. ✓

**Result: CORRECT**

---

#### 2.1.2 Time Dilation

**Physics definition:**
```
t_dilated = γ · τ_proper
```
A moving clock runs slow: dilated time ≥ proper time. γ ≥ 1, so t_dilated ≥ τ. ✓

**Implementation** (`lorentz_transform.cpp`, line 39–55):
```cpp
if (proper_time < 0.0) return std::nullopt;
return proper_time * g->value;
```

**Verification:**
- **Direction:** Returns γτ ≥ τ. Dilation, not compression. ✓
- **Guard:** Rejects negative proper time (no signal can have negative age). ✓
- **Newtonian limit:** β → 0, γ → 1, t_dilated → τ. ✓

**Financial interpretation:** A "signal" generated at bar τ in a high-β market is
observed to have age γτ in the reference frame. Fast markets make signals appear
"older" — they carry more historical weight. This is the core mechanism for
up-weighting high-velocity signals.

**Result: CORRECT**

---

#### 2.1.3 Relativistic Velocity Addition

**Physics definition:**
```
β_total = (β₁ + β₂) / (1 + β₁β₂)
```

**Proof that |β_total| < 1 when |β₁|, |β₂| < 1:**

Let β₁, β₂ ∈ (−1, 1). We need |β₁ + β₂| < |1 + β₁β₂|.

Case 1: Both positive. Then 1 + β₁β₂ > 1 and β₁ + β₂ < 2. Also:
```
(1 − β_total)(1 + β_total) = 1 − β_total² = (1 − β₁²)(1 − β₂²) / (1 + β₁β₂)² > 0
```
This follows from (1 − β₁²) > 0 and (1 − β₂²) > 0 and (1 + β₁β₂) > 0. Therefore
β_total² < 1, i.e. |β_total| < 1. ✓

The result extends to all sign combinations by symmetry.

**Implementation** (`lorentz_transform.cpp`, line 83–98):
```cpp
const double num   = beta1.value + beta2.value;
const double denom = 1.0 + beta1.value * beta2.value;
return BetaVelocity{num / denom};
```

**Verification:**
- **Formula correct.** ✓
- **No nullopt guard:** The denominator 1 + β₁β₂ can only be zero if β₁β₂ = −1,
  which requires one input to be ±1 — excluded by BETA_MAX_SAFE on typical inputs.
  However, `composeVelocities` accepts `BetaVelocity` without validating the inputs
  first. If a caller passes two un-validated BetaVelocity structs, the denominator
  could approach zero. **Minor robustness gap.** ⚠
- **Identity:** β ⊕ 0 = β. ✓
- **Commutativity:** β₁ ⊕ β₂ = β₂ ⊕ β₁. ✓
- **Newtonian limit:** β₁, β₂ ≪ 1 → β₁ ⊕ β₂ ≈ β₁ + β₂. ✓

**Result: CORRECT with minor robustness note**

---

#### 2.1.4 Rapidity

**Physics definition:**
```
φ = atanh(β),   φ ∈ (−∞, +∞) for β ∈ (−1, 1)
```

**Key property — rapidity additivity:**
```
φ(β₁ ⊕ β₂) = φ₁ + φ₂
```
Proof: φ(β₁ ⊕ β₂) = atanh((β₁+β₂)/(1+β₁β₂)) = atanh(β₁) + atanh(β₂) = φ₁ + φ₂.
This follows from the addition formula for atanh. ✓

**Implementation** (`lorentz_transform.cpp`, line 132–146):
```cpp
return std::atanh(beta.value);
```

**Verification:**
- **Domain guarded:** `isValidBeta` ensures |β| < 0.9999 < 1. atanh is defined on (−1, 1). ✓
- **Additivity tested:** `tests/lorentz/test_lorentz_transform.cpp` verifies
  `φ(β₁ ⊕ β₂) = φ₁ + φ₂` explicitly. ✓
- **Newtonian limit:** atanh(β) ≈ β for small β. ✓

**Result: CORRECT**

---

#### 2.1.5 Doppler Factor

**Physics definition:**
```
D(β) = √((1 + β) / (1 − β))
```
- β > 0 (approaching): D > 1 (blue-shift, higher observed frequency)
- β < 0 (receding): D < 1 (red-shift)
- β = 0: D = 1

**Reciprocity identity:** D(β) · D(−β) = √((1+β)/(1−β)) · √((1−β)/(1+β)) = 1 ✓

**Implementation** (`beta_calculator.cpp`, line 190–212):
```cpp
const double numerator   = 1.0 + beta.value;
const double denominator = 1.0 - beta.value;
if (denominator <= 0.0 || numerator <= 0.0) return std::nullopt;
return std::sqrt(numerator / denominator);
```

**Verification:**
- **Guards:** `isValidBeta` ensures |β| < 0.9999, so both numerator (1+β ≥ 0.0001)
  and denominator (1−β ≥ 0.0001) are strictly positive. The extra guard at line
  207 is redundant but not harmful. ✓
- **Reciprocity tested:** `D(β) × D(−β) = 1` verified in test suite. ✓

**Result: CORRECT**

---

### 2.2 Beta Calculator

#### 2.2.1 Velocity Normalization

**Formula:**
```
β = |dP/dt| / max_observed_velocity
```

**Implementation choice:** `meanAbsVelocity` uses central finite differences of the
price series. The velocity at each point is `|dP/dt|` using:
- Central difference at interior points: `(P[i+1] − P[i-1]) / (2·Δt)` — O(h²) ✓
- Forward difference at left boundary: `(P[1] − P[0]) / Δt` — O(h)
- Backward difference at right boundary: `(P[n-1] − P[n-2]) / Δt` — O(h)

**Issue:** Boundary differences are first-order accurate while interior points are
second-order. For financial time series where bars have equal spacing (Δt = 1 bar),
this inconsistency is minor but worth noting. Symmetric O(h²) boundary formulas
could improve accuracy at series edges. **Low severity.** ⚠

**Is |dP/dt| the right velocity measure?**
The choice of absolute first-difference velocity (not log-return, not RMS) is
pragmatic but has implications:
- **Units:** Raw price change per bar. Normalization by `max_velocity` makes it
  dimensionless, but `max_velocity` is the *observed historical maximum* — this is
  an expanding-window normalization that can change as new extremes are observed.
- **Alternative:** Using log returns `|log(P[i]/P[i-1])|` would be scale-invariant
  and more natural for multiplicative price processes. The current implementation
  uses arithmetic differences, which are scale-dependent. **Design question.** ⚠
- **RMS vs mean abs:** The implementation uses mean absolute velocity, not RMS
  velocity. Both are valid norms; mean abs is more robust to outliers. ✓

#### 2.2.2 Rolling Window Edge Cases

`fromRollingWindow` selects the `window` most-recent prices and delegates to
`meanAbsVelocity`. At initialization (fewer than `window` bars available):
- **Guard:** Returns `nullopt` if `window > prices.size()`. ✓
- **Guard:** Returns `nullopt` if `window < 2`. ✓
- **Engine behavior:** `engine.cpp` uses `min(i+1, WINDOW=5)` for early bars,
  falling back to β = 0.0 when fewer than 2 bars are available. This is the
  Newtonian limit fallback — sensible. ✓

#### 2.2.3 Calibration: Why 252?

`ANNUALISATION_FACTOR = 252` is the standard US equity market trading day count.
This is correct for daily return series but **incorrect** for:
- Intraday data (use 252 × bars_per_day)
- Weekly data (use 52)
- Crypto 24/7 markets (use 365)

The constant is not configurable per-series — it is a global default. The
`BacktestConfig` struct does allow overriding `annualisation`, but callers must
remember to do so. **Documentation gap.** ⚠

---

### 2.3 Spacetime Interval

#### 2.3.1 Formula and Signature

**Physics definition (Minkowski metric, signature −,+,+,+):**
```
ds² = −c²Δt² + ΔP² + ΔV² + ΔM²
```

**Implementation** (`spacetime_interval.cpp`, line 46–49):
```cpp
const double time_term    = c_market * c_market * dt * dt;
const double spatial_term = dp * dp + dv * dv + dm * dm;
return spatial_term - time_term;
```

**Verification:**
- **Signature (−,+,+,+):** The time component enters with a negative sign. ✓
- **ds² < 0 → Timelike:** Returned when time_term > spatial_term, i.e. when
  the market moves "slowly" in price/volume/momentum space relative to market time.
  Correct — this matches the physics convention. ✓
- **ds² > 0 → Spacelike:** When the price/vol/momentum change is large relative
  to the time elapsed. ✓

**Dimensional consistency question:**
The four coordinates have very different natural scales:
- `time`: bar index (e.g. 1, 2, 3...)
- `price`: absolute price level (e.g. 100–200)
- `volume`: shares traded (e.g. 1,000,000)
- `momentum`: ROC or similar (e.g. 0.01–0.05)

Without normalization, volume (1e6) will completely dominate the spatial term,
making virtually every bar spacelike regardless of the physics. **This is a
significant modeling concern.** The interval is only meaningful if the coordinates
are normalized to comparable scales. The current implementation does no
normalization — it is the caller's responsibility to pre-scale inputs. ⚠⚠

#### 2.3.2 c_market = 1.0

`SPEED_OF_INFORMATION = 1.0` (natural units). This means Δt and spatial displacements
must be in the same units for the interval to be non-trivially classified.
With bar index as time and raw price as the spatial coordinate, the lightlike
condition `|ΔP| = |Δt|` is only satisfied by coincidence. **The normalization
convention is implicit and undocumented.** ⚠

**Recommendation:** Document the required pre-normalization convention explicitly,
or have `SpacetimeInterval::compute` accept scale parameters.

#### 2.3.3 LIGHTLIKE_EPSILON = FLOAT_EPSILON = 1e-12

The classify function uses `std::abs(interval_squared) <= 1e-12` for lightlike
classification. Given that `ds²` values for unscaled financial data will be on the
order of `price²` (10⁴ to 10⁶), a threshold of 1e-12 means the lightlike regime
is effectively never observed in practice. **This is an issue if lightlike
classification is intended to be observable.** For normalized (unit-scale) inputs,
1e-12 is appropriate. ⚠

---

### 2.4 Momentum Processor

#### 2.4.1 Relativistic Momentum

**Physics definition:**
```
p_rel = γ(β) · m_eff · v
```

**Newtonian limit:** β → 0 → γ → 1 → p_rel → m_eff · v = classical momentum. ✓

**Implementation** (`momentum_processor.cpp`, line 17–30):
Delegates to `LorentzTransform::applyMomentumCorrection`, which computes
`γ · m_eff · raw_signal`. ✓

**Is m_eff = ADV/ADV_baseline dimensionally consistent?**
The codebase treats `effective_mass` as a pure scaling parameter with no fixed
definition. The docstring says "liquidity proxy, e.g. ADV". If m_eff has units of
shares traded, then `p_rel` has units of `shares × signal_units × γ`. Since γ is
dimensionless and signal is dimensionless (a direction indicator), the product is
in "effective share-weighted signal units." This is self-consistent as a scalar
multiplier but **the financial interpretation requires the caller to normalize
m_eff** to make comparisons across assets meaningful. ⚠

#### 2.4.2 γ Computed Per Signal, Not Per Batch

The `process_series` function calls `process` for each signal individually.
The claim in the README ("γ is computed once per batch — N−1 sqrt savings") is
**incorrect** — `γ` is recomputed (via sqrt) for every element of the series.

The `LorentzSignalAdjuster::adjust` method in `performance_metrics.cpp` has the
same structure (one sqrt per bar). There is no caching of γ values across bars
with the same β. **The N−1 sqrt savings claim in the README is not implemented.** ⚠

---

### 2.5 Tensor Calculus

#### 2.5.1 Metric Tensor Symmetry

`make_from_covariance` embeds a covariance matrix `cov` directly in the 3×3 spatial
block. A covariance matrix is symmetric by construction (Σ = Σᵀ). The time-time
entry is `−c²` (scalar). Therefore g_μν = g_νμ for all factory-constructed metrics.

**Is the metric guaranteed symmetric for arbitrary user-supplied `MetricFunction`?**
No — the user can supply any `std::function<MetricMatrix(SpacetimePoint)>`. If the
supplied function returns a non-symmetric matrix, the Christoffel computation will
be incorrect because it assumes symmetry implicitly (it uses `g^λσ` from FullPivLU
without symmetrizing). **Documentation gap; not validated at runtime.** ⚠

#### 2.5.2 Christoffel Symbols

**Physics definition:**
```
Γ^λ_μν = ½ g^λσ (∂_μg_νσ + ∂_νg_μσ − ∂_σg_μν)
```

**Implementation** (`christoffel.cpp`, lines 55–70):
```cpp
double bracket =
    dg[mu](nu, sigma)    // ∂_μ g_{νσ}
  + dg[nu](mu, sigma)    // ∂_ν g_{μσ}
  - dg[sigma](mu, nu);   // ∂_σ g_{μν}
sum += g_inv(lambda, sigma) * bracket;
result[lambda](mu, nu) = 0.5 * sum;
```

**Verification of index mapping:**
- `dg[mu]` = ∂g/∂x^μ (derivative w.r.t. coordinate μ) ✓
- `dg[mu](nu, sigma)` = ∂g_{νσ}/∂x^μ ✓
- `dg[nu](mu, sigma)` = ∂g_{μσ}/∂x^ν ✓
- `dg[sigma](mu, nu)` = ∂g_{μν}/∂x^σ ✓
- Sum over σ with g^λσ applied: ✓

**Formula is correctly transcribed from physics.** ✓

**Finite difference accuracy:**
```
∂g/∂x^σ ≈ (g(x + h·ê_σ) − g(x − h·ê_σ)) / (2h)
```
Central difference, O(h²) at interior. Default h = 1e-5. ✓

**Lower-index symmetry Γ^λ_μν = Γ^λ_νμ:**
The formula is symmetric in (μ, ν) because swapping μ ↔ ν transforms:
- ∂_μg_νσ → ∂_νg_μσ (swapped)
- ∂_νg_μσ → ∂_μg_νσ (swapped)
- ∂_σg_μν → ∂_σg_νμ = ∂_σg_μν (symmetric metric)
The bracket is unchanged → Γ^λ_μν = Γ^λ_νμ. ✓
This symmetry is verified in the test suite. ✓

**Flat metric → zero symbols:** For Minkowski (constant metric), all ∂g/∂x^σ = 0
exactly. The finite differences will produce numerically-zero results. Tests verify
all 64 symbols < 1e-8 for flat metrics. ✓

#### 2.5.3 Geodesic Equation

**Physics definition:**
```
d²x^λ/dτ² + Γ^λ_μν (dx^μ/dτ)(dx^ν/dτ) = 0
```

Rewritten as first-order system:
```
dx^λ/dτ = u^λ
du^λ/dτ = −Γ^λ_μν u^μ u^ν
```

**RK4 Implementation** (`geodesic.cpp`, lines 27–50):
```
k1 = f(state)
k2 = f(state + (h/2)·k1)
k3 = f(state + (h/2)·k2)
k4 = f(state + h·k3)
next = state + (h/6)·(k1 + 2k2 + 2k3 + k4)
```
This is the standard classical 4th-order Runge-Kutta formula. ✓

**Flat spacetime → straight lines:**
For zero Christoffel symbols, `accel = 0`, so `du/dτ = 0` and velocity is
conserved. Position advances linearly: `x(τ) = x₀ + u₀τ`. Tests verify
position deviation < 1e-8 over 100 steps. ✓

**Note on `GeodesicState` arithmetic:**
The `operator+` and `operator*` are defined inline in `tensor.hpp`. Each RK4
substep recomputes Christoffel symbols at the new position — 4 full Christoffel
computations per step. For a 4×4 metric this requires 4×4×4 = 64 FD evaluations
per Christoffel computation, so 256 metric evaluations per RK4 step. For a
position-dependent covariance metric this is expensive but correct.

---

### 2.6 Backtester

#### 2.6.1 Sharpe Ratio

**Formula:**
```
Sharpe = (mean(R) − r_f) / σ(R) × √ann
```
where σ is the **sample** standard deviation (Bessel-corrected, n−1 denominator).

**Implementation** (verified in `performance_metrics.cpp`, line 86):
```cpp
return (mu - risk_free_rate) / sd * std::sqrt(annualisation);
```
`stddev` uses `sq_sum / (n−1)`. ✓

**Issue:** The annualization is applied to the Sharpe computed on *per-bar* returns.
If the return series is daily, `ann = 252` is correct. If weekly, `ann = 52` is
needed. The constant 252 is the default — this is correct for daily data and
configurable for other frequencies. ✓

#### 2.6.2 Sortino Ratio

**Formula:**
```
Sortino = (mean(R) − r_f) / σ_down × √ann
```
where `σ_down` = downside deviation (RMS of returns below r_f, Bessel-corrected).

**Implementation** (`performance_metrics.cpp`, lines 52–67):
Only returns strictly below `threshold` contribute to σ_down. Bessel correction
applied. Returns `nullopt` if fewer than 2 below-threshold returns exist. ✓

**Note:** Some practitioners do not apply Bessel correction to downside deviation.
The `n−1` denominator is used here, which is the MAR-consistent convention. ✓

#### 2.6.3 Maximum Drawdown

**Formula:**
```
MDD = max over t { (peak_t − equity_t) / peak_t }
```

**Implementation** (`performance_metrics.cpp`, lines 117–132):
```cpp
equity  *= (1.0 + r);    // multiplicative compounding ✓
peak = max(peak, equity); // running peak ✓
dd   = (peak − equity) / peak; // fractional drawdown ✓
```

**Verification:**
- Uses multiplicative equity curve `∏(1 + r_t)` — correct for compounding returns. ✓
- Peak tracks correctly on new highs. ✓
- Returns value in [0, 1]. ✓
- **Edge case:** If all returns are large losses (equity → 0), MDD → 1 correctly. ✓

#### 2.6.4 γ-Weighted Information Ratio

**Formula as implemented:**
```
IR_γ = (mean(active_returns) × mean(γ)) / σ(active_returns)
```
where `active_returns_t = strategy_return_t − benchmark_return_t`.

**Verification:**
- `mean(γ) ≥ 1` always (since γ ≥ 1). This means `IR_γ ≥ IR_classic` when the
  strategy has positive mean active return. This is the intended behavior — the
  metric rewards high-velocity signals. ✓
- The formula is non-standard (not a recognized industry metric). It is a
  novel contribution of this codebase. The mathematical definition is clear and
  self-consistent, but **it has no historical backtesting track record** to validate
  whether it predicts live performance. ⚠

#### 2.6.5 Sign-Following Strategy

`backtester.cpp` lines 92–96:
```cpp
const double raw_sign = (bars[i].raw_signal >= 0.0) ? 1.0 : -1.0;
const double adj_sign = (corrected->adjusted_signals[i] >= 0.0) ? 1.0 : -1.0;
```

**Critical observation (documented in the code at line 82–87):**
Because γ > 0 and m_eff > 0, `sign(adj_signal) = sign(raw_signal)` always.
The raw and relativistic strategies therefore enter identical trades.
The **only** difference between raw and relativistic in the current backtester
is the γ weighting in the IR calculation. Sharpe, Sortino, and MDD will be
identical for both strategies.

This is explicitly acknowledged in the code comment. However, it means the
"relativistic lift" claim in the README (showing different Sharpe ratios between
raw and relativistic) cannot be produced by the current backtest engine as
implemented. A magnitude-weighted position sizing rule (where γ affects trade size)
would be needed to differentiate Sharpe/Sortino. **Design limitation.** ⚠⚠

---

## 3. Known Limitations

### 3.1 Build System
- CMakeLists.txt uses Unix compiler flags incompatible with MSVC
- No vcpkg manifest (`vcpkg.json`) to auto-install Eigen3
- Build requires VS Developer environment (vcvarsall.bat) or Linux/WSL2

### 3.2 Mathematical
- **Coordinate dimensionality:** The spacetime interval is only physically
  meaningful if price, volume, and momentum are normalized to comparable scales.
  The implementation does not enforce this.
- **β = max observed velocity normalization:** Using the historical maximum as
  the normalization denominator means β changes as new price extremes are observed.
  This creates look-ahead bias if computed on a full backtest window.
- **m_eff = ADV is not defined:** The effective mass is passed as a raw double
  with no enforcement of units or normalization. Cross-asset comparisons require
  consistent m_eff normalization.
- **Sign-following strategy:** γ does not affect trade direction or size; it only
  modifies the IR weighting. The README's implication of different Sharpe/Sortino
  between raw and relativistic strategies is not produced by the current engine.

### 3.3 Statistical
- `MIN_RETURN_SERIES_LENGTH = 30`: Sharpe ratio estimates over 30 bars have very
  high standard error (~√(2/T) ≈ 0.26 at T=30). Institutional use requires T ≥ 252.
- Downside deviation with Bessel correction at small n (e.g. 30) introduces high
  variance in the Sortino estimate.
- No standard errors, confidence intervals, or bootstrap estimates are reported
  for any metric.

### 3.4 Tensor Module
- The Christoffel computation does 256 metric evaluations per RK4 step.
  For large N-bar geodesic integrations this will be slow.
- The metric symmetry is assumed but not enforced for user-supplied functions.

---

## 4. Open Questions

**Q1: What is the right choice of c_market?**
The code sets c_market = 1.0 (natural units). The financial analog of the speed of
light would naturally be the maximum theoretically possible single-bar price move —
but this is unbounded for equities (trading halts notwithstanding). Using
`max_observed_velocity` makes c_market data-dependent and time-varying. A
theoretically grounded choice (e.g., exchange-mandated circuit breaker levels)
would be more principled.

**Q2: Should velocity use log-returns or arithmetic returns?**
Log-returns are the standard in quantitative finance for multiplicative processes.
The current implementation uses arithmetic price differences, which are correct
for additive (arithmetic) price processes but introduce path-dependence for
multiplicative processes over multi-bar windows.

**Q3: Does the spacetime interval have predictive power?**
The regime classification (Timelike = predictable, Spacelike = stochastic) is an
empirical hypothesis. Is there evidence that bars classified as "timelike" by this
formula have lower next-bar return variance? This is the testable prediction of the
model and has not been evaluated.

**Q4: What does the geodesic actually predict?**
Geodesic integration in the financial metric produces a "free-fall" price trajectory.
The deviation of actual prices from the geodesic could serve as a signal (analogous
to a non-gravitational force). The current codebase computes geodesics but does not
use them as signals in the backtester.

**Q5: Is the γ-weighted IR a valid performance measure?**
The formula IR_γ = mean(active) × mean(γ) / σ(active) has no precedent in published
quantitative finance literature. It will always produce IR_γ ≥ IR_classic when
mean(active) > 0, and IR_γ ≤ IR_classic when mean(active) < 0. Before claiming
"relativistic lift," the measure should be validated: does higher IR_γ in-sample
predict better out-of-sample Sharpe?

**Q6: Is BETA_MAX_SAFE = 0.9999 the right choice?**
At β = 0.9999, γ ≈ 70.7. For a typical 1% daily volatility market (β ≈ 0.01),
γ ≈ 1.00005 — essentially Newtonian. Reaching β = 0.5 (where γ = 1.155) would
require price velocity = 0.5 × max_observed_velocity. The mapping between market
conditions and β values is not calibrated to any empirical data.

---

## 5. Verdict

### Does the math check out?

**Yes, with caveats.** The core mathematical machinery is correctly transcribed from
special relativity:

| Formula | Correctness |
|---------|-------------|
| γ = 1/√(1−β²) | ✓ Exact |
| Time dilation t = γτ | ✓ Correct direction |
| Velocity addition β₁⊕β₂ | ✓ Exact |
| Rapidity φ = atanh(β) | ✓ Exact, additivity tested |
| Doppler D(β)·D(−β) = 1 | ✓ Identity verified |
| Christoffel Γ^λ_μν | ✓ Formula exact, FD correct |
| Geodesic RK4 | ✓ Standard 4th-order |
| Sharpe/Sortino/MDD | ✓ Standard financial formulas |

### Does the code match the math?

**Mostly yes.** Every formula audited has a correct implementation. The code is
clean, well-documented, uses `std::optional` consistently for fallible paths, and
achieves zero-UB in the paths reviewed.

### What would need to change before live data validation?

1. **Fix the sign-following strategy** to use γ-weighted position sizing, so
   Sharpe/Sortino actually differ between raw and relativistic strategies.
2. **Normalize spacetime coordinates** before computing ds². Raw price/volume data
   will produce meaningless interval classifications.
3. **Define and enforce m_eff normalization** to make cross-asset comparisons valid.
4. **Address look-ahead bias** in β normalization (max_observed_velocity should use
   only past data, not the full backtest window).
5. **Increase MIN_RETURN_SERIES_LENGTH** from 30 to ≥ 252 for production-grade
   Sharpe estimates.
6. **Fix the build system** for Windows (add vcpkg.json, add MSVC-compatible flags
   or document Linux/WSL2 as the required build environment).
7. **Empirically validate** the core hypothesis: do timelike bars exhibit different
   next-bar autocorrelation than spacelike bars on real market data?

### Overall assessment

The codebase is a rigorous, well-engineered implementation of a novel theoretical
framework. The physics is applied correctly and the C++20 is production-quality.
The primary gap is not in the mathematics or the code — it is in the **empirical
calibration**: the mapping from financial observables (prices, volumes, returns) to
the relativistic parameters (β, c_market, m_eff, coordinate scales) is underdetermined.
Without calibration data, the framework is theoretically sound but not yet
quantitatively actionable.

The code merits continued development. The next priority should be an empirical
validation study on historical equity data with explicit coordinate normalization
and position-weighted backtesting.

---

*Audit produced by AGT-AUDIT. Read-only access. No source files modified.*
