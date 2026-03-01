# CLAUDE.md — Behavioral Instructions for SRFM Agents

## Project Identity

**Special Relativity in Financial Modeling (SRFM)**
C++20 library applying Lorentz transforms, spacetime manifolds, and tensor calculus
to financial signal processing, momentum modeling, and relativistic backtesting.

This is real mathematics applied to real financial data.
Every claim in the codebase must be mathematically grounded.
No placeholder physics. No metaphorical relativity. Actual transforms.

---

## Your Role

You are one of 6 specialized agents. You own exactly one module (see AGENT.md).
You ship production-quality C++20 with a 1.5:1 test-to-production ratio.
You do not touch files outside your ownership zone.

---

## C++ Standards

- **Standard:** C++20
- **Compiler:** clang++ or g++ with `-std=c++20 -Wall -Wextra -Werror`
- **Zero warnings policy:** code must compile clean
- **No raw pointers:** use `std::unique_ptr`, `std::shared_ptr`, or references
- **No undefined behavior:** run with `-fsanitize=address,undefined` in test builds
- **Eigen3** for matrix/tensor operations
- **Google Test** for all tests
- **{fmt}** for formatting output
- **nlohmann/json** for data ingestion

---

## Mathematical Requirements

### Lorentz Transform (AGT-01)
The core velocity parameter β = v/c must be mapped to a financial analog.
In SRFM, β represents normalized market velocity:

```
β_market = (price_velocity) / (max_observed_velocity)
```

Where price_velocity = dP/dt over a rolling window.
Lorentz factor: γ = 1 / sqrt(1 - β²)

γ dilates the time-weight of financial signals — fast markets compress signal time.
This is not metaphor. γ must be computed and applied to indicator weights.

### Spacetime Interval (AGT-02)
Financial spacetime interval:
```
ds² = -c²dt² + dP² + dV² + dM²
```
Where dP = price displacement, dV = volume displacement, dM = momentum displacement.
Timelike intervals (ds² < 0): market is in causal regime — past predicts future.
Spacelike intervals (ds² > 0): market is in stochastic regime — signal decorrelated.

### Momentum-Energy Relation (AGT-03)
Relativistic momentum analog:
```
p_rel = γ * m_eff * v_market
```
Where m_eff = effective market mass (liquidity proxy, e.g. ADV).
Rest energy analog: E₀ = m_eff * c²_market (baseline volatility × liquidity).

### Tensor Covariance (AGT-04)
Metric tensor g_μν for the financial manifold:
Multi-asset covariance modeled as a 4×4 metric tensor.
Christoffel symbols Γ^λ_μν capture curvature = rate of change of correlations.
Geodesic equation describes "natural" price path in curved market space.

### Backtester (AGT-05)
All signals fed through relativistic corrections before strategy evaluation.
Benchmark: compare relativistic-adjusted signals vs. raw signals.
Report: Sharpe, Sortino, max drawdown, and γ-weighted information ratio.

---

## Code Style

```cpp
// Good: explicit types, clear names, documented math
[[nodiscard]] double computeLorentzFactor(double beta) noexcept {
    // γ = 1 / sqrt(1 - β²), undefined for |β| >= 1
    assert(std::abs(beta) < 1.0);
    return 1.0 / std::sqrt(1.0 - beta * beta);
}

// Bad: magic numbers, no documentation, implicit conversions
auto lf(auto b) { return 1/sqrt(1-b*b); }
```

- Every public function has a doc comment explaining the math
- Every non-obvious constant is named and justified
- No `using namespace std;`
- Structs over classes for pure data
- Classes for objects with behavior

---

## Test Requirements

**Framework:** Google Test
**Ratio:** 1.5 lines of test per 1 line of production code (measured by `cloc`)

Test categories required per module:
1. **Unit tests** — every public function, nominal + edge cases
2. **Boundary tests** — β → 0 (Newtonian limit), β → 1 (relativistic limit)
3. **Mathematical identity tests** — verify transforms compose correctly
4. **Precision tests** — numerical stability at extreme inputs
5. **Performance tests** — latency benchmarks for hot paths

Example boundary test:
```cpp
TEST(LorentzTransform, NewtonianLimit) {
    // At low velocity, γ → 1 (classical physics recovered)
    LorentzTransform lt;
    EXPECT_NEAR(lt.gamma(0.001), 1.0, 1e-6);
}

TEST(LorentzTransform, RelativisticLimit) {
    // At β → 1, γ → ∞
    LorentzTransform lt;
    EXPECT_GT(lt.gamma(0.9999), 70.0);
}
```

---

## Ship Sequence (Repeat for Every Function)

1. Write the header declaration
2. Write the test (failing)
3. Write the implementation (test passes)
4. Verify ratio with `cloc src/ tests/`
5. Compile with `-Wall -Wextra -Werror`
6. Log to PROGRESS.md

---

## PROGRESS.md Format

Append one line per completed step:
```
[AGT-01][2026-02-28T06:00Z] SCAFFOLD complete — lorentz/ created, types declared
[AGT-01][2026-02-28T06:30Z] IMPL complete — lorentz_transform.cpp 87 LOC, tests 134 LOC (1.54:1)
```

---

## What This Project Is

This is a research-grade C++ library designed to attract institutional interest.
The code will be read by ML Quants at top funds.
Every line is a signal about the quality of thinking behind it.
Ship accordingly.
