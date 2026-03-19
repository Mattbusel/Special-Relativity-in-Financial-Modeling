# Multi-Asset Manifold

This document describes the N-asset extension of SRFM's spacetime geometry,
covering the public API, construction patterns, and financial interpretation of
TIMELIKE, SPACELIKE, and LIGHTLIKE intervals.

---

## Overview

The single-asset engine models one price series as a 1+1 dimensional spacetime
(time + one price axis). The N-asset manifold generalises this to an arbitrary
number of assets: N asset price axes plus one time axis, giving a
(N+1)-dimensional pseudo-Riemannian manifold with Lorentzian signature
(-,+,+,...,+).

The time-time component of the metric encodes the "market speed of light"
`c_market`, a parameter that controls how quickly causal information can
propagate between events. The spatial block is given by the N×N asset
covariance matrix Σ, so correlated assets naturally pull nearby in the geometry.

The core formula for the squared spacetime interval between two events is:

```
ds² = g_μν Δx^μ Δx^ν
    = -c_market² (Δt)² + Σ_ij Δp^i Δp^j
```

where Δt is the time separation and Δp^i are the price-coordinate separations.

---

## Key Classes

### `srfm::tensor::NAssetManifold`

Header: `include/srfm/tensor/n_asset_manifold.hpp`

Represents the (N+1)-dimensional flat Lorentzian manifold. Stores the metric
tensor g and its inverse g^{-1} as pre-built constant matrices.

| Method | Description |
|---|---|
| `NAssetManifold::make(n, cov, c_market)` | Validated factory; returns `std::nullopt` if inputs are invalid. |
| `dim()` | Returns N+1 (total manifold dimension). |
| `n_assets()` | Returns N (number of spatial/price axes). |
| `c_market()` | Returns the market speed-of-light parameter. |
| `metric_at(x)` | Returns the (N+1)×(N+1) metric matrix (constant; x is ignored for flat manifolds). |
| `inverse_metric_at(x)` | Returns g^{-1}, computed via LU decomposition. |
| `line_element_sq(x, dx)` | Computes ds² = dx^T g dx directly. |
| `covariance()` | Returns the N×N covariance matrix used to build the metric. |
| `is_flat()` | Always returns `true` for this class (constant metric). |
| `reduces_to_4d(other)` | Returns `true` when `other.n_assets() == 3` and `this->n_assets() >= 3`. |

Validation performed by `make()`:
- `n >= 1`
- covariance matrix is N×N, symmetric, and positive-definite
- `c_market > 0`

### `srfm::manifold::NAssetEvent`

Header: `include/srfm/manifold/n_asset_interval.hpp`

A spacetime event: a timestamp paired with N asset prices.

| Field / Method | Description |
|---|---|
| `t` | Time coordinate (double). |
| `prices` | Eigen::VectorXd of length N. |
| `NAssetEvent::make(t, prices)` | Validated factory; returns `std::nullopt` if prices is empty. |
| `to_coords()` | Returns a single (N+1)-vector `[t, p_0, p_1, ..., p_{N-1}]`. |

### `srfm::manifold::NAssetInterval`

Header: `include/srfm/manifold/n_asset_interval.hpp`

Stateless calculator. All methods are `const` and `noexcept`.

| Method | Description |
|---|---|
| `compute(a, b, manifold)` | Computes ds² between events a and b; returns `std::nullopt` on dimension mismatch. |
| `batch_from_reference(ref, events, manifold)` | Computes intervals from one reference event to many target events. |

### `srfm::manifold::IntervalResult`

Returned by `NAssetInterval::compute()` and `batch_from_reference()`.

| Field | Type | Description |
|---|---|---|
| `ds_sq` | double | The raw squared interval value. |
| `type` | `IntervalType` | `TIMELIKE`, `SPACELIKE`, or `LIGHTLIKE`. |
| `magnitude` | double | `sqrt(|ds_sq|)`, always non-negative. |

### `srfm::manifold::IntervalType`

```cpp
enum class IntervalType {
    TIMELIKE,   // ds² < 0
    SPACELIKE,  // ds² > 0
    LIGHTLIKE,  // |ds²| < 1e-10
};
```

---

## Constructing an NAssetManifold

```cpp
#include "srfm/tensor/n_asset_manifold.hpp"
#include <Eigen/Dense>

using srfm::tensor::NAssetManifold;

// Two uncorrelated assets with annualised volatilities σ1=20%, σ2=30%.
// Covariance matrix entries are variance = σ².
Eigen::MatrixXd cov(2, 2);
cov << 0.04, 0.00,
       0.00, 0.09;

// Market speed of light = 1.0 (default).
auto manifold_opt = NAssetManifold::make(2, cov, /*c_market=*/1.0);
if (!manifold_opt) {
    // inputs failed validation — handle error
}
NAssetManifold& manifold = *manifold_opt;

// Inspect dimensions.
int dim     = manifold.dim();      // 3  (N+1)
int n       = manifold.n_assets(); // 2
double c    = manifold.c_market(); // 1.0

// Inspect the metric tensor.
// g = [[-1, 0, 0], [0, 0.04, 0], [0, 0, 0.09]]
const Eigen::MatrixXd& g = manifold.metric();
```

### With asset correlation

```cpp
// Two assets with σ1=20%, σ2=30%, correlation ρ=0.6.
double s1 = 0.20, s2 = 0.30, rho = 0.60;
Eigen::MatrixXd cov(2, 2);
cov(0, 0) = s1 * s1;
cov(0, 1) = rho * s1 * s2;
cov(1, 0) = rho * s1 * s2;
cov(1, 1) = s2 * s2;

auto manifold_opt = NAssetManifold::make(2, cov, 1.0);
```

---

## Computing Spacetime Intervals

```cpp
#include "srfm/manifold/n_asset_interval.hpp"

using srfm::manifold::NAssetEvent;
using srfm::manifold::NAssetInterval;
using srfm::manifold::IntervalType;

// Build events.
// Event A: t=0, prices=[100, 200]
auto ev_a = NAssetEvent::make(0.0, Eigen::Vector2d(100.0, 200.0));

// Event B: t=1 (one bar later), prices=[101, 202]
auto ev_b = NAssetEvent::make(1.0, Eigen::Vector2d(101.0, 202.0));

// Compute the interval.
NAssetInterval calc;
auto result_opt = calc.compute(*ev_a, *ev_b, manifold);
if (!result_opt) { /* dimension mismatch */ }

auto& r = *result_opt;
double ds2       = r.ds_sq;     // e.g. -0.84
double magnitude = r.magnitude; // sqrt(|ds2|)

switch (r.type) {
    case IntervalType::TIMELIKE:   /* ... */ break;
    case IntervalType::SPACELIKE:  /* ... */ break;
    case IntervalType::LIGHTLIKE:  /* ... */ break;
}
```

### Batch computation from a reference event

```cpp
std::vector<NAssetEvent> history; // populate with past events
auto results_opt = calc.batch_from_reference(*ev_a, history, manifold);
// results_opt contains one IntervalResult per element of history.
```

---

## Interpreting TIMELIKE, SPACELIKE, LIGHTLIKE for N Assets

### Classification rule

| Condition | Classification | Meaning |
|---|---|---|
| `ds² < 0` | TIMELIKE | Price changes are small relative to elapsed time. |
| `ds² > 0` | SPACELIKE | Price changes are large relative to elapsed time. |
| `|ds²| < 1e-10` | LIGHTLIKE | Event pair lies on the market light cone. |

### Financial interpretation

**TIMELIKE** (`ds² < 0`): The market is in a low-volatility or trend-persistent
regime. The time separation dominates over the spatial (price) separation.
Causally connected events: information from event A can reach event B at or
below the market speed of light. Signals in this regime tend to exhibit
momentum (the relativistic momentum correction γ > 1 amplifies the signal).

**SPACELIKE** (`ds² > 0`): The market has moved further in price-space than
would be consistent with causal propagation at `c_market`. This indicates a
high-volatility or mean-reverting regime. Events are causally disconnected:
no signal travelling at `c_market` can link them. In a multi-asset context, a
large off-diagonal jump in one asset while another is flat produces a spacelike
separation even if the time step is non-zero.

**LIGHTLIKE** (`|ds²| < 1e-10`): The event pair lies exactly on the light
cone. This is a boundary case — a transitional regime between momentum and
mean-reversion. In practice this appears at near-zero price moves or
specifically chosen (Δt, Δp) pairs satisfying `c_market² Δt² = Σ_ij Δp^i Δp^j`.

### Effect of the covariance matrix

In the N-asset case, the spatial block is `Σ_ij Δp^i Δp^j`, so correlated
assets contribute more to the spatial separation than uncorrelated ones for the
same raw price displacements. Two highly correlated assets moving together
produce a larger spacelike contribution than two uncorrelated assets with
identical individual moves.

### Effect of c_market

`c_market` scales the time-time component as `g_00 = -c_market²`. A larger
`c_market` makes the timelike region wider: price moves that were previously
spacelike become timelike. Calibrating `c_market` to the typical price
velocity of the assets under study is the primary tuning lever.

---

## Analytic Example (N=2, diagonal covariance)

Given:
- `c_market = 1`, `σ_1² = 0.04`, `σ_2² = 0.09` (uncorrelated)
- Event A: (t=0, p_1=0, p_2=0)
- Event B: (t=1, p_1=2, p_2=1)

```
ds² = -1*(1)² + 0.04*(2)² + 0.09*(1)²
    = -1 + 0.16 + 0.09
    = -0.75   → TIMELIKE
```

The negative value confirms the time separation dominates: the two-asset price
move of (2, 1) in one time unit is not enough to overcome the causal metric.

---

## N-Asset Geodesic and Christoffel Symbols

For the curved (non-flat) generalisation, see:
- `include/srfm/manifold/geodesic_n.hpp` — N-dimensional geodesic solver.
- `include/srfm/manifold/christoffel_n.hpp` — N-dimensional Christoffel symbols.

These are built on top of `NAssetManifold` and accept the same `NAssetEvent`
types.

---

## Test Coverage

The N-asset manifold is covered by three test binaries in `tests/n_asset/`:

| File | What it tests |
|---|---|
| `test_n_asset_manifold.cpp` | `NAssetManifold` construction, metric structure, inverse, line element, c_market scaling, N sweeps up to N=100. |
| `test_n_asset_interval.cpp` | `NAssetEvent`, `NAssetInterval::compute`, `batch_from_reference`, classification, analytic values for N=1,2,4. |
| `test_n_asset_engine.cpp` | End-to-end pipeline using the N-asset manifold. |
