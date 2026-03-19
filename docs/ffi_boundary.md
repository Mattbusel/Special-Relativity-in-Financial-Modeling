# FFI Boundary Documentation

This document describes the boundary between the C++ SRFM signal-processing
library and the Rust `tokio-prompt-orchestrator` crate.

---

## Architecture overview

The project consists of two distinct language layers that currently operate
**independently** rather than through a live FFI call boundary:

| Layer | Language | Role |
|-------|----------|------|
| Signal-processing core | C++ (ISO C++20) | OHLCV data loading, relativistic transformations, backtesting |
| Orchestration runtime | Rust (`tokio-prompt-orchestrator`) | Async LLM worker coordination, circuit breaking, cost tracking, TUI/Web API |

The Rust crate (`src/lib.rs`, `src/main.rs`) does **not** link directly against
the C++ shared library at runtime.  Instead the two layers communicate through:

- **File-based hand-off**: the C++ CLI (`srfm --backtest <csv>`) writes
  results to stdout/files; the Rust orchestrator can shell out to that binary
  via `std::process::Command`.
- **Shared data schemas**: both sides read/write the same OHLCV CSV format
  (`timestamp,open,high,low,close,volume`).

If a future integration adds a direct C-callable ABI, it should follow the
conventions documented in the sections below.

---

## C++ functions available for FFI export

The following C++ functions are candidates for `extern "C"` export should a
direct linkage layer be introduced.  They are currently callable only from
C++ code (tests, CLI, benchmarks).

### `srfm::core::DataLoader`

```cpp
// Load OHLCV bars from a CSV file.
// Returns nullopt if the file cannot be opened.
static std::optional<std::vector<OHLCV>>
    DataLoader::load_csv(const std::string& filepath) noexcept;

// Parse OHLCV bars from an in-memory CSV string.
// Never returns nullopt; returns empty vector on bad input.
static std::vector<OHLCV>
    DataLoader::parse_csv_string(const std::string& csv_content) noexcept;
```

### `srfm::core::Engine`

```cpp
// Run a complete batch backtest over a span of OHLCV bars.
// Returns nullopt when fewer than MIN_RETURN_SERIES_LENGTH bars are supplied
// or any downstream calculation is numerically degenerate.
std::optional<backtest::BacktestComparison>
    Engine::run_backtest(std::span<const OHLCV> bars) const noexcept;

// Process one bar in streaming mode.
// Returns nullopt during the warm-up phase (fewer than 2 bars seen).
std::optional<PipelineBar>
    Engine::process_stream_bar(const OHLCV& bar) noexcept;
```

### `srfm::backtest::PerformanceCalculator`

```cpp
// Annualised Sharpe ratio.  Returns nullopt when σ = 0 or input is degenerate.
static std::optional<double>
    PerformanceCalculator::sharpe(std::span<const double> returns,
                                  double risk_free_rate,
                                  double annualisation) noexcept;

// Annualised Sortino ratio.  Returns nullopt when downside vol = 0.
static std::optional<double>
    PerformanceCalculator::sortino(std::span<const double> returns,
                                   double risk_free_rate,
                                   double annualisation) noexcept;

// Maximum drawdown in [0, 1].  Returns nullopt on empty input.
static std::optional<double>
    PerformanceCalculator::max_drawdown(std::span<const double> returns) noexcept;
```

### `srfm::backtest::LorentzSignalAdjuster`

```cpp
// Apply γ-weighted Lorentz corrections to a bar series.
// Returns nullopt when bars is empty or effective_mass <= 0.
std::optional<LorentzCorrectedSeries>
    LorentzSignalAdjuster::adjust(std::span<const BarData> bars) const noexcept;

// Compute γ for a single β value.
// Returns nullopt for |β| >= BETA_MAX_SAFE or non-finite β.
static std::optional<double>
    LorentzSignalAdjuster::lorentz_gamma(BetaVelocity beta) noexcept;
```

---

## Error type mapping: C++ `std::nullopt` → Rust `None`/`Err`

All C++ fallible functions in the SRFM library use `std::optional<T>` as
their error channel — they never throw exceptions and never return raw
error codes.  The mapping to Rust idioms is:

| C++ return type | Rust equivalent | Notes |
|----------------|-----------------|-------|
| `std::optional<T>` with value | `Ok(T)` / `Some(T)` | Successful computation |
| `std::optional<T>` as `nullopt` | `Err(SrfmError)` / `None` | Degenerate input or numeric failure |
| `std::vector<T>` (possibly empty) | `Vec<T>` | Empty vec is a valid "no results" signal, not an error |
| `bool` return | `bool` | Used for validation predicates only |

When wrapping C++ `optional`-returning functions via an `extern "C"` shim,
the recommended pattern is to use a two-field C struct:

```c
// C shim pattern for an optional double
typedef struct {
    int   has_value;   // 1 if valid, 0 if nullopt
    double value;
} SrfmOptDouble;
```

On the Rust side this maps to:

```rust
#[repr(C)]
pub struct SrfmOptDouble {
    pub has_value: i32,
    pub value: f64,
}

impl From<SrfmOptDouble> for Option<f64> {
    fn from(s: SrfmOptDouble) -> Self {
        if s.has_value != 0 { Some(s.value) } else { None }
    }
}
```

---

## Panic safety conventions

### C++ side

- **No exceptions**: all public SRFM functions are marked `noexcept`.
  Internal assertions use `assert()` in debug builds only.
- **No UB on bad input**: invalid inputs (NaN, Inf, empty spans) produce
  `nullopt` or an empty collection — they never invoke undefined behaviour.
- **No raw pointers across the boundary**: if an FFI shim is added, all
  buffers must be passed as `(ptr, len)` pairs or as opaque handle types;
  raw C++ object pointers must never cross the boundary.

### Rust side

- **Unwrap policy**: `unwrap()` / `expect()` are forbidden in library code
  (`src/lib.rs` and all public modules).  The `main.rs` entry point may use
  `expect` on truly-unrecoverable startup errors (e.g., Tokio runtime
  creation).
- **FFI callbacks**: any Rust closure passed as a function pointer to C++
  must be wrapped in `catch_unwind` to convert panics to error codes before
  returning across the FFI boundary; a Rust panic unwinding through C++
  frames is undefined behaviour.
- **`#[no_mangle] extern "C"` functions**: every such function in a
  hypothetical Rust→C++ export must begin with a `std::panic::catch_unwind`
  block and return a sentinel error value (e.g., `SrfmOptDouble{0, 0.0}`)
  on panic.

---

## Memory ownership conventions

### Allocations that stay on one side

- C++ `std::vector`, `std::string`, and `std::optional` objects must never
  be freed from Rust, and vice versa for Rust heap allocations.
- If a C++ function returns a heap-allocated buffer via an `extern "C"` shim,
  the shim must also expose a corresponding `srfm_free_*` function that the
  Rust caller must invoke.  Rust must not call `dealloc` directly on
  C++-allocated memory.

### Passing data across the boundary

- **Read-only slices**: pass as `(const T* ptr, size_t len)` from C++ to
  Rust and as `*const T` + `usize` from Rust.  The callee must not store
  the pointer beyond the call duration.
- **Owned buffers returned to Rust**: allocate with a C-compatible allocator
  (`malloc`/`free`) and document that the Rust caller owns the memory and
  must free it via the provided `srfm_free_*` function.
- **String data**: use null-terminated `const char*` for fixed-lifetime
  string results (e.g., error messages), or a `(char* ptr, size_t len)`
  pair for owned strings.  Rust should convert to `CStr` / `String`
  immediately and release using the paired free function.

### Current state (no live FFI)

Because the Rust and C++ layers are not yet linked at runtime, these
conventions are forward-looking design guidelines.  The C++ CLI binary
(`srfm`) is spawned as a subprocess by the Rust orchestrator when needed;
stdout/stdin is the only actual IPC mechanism today.
