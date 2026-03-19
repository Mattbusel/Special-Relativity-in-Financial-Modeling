# Troubleshooting Guide — SRFM

This guide covers the most common problems encountered when building, running,
or using SRFM. Each section describes the symptom, the root cause, and the
recommended fix.

---

## Build Failures

### Missing vcpkg / `fmt/core.h: No such file or directory`

**Symptom:** CMake configure or compile step fails with a message like:
```
fatal error: fmt/core.h: No such file or directory
```

**Cause:** The `fmt` library is not installed, or CMake cannot find vcpkg.

**Fix:**
1. Clone vcpkg and bootstrap it:
   ```bash
   git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
   ~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
   ```
2. Install the required packages:
   ```bash
   ~/vcpkg/vcpkg install fmt gtest rapidcheck
   ```
3. Pass the toolchain file to every CMake invocation:
   ```bash
   cmake -S . -B build \
     -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
   ```
4. Set `VCPKG_ROOT` in your shell profile so the variable is always available:
   ```bash
   export VCPKG_ROOT=~/vcpkg
   ```

---

### CMake configure errors (`CMake Error at CMakeLists.txt`)

**Symptom:** CMake exits with an error during the configure step, such as:
```
CMake Error: could not find CMAKE_TOOLCHAIN_FILE
```
or
```
CMake Error: Generator "Ninja" not found
```

**Fix — toolchain file not found:**
Verify `VCPKG_ROOT` is set and the path contains `scripts/buildsystems/vcpkg.cmake`.

**Fix — Ninja not installed:**
Install Ninja, or remove `-G Ninja` and let CMake pick the default generator:
```bash
# Linux / macOS
sudo apt-get install ninja-build   # Debian / Ubuntu
brew install ninja                  # macOS

# Or omit the generator flag
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

**Fix — CMake version too old:**
SRFM requires CMake 3.25 or newer. Check with `cmake --version` and upgrade
from https://cmake.org/download/.

---

### Rust toolchain issues

**Symptom:** `cargo build` or `cargo check` fails with:
```
error[E0554]: #![feature(...)] may not be used on the stable release channel
```
or a missing feature flag error.

**Fix:**
1. Install the stable Rust toolchain via rustup:
   ```bash
   rustup update stable
   rustup default stable
   ```
2. If the project requires a specific Minimum Supported Rust Version (MSRV),
   check `Cargo.toml` for the `rust-version` field and install that version:
   ```bash
   rustup install 1.XX.0
   rustup default 1.XX.0
   ```
3. Ensure the `tui` feature compiles correctly by passing `--all-features`:
   ```bash
   cargo check --all-features
   cargo test --all-features
   ```

---

### C++ standard not met (`error: expected` / `requires C++20`)

**Symptom:** Compiler emits errors about missing language features (concepts,
`std::span`, coroutines, etc.).

**Fix:**
SRFM requires C++20. GCC 12+, Clang 16+, or MSVC 2022 are the minimum
supported compilers. Pass `-DCMAKE_CXX_STANDARD=20` explicitly if your
toolchain defaults to an earlier standard:
```bash
cmake -S . -B build -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

---

## Runtime Errors

### WebSocket connection refused

**Symptom:** The streaming CLI mode (`--stream`) or the TUI live-metrics mode
exits immediately with:
```
HTTP error: error sending request for url (http://localhost:9090/metrics)
```
or a connection refused / timeout error.

**Cause:** The orchestrator's Prometheus metrics server is not running, or it
is listening on a different address or port.

**Fix:**
1. Confirm the orchestrator process is running.
2. Check which address the metrics endpoint is bound to:
   ```bash
   curl http://localhost:9090/metrics
   ```
3. If the port or host differs, pass the correct URL when invoking the TUI in
   live mode. The default URL is `http://localhost:9090/metrics`.
4. Check firewall rules if the server is on a remote host.

---

### C++ shared library not found at runtime

**Symptom:** The `srfm` binary starts but immediately crashes with:
```
error while loading shared libraries: libsrfm_engine.so: cannot open shared object file
```

**Cause:** The installed shared library is not on the dynamic linker search
path.

**Fix:**
1. After `cmake --install build --prefix /usr/local`, update the linker cache:
   ```bash
   sudo ldconfig
   ```
2. Or set `LD_LIBRARY_PATH` temporarily:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   ./srfm --help
   ```
3. On macOS, use `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH`.

---

## Performance Issues

### Slow gamma computation

**Symptom:** Batch beta/gamma computation is slower than expected, especially
on large datasets.

**Cause:** SIMD dispatch may have fallen back to the scalar kernel because the
CPU does not support AVX2 or AVX-512, or because the library was not compiled
with the appropriate ISA flags.

**Fix:**
1. Check which SIMD level was selected at runtime by enabling debug logging.
2. To force a specific level for benchmarking, define the compile-time flag:
   ```bash
   cmake -S . -B build -DSRFM_FORCE_SCALAR=ON ...
   cmake -S . -B build -DSRFM_FORCE_AVX2=ON ...
   ```
3. If AVX-512 is available on your CPU but not being used, confirm that the
   compiler supports it:
   ```bash
   g++ -march=native -dM -E - < /dev/null | grep AVX512
   ```

### Disabling SIMD

If SIMD causes incorrect results (e.g., after cross-compiling to a target that
does not match the build host), disable it entirely:
```bash
cmake -S . -B build -DSRFM_DISABLE_SIMD=ON \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```
The scalar implementation in `src/simd/beta_scalar.cpp` is always correct;
SIMD kernels are only for throughput.

---

### GeodesicSolver is slow for large step counts

**Symptom:** `GeodesicSolver::solve()` with a large `steps` value causes
noticeable latency.

**Cause:** The RK4 integrator is O(steps) with four metric-tensor evaluations
per step.

**Fix:** Reduce `steps` (clamped internally to [1, 100000]). Typical financial
backtests use 50–500 steps. The accuracy gain beyond 1000 steps is negligible
for well-behaved market data.

---

## Common API Misuse

### Passing beta >= 1.0

**Symptom:** `BetaVelocity::make(beta)` returns `std::nullopt`, or the Lorentz
factor `gamma` becomes infinite or NaN.

**Cause:** Beta must satisfy `|beta| < BETA_MAX_SAFE` (0.9999). Values at or
above 1.0 are physically meaningless (superluminal) and trigger construction
failure.

**Fix:**
- Always check the return value of `::make()` before use.
- Clamp your computed beta before constructing `BetaVelocity`:
  ```cpp
  constexpr double BETA_MAX_SAFE = 0.9999;
  double safe_beta = std::clamp(raw_beta, -BETA_MAX_SAFE, BETA_MAX_SAFE);
  auto bv = BetaVelocity::make(safe_beta);
  ```
- In the Rust TUI layer, `MockMetrics` already clamps beta to `(0.01, 0.9999)`
  using `.clamp()` before computing gamma.

---

### Passing empty price arrays

**Symptom:** `BetaCalculator::fromPriceVelocityOnline()` or
`DataLoader::load_csv()` returns an empty result or `std::nullopt` when called
with an empty input.

**Cause:** These functions require at least two data points to compute a price
velocity (first difference). A single price or an empty array produces no
meaningful beta.

**Fix:**
- Guard the call site:
  ```cpp
  if (prices.size() < 2) {
      // Not enough data — handle gracefully
      return;
  }
  auto betas = BetaCalculator::fromPriceVelocityOnline(prices);
  ```
- Check the return type: all fallible operations return `std::optional`. Treat
  `std::nullopt` as "insufficient data", not as an error.

---

### NaN or Inf propagation

**Symptom:** Downstream metrics (Sharpe, Sortino, gamma-weighted IR) are NaN
or Inf.

**Cause:** A NaN or Inf leaked through a public API boundary.

**Fix:**
- Every public factory function validates inputs. If you are constructing types
  directly (bypassing factories), ensure all inputs are finite before doing so.
- Run with ASAN + UBSAN to catch undefined behaviour early:
  ```bash
  cmake -S . -B build-asan -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" ...
  cmake --build build-asan && ctest --test-dir build-asan
  ```

---

## TUI Display Issues

### Terminal size too small

**Symptom:** The TUI renders garbled output, overlapping widgets, or exits with
a message such as "Terminal too small".

**Cause:** The dashboard requires a minimum terminal size of 100 columns by 40
rows (`MIN_COLS = 60`, `MIN_ROWS = 20` are the hard limits in `app.rs`, but
the full layout needs more space).

**Fix:**
- Resize your terminal window to at least 120×40 before launching the TUI.
- On tmux or screen, check the active window size with `echo $COLUMNS $LINES`.
- If your terminal emulator does not support resize, switch to one that does
  (e.g., Alacritty, kitty, Windows Terminal).

---

### No color support / monochrome output

**Symptom:** All text appears in the default foreground color with no
highlighting.

**Cause:** The terminal does not advertise color support, or the `TERM`
variable is set to a non-color terminal type.

**Fix:**
1. Set `TERM` to a color-capable value:
   ```bash
   export TERM=xterm-256color
   ```
2. On Windows, use Windows Terminal or enable Virtual Terminal Processing in
   the console host.
3. If running over SSH, ensure the SSH client forwards `TERM` correctly:
   ```bash
   ssh -o "SendEnv TERM" user@host
   ```

---

### Unicode block characters missing (sparkline appears as question marks)

**Symptom:** Sparkline bars render as `?` or boxes instead of the Braille/block
characters.

**Cause:** The font in use does not contain the Unicode block elements used by
Ratatui's Sparkline widget (U+2581 to U+2588).

**Fix:**
- Use a font with full Unicode block element coverage, such as Nerd Fonts,
  Fira Code, or JetBrains Mono.
- On macOS, the default terminal font (Menlo) supports block elements. On
  Linux, install a Nerd Font and configure your terminal to use it.

---

## Web API Issues

### Port already in use

**Symptom:** The web API server fails to start with:
```
Error: Address already in use (os error 98)
```

**Cause:** Another process is already listening on the configured port
(default 8080).

**Fix:**
1. Find and stop the conflicting process:
   ```bash
   lsof -i :8080
   kill <PID>
   ```
2. Or configure the server to listen on a different port by passing the
   `--port` flag (if supported by your build) or setting the appropriate
   environment variable.

---

### CORS issues (browser requests blocked)

**Symptom:** API calls from the React/Vite dashboard (served from
`http://localhost:5173`) fail in the browser console with:
```
Access to fetch at 'http://localhost:8080/...' from origin 'http://localhost:5173'
has been blocked by CORS policy
```

**Cause:** The web API server does not include the required
`Access-Control-Allow-Origin` response header, or it does not respond to
`OPTIONS` preflight requests.

**Fix:**
1. Confirm the API server has CORS middleware enabled. Look for the CORS
   configuration in the server setup code.
2. For local development, set the allowed origins to `http://localhost:5173`
   or use the wildcard `*`.
3. If you are proxying the API through Vite's dev server (`vite.config.ts`),
   add a proxy rule to avoid cross-origin requests entirely:
   ```typescript
   // vite.config.ts
   export default {
     server: {
       proxy: {
         '/api': 'http://localhost:8080',
       },
     },
   };
   ```

---

## Still Stuck?

1. Run the full test suite to confirm your build is sound:
   ```bash
   cargo test --all-features
   ctest --test-dir build --output-on-failure
   ```
2. Re-read `BUILD.md` for the canonical build procedure.
3. Check `ARCHITECTURE.md` for module contracts — many runtime failures trace
   back to violating an API pre-condition documented there.
4. Open an issue on GitHub with:
   - Your OS, compiler version, and CMake version.
   - The exact command you ran.
   - The full error output.
