@echo off
REM ============================================================================
REM  scripts/build_simd.bat
REM  AGT-08 — Direct MSVC build for the SIMD acceleration layer
REM  (no CMake required; useful for rapid iteration or CI environments
REM   that have cl.exe but not a CMake installation)
REM
REM  Usage:
REM    scripts\build_simd.bat              — build + run tests
REM    scripts\build_simd.bat bench        — build + run benchmarks (requires
REM                                          Google Benchmark in %GBENCH_ROOT%)
REM
REM  Prerequisites:
REM    - Run from a Visual Studio Developer Command Prompt (x64)
REM      (or run vcvars64.bat first)
REM    - REPO environment variable set to the repository root, OR
REM      run the script from the repository root
REM
REM  Optional:
REM    set GBENCH_ROOT=C:\path\to\google-benchmark  (for bench target)
REM    set GTEST_ROOT=C:\path\to\googletest         (for GTest-linked tests)
REM ============================================================================

setlocal EnableDelayedExpansion

REM ── Repository root ──────────────────────────────────────────────────────────
if not defined REPO (
    set "REPO=%~dp0.."
)
echo [build_simd] REPO = %REPO%

REM ── Output directory ─────────────────────────────────────────────────────────
set "OUT=%REPO%\build_simd_direct"
if not exist "%OUT%" mkdir "%OUT%"

REM ── Common compile flags ─────────────────────────────────────────────────────
set "CFLAGS=/std:c++20 /W4 /WX /permissive- /EHsc /O2 /DNDEBUG"
set "INC=/I"%REPO%\src" /I"%REPO%\include" /I"%REPO%\src\simd" /I"%REPO%""

REM ── Per-TU SIMD flags ────────────────────────────────────────────────────────
set "AVX2_FLAGS=/arch:AVX2"
set "AVX512_FLAGS=/arch:AVX512"

echo.
echo [build_simd] ── Compiling scalar kernels ──────────────────────────────────

cl.exe %CFLAGS% %INC% /c /Fo"%OUT%\beta_scalar.obj"  "%REPO%\src\simd\beta_scalar.cpp"
if errorlevel 1 goto :error

cl.exe %CFLAGS% %INC% /c /Fo"%OUT%\gamma_scalar.obj" "%REPO%\src\simd\gamma_scalar.cpp"
if errorlevel 1 goto :error

echo [build_simd] ── Compiling AVX2 kernels ───────────────────────────────────

cl.exe %CFLAGS% %AVX2_FLAGS% %INC% /c /Fo"%OUT%\beta_avx2.obj"  "%REPO%\src\simd\beta_avx2.cpp"
if errorlevel 1 goto :error

cl.exe %CFLAGS% %AVX2_FLAGS% %INC% /c /Fo"%OUT%\gamma_avx2.obj" "%REPO%\src\simd\gamma_avx2.cpp"
if errorlevel 1 goto :error

echo [build_simd] ── Compiling AVX-512F kernels ─────────────────────────────────

cl.exe %CFLAGS% %AVX512_FLAGS% %INC% /c /Fo"%OUT%\beta_avx512.obj"  "%REPO%\src\simd\beta_avx512.cpp"
if errorlevel 1 goto :error

cl.exe %CFLAGS% %AVX512_FLAGS% %INC% /c /Fo"%OUT%\gamma_avx512.obj" "%REPO%\src\simd\gamma_avx512.cpp"
if errorlevel 1 goto :error

echo [build_simd] ── Compiling dispatch layer ───────────────────────────────────

cl.exe %CFLAGS% %INC% /c /Fo"%OUT%\simd_dispatch.obj" "%REPO%\src\simd\simd_dispatch.cpp"
if errorlevel 1 goto :error

REM ── Static library ───────────────────────────────────────────────────────────
echo [build_simd] ── Linking srfm_simd.lib ─────────────────────────────────────

lib.exe /nologo /OUT:"%OUT%\srfm_simd.lib" ^
    "%OUT%\beta_scalar.obj"  ^
    "%OUT%\gamma_scalar.obj" ^
    "%OUT%\beta_avx2.obj"    ^
    "%OUT%\gamma_avx2.obj"   ^
    "%OUT%\beta_avx512.obj"  ^
    "%OUT%\gamma_avx512.obj" ^
    "%OUT%\simd_dispatch.obj"
if errorlevel 1 goto :error

echo [build_simd] srfm_simd.lib built OK

REM ── Unit test executable ─────────────────────────────────────────────────────
echo [build_simd] ── Building test_simd.exe ────────────────────────────────────

if defined GTEST_ROOT (
    REM Link against GTest (recommended)
    set "GTEST_INC=/I"%GTEST_ROOT%\include""
    set "GTEST_LIB="%GTEST_ROOT%\lib\gtest.lib" "%GTEST_ROOT%\lib\gtest_main.lib""

    cl.exe %CFLAGS% %INC% %GTEST_INC% ^
        /Fo"%OUT%\test_simd.obj" ^
        /Fe"%OUT%\test_simd.exe" ^
        "%REPO%\tests\simd\test_simd.cpp" ^
        "%OUT%\srfm_simd.lib" ^
        %GTEST_LIB%
    if errorlevel 1 goto :error
) else (
    echo [build_simd] GTEST_ROOT not set — skipping test_simd.exe
    echo              Set GTEST_ROOT=^<path^> to build with GTest.
    goto :bench_section
)

echo [build_simd] ── Running test_simd.exe ─────────────────────────────────────
"%OUT%\test_simd.exe"
if errorlevel 1 (
    echo [build_simd] TESTS FAILED
    exit /b 1
)
echo [build_simd] All tests passed.

:bench_section
REM ── Benchmark executable (optional) ─────────────────────────────────────────
if /i not "%1"=="bench" goto :done

echo.
echo [build_simd] ── Building bench_beta_gamma.exe ────────────────────────────

if not defined GBENCH_ROOT (
    echo [build_simd] GBENCH_ROOT not set — cannot build benchmarks.
    echo              Set GBENCH_ROOT=^<path to google-benchmark install^>
    goto :done
)

set "GBENCH_INC=/I"%GBENCH_ROOT%\include""
set "GBENCH_LIB="%GBENCH_ROOT%\lib\benchmark.lib" "%GBENCH_ROOT%\lib\benchmark_main.lib" Shlwapi.lib"

cl.exe %CFLAGS% %INC% %GBENCH_INC% ^
    /Fo"%OUT%\bench_beta_gamma.obj" ^
    /Fe"%OUT%\bench_beta_gamma.exe" ^
    "%REPO%\bench\bench_beta_gamma.cpp" ^
    "%OUT%\srfm_simd.lib" ^
    %GBENCH_LIB%
if errorlevel 1 goto :error

echo [build_simd] ── Running bench_beta_gamma.exe ────────────────────────────
"%OUT%\bench_beta_gamma.exe" --benchmark_format=console
if errorlevel 1 goto :error

goto :done

:error
echo.
echo [build_simd] BUILD FAILED (exit code %ERRORLEVEL%)
exit /b 1

:done
echo.
echo [build_simd] Done.  Artifacts in: %OUT%
endlocal
