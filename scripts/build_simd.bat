@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  build_simd.bat  —  AGT-08 AVX-512 SIMD build + test
REM  Requires: MSVC 2022 (Visual Studio Build Tools or Community)
REM  Usage:    scripts\build_simd.bat
REM
REM  Builds:
REM    tests\simd\build\test_simd.exe     — unit + correctness tests
REM    bench\build\bench_beta_gamma.exe   — throughput benchmarks
REM                                         (requires Google Benchmark)
REM ============================================================

set "REPO=%~dp0.."
set "SRC=%REPO%\src"
set "SIMD_SRC=%REPO%\src\simd"
set "MOMENTUM_SRC=%REPO%\src\momentum"
set "TESTS=%REPO%\tests\momentum"
set "BENCH=%REPO%\bench"
set "INCLUDE=%REPO%\include"
set "OUT_TEST=%REPO%\tests\simd\build"
set "OUT_BENCH=%REPO%\bench\build"

REM ── Initialise MSVC environment ───────────────────────────────────────────────
set "VSDEV=%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEV%" (
    set "VSDEV=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
)
if not exist "%VSDEV%" (
    echo ERROR: Cannot find VsDevCmd.bat — is VS 2022 installed?
    exit /b 1
)
call "%VSDEV%" -arch=x64 -host_arch=x64 2>nul
if errorlevel 1 (
    echo ERROR: VsDevCmd.bat failed to initialise
    exit /b 1
)

REM ── Create output directories ─────────────────────────────────────────────────
if not exist "%OUT_TEST%"  mkdir "%OUT_TEST%"
if not exist "%OUT_BENCH%" mkdir "%OUT_BENCH%"

REM ── Common compiler flags ─────────────────────────────────────────────────────
REM  /std:c++20    — C++20 required for std::span, concepts, etc.
REM  /W4 /WX       — high warnings, warnings-as-errors
REM  /EHsc         — standard C++ exception model
REM  /O2           — full optimisation
REM  /fp:precise   — IEEE-754 conformant FP (required for scalar/SIMD parity)
set "COMMON=/std:c++20 /W4 /WX /EHsc /nologo /O2 /fp:precise"
set "INCLUDES=/I"%SRC%" /I"%INCLUDE%" /I"%TESTS%" /I"%SIMD_SRC%""

REM ── Compile momentum core (scalar physics) ────────────────────────────────────
echo.
echo [build] Compiling momentum core (scalar)...
cl.exe %COMMON% %INCLUDES% ^
    /c "%MOMENTUM_SRC%\momentum.cpp" ^
    /Fo:"%OUT_TEST%\momentum.obj" ^
    /Fd:"%OUT_TEST%\vc.pdb"
if errorlevel 1 goto :fail

REM ── Compile SIMD scalar kernels ───────────────────────────────────────────────
echo [build] Compiling SIMD scalar kernels...
cl.exe %COMMON% %INCLUDES% ^
    /c "%SIMD_SRC%\beta_scalar.cpp" ^
    /Fo:"%OUT_TEST%\beta_scalar.obj" ^
    /Fd:"%OUT_TEST%\vc.pdb"
if errorlevel 1 goto :fail

cl.exe %COMMON% %INCLUDES% ^
    /c "%SIMD_SRC%\gamma_scalar.cpp" ^
    /Fo:"%OUT_TEST%\gamma_scalar.obj" ^
    /Fd:"%OUT_TEST%\vc.pdb"
if errorlevel 1 goto :fail

REM ── Compile AVX2 kernels  (/arch:AVX2) ───────────────────────────────────────
echo [build] Compiling AVX2 kernels (/arch:AVX2)...
cl.exe %COMMON% %INCLUDES% /arch:AVX2 ^
    /c "%SIMD_SRC%\beta_avx2.cpp" ^
    /Fo:"%OUT_TEST%\beta_avx2.obj" ^
    /Fd:"%OUT_TEST%\vc.pdb"
if errorlevel 1 goto :fail

cl.exe %COMMON% %INCLUDES% /arch:AVX2 ^
    /c "%SIMD_SRC%\gamma_avx2.cpp" ^
    /Fo:"%OUT_TEST%\gamma_avx2.obj" ^
    /Fd:"%OUT_TEST%\vc.pdb"
if errorlevel 1 goto :fail

REM ── Compile AVX-512 kernels (/arch:AVX512) ────────────────────────────────────
echo [build] Compiling AVX-512 kernels (/arch:AVX512)...
cl.exe %COMMON% %INCLUDES% /arch:AVX512 ^
    /c "%SIMD_SRC%\beta_avx512.cpp" ^
    /Fo:"%OUT_TEST%\beta_avx512.obj" ^
    /Fd:"%OUT_TEST%\vc.pdb"
if errorlevel 1 goto :fail

cl.exe %COMMON% %INCLUDES% /arch:AVX512 ^
    /c "%SIMD_SRC%\gamma_avx512.cpp" ^
    /Fo:"%OUT_TEST%\gamma_avx512.obj" ^
    /Fd:"%OUT_TEST%\vc.pdb"
if errorlevel 1 goto :fail

REM ── Compile dispatch layer ────────────────────────────────────────────────────
echo [build] Compiling SIMD dispatch layer...
cl.exe %COMMON% %INCLUDES% ^
    /c "%SIMD_SRC%\simd_dispatch.cpp" ^
    /Fo:"%OUT_TEST%\simd_dispatch.obj" ^
    /Fd:"%OUT_TEST%\vc.pdb"
if errorlevel 1 goto :fail

REM ── Link test_simd ────────────────────────────────────────────────────────────
echo [build] Linking test_simd.exe...
cl.exe %COMMON% %INCLUDES% ^
    "%TESTS%\test_simd.cpp" ^
    "%OUT_TEST%\momentum.obj" ^
    "%OUT_TEST%\beta_scalar.obj" ^
    "%OUT_TEST%\gamma_scalar.obj" ^
    "%OUT_TEST%\beta_avx2.obj" ^
    "%OUT_TEST%\gamma_avx2.obj" ^
    "%OUT_TEST%\beta_avx512.obj" ^
    "%OUT_TEST%\gamma_avx512.obj" ^
    "%OUT_TEST%\simd_dispatch.obj" ^
    /Fe:"%OUT_TEST%\test_simd.exe" ^
    /Fd:"%OUT_TEST%\vc.pdb"
if errorlevel 1 goto :fail

echo.
echo [build] All objects compiled successfully.

REM ── Run SIMD tests ────────────────────────────────────────────────────────────
echo.
echo [test]  Running test_simd.exe
echo.
"%OUT_TEST%\test_simd.exe"
set "TEST_RESULT=%ERRORLEVEL%"
echo.
if %TEST_RESULT% equ 0 (
    echo [test]  PASS
) else (
    echo [test]  FAIL  (exit code %TEST_RESULT%)
    exit /b %TEST_RESULT%
)

REM ── Check for Google Benchmark ────────────────────────────────────────────────
REM  Google Benchmark is not bundled; install via vcpkg:
REM    vcpkg install benchmark --triplet x64-windows
REM  Then set BENCHMARK_INCLUDE and BENCHMARK_LIB below.
REM  If not found, skip the benchmark build.

set "BENCHMARK_INCLUDE=%VCPKG_ROOT%\installed\x64-windows\include"
set "BENCHMARK_LIB=%VCPKG_ROOT%\installed\x64-windows\lib\benchmark.lib"
set "BENCHMARK_MAIN_LIB=%VCPKG_ROOT%\installed\x64-windows\lib\benchmark_main.lib"

if not defined VCPKG_ROOT (
    echo.
    echo [bench]  Skipping benchmark build: VCPKG_ROOT not set.
    echo          To build benchmarks:
    echo            1. Install vcpkg (https://github.com/microsoft/vcpkg)
    echo            2. vcpkg install benchmark:x64-windows
    echo            3. Set VCPKG_ROOT and re-run this script.
    echo.
    exit /b 0
)

if not exist "%BENCHMARK_LIB%" (
    echo.
    echo [bench]  Skipping benchmark build: %BENCHMARK_LIB% not found.
    echo          Run: vcpkg install benchmark:x64-windows
    echo.
    exit /b 0
)

echo [build] Compiling bench_beta_gamma.exe (Google Benchmark)...
cl.exe %COMMON% %INCLUDES% /I"%BENCHMARK_INCLUDE%" ^
    "%BENCH%\bench_beta_gamma.cpp" ^
    "%OUT_TEST%\momentum.obj" ^
    "%OUT_TEST%\beta_scalar.obj" ^
    "%OUT_TEST%\gamma_scalar.obj" ^
    "%OUT_TEST%\beta_avx2.obj" ^
    "%OUT_TEST%\gamma_avx2.obj" ^
    "%OUT_TEST%\beta_avx512.obj" ^
    "%OUT_TEST%\gamma_avx512.obj" ^
    "%OUT_TEST%\simd_dispatch.obj" ^
    /Fe:"%OUT_BENCH%\bench_beta_gamma.exe" ^
    /Fd:"%OUT_BENCH%\vc.pdb" ^
    /link "%BENCHMARK_LIB%" "%BENCHMARK_MAIN_LIB%" shlwapi.lib
if errorlevel 1 (
    echo [bench]  Build failed — see compiler output above.
    exit /b 1
)

echo.
echo [bench]  Running bench_beta_gamma.exe
echo.
"%OUT_BENCH%\bench_beta_gamma.exe" --benchmark_format=console
echo.

goto :eof

:fail
echo.
echo [build] FAILED — see compiler output above.
exit /b 1
