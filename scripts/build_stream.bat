@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  build_stream.bat  —  AGT-10 streaming pipeline build + test
REM  Requires: MSVC 2022 (Visual Studio Build Tools)
REM  Usage:    scripts\build_stream.bat [test_name]
REM            test_name: spsc_ring | tick_validation | coordinate_normalizer
REM                       beta_calculator | lorentz_manifold | stream_engine
REM                       pipeline_invariants | signal_chain | spsc_stress
REM                       all  (default)
REM ============================================================

set "REPO=%~dp0.."
set "INC=%REPO%\include"
set "SRC=%REPO%\src"
set "TESTS=%REPO%\tests\stream"
set "OUT=%REPO%\tests\stream\build"

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=all"

REM ── Initialise MSVC environment ───────────────────────────────────────────
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

REM ── Create output directory ───────────────────────────────────────────────
if not exist "%OUT%" mkdir "%OUT%"

set "CFLAGS=/std:c++20 /W4 /WX /EHsc /nologo /O2"
set "IFLAGS=/I"%INC%" /I"%TESTS%""
set "FAILED=0"
set "PASSED=0"

REM ── Shared compilation helper ─────────────────────────────────────────────
REM Usage: CALL :build_and_run <exe_name> <sources...>

goto :main

:build_one
    set "EXE=%OUT%\%~1.exe"
    shift
    REM Collect remaining args as source files
    set "SRCS="
:collect_loop
    if "%~1"=="" goto :compile
    set "SRCS=!SRCS! "%~1""
    shift
    goto :collect_loop
:compile
    echo.
    echo [build] %EXE%
    cl.exe %CFLAGS% %IFLAGS% !SRCS! /Fe:"%EXE%" /Fo:"%OUT%\\" /Fd:"%OUT%\vc.pdb" >nul
    if errorlevel 1 (
        echo [build] FAILED
        set /a FAILED+=1
    ) else (
        echo [test]  Running...
        "%EXE%"
        if errorlevel 1 (
            set /a FAILED+=1
        ) else (
            set /a PASSED+=1
        )
    )
    exit /b 0

:main

REM ── test_spsc_ring ────────────────────────────────────────────────────────
if "%TARGET%"=="spsc_ring" goto :build_spsc_ring
if "%TARGET%"=="all"       goto :build_spsc_ring
goto :skip_spsc_ring

:build_spsc_ring
echo.
echo ================================================================
echo  Building: test_spsc_ring
echo ================================================================
set "EXE=%OUT%\test_spsc_ring.exe"
cl.exe %CFLAGS% %IFLAGS% ^
    "%TESTS%\test_spsc_ring.cpp" ^
    /Fe:"%EXE%" /Fo:"%OUT%\\" /Fd:"%OUT%\vc.pdb"
if errorlevel 1 (
    echo [build] FAILED — test_spsc_ring
    set /a FAILED+=1
) else (
    "%EXE%"
    if errorlevel 1 (set /a FAILED+=1) else (set /a PASSED+=1)
)
:skip_spsc_ring

REM ── test_tick_validation ──────────────────────────────────────────────────
if "%TARGET%"=="tick_validation" goto :build_tick_validation
if "%TARGET%"=="all"             goto :build_tick_validation
goto :skip_tick_validation

:build_tick_validation
echo.
echo ================================================================
echo  Building: test_tick_validation
echo ================================================================
set "EXE=%OUT%\test_tick_validation.exe"
cl.exe %CFLAGS% %IFLAGS% ^
    "%TESTS%\test_tick_validation.cpp" ^
    /Fe:"%EXE%" /Fo:"%OUT%\\" /Fd:"%OUT%\vc.pdb"
if errorlevel 1 (
    echo [build] FAILED — test_tick_validation
    set /a FAILED+=1
) else (
    "%EXE%"
    if errorlevel 1 (set /a FAILED+=1) else (set /a PASSED+=1)
)
:skip_tick_validation

REM ── test_coordinate_normalizer ────────────────────────────────────────────
if "%TARGET%"=="coordinate_normalizer" goto :build_coord
if "%TARGET%"=="all"                   goto :build_coord
goto :skip_coord

:build_coord
echo.
echo ================================================================
echo  Building: test_coordinate_normalizer
echo ================================================================
set "EXE=%OUT%\test_coordinate_normalizer.exe"
cl.exe %CFLAGS% %IFLAGS% ^
    "%TESTS%\test_coordinate_normalizer.cpp" ^
    /Fe:"%EXE%" /Fo:"%OUT%\\" /Fd:"%OUT%\vc.pdb"
if errorlevel 1 (
    echo [build] FAILED — test_coordinate_normalizer
    set /a FAILED+=1
) else (
    "%EXE%"
    if errorlevel 1 (set /a FAILED+=1) else (set /a PASSED+=1)
)
:skip_coord

REM ── test_beta_calculator ──────────────────────────────────────────────────
if "%TARGET%"=="beta_calculator" goto :build_beta
if "%TARGET%"=="all"             goto :build_beta
goto :skip_beta

:build_beta
echo.
echo ================================================================
echo  Building: test_beta_calculator
echo ================================================================
set "EXE=%OUT%\test_beta_calculator.exe"
cl.exe %CFLAGS% %IFLAGS% ^
    "%TESTS%\test_beta_calculator.cpp" ^
    /Fe:"%EXE%" /Fo:"%OUT%\\" /Fd:"%OUT%\vc.pdb"
if errorlevel 1 (
    echo [build] FAILED — test_beta_calculator
    set /a FAILED+=1
) else (
    "%EXE%"
    if errorlevel 1 (set /a FAILED+=1) else (set /a PASSED+=1)
)
:skip_beta

REM ── test_lorentz_manifold ─────────────────────────────────────────────────
if "%TARGET%"=="lorentz_manifold" goto :build_lorentz
if "%TARGET%"=="all"              goto :build_lorentz
goto :skip_lorentz

:build_lorentz
echo.
echo ================================================================
echo  Building: test_lorentz_manifold
echo ================================================================
set "EXE=%OUT%\test_lorentz_manifold.exe"
cl.exe %CFLAGS% %IFLAGS% ^
    "%TESTS%\test_lorentz_manifold.cpp" ^
    /Fe:"%EXE%" /Fo:"%OUT%\\" /Fd:"%OUT%\vc.pdb"
if errorlevel 1 (
    echo [build] FAILED — test_lorentz_manifold
    set /a FAILED+=1
) else (
    "%EXE%"
    if errorlevel 1 (set /a FAILED+=1) else (set /a PASSED+=1)
)
:skip_lorentz

REM ── test_stream_engine (integration) ─────────────────────────────────────
if "%TARGET%"=="stream_engine" goto :build_engine
if "%TARGET%"=="all"           goto :build_engine
goto :skip_engine

:build_engine
echo.
echo ================================================================
echo  Building: test_stream_engine  (integration)
echo ================================================================
set "EXE=%OUT%\test_stream_engine.exe"
cl.exe %CFLAGS% %IFLAGS% ^
    "%SRC%\stream\tick_ingester.cpp" ^
    "%SRC%\stream\signal_processor.cpp" ^
    "%SRC%\stream\signal_consumer.cpp" ^
    "%SRC%\engine\stream_engine.cpp" ^
    "%TESTS%\test_stream_engine.cpp" ^
    /Fe:"%EXE%" /Fo:"%OUT%\\" /Fd:"%OUT%\vc.pdb"
if errorlevel 1 (
    echo [build] FAILED — test_stream_engine
    set /a FAILED+=1
) else (
    "%EXE%"
    if errorlevel 1 (set /a FAILED+=1) else (set /a PASSED+=1)
)
:skip_engine

REM ── test_pipeline_invariants (property-based invariants) ──────────────────
if "%TARGET%"=="pipeline_invariants" goto :build_invariants
if "%TARGET%"=="all"                 goto :build_invariants
goto :skip_invariants

:build_invariants
echo.
echo ================================================================
echo  Building: test_pipeline_invariants
echo ================================================================
set "EXE=%OUT%\test_pipeline_invariants.exe"
cl.exe %CFLAGS% %IFLAGS% ^
    "%SRC%\stream\tick_ingester.cpp" ^
    "%SRC%\stream\signal_processor.cpp" ^
    "%SRC%\stream\signal_consumer.cpp" ^
    "%SRC%\engine\stream_engine.cpp" ^
    "%TESTS%\test_pipeline_invariants.cpp" ^
    /Fe:"%EXE%" /Fo:"%OUT%\\" /Fd:"%OUT%\vc.pdb"
if errorlevel 1 (
    echo [build] FAILED — test_pipeline_invariants
    set /a FAILED+=1
) else (
    "%EXE%"
    if errorlevel 1 (set /a FAILED+=1) else (set /a PASSED+=1)
)
:skip_invariants

REM ── test_signal_chain (unit + integration, header-only) ──────────────────
if "%TARGET%"=="signal_chain" goto :build_signal_chain
if "%TARGET%"=="all"          goto :build_signal_chain
goto :skip_signal_chain

:build_signal_chain
echo.
echo ================================================================
echo  Building: test_signal_chain
echo ================================================================
set "EXE=%OUT%\test_signal_chain.exe"
cl.exe %CFLAGS% %IFLAGS% ^
    "%TESTS%\test_signal_chain.cpp" ^
    /Fe:"%EXE%" /Fo:"%OUT%\\" /Fd:"%OUT%\vc.pdb"
if errorlevel 1 (
    echo [build] FAILED — test_signal_chain
    set /a FAILED+=1
) else (
    "%EXE%"
    if errorlevel 1 (set /a FAILED+=1) else (set /a PASSED+=1)
)
:skip_signal_chain

REM ── test_spsc_stress (ring boundary + tick validation stress) ─────────────
if "%TARGET%"=="spsc_stress" goto :build_spsc_stress
if "%TARGET%"=="all"         goto :build_spsc_stress
goto :skip_spsc_stress

:build_spsc_stress
echo.
echo ================================================================
echo  Building: test_spsc_stress
echo ================================================================
set "EXE=%OUT%\test_spsc_stress.exe"
cl.exe %CFLAGS% %IFLAGS% ^
    "%TESTS%\test_spsc_stress.cpp" ^
    /Fe:"%EXE%" /Fo:"%OUT%\\" /Fd:"%OUT%\vc.pdb"
if errorlevel 1 (
    echo [build] FAILED — test_spsc_stress
    set /a FAILED+=1
) else (
    "%EXE%"
    if errorlevel 1 (set /a FAILED+=1) else (set /a PASSED+=1)
)
:skip_spsc_stress

REM ── Summary ───────────────────────────────────────────────────────────────
echo.
echo ================================================================
echo  AGT-10 Stream Build Summary
echo  Suites passed: %PASSED%    Suites failed: %FAILED%
echo ================================================================

if %FAILED% gtr 0 (
    echo [result] FAIL
    exit /b 1
)
echo [result] PASS
exit /b 0
