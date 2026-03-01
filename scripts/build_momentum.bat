@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  build_momentum.bat  —  AGT-03 Momentum module build + test
REM  Requires: MSVC 2022 (Visual Studio Build Tools)
REM  Usage:    scripts\build_momentum.bat
REM ============================================================

REM Resolve repository root (one level above this script)
set "REPO=%~dp0.."
set "SRC=%REPO%\src\momentum"
set "TESTS=%REPO%\tests\momentum"
set "OUT=%REPO%\tests\momentum\build"

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

REM ── Compile ───────────────────────────────────────────────────────────────
echo.
echo [build] Compiling src\momentum\momentum.cpp + tests\momentum\test_momentum.cpp
echo.

cl.exe /std:c++20 /W4 /WX /EHsc /nologo /O2 ^
    /I"%REPO%\src" ^
    /I"%TESTS%" ^
    "%SRC%\momentum.cpp" ^
    "%TESTS%\test_momentum.cpp" ^
    /Fe:"%OUT%\test_momentum.exe" ^
    /Fo:"%OUT%\\" ^
    /Fd:"%OUT%\vc.pdb"

if errorlevel 1 (
    echo.
    echo [build] FAILED — see compiler output above
    exit /b 1
)

echo.
echo [build] Compilation succeeded.

REM ── Run tests ─────────────────────────────────────────────────────────────
echo [test]  Running %OUT%\test_momentum.exe
echo.
"%OUT%\test_momentum.exe"
set "TEST_RESULT=%ERRORLEVEL%"

echo.
if %TEST_RESULT% equ 0 (
    echo [test]  PASS
) else (
    echo [test]  FAIL  ^(exit code %TEST_RESULT%^)
)
exit /b %TEST_RESULT%
