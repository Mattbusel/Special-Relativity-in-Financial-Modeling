@echo off
setlocal

set "REPO=C:\Users\Matthew\Tokio Prompt"
set "VSDEV=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"

call "%VSDEV%" -arch=x64 -host_arch=x64 2>nul

mkdir "%REPO%\tests\simd\build" 2>nul

set "CL_FLAGS=/std:c++20 /W4 /EHsc /nologo /O2 /fp:precise"
set "INC=/I"%REPO%\src" /I"%REPO%\include""

echo [1/8] momentum.cpp
cl.exe %CL_FLAGS% %INC% /c "%REPO%\src\momentum\momentum.cpp" /Fo:"%REPO%\tests\simd\build\momentum.obj" 2>&1
if errorlevel 1 ( echo FAILED & exit /b 1 )

echo [2/8] beta_scalar.cpp
cl.exe %CL_FLAGS% %INC% /c "%REPO%\src\simd\beta_scalar.cpp" /Fo:"%REPO%\tests\simd\build\beta_scalar.obj" 2>&1
if errorlevel 1 ( echo FAILED & exit /b 1 )

echo [3/8] gamma_scalar.cpp
cl.exe %CL_FLAGS% %INC% /c "%REPO%\src\simd\gamma_scalar.cpp" /Fo:"%REPO%\tests\simd\build\gamma_scalar.obj" 2>&1
if errorlevel 1 ( echo FAILED & exit /b 1 )

echo [4/8] beta_avx2.cpp
cl.exe %CL_FLAGS% %INC% /arch:AVX2 /c "%REPO%\src\simd\beta_avx2.cpp" /Fo:"%REPO%\tests\simd\build\beta_avx2.obj" 2>&1
if errorlevel 1 ( echo FAILED & exit /b 1 )

echo [5/8] gamma_avx2.cpp
cl.exe %CL_FLAGS% %INC% /arch:AVX2 /c "%REPO%\src\simd\gamma_avx2.cpp" /Fo:"%REPO%\tests\simd\build\gamma_avx2.obj" 2>&1
if errorlevel 1 ( echo FAILED & exit /b 1 )

echo [6/8] beta_avx512.cpp
cl.exe %CL_FLAGS% %INC% /arch:AVX512 /c "%REPO%\src\simd\beta_avx512.cpp" /Fo:"%REPO%\tests\simd\build\beta_avx512.obj" 2>&1
if errorlevel 1 ( echo FAILED & exit /b 1 )

echo [7/8] gamma_avx512.cpp
cl.exe %CL_FLAGS% %INC% /arch:AVX512 /c "%REPO%\src\simd\gamma_avx512.cpp" /Fo:"%REPO%\tests\simd\build\gamma_avx512.obj" 2>&1
if errorlevel 1 ( echo FAILED & exit /b 1 )

echo [8/8] simd_dispatch.cpp + test_simd.cpp
cl.exe %CL_FLAGS% %INC% /I"%REPO%\tests\momentum" /I"%REPO%\src\simd" ^
    "%REPO%\src\simd\simd_dispatch.cpp" ^
    "%REPO%\tests\momentum\test_simd.cpp" ^
    "%REPO%\tests\simd\build\momentum.obj" ^
    "%REPO%\tests\simd\build\beta_scalar.obj" ^
    "%REPO%\tests\simd\build\gamma_scalar.obj" ^
    "%REPO%\tests\simd\build\beta_avx2.obj" ^
    "%REPO%\tests\simd\build\gamma_avx2.obj" ^
    "%REPO%\tests\simd\build\beta_avx512.obj" ^
    "%REPO%\tests\simd\build\gamma_avx512.obj" ^
    /Fe:"%REPO%\tests\simd\build\test_simd.exe" 2>&1
if errorlevel 1 ( echo LINK FAILED & exit /b 1 )

echo.
echo === Build succeeded — running tests ===
"%REPO%\tests\simd\build\test_simd.exe"
exit /b %errorlevel%
