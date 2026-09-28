@echo off
cd /d "%~dp0"
echo SRFM demo: label 1,256 SPY daily bars as timelike or spacelike
echo.
.\regime_validator.exe --input validation\data\SPY_1m.csv --output spy_regime.csv --ticker SPY
if errorlevel 1 goto end
echo.
echo The first rows of spy_regime.csv:
powershell -NoProfile -Command "Get-Content spy_regime.csv -TotalCount 6"
echo.
echo The library example: beta, gamma and two interval labels
.\lorentz_basics.exe
echo.
echo Next: run any tool with --help, e.g.  .\regime_validator.exe --help
:end
echo.
pause
