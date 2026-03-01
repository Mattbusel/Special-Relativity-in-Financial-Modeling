# build_n_asset.ps1 - Build and run all Stage 4 N-Asset Manifold tests.
# Direct cl.exe invocation without VsDevCmd (mirrors build_momentum.ps1).
param([switch]$Verbose)

$ErrorActionPreference = 'Stop'

$RepoRoot    = Split-Path $PSScriptRoot -Parent
$SrcTensor   = Join-Path $RepoRoot "src\tensor"
$SrcManifold = Join-Path $RepoRoot "src\manifold"
$SrcEngine   = Join-Path $RepoRoot "src\engine"
$TestsDir    = Join-Path $RepoRoot "tests\n_asset"
$BuildDir    = Join-Path $TestsDir  "build"
$EigenInc    = Join-Path $RepoRoot "third_party\eigen"

# ── Locate MSVC ───────────────────────────────────────────────────────────────
$MsvcBase = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC'
$MsvcVer  = (Get-ChildItem $MsvcBase | Sort-Object Name | Select-Object -Last 1).Name
$MsvcDir  = Join-Path $MsvcBase $MsvcVer
$ClExe    = Join-Path $MsvcDir  'bin\Hostx64\x64\cl.exe'

if (-not (Test-Path $ClExe)) {
    Write-Error "cl.exe not found at $ClExe"
    exit 1
}

# ── Locate Windows SDK ────────────────────────────────────────────────────────
$SdkBase = 'C:\Program Files (x86)\Windows Kits\10'
$SdkVer  = (Get-ChildItem (Join-Path $SdkBase 'Include') |
            Sort-Object Name | Select-Object -Last 1).Name

$Includes = @(
    (Join-Path $MsvcDir  'include'),
    (Join-Path $SdkBase  "Include\$SdkVer\ucrt"),
    (Join-Path $SdkBase  "Include\$SdkVer\um"),
    (Join-Path $SdkBase  "Include\$SdkVer\shared"),
    $RepoRoot,
    $EigenInc
)
$LibDirs  = @(
    (Join-Path $MsvcDir  'lib\x64'),
    (Join-Path $SdkBase  "lib\$SdkVer\ucrt\x64"),
    (Join-Path $SdkBase  "lib\$SdkVer\um\x64")
)

# ── Prerequisites ─────────────────────────────────────────────────────────────
$EigenDense = Join-Path $EigenInc "Eigen\Dense"
if (-not (Test-Path $EigenDense)) {
    Write-Host "Eigen not found. Running fetch_eigen.ps1 ..."
    & (Join-Path $PSScriptRoot "fetch_eigen.ps1")
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

# ── Common flags ──────────────────────────────────────────────────────────────
$IncFlags = $Includes | ForEach-Object { "/I`"$_`"" }
$LibFlags = $LibDirs  | ForEach-Object { "/LIBPATH:`"$_`"" }

$BaseFlags = @(
    '/std:c++20', '/W3', '/WX', '/EHsc', '/nologo', '/O2',
    '/DEIGEN_MPL2_ONLY'
) + $IncFlags

# ── Source files ──────────────────────────────────────────────────────────────
$SrcManifoldCpp = Join-Path $SrcTensor   "n_asset_manifold.cpp"
$SrcChristoffel  = Join-Path $SrcTensor   "christoffel_n.cpp"
$SrcGeodesic     = Join-Path $SrcTensor   "geodesic_n.cpp"
$SrcIntervalCpp  = Join-Path $SrcManifold "n_asset_interval.cpp"
$SrcEngineCpp    = Join-Path $SrcEngine   "n_asset_engine.cpp"

$Tests = @(
    @{
        Name = "test_n_asset_manifold"
        Main = Join-Path $TestsDir "test_n_asset_manifold.cpp"
        Deps = @($SrcManifoldCpp)
    },
    @{
        Name = "test_christoffel_n"
        Main = Join-Path $TestsDir "test_christoffel_n.cpp"
        Deps = @($SrcManifoldCpp, $SrcChristoffel)
    },
    @{
        Name = "test_geodesic_n"
        Main = Join-Path $TestsDir "test_geodesic_n.cpp"
        Deps = @($SrcManifoldCpp, $SrcChristoffel, $SrcGeodesic)
    },
    @{
        Name = "test_n_asset_interval"
        Main = Join-Path $TestsDir "test_n_asset_interval.cpp"
        Deps = @($SrcManifoldCpp, $SrcIntervalCpp)
    },
    @{
        Name = "test_n_asset_engine"
        Main = Join-Path $TestsDir "test_n_asset_engine.cpp"
        Deps = @($SrcManifoldCpp, $SrcChristoffel, $SrcGeodesic, $SrcIntervalCpp, $SrcEngineCpp)
    }
)

# ── Build loop ────────────────────────────────────────────────────────────────
$TotalPass  = 0
$TotalFail  = 0
$BuildFailed = @()

foreach ($t in $Tests) {
    Write-Host ""
    Write-Host "[build] $($t.Name)" -ForegroundColor Cyan

    $Exe     = Join-Path $BuildDir "$($t.Name).exe"
    $ObjDir  = Join-Path $BuildDir "$($t.Name)_obj"
    New-Item -ItemType Directory -Force -Path $ObjDir | Out-Null

    $Sources = @($t.Main) + $t.Deps | ForEach-Object { "`"$_`"" }
    $Args = $BaseFlags + $Sources + @(
        "/Fe:`"$Exe`"",
        "/Fo:`"$ObjDir\\`"",
        "/Fd:`"$ObjDir\vc.pdb`"",
        '/link'
    ) + $LibFlags

    if ($Verbose) { Write-Host "cl $($Args -join ' ')" }

    $proc = Start-Process -FilePath $ClExe `
        -ArgumentList $Args `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$ObjDir\stdout.txt" `
        -RedirectStandardError  "$ObjDir\stderr.txt"

    Get-Content "$ObjDir\stdout.txt" | Write-Host
    $errTxt = Get-Content "$ObjDir\stderr.txt" -Raw -ErrorAction SilentlyContinue
    if ($errTxt) { Write-Host $errTxt }

    if ($proc.ExitCode -ne 0) {
        Write-Host "[build] FAILED for $($t.Name) (exit $($proc.ExitCode))" -ForegroundColor Red
        $BuildFailed += $t.Name
        continue
    }
    Write-Host "[build] OK"

    Write-Host "[test]  Running $($t.Name) ..."
    $run = Start-Process -FilePath $Exe -NoNewWindow -Wait -PassThru
    if ($run.ExitCode -eq 0) {
        Write-Host "[test]  PASS" -ForegroundColor Green
        $TotalPass++
    } else {
        Write-Host "[test]  FAIL (exit $($run.ExitCode))" -ForegroundColor Red
        $TotalFail++
    }
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Stage 4 N-Asset Manifold Summary ==="
if ($BuildFailed.Count -gt 0) {
    Write-Host "Build failures: $($BuildFailed -join ', ')" -ForegroundColor Red
}
Write-Host "Suites passed: $TotalPass" -ForegroundColor Green
Write-Host "Suites failed: $TotalFail" -ForegroundColor $(if ($TotalFail -gt 0) { 'Red' } else { 'Green' })

if ($TotalFail -gt 0 -or $BuildFailed.Count -gt 0) { exit 1 }
Write-Host "All Stage 4 tests PASSED." -ForegroundColor Green
exit 0
