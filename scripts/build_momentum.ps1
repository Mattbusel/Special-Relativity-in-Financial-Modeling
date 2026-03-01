# build_momentum.ps1 — Direct cl.exe invocation without VsDevCmd
# Resolves all MSVC and Windows SDK paths explicitly.
param([switch]$Verbose)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Src      = Join-Path $RepoRoot 'src\momentum'
$Tests    = Join-Path $RepoRoot 'tests\momentum'
$Out      = Join-Path $Tests    'build'

# ── Locate MSVC ──────────────────────────────────────────────────────────────
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
    (Join-Path $SdkBase  "Include\$SdkVer\shared")
)
$LibDirs  = @(
    (Join-Path $MsvcDir  'lib\x64'),
    (Join-Path $SdkBase  "lib\$SdkVer\ucrt\x64"),
    (Join-Path $SdkBase  "lib\$SdkVer\um\x64")
)

# ── Create output dir ─────────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path $Out | Out-Null

# ── Build argument list ───────────────────────────────────────────────────────
$IncFlags = $Includes | ForEach-Object { "/I`"$_`"" }
$LibFlags = $LibDirs  | ForEach-Object { "/LIBPATH:`"$_`"" }

$Args = @(
    '/std:c++20', '/W4', '/WX', '/EHsc', '/nologo', '/O2',
    "/I`"$RepoRoot\src`"",
    "/I`"$Tests`""
) + $IncFlags + @(
    "`"$Src\momentum.cpp`"",
    "`"$Tests\test_momentum.cpp`"",
    "/Fe:`"$Out\test_momentum.exe`"",
    "/Fo:`"$Out\\`"",
    "/Fd:`"$Out\vc.pdb`"",
    '/link'
) + $LibFlags

Write-Host "`n[build] cl.exe $MsvcVer  C++20  /W4 /WX"
if ($Verbose) { Write-Host "Args: $Args" }

# ── Compile ───────────────────────────────────────────────────────────────────
$proc = Start-Process -FilePath $ClExe `
    -ArgumentList $Args `
    -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput "$Out\compile_stdout.txt" `
    -RedirectStandardError  "$Out\compile_stderr.txt"

Get-Content "$Out\compile_stdout.txt" | Write-Host
$errText = Get-Content "$Out\compile_stderr.txt" -Raw
if ($errText) { Write-Host $errText }

if ($proc.ExitCode -ne 0) {
    Write-Host "`n[build] FAILED (exit $($proc.ExitCode))"
    exit $proc.ExitCode
}
Write-Host "`n[build] Compilation succeeded."

# ── Run tests ─────────────────────────────────────────────────────────────────
Write-Host "[test]  Running tests..."
$run = Start-Process -FilePath "$Out\test_momentum.exe" `
    -NoNewWindow -Wait -PassThru
Write-Host ""
if ($run.ExitCode -eq 0) {
    Write-Host "[test]  PASS"
} else {
    Write-Host "[test]  FAIL (exit $($run.ExitCode))"
}
exit $run.ExitCode
