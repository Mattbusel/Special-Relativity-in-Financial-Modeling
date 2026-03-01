# fetch_eigen.ps1
# Downloads Eigen 3.4.0 (header-only, MPL2 licence) to third_party/eigen/.
#
# Usage:
#   .\scripts\fetch_eigen.ps1
#
# After this script completes, the Eigen headers will be available at:
#   third_party/eigen/Eigen/Dense
#   third_party/eigen/Eigen/Core
#
# Eigen is header-only; no compilation step is needed.

$ErrorActionPreference = 'Stop'

$EigenVersion = "3.4.0"
$EigenUrl     = "https://gitlab.com/libeigen/eigen/-/archive/$EigenVersion/eigen-$EigenVersion.zip"
$EigenUrlAlt  = "https://gitlab.com/libeigen/eigen/-/archive/$EigenVersion/eigen-$EigenVersion.zip"

$RepoRoot    = Split-Path $PSScriptRoot -Parent
$ThirdParty  = Join-Path $RepoRoot "third_party"
$EigenDest   = Join-Path $ThirdParty "eigen"
$ZipPath     = Join-Path $ThirdParty "eigen-$EigenVersion.zip"
$ExtractPath = Join-Path $ThirdParty "eigen-extract-tmp"

Write-Host "=== fetch_eigen.ps1 : Eigen $EigenVersion ==="
Write-Host "Repo root : $RepoRoot"
Write-Host "Dest dir  : $EigenDest"

$DensePath = Join-Path (Join-Path $EigenDest "Eigen") "Dense"
if (Test-Path $DensePath) {
    Write-Host "Eigen already present at $EigenDest -- nothing to do."
    exit 0
}

if (-not (Test-Path $ThirdParty)) {
    Write-Host "Creating $ThirdParty ..."
    New-Item -ItemType Directory -Path $ThirdParty | Out-Null
}

$downloaded = $false
foreach ($url in @($EigenUrl, $EigenUrlAlt)) {
    try {
        Write-Host "Downloading from $url ..."
        Invoke-WebRequest -Uri $url -OutFile $ZipPath -UseBasicParsing
        $downloaded = $true
        Write-Host "Download complete."
        break
    } catch {
        Write-Warning "Failed from ${url}: $_"
    }
}

if (-not $downloaded) {
    Write-Error "All download attempts failed."
    exit 1
}

Write-Host "Extracting to $ExtractPath ..."
if (Test-Path $ExtractPath) {
    Remove-Item $ExtractPath -Recurse -Force
}
Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath

$Extracted = Get-ChildItem -Path $ExtractPath -Directory | Select-Object -First 1
if (-not $Extracted) {
    Write-Error "Could not find extracted Eigen directory inside $ExtractPath"
    exit 1
}

Write-Host "Extracted: $($Extracted.FullName)"

if (Test-Path $EigenDest) {
    Write-Host "Removing existing $EigenDest ..."
    Remove-Item $EigenDest -Recurse -Force
}

Write-Host "Moving to $EigenDest ..."
Move-Item -Path $Extracted.FullName -Destination $EigenDest

if (Test-Path $ZipPath)     { Remove-Item $ZipPath     -Force }
if (Test-Path $ExtractPath) { Remove-Item $ExtractPath -Recurse -Force }

if (Test-Path $DensePath) {
    Write-Host "Eigen $EigenVersion successfully installed at $EigenDest"
} else {
    Write-Error "Verification failed: $DensePath not found after extraction."
    exit 1
}
