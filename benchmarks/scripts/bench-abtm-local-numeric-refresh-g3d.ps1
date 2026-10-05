[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,
    [int] $Regions = 64,
    [int] $Hops = 2,
    [int] $Repeats = 5
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if ($Matrix.Count -eq 0) {
    throw "at least one -Matrix path is required"
}
if ($Regions -lt 1) {
    throw "-Regions must be >= 1"
}
if ($Hops -lt 0) {
    throw "-Hops must be >= 0"
}
if ($Repeats -lt 1) {
    throw "-Repeats must be >= 1"
}

& cargo build --release -p hybit --example abtm_local_numeric_refresh_g3d
if ($LASTEXITCODE -ne 0) {
    throw "failed to build abtm_local_numeric_refresh_g3d"
}

$exe = Join-Path $root "target\release\examples\abtm_local_numeric_refresh_g3d.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "ABTM G3d executable not found: $exe"
}

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== ABTM G3d matrix: {0} ===" -f $resolved)

    & $exe `
        --matrix $resolved `
        --regions $Regions `
        --hops $Hops `
        --repeats $Repeats
    if ($LASTEXITCODE -ne 0) {
        throw "ABTM G3d benchmark failed for '$resolved' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== ABTM G3d prepared numeric-refresh corpus complete ==="
