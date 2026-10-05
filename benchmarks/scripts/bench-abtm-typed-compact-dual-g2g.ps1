[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,
    [int] $PairsPerRow = 8,
    [int] $Repeats = 5
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if ($Matrix.Count -eq 0) {
    throw "at least one -Matrix path is required"
}
if ($PairsPerRow -lt 1) {
    throw "-PairsPerRow must be >= 1"
}
if ($Repeats -lt 1) {
    throw "-Repeats must be >= 1"
}

& cargo build --release -p hybit --example abtm_typed_compact_dual_g2g
if ($LASTEXITCODE -ne 0) {
    throw "failed to build abtm_typed_compact_dual_g2g"
}

$exe = Join-Path $root "target\release\examples\abtm_typed_compact_dual_g2g.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "ABTM G2g executable not found: $exe"
}

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== ABTM G2g matrix: {0} ===" -f $resolved)

    & $exe `
        --matrix $resolved `
        --pairs-per-row $PairsPerRow `
        --repeats $Repeats
    if ($LASTEXITCODE -ne 0) {
        throw "ABTM G2g benchmark failed for '$resolved' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== ABTM G2g typed-compact dual-numeric corpus complete ==="
