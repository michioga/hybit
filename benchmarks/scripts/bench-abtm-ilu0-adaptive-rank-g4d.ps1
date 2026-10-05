[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,
    [int] $Repeats = 5,
    [string] $Thresholds = "1,2,4,6,8,12,16,24,32"
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if ($Matrix.Count -eq 0) {
    throw "at least one -Matrix path is required"
}
if ($Repeats -lt 1) {
    throw "-Repeats must be >= 1"
}

& cargo build --release -p hybit --example abtm_ilu0_adaptive_rank_g4d
if ($LASTEXITCODE -ne 0) {
    throw "failed to build abtm_ilu0_adaptive_rank_g4d"
}

$exe = Join-Path $root "target\release\examples\abtm_ilu0_adaptive_rank_g4d.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "ABTM G4d executable not found: $exe"
}

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== ABTM G4d matrix: {0} ===" -f $resolved)

    & $exe `
        --matrix $resolved `
        --repeats $Repeats `
        --thresholds $Thresholds
    if ($LASTEXITCODE -ne 0) {
        throw "ABTM G4d benchmark failed for '$resolved' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== ABTM G4d adaptive rank-LUT corpus complete ==="
