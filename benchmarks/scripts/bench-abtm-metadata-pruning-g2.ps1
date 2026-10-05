[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,
    [int] $Repeats = 5
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

& cargo build --release -p hybit --example abtm_metadata_pruning_g2
if ($LASTEXITCODE -ne 0) {
    throw "failed to build abtm_metadata_pruning_g2"
}

$exe = Join-Path $root "target\release\examples\abtm_metadata_pruning_g2.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "ABTM G2 executable not found: $exe"
}

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== ABTM G2 matrix: {0} ===" -f $resolved)

    & $exe --matrix $resolved --repeats $Repeats
    if ($LASTEXITCODE -ne 0) {
        throw "ABTM G2 benchmark failed for '$resolved' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== ABTM G2 metadata-first pruning corpus complete ==="
