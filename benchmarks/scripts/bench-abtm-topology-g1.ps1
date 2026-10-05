[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if ($Matrix.Count -eq 0) { throw "at least one -Matrix path is required" }

& cargo build --release -p hybit --example abtm_topology_g1
if ($LASTEXITCODE -ne 0) { throw "failed to build abtm_topology_g1" }

$exe = Join-Path $root "target\release\examples\abtm_topology_g1.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "ABTM G1 executable not found: $exe" }

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== ABTM G1 matrix: {0} ===" -f $resolved)
    & $exe --matrix $resolved
    if ($LASTEXITCODE -ne 0) { throw "ABTM G1 profile failed for '$resolved' with exit code $LASTEXITCODE" }
}

Write-Host ""
Write-Host "=== ABTM G1 scalar topology corpus complete ==="
