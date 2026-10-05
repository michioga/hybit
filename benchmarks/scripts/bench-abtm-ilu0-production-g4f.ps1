[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,
    [int] $Repeats = 5
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

& cargo build --release -p hybit --example abtm_ilu0_production_g4f
if ($LASTEXITCODE -ne 0) {
    throw "failed to build abtm_ilu0_production_g4f"
}

$exe = Join-Path $root "target\release\examples\abtm_ilu0_production_g4f.exe"

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== ABTM G4f matrix: {0} ===" -f $resolved)

    & $exe --matrix $resolved --repeats $Repeats
    if ($LASTEXITCODE -ne 0) {
        throw "ABTM G4f benchmark failed for '$resolved' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== ABTM G4f production corpus complete ==="
