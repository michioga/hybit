[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if ($Matrix.Count -eq 0) {
    throw "at least one -Matrix path is required"
}

& cargo build --release -p hybit --example general_square_preconditioner_probe
if ($LASTEXITCODE -ne 0) {
    throw "failed to build general_square_preconditioner_probe"
}

$exe = Join-Path $root "target\release\examples\general_square_preconditioner_probe.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "F7b executable not found: $exe"
}

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== F7b matrix: {0} ===" -f $resolved)

    & $exe --matrix $resolved
    if ($LASTEXITCODE -ne 0) {
        throw "F7b probe failed for '$resolved' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== GeneralSquare F7b preconditioner quality/cost probe complete ==="
