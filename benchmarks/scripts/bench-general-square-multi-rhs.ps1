[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,

    [int] $RhsCount = 5,

    [double] $Tolerance = 1.0e-8,

    [int] $MaxIterations = 5000,

    [int] $Restart = 30
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if ($Matrix.Count -eq 0) { throw "at least one -Matrix path is required" }
if ($RhsCount -le 0) { throw "-RhsCount must be > 0" }
if ($Tolerance -le 0.0 -or [double]::IsNaN($Tolerance) -or [double]::IsInfinity($Tolerance)) {
    throw "-Tolerance must be finite and > 0"
}
if ($MaxIterations -le 0) { throw "-MaxIterations must be > 0" }
if ($Restart -le 0) { throw "-Restart must be > 0" }

Write-Host "=== HyBIT GeneralSquare prepared multi-RHS benchmark ==="
Write-Host ("branch    : {0}" -f (git branch --show-current))
Write-Host ("HEAD      : {0}" -f (git rev-parse --short HEAD))
Write-Host ("RHS count : {0}" -f $RhsCount)
Write-Host ("tol       : {0:e3}" -f $Tolerance)
Write-Host ("iters     : {0}" -f $MaxIterations)
Write-Host ("restart   : {0}" -f $Restart)

& cargo build --release -p hybit --example general_square_multi_rhs
if ($LASTEXITCODE -ne 0) {
    throw "failed to build general_square_multi_rhs"
}

$exe = Join-Path $root "target\release\examples\general_square_multi_rhs.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "benchmark executable not found: $exe"
}

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== matrix: {0} ===" -f $resolved)

    & $exe `
        --matrix $resolved `
        --rhs-count $RhsCount `
        --tol $Tolerance.ToString("R", [Globalization.CultureInfo]::InvariantCulture) `
        --max-iters $MaxIterations `
        --restart $Restart

    if ($LASTEXITCODE -ne 0) {
        throw "multi-RHS benchmark failed for '$resolved' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== prepared multi-RHS benchmark complete ==="
