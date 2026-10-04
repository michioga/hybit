[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,
    [int] $RhsCount = 1,
    [int] $Restart = 30,
    [int] $MaxIterations = 5000,
    [double] $Tolerance = 1.0e-8
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if ($Matrix.Count -eq 0) { throw "at least one -Matrix path is required" }
if ($RhsCount -le 0) { throw "-RhsCount must be > 0" }
if ($Restart -le 0) { throw "-Restart must be > 0" }
if ($MaxIterations -le 0) { throw "-MaxIterations must be > 0" }
if (-not [double]::IsFinite($Tolerance) -or $Tolerance -le 0.0) {
    throw "-Tolerance must be finite and > 0"
}

& cargo build --release -p hybit --example general_square_preconditioner_compare
if ($LASTEXITCODE -ne 0) {
    throw "failed to build general_square_preconditioner_compare"
}

$exe = Join-Path $root "target\release\examples\general_square_preconditioner_compare.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "F7a executable not found: $exe"
}

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== F7a matrix: {0} ===" -f $resolved)

    & $exe `
        --matrix $resolved `
        --rhs-count $RhsCount `
        --restart $Restart `
        --max-iters $MaxIterations `
        --tol $Tolerance

    if ($LASTEXITCODE -ne 0) {
        throw "F7a comparison failed for '$resolved' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== GeneralSquare F7a Jacobi-vs-ILU0 comparison complete ==="
