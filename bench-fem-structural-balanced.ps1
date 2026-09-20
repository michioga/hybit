param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Matrix,

    [string]$Coordinates,
    [string]$Rhs,

    [int]$TargetCoarseDimension = 1536,
    [double]$Tolerance = 1e-8,
    [int]$MaxIterations = 3000,

    [ValidateSet("graph", "contiguous")]
    [string]$Aggregation = "graph"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Matrix)) { throw "Matrix file not found: $Matrix" }
if ([string]::IsNullOrWhiteSpace($Coordinates)) {
    $Coordinates = [System.IO.Path]::ChangeExtension($Matrix, ".coords")
}
if (-not (Test-Path -LiteralPath $Coordinates)) { throw "Coordinate sidecar not found: $Coordinates" }
if (-not [string]::IsNullOrWhiteSpace($Rhs) -and -not (Test-Path -LiteralPath $Rhs)) { throw "RHS file not found: $Rhs" }
if ($TargetCoarseDimension -lt 6) { throw "TargetCoarseDimension must be >= 6" }
if (-not [double]::IsFinite($Tolerance) -or $Tolerance -le 0.0) { throw "Tolerance must be finite and > 0" }
if ($MaxIterations -le 0) { throw "MaxIterations must be > 0" }

Write-Host "== HyBIT 0.6.0 additive vs balanced structural benchmark =="
Write-Host "Matrix: $Matrix"
Write-Host "Coordinates: $Coordinates"
if (-not [string]::IsNullOrWhiteSpace($Rhs)) { Write-Host "RHS: $Rhs" }
else { Write-Host "RHS: generated as b=A*1" }
Write-Host "Aggregation: $Aggregation"
Write-Host "Target coarse dimension: $TargetCoarseDimension"

$cargoArgs = @(
    "run", "--release", "-p", "hybit", "--example", "fem_structural_balance", "--",
    "--matrix", $Matrix,
    "--coords", $Coordinates,
    "--tol", $Tolerance.ToString("R", [System.Globalization.CultureInfo]::InvariantCulture),
    "--max-iters", $MaxIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--target-coarse-dim", $TargetCoarseDimension.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--aggregation", $Aggregation
)
if (-not [string]::IsNullOrWhiteSpace($Rhs)) { $cargoArgs += @("--rhs", $Rhs) }

& cargo @cargoArgs
if ($LASTEXITCODE -ne 0) {
    throw "additive-vs-balanced structural benchmark failed with exit code $LASTEXITCODE"
}
