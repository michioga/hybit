param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Matrix,

    [string]$Coordinates,
    [string]$Rhs,

    [int[]]$TargetCoarseDimensions = @(384, 768, 1536, 3072),

    [double]$Tolerance = 1e-8,
    [int]$MaxIterations = 3000,

    [ValidateSet("auto", "graph", "contiguous")]
    [string]$Aggregation = "graph"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Matrix)) {
    throw "Matrix file not found: $Matrix"
}

if ([string]::IsNullOrWhiteSpace($Coordinates)) {
    $Coordinates = [System.IO.Path]::ChangeExtension($Matrix, ".coords")
}
if (-not (Test-Path -LiteralPath $Coordinates)) {
    throw "Coordinate sidecar not found: $Coordinates"
}
if (-not [string]::IsNullOrWhiteSpace($Rhs) -and -not (Test-Path -LiteralPath $Rhs)) {
    throw "RHS file not found: $Rhs"
}
if ($TargetCoarseDimensions.Count -eq 0) {
    throw "TargetCoarseDimensions may not be empty"
}
foreach ($target in $TargetCoarseDimensions) {
    if ($target -lt 6) {
        throw "Every target coarse dimension must be >= 6 (got $target)"
    }
}
if (-not [double]::IsFinite($Tolerance) -or $Tolerance -le 0.0) {
    throw "Tolerance must be finite and > 0"
}
if ($MaxIterations -le 0) {
    throw "MaxIterations must be > 0"
}

Write-Host "== HyBIT 0.6.0 structural Graph coarse-dimension sweep =="
Write-Host "Matrix: $Matrix"
Write-Host "Coordinates: $Coordinates"
if (-not [string]::IsNullOrWhiteSpace($Rhs)) {
    Write-Host "RHS: $Rhs"
} else {
    Write-Host "RHS: generated as b=A*1"
}
Write-Host "Aggregation: $Aggregation"
Write-Host "Targets: $($TargetCoarseDimensions -join ', ')"
Write-Host "Tolerance: $Tolerance"
Write-Host "Max iterations: $MaxIterations"

foreach ($target in $TargetCoarseDimensions) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host " target coarse dimension = $target"
    Write-Host "============================================================"

    $cargoArgs = @(
        "run", "--release", "-p", "hybit", "--example", "fem_structural_auto", "--",
        "--matrix", $Matrix,
        "--coords", $Coordinates,
        "--tol", $Tolerance.ToString("R", [System.Globalization.CultureInfo]::InvariantCulture),
        "--max-iters", $MaxIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--target-coarse-dim", $target.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--aggregation", $Aggregation
    )

    if (-not [string]::IsNullOrWhiteSpace($Rhs)) {
        $cargoArgs += @("--rhs", $Rhs)
    }

    & cargo @cargoArgs
    if ($LASTEXITCODE -ne 0) {
        throw "coarse-dimension sweep failed at target $target with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "== coarse-dimension sweep complete =="
