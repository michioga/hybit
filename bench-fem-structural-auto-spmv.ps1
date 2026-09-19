param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Matrix,
    [string]$Coordinates,
    [string]$Rhs,
    [int]$TargetCoarseDimension = 1536,
    [ValidateSet("auto", "contiguous", "graph")]
    [string]$Aggregation = "auto",
    [int]$RayonThreads = 4,
    [double]$Tolerance = 1e-8,
    [int]$MaxIterations = 3000
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Matrix)) { throw "Matrix file not found: $Matrix" }
if ([string]::IsNullOrWhiteSpace($Coordinates)) {
    $Coordinates = [System.IO.Path]::ChangeExtension($Matrix, ".coords")
}
if (-not (Test-Path -LiteralPath $Coordinates)) { throw "Coordinate sidecar not found: $Coordinates" }
if (-not [string]::IsNullOrWhiteSpace($Rhs) -and -not (Test-Path -LiteralPath $Rhs)) {
    throw "RHS file not found: $Rhs"
}
if ($TargetCoarseDimension -lt 6) { throw "TargetCoarseDimension must be >= 6" }
if ($RayonThreads -le 0) { throw "RayonThreads must be > 0" }

Write-Host "== HyBIT 0.6.0 integrated structural SpMV A/B benchmark =="
Write-Host "Matrix: $Matrix"
Write-Host "Coordinates: $Coordinates"
if (-not [string]::IsNullOrWhiteSpace($Rhs)) { Write-Host "RHS: $Rhs" }
Write-Host "Aggregation: $Aggregation"
Write-Host "Target coarse dimension: $TargetCoarseDimension"
Write-Host "Rayon threads for parallel run: $RayonThreads"
Write-Host "Preconditioner fixed: serial"

function Invoke-Case([string]$Spmv) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Structural SpMV = $Spmv"
    Write-Host "============================================================"

    $cargoArgs = @(
        "run", "--release", "-p", "hybit", "--example", "fem_structural_auto", "--",
        "--matrix", $Matrix,
        "--coords", $Coordinates,
        "--tol", $Tolerance.ToString("R", [System.Globalization.CultureInfo]::InvariantCulture),
        "--max-iters", $MaxIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--target-coarse-dim", $TargetCoarseDimension.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--aggregation", $Aggregation,
        "--spmv", $Spmv,
        "--precond", "serial",
        "--pcg-vectors", "serial"
    )
    if (-not [string]::IsNullOrWhiteSpace($Rhs)) { $cargoArgs += @("--rhs", $Rhs) }

    & cargo @cargoArgs
    if ($LASTEXITCODE -ne 0) {
        throw "structural SpMV '$Spmv' benchmark failed with exit code $LASTEXITCODE"
    }
}

$oldRayonThreads = $env:RAYON_NUM_THREADS
try {
    Invoke-Case "serial"
    $env:RAYON_NUM_THREADS = $RayonThreads.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    Invoke-Case "auto"
}
finally {
    $env:RAYON_NUM_THREADS = $oldRayonThreads
}

Write-Host ""
Write-Host "== integrated structural SpMV A/B benchmark complete =="
