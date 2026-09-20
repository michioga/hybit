param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Matrix,
    [string]$Coordinates,
    [string]$Rhs,
    [int]$TargetCoarseDimension = 1536,
    [ValidateSet("auto", "contiguous", "graph")]
    [string]$Aggregation = "auto",
    [ValidateSet("auto", "serial", "parallel")]
    [string]$Spmv = "auto",
    [ValidateSet("auto", "serial", "parallel")]
    [string]$Preconditioner = "auto",
    [int]$RayonThreads = 0,
    [double]$Tolerance = 1e-8,
    [int]$MaxIterations = 3000
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $Matrix)) { throw "matrix file not found: $Matrix" }
if ([string]::IsNullOrWhiteSpace($Coordinates)) {
    $Coordinates = [System.IO.Path]::ChangeExtension($Matrix, ".coords")
}
if (-not (Test-Path $Coordinates)) { throw "coordinate sidecar not found: $Coordinates" }
if (-not [string]::IsNullOrWhiteSpace($Rhs) -and -not (Test-Path $Rhs)) { throw "RHS file not found: $Rhs" }
if ($TargetCoarseDimension -lt 6) { throw "TargetCoarseDimension must be >= 6" }
if ($RayonThreads -lt 0) { throw "RayonThreads must be >= 0 (0 keeps the environment/default pool)" }

Write-Host "== HyBIT 0.6.0 structural-auto FEM benchmark =="
Write-Host "Matrix: $Matrix"
Write-Host "Coordinates: $Coordinates"
if (-not [string]::IsNullOrWhiteSpace($Rhs)) { Write-Host "RHS: $Rhs" }
Write-Host "Target coarse dimension: $TargetCoarseDimension"
Write-Host "Aggregation: $Aggregation"
Write-Host "SpMV policy: $Spmv"
Write-Host "Preconditioner policy: $Preconditioner"
if ($RayonThreads -gt 0) { Write-Host "Rayon threads: $RayonThreads" }

$argsList = @(
    "run", "--release", "-p", "hybit", "--example", "fem_structural_auto", "--",
    "--matrix", $Matrix,
    "--coords", $Coordinates,
    "--tol", $Tolerance,
    "--max-iters", $MaxIterations,
    "--target-coarse-dim", $TargetCoarseDimension,
    "--aggregation", $Aggregation,
    "--spmv", $Spmv,
    "--precond", $Preconditioner
)
if (-not [string]::IsNullOrWhiteSpace($Rhs)) {
    $argsList += @("--rhs", $Rhs)
}
$oldRayonThreads = $env:RAYON_NUM_THREADS
try {
    if ($RayonThreads -gt 0) {
        $env:RAYON_NUM_THREADS = $RayonThreads.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    }
    & cargo @argsList
    if ($LASTEXITCODE -ne 0) { throw "structural-auto FEM benchmark failed with exit code $LASTEXITCODE" }
}
finally {
    $env:RAYON_NUM_THREADS = $oldRayonThreads
}
