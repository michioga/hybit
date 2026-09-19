param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Matrix,

    [string]$Coordinates,
    [string]$Rhs,

    [int[]]$Threads = @(2, 4, 8, 16),
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
if (-not [string]::IsNullOrWhiteSpace($Rhs) -and -not (Test-Path -LiteralPath $Rhs)) {
    throw "RHS file not found: $Rhs"
}
if ($Threads.Count -eq 0) { throw "Threads may not be empty" }
if ($Threads | Where-Object { $_ -le 0 }) { throw "all thread counts must be > 0" }

Write-Host "== HyBIT 0.6.0 production vs parallel/fused PCG vector benchmark =="
Write-Host "Matrix: $Matrix"
Write-Host "Coordinates: $Coordinates"
if (-not [string]::IsNullOrWhiteSpace($Rhs)) { Write-Host "RHS: $Rhs" }
Write-Host "Aggregation: $Aggregation"
Write-Host "Target coarse dimension: $TargetCoarseDimension"
Write-Host "Rayon thread sweep: $($Threads -join ', ')"

$oldRayonThreads = $env:RAYON_NUM_THREADS
try {
    foreach ($threadCount in $Threads) {
        Write-Host ""
        Write-Host "============================================================"
        Write-Host " RAYON_NUM_THREADS = $threadCount"
        Write-Host "============================================================"
        $env:RAYON_NUM_THREADS = $threadCount.ToString([System.Globalization.CultureInfo]::InvariantCulture)

        $cargoArgs = @(
            "run", "--release", "-p", "hybit", "--example", "fem_structural_pcg_parallel", "--",
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
            throw "parallel PCG vector benchmark failed for $threadCount threads with exit code $LASTEXITCODE"
        }
    }
}
finally {
    $env:RAYON_NUM_THREADS = $oldRayonThreads
}

Write-Host ""
Write-Host "== PCG vector-kernel thread sweep complete =="
