param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Matrix,

    [string]$Coordinates,
    [string]$Rhs,

    [int[]]$TargetCoarseDimensions = @(1536, 3072),

    [double]$Tolerance = 1e-8,
    [int]$MaxIterations = 3000,
    [int]$KernelRepeats = 20,

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
if ($TargetCoarseDimensions.Count -eq 0) { throw "TargetCoarseDimensions may not be empty" }
if ($KernelRepeats -le 0) { throw "KernelRepeats must be > 0" }

Write-Host "== HyBIT 0.6.0 structural kernel profile =="
Write-Host "Matrix: $Matrix"
Write-Host "Coordinates: $Coordinates"
if (-not [string]::IsNullOrWhiteSpace($Rhs)) { Write-Host "RHS: $Rhs" }
Write-Host "Aggregation: $Aggregation"
Write-Host "Targets: $($TargetCoarseDimensions -join ', ')"
Write-Host "Kernel repeats: $KernelRepeats"

foreach ($target in $TargetCoarseDimensions) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host " target coarse dimension = $target"
    Write-Host "============================================================"

    $cargoArgs = @(
        "run", "--release", "-p", "hybit", "--example", "fem_structural_profile", "--",
        "--matrix", $Matrix,
        "--coords", $Coordinates,
        "--tol", $Tolerance.ToString("R", [System.Globalization.CultureInfo]::InvariantCulture),
        "--max-iters", $MaxIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--target-coarse-dim", $target.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--aggregation", $Aggregation,
        "--kernel-repeats", $KernelRepeats.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    )
    if (-not [string]::IsNullOrWhiteSpace($Rhs)) { $cargoArgs += @("--rhs", $Rhs) }

    & cargo @cargoArgs
    if ($LASTEXITCODE -ne 0) {
        throw "structural kernel profile failed at target $target with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "== structural kernel profile complete =="
