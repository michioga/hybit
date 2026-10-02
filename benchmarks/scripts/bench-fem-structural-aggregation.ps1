param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Matrix,
    [string]$Coordinates,
    [string]$Rhs,
    [int]$TargetCoarseDimension = 1536,
    [double]$Tolerance = 1e-8,
    [int]$MaxIterations = 3000
)

$ErrorActionPreference = "Stop"
$HybitCallerDirectory = (Get-Location).Path
function Resolve-HybitCallerPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or [IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return [IO.Path]::GetFullPath((Join-Path $HybitCallerDirectory $Path))
}
foreach ($HybitPathVariable in @(
    'Matrix', 'Coordinates', 'Rhs', 'CsvPath', 'SummaryCsvPath', 'RunsCsvPath'
)) {
    $HybitVariable = Get-Variable -Name $HybitPathVariable -ErrorAction SilentlyContinue
    if ($null -ne $HybitVariable -and
        -not [string]::IsNullOrWhiteSpace([string]$HybitVariable.Value)) {
        Set-Variable -Name $HybitPathVariable -Value (
            Resolve-HybitCallerPath ([string]$HybitVariable.Value)
        )
    }
}
$HybitRepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $HybitRepoRoot
$HybitResultsDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\results'))
New-Item -ItemType Directory -Force -Path $HybitResultsDirectory | Out-Null
if (-not (Test-Path $Matrix)) { throw "matrix file not found: $Matrix" }
if ([string]::IsNullOrWhiteSpace($Coordinates)) {
    $Coordinates = [System.IO.Path]::ChangeExtension($Matrix, ".coords")
}
if (-not (Test-Path $Coordinates)) { throw "coordinate sidecar not found: $Coordinates" }
if (-not [string]::IsNullOrWhiteSpace($Rhs) -and -not (Test-Path $Rhs)) { throw "RHS file not found: $Rhs" }
if ($TargetCoarseDimension -lt 6) { throw "TargetCoarseDimension must be >= 6" }

Write-Host "== HyBIT 0.7.0 structural aggregation A/B benchmark =="
Write-Host "Matrix: $Matrix"
Write-Host "Coordinates: $Coordinates"
if (-not [string]::IsNullOrWhiteSpace($Rhs)) { Write-Host "RHS: $Rhs" }
Write-Host "Target coarse dimension: $TargetCoarseDimension"

foreach ($aggregation in @("contiguous", "graph")) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host " aggregation = $aggregation"
    Write-Host "============================================================"

    $argsList = @(
        "run", "--release", "-p", "hybit", "--example", "fem_structural_auto", "--",
        "--matrix", $Matrix,
        "--coords", $Coordinates,
        "--tol", $Tolerance,
        "--max-iters", $MaxIterations,
        "--target-coarse-dim", $TargetCoarseDimension,
        "--aggregation", $aggregation
    )
    if (-not [string]::IsNullOrWhiteSpace($Rhs)) {
        $argsList += @("--rhs", $Rhs)
    }

    & cargo @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "structural aggregation '$aggregation' benchmark failed with exit code $LASTEXITCODE"
    }
}
