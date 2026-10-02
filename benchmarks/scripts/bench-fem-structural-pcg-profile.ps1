param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Matrix,

    [Parameter(Mandatory = $true)]
    [string]$Coordinates,

    [Parameter(Mandatory = $true)]
    [string]$Rhs,

    [int]$TargetCoarseDimension = 1536,
    [ValidateSet("graph", "contiguous")]
    [string]$Aggregation = "graph",
    [int]$RayonThreads = 8,
    [double]$Tolerance = 1.0e-8,
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
if ($RayonThreads -lt 1) { throw "RayonThreads must be >= 1" }
if ($TargetCoarseDimension -lt 6) { throw "TargetCoarseDimension must be >= 6" }
if ($MaxIterations -lt 1) { throw "MaxIterations must be >= 1" }
if ($Tolerance -le 0) { throw "Tolerance must be > 0" }

Write-Host "== HyBIT 0.7.0 PCG vector-kernel profile =="
Write-Host "Matrix: $Matrix"
Write-Host "Coordinates: $Coordinates"
Write-Host "RHS: $Rhs"
Write-Host "Aggregation: $Aggregation"
Write-Host "Target coarse dimension: $TargetCoarseDimension"
Write-Host "Rayon threads: $RayonThreads"

$oldRayon = $env:RAYON_NUM_THREADS
try {
    $env:RAYON_NUM_THREADS = "$RayonThreads"
    cargo run --release -p hybit --example fem_structural_pcg_profile -- `
        --matrix $Matrix `
        --coords $Coordinates `
        --rhs $Rhs `
        --tol $Tolerance `
        --max-iters $MaxIterations `
        --target-coarse-dim $TargetCoarseDimension `
        --aggregation $Aggregation
    if ($LASTEXITCODE -ne 0) {
        throw "PCG vector-kernel profile failed with exit code $LASTEXITCODE"
    }
}
finally {
    if ($null -eq $oldRayon) {
        Remove-Item Env:RAYON_NUM_THREADS -ErrorAction SilentlyContinue
    }
    else {
        $env:RAYON_NUM_THREADS = $oldRayon
    }
}
