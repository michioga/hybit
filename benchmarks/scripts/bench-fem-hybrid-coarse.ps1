param(
    [Parameter(Mandatory=$true)][string]$Matrix,
    [Parameter(Mandatory=$false)][string]$Rhs,
    [int]$FactorBudgetMiB = 64,
    [int]$MaxIterations = 3000,
    [double]$Tolerance = 1.0e-8,
    [int]$ProbeIterations = 12,
    [int]$MaxRegion = 128,
    [int]$MaxRegions = 8,
    [int]$Overlap = 1,
    [int]$MaxEscalations = 3,
    [int]$StageIterations = 24,
    [int]$CoarseDofs = 3,
    [int]$CoarseTarget = 1536,
    [ValidateSet('auto','csr','abtm')][string]$Backend = 'auto'
)

$ErrorActionPreference = 'Stop'
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

$common = @(
    '--matrix', $Matrix,
    '--tol', $Tolerance,
    '--max-iters', $MaxIterations,
    '--backend', $Backend,
    '--probe-iters', $ProbeIterations,
    '--overlap', $Overlap,
    '--max-region', $MaxRegion,
    '--max-regions', $MaxRegions,
    '--factor-budget-mib', $FactorBudgetMiB,
    '--max-escalations', $MaxEscalations,
    '--stage-iters', $StageIterations,
    '--selector', 'jacobi-byte'
)

if ($Rhs) {
    $common += @('--rhs', $Rhs)
}

Write-Host '=== HyBIT selective-direct only ==='
& cargo run --release -p hybit --example fem_bench -- @common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ''
Write-Host '=== HyBIT algebraic two-level + selective-direct ==='
& cargo run --release -p hybit --example fem_bench -- @common `
    --hybrid-coarse `
    --coarse-dofs $CoarseDofs `
    --coarse-target $CoarseTarget
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
