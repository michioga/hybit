param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Matrix,
    [double]$Tolerance = 1.0e-8,
    [int]$MaxIterations = 3000,
    [int]$DofsPerNode = 3,
    [int]$AggregateNodes = 1024
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

Write-Host "== HyBIT 0.7.0 two-level FEM benchmark =="
Write-Host "Matrix: $Matrix"

& cargo run --release -p hybit --example fem_twolevel_bench -- `
    --matrix $Matrix `
    --tol $Tolerance `
    --max-iters $MaxIterations `
    --dofs-per-node $DofsPerNode `
    --aggregate-nodes $AggregateNodes

if ($LASTEXITCODE -ne 0) {
    throw "two-level FEM benchmark failed with exit code $LASTEXITCODE"
}
