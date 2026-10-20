# Single physical node, native MS-MPI only. Do not claim multi-node performance.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$Matrix,
    [int[]]$Ranks = @(4, 8),
    [string]$OutputDir = 'D:\Work',
    [string]$Prefix = 'k5b_thermal1_spd'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = [IO.Path]::GetFullPath($RepoRoot)
$matrixPath = [IO.Path]::GetFullPath($Matrix)
if (-not (Test-Path -LiteralPath $matrixPath -PathType Leaf)) { throw "Matrix not found: $matrixPath" }
if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) { throw "Output directory not found: $OutputDir" }
if ($Ranks.Count -eq 0) { throw 'At least one rank count is required' }
foreach ($r in $Ranks) { if ($r -notin @(1,2,4,8)) { throw "Unsupported single-node rank count: $r" } }
$envScript = Join-Path $root 'tools\env\Enter-HybitMsMpi.ps1'
if (-not (Test-Path -LiteralPath $envScript -PathType Leaf)) { throw 'MS-MPI environment script not found' }
& { Set-StrictMode -Off; . $envScript }
$mpi = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
$exe = Join-Path $root 'target\release\examples\mpi_k5b_assignment_solve_many.exe'
if (-not (Test-Path -LiteralPath $mpi -PathType Leaf)) { throw 'Native MS-MPI mpiexec not found' }
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "K5-B executable not found: $exe" }
foreach ($r in $Ranks) {
    $csv = Join-Path $OutputDir ("{0}_r{1}.csv" -f $Prefix,$r)
    $raw = [IO.Path]::ChangeExtension($csv,'rhs.csv')
    if ((Test-Path -LiteralPath $csv) -or (Test-Path -LiteralPath $raw)) { throw "Refusing overwrite: $csv / $raw" }
    Write-Host "=== K5-B native MS-MPI solve-many: $r ranks, 1 physical node ==="
    & $mpi -n $r $exe $r $matrixPath 16 3000 1e-10 0.01 $csv
    if ($LASTEXITCODE -ne 0) { throw "K5-B failed: $r ranks, exit=$LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $csv -PathType Leaf) -or -not (Test-Path -LiteralPath $raw -PathType Leaf)) { throw "Missing K5-B CSV output for $r ranks" }
    $summary = @(Import-Csv -LiteralPath $csv)
    $samples = @(Import-Csv -LiteralPath $raw)
    if ($summary.Count -ne 25 -or $samples.Count -ne 320) { throw "Unexpected K5-B CSV row count: $($summary.Count) / $($samples.Count)" }
    if (@($summary | Where-Object { [int]$_.ranks -ne $r }).Count -ne 0) { throw 'Unexpected ranks in K5-B CSV' }
    if (@($summary | Where-Object { [double]$_.worst_true_rel -gt 1e-10 }).Count -ne 0) { throw 'True relative residual exceeded tolerance' }
    Write-Host "=== K5-B $r ranks PASS: 25 checkpoints, 320 RHS ==="
}
Write-Host '=== K5-B SINGLE-NODE BENCHMARK SUITE PASS ==='
