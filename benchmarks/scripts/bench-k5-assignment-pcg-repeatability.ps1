# H1-C3K5: independent process-launch replication for one-node native MS-MPI.
# Does not build or modify the repository. It writes separate summary/raw CSVs.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$Matrix,
    [string]$OutputDir = 'D:\Work',
    [string]$Prefix = 'k5_thermal1_spd',
    [int[]]$Ranks = @(4,8),
    [ValidateRange(2,10)][int]$Launches = 5,
    [ValidateRange(0,10)][int]$Warmups = 2,
    [ValidateRange(1,30)][int]$Repeats = 5,
    [ValidateRange(1,5000)][int]$MaxIters = 3000,
    [ValidateRange(1,256)][int]$Radius = 16,
    [double]$Tol = 1e-10,
    [double]$Shift = 0.01
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = [IO.Path]::GetFullPath($RepoRoot)
if ($Prefix -notmatch '^[A-Za-z0-9_-]{1,70}$') { throw 'Prefix must be a simple filename stem' }
if ($Ranks.Count -lt 1 -or @($Ranks | Where-Object { $_ -notin @(1,2,4,8) }).Count -gt 0) { throw 'Ranks must contain only 1, 2, 4 or 8' }
if ($Tol -lt 1e-14 -or $Tol -gt 1e-3 -or $Shift -lt 1e-5 -or $Shift -gt 10) { throw 'Tolerance or SPD shift outside K5 guard' }
if (-not (Test-Path -LiteralPath $Matrix -PathType Leaf)) { throw "Matrix not found: $Matrix" }
if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) { throw "Output directory not found: $OutputDir" }
$branch = (& git -C $root branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne 'develop/0.9.0') { throw 'Expected develop/0.9.0 branch' }
$envScript = Join-Path $root 'tools/env/Enter-HybitMsMpi.ps1'
if (-not (Test-Path -LiteralPath $envScript -PathType Leaf)) { throw 'MS-MPI env script missing' }
# The helper is PowerShell and MUST run in-process to propagate VS/MS-MPI env vars.
& { Set-StrictMode -Off; . $envScript }
$mpi = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
$exe = Join-Path $root 'target/release/examples/mpi_k5_assignment_pcg_profile.exe'
if (-not (Test-Path -LiteralPath $mpi -PathType Leaf)) { throw "MS-MPI launcher not found: $mpi" }
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "K5 release executable missing: $exe" }
$aggregateFile = Join-Path $OutputDir ($Prefix + '_independent_launches.csv')
if (Test-Path -LiteralPath $aggregateFile) { throw "Aggregate CSV already exists: $aggregateFile" }
$culture = [Globalization.CultureInfo]::InvariantCulture
function Parse-F64([string]$s) {
    [double]::Parse($s, [Globalization.NumberStyles]::Float, $culture)
}
$records = [System.Collections.Generic.List[object]]::new()
foreach ($rank in $Ranks) {
    for ($run = 1; $run -le $Launches; $run++) {
        $stem = ('{0}_r{1}_run{2:d2}' -f $Prefix, $rank, $run)
        $summaryFile = Join-Path $OutputDir ($stem + '.csv')
        $rawFile = Join-Path $OutputDir ($stem + '.samples.csv')
        if ((Test-Path -LiteralPath $summaryFile) -or (Test-Path -LiteralPath $rawFile)) {
            throw "Refuse to overwrite existing run output for $stem"
        }
        Write-Host "=== K5 MS-MPI independent run $run/$Launches, $rank ranks ==="
        & $mpi -n $rank $exe $rank $Matrix $Radius $Warmups $Repeats $MaxIters $Tol $Shift $summaryFile
        if ($LASTEXITCODE -ne 0) { throw "K5 MPI benchmark failed for $stem (exit $LASTEXITCODE)" }
        if (-not (Test-Path -LiteralPath $summaryFile) -or -not (Test-Path -LiteralPath $rawFile)) {
            throw "K5 summary or per-trial samples missing for $stem"
        }
        $rows = @(Import-Csv -LiteralPath $summaryFile)
        if ($rows.Count -ne 5) { throw "Expected five policies in $summaryFile" }
        $baseRows = @($rows | Where-Object { $_.method -eq 'equal' })
        if ($baseRows.Count -ne 1) { throw 'Equal baseline must occur exactly once per independent run' }
        $baseline = Parse-F64 $baseRows[0].median_max_solve_ms
        if ($baseline -le 0) { throw 'Invalid Equal baseline' }
        $iterations = @($rows | Select-Object -ExpandProperty iterations | Sort-Object -Unique)
        if ($iterations.Count -ne 1) { throw "Inconsistent PCG iteration counts in $stem" }
        foreach ($row in $rows) {
            $solve = Parse-F64 $row.median_max_solve_ms
            $dot = Parse-F64 $row.median_max_dot_allreduce_sum_ms
            $exchange = Parse-F64 $row.median_max_exchange_sum_ms
            $rel = Parse-F64 $row.true_rel_residual
            if ($solve -le 0 -or $dot -lt 0 -or $exchange -lt 0 -or $rel -gt $Tol) {
                throw "Invalid PCG numerical/performance result for $stem/$($row.method)"
            }
            $records.Add([pscustomobject]@{
                scope = 'synthetic_spd_single_node_mpi_experiment'
                ranks = $rank
                independent_run = $run
                method = $row.method
                iterations = [int]$row.iterations
                true_rel_residual = $row.true_rel_residual
                solve_median_ms = $row.median_max_solve_ms
                halo_exchange_median_ms = $row.median_max_exchange_sum_ms
                dot_allreduce_median_ms = $row.median_max_dot_allreduce_sum_ms
                dot_local_median_ms = $row.median_max_dot_local_sum_ms
                partition_ms = $row.partition_max_ms
                owned_extract_ms = $row.owned_extract_max_ms
                halo_setup_ms = $row.halo_setup_max_ms
                rhs_setup_ms = $row.rhs_setup_max_ms
                paired_equal_improvement_pct = [math]::Round(100.0 * ($baseline - $solve) / $baseline, 5)
                summary_csv = $summaryFile
                raw_csv = $rawFile
            })
        }
    }
}
$records | Export-Csv -LiteralPath $aggregateFile -NoTypeInformation -Encoding utf8
Write-Host "=== K5 INDEPENDENT PROCESS RUNS PASS: $($records.Count) policy records, $aggregateFile ==="
Write-Host 'NOTE: one physical node only; full CSR is replicated on every rank.'
