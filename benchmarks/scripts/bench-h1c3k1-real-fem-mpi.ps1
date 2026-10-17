# H1-C3K1: actual native MPI-PCG reference for existing Matrix Market FEM input.
# Root-only Matrix Market ingestion, owned-only CSR, Equal partition and Jacobi.
# Neither successful numerical convergence nor this launcher certifies global SPD.
[CmdletBinding()]
param(
    [string]$RepoRoot = 'D:\Work\hybit\hybit',
    [Parameter(Mandatory = $true)][string]$Matrix,
    [Parameter(Mandatory = $true)][string]$Label,
    [ValidateSet('Certified','ExperimentalUnverified')][string]$Policy = 'Certified',
    [switch]$PermitUnverifiedSpd,
    [ValidateRange(1,20)][int]$LaunchRepeats = 3,
    [ValidateRange(0,100)][int]$Warmups = 3,
    [ValidateRange(1,100)][int]$InnerRepeats = 5,
    [ValidateRange(1,100000)][int]$MaxIterations = 5000,
    [double]$RelativeTolerance = 1e-9,
    [ValidateRange(1,1000000)][long]$MaxDofs = 250000,
    [ValidateRange(1,20000000)][long]$MaxEstimatedNnz = 12000000,
    [string]$OutputDirectory = 'D:\Work\h1c3k1_results',
    [switch]$PreflightOnly,
    [switch]$Run
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ([bool]$PreflightOnly -eq [bool]$Run) { throw 'Specify exactly one of -PreflightOnly and -Run' }
function Require([bool]$ok, [string]$why) { if (-not $ok) { throw $why } }
function Checked([string]$name) { if ($LASTEXITCODE -ne 0) { throw "$name failed (exit=$LASTEXITCODE)" } }
function Parse-Number([string]$value) {
    return [double]::Parse($value, [Globalization.CultureInfo]::InvariantCulture)
}
function Inspect-Matrix([string]$path) {
    $reader = [IO.File]::OpenText($path)
    try {
        $header = $reader.ReadLine()
        Require ($null -ne $header) 'Empty Matrix Market input'
        $parts = @($header.Trim().Split([char[]]@(' ',"`t"), [StringSplitOptions]::RemoveEmptyEntries))
        Require ($parts.Count -eq 5) 'Invalid Matrix Market banner'
        Require ($parts[0] -ieq '%%MatrixMarket' -and $parts[1] -ieq 'matrix' -and $parts[2] -ieq 'coordinate') 'Matrix must use Matrix Market coordinate format'
        Require ($parts[3] -ieq 'real' -or $parts[3] -ieq 'integer') 'Only real/integer data supported'
        Require ($parts[4] -ieq 'symmetric' -or $parts[4] -ieq 'general') 'Only general/symmetric storage supported'
        $dims = $null
        while ($null -eq $dims) {
            $line = $reader.ReadLine()
            Require ($null -ne $line) 'Matrix Market size line missing'
            $line = $line.Trim()
            if ($line -ne '' -and -not $line.StartsWith('%')) { $dims = $line }
        }
        $fields = @($dims.Split([char[]]@(' ',"`t"), [StringSplitOptions]::RemoveEmptyEntries))
        Require ($fields.Count -eq 3) 'Matrix Market dimensions malformed'
        $n = [long]::Parse($fields[0], [Globalization.CultureInfo]::InvariantCulture)
        $m = [long]::Parse($fields[1], [Globalization.CultureInfo]::InvariantCulture)
        $entries = [long]::Parse($fields[2], [Globalization.CultureInfo]::InvariantCulture)
        Require ($n -ge 8 -and $n -eq $m -and $entries -gt 0) 'Matrix must be square and have at least 8 DOFs for the 8-rank gate'
        $worst = if ($parts[4] -ieq 'symmetric') { [long]($entries * 2) } else { $entries }
        Require ($n -le $MaxDofs) "n=$n exceeds -MaxDofs=$MaxDofs"
        Require ($worst -le $MaxEstimatedNnz) "Upper-bound nnz=$worst exceeds -MaxEstimatedNnz=$MaxEstimatedNnz"
        return [pscustomobject]@{ n=$n; declared_nnz=$entries; expanded_nnz_upper_bound=$worst; format=$parts[4] }
    } finally { $reader.Dispose() }
}
Write-Host '=== HYBIT 0.9 H1-C3K1 / REAL FEM MPI-PCG BASELINE ==='
$root = [IO.Path]::GetFullPath($RepoRoot)
Require (Test-Path -LiteralPath (Join-Path $root '.git') -PathType Container) 'Invalid HyBIT repo root'
$branch = (& git -C $root branch --show-current).Trim(); Checked 'git branch'
$head = (& git -C $root rev-parse HEAD).Trim(); Checked 'git HEAD'
Require ($branch -ceq 'develop/0.9.0') "Expected develop/0.9.0, got $branch"
Require ($head.StartsWith('b28dd0c')) "Expected H1-C3J HEAD b28dd0c; got $head"
Require ($Label -cmatch '^[A-Za-z0-9_-]{1,48}$') 'Label must use 1..48 ASCII letters, digits, underscores or dashes'
Require ((Test-Path -LiteralPath $Matrix -PathType Leaf)) "Missing matrix $Matrix"
Require ([double]::IsFinite($RelativeTolerance) -and $RelativeTolerance -ge 1e-13 -and $RelativeTolerance -le 1e-8) 'RelativeTolerance must be finite and in 1e-13..1e-8'
$tolText = $RelativeTolerance.ToString('G17', [Globalization.CultureInfo]::InvariantCulture)
Require (($Policy -ne 'ExperimentalUnverified') -or $PermitUnverifiedSpd) 'Explicit -PermitUnverifiedSpd required; PCG acceptance does not certify SPD'
$matrixPath = [IO.Path]::GetFullPath($Matrix)
$dim = Inspect-Matrix $matrixPath
$baseline = Join-Path $root 'benchmarks/data/cube8_identity.mtx'
Require (Test-Path -LiteralPath $baseline -PathType Leaf) 'Missing certified control matrix cube8_identity.mtx'
$exePath = Join-Path $root 'target/release/examples/mpi_mtx_profile.exe'
$policyArg = if ($Policy -eq 'Certified') { 'certified' } else { 'experimental' }
Write-Host "branch=$branch HEAD=$head matrix=$matrixPath policy=$Policy relative_tolerance=$tolText"
Write-Host "input n=$($dim.n) declared_nnz=$($dim.declared_nnz) expanded_upper_bound=$($dim.expanded_nnz_upper_bound)"
Write-Host 'Scope: Equal contiguous ownership, owned CSR, Jacobi MPI-PCG; NOT ABTM/Graph selection'
Write-Host 'Memory: peak RSS is NOT measured by this baseline (separate K3 gate).'
& pwsh -NoProfile -File (Join-Path $root 'tools/gates/source-integrity-gate.ps1'); Checked 'source integrity'
if ($PreflightOnly) {
    Write-Host '=== H1-C3K1 PREFLIGHT PASS: NO BUILD / MPI LAUNCH ==='
    return
}
Push-Location $root
try {
    & cargo build --release --locked -p hybit --features mpi --example mpi_mtx_profile
    Checked 'release MPI benchmark build'
} finally { Pop-Location }
Require (Test-Path -LiteralPath $exePath -PathType Leaf) 'Missing built MPI profile executable'
[IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$csvSummary = Join-Path $OutputDirectory "k1_summary_$stamp.csv"
$log = Join-Path $OutputDirectory "k1_runs_$stamp.log"
$rows = [Collections.Generic.List[object]]::new()
$failed = 0
function Invoke-Case([int]$ranks,[int]$rep,[string]$path,[string]$caseName,[string]$policy,[int]$warm,[int]$inner,[int]$iters) {
    $csv = Join-Path $OutputDirectory "k1_$($stamp)_$($caseName)_r$($ranks)_rep$($rep).csv"
    Require (-not (Test-Path -LiteralPath $csv)) 'Refusing to append to existing run CSV'
    "=== case=$caseName ranks=$ranks launch=$rep policy=$policy ===" | Tee-Object -FilePath $log -Append
    $clock = [Diagnostics.Stopwatch]::StartNew()
    & mpiexec -n $ranks $exePath $ranks $csv $path $caseName $policy $warm $inner $iters $tolText 2>&1 |
        Tee-Object -FilePath $log -Append
    $exit = $LASTEXITCODE
    $clock.Stop()
    if ($exit -ne 0) { throw "MPI failed case=$caseName ranks=$ranks launch=$rep exit=$exit; see $log" }
    Require (Test-Path -LiteralPath $csv -PathType Leaf) "No result CSV from case=$caseName ranks=$ranks"
    $result = @(Import-Csv -LiteralPath $csv)
    Require ($result.Count -eq 1) "Expected 1 CSV result, received $($result.Count)"
    $r = $result[0]
    $residual = Parse-Number $r.true_relative_residual
    $error = Parse-Number $r.max_abs_error
    $pass = ($r.status -ceq 'Converged') -and [double]::IsFinite($residual) -and
        ($residual -le (1.1 * $RelativeTolerance)) -and [double]::IsFinite($error) -and ($error -le 2.0e-7)
    $entry = [pscustomobject][ordered]@{
        case=$caseName; policy=$policy; relative_tolerance=$tolText; ranks=$ranks; launch=$rep;
        n=$r.n; nnz=$r.nnz; iterations=$r.iterations; status=$r.status;
        numeric_pass=$pass; spmv_calls=$r.spmv_calls; allreduce_calls=$r.allreduce_calls;
        true_relative_residual=$r.true_relative_residual; max_abs_error=$r.max_abs_error;
        global_ghosts=$r.global_ghosts; root_read_ms=$r.root_read_ms;
        max_distribute_ms=$r.max_distribute_ms; max_halo_ms=$r.max_halo_ms;
        max_pcg_prepare_ms=$r.max_pcg_prepare_ms; median_max_solve_ms=$r.median_max_solve_ms;
        min_max_solve_ms=$r.min_max_solve_ms; max_max_solve_ms=$r.max_max_solve_ms;
        median_max_over_mean=$r.median_max_over_mean;
        process_wall_ms=[math]::Round($clock.Elapsed.TotalMilliseconds, 6)
    }
    $rows.Add($entry)
    $verdict = if ($pass) { 'NUMERICAL_PASS' } else { 'NUMERICAL_FAIL' }
    Write-Host "K1 $verdict case=$caseName ranks=$ranks launch=$rep rtol=$tolText iters=$($r.iterations) true_rel=$residual max_error=$error solve_median_ms=$($r.median_max_solve_ms) process_ms=$($entry.process_wall_ms)"
    if (-not $pass) { $script:failed++ }
}
# Certified control checks MPI transport and PCG at every requested rank count.
foreach ($ranks in @(1,2,4,8)) {
    Invoke-Case $ranks 1 $baseline 'certified_control' 'certified' 1 2 100
}
# Genuine supplied Matrix Market data: each trial gets a fresh MPI process.
foreach ($ranks in @(1,2,4,8)) {
    foreach ($repeat in 1..$LaunchRepeats) {
        Invoke-Case $ranks $repeat $matrixPath $Label $policyArg $Warmups $InnerRepeats $MaxIterations
    }
}
$rows | Export-Csv -LiteralPath $csvSummary -NoTypeInformation -Encoding utf8
Write-Host "Summary: $csvSummary"
Write-Host "Full log: $log"
Write-Host "Numerical failures: $failed / $($rows.Count)"
if ($failed -ne 0) { throw "H1-C3K1 numerical gate FAIL ($failed cases); do not treat as successful SPD solve" }
Write-Host '=== H1-C3K1 MPI-PCG BASELINE NUMERICAL PASS ==='
if ($Policy -eq 'ExperimentalUnverified') {
    Write-Warning 'SPD remains UNVERIFIED; convergence and manufactured-solution error are numerical observations only.'
}
