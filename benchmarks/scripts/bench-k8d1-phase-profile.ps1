[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$Matrix,
    [ValidateSet('equal','region','multisource')][string[]]$Policies=@('equal','region','multisource'),
    [int[]]$Ranks=@(1,4,8),
    [ValidateSet('jacobi','ic0','block')][string[]]$Preconditioners=@('jacobi','ic0','block'),
    [ValidateRange(2,512)][int]$BlockSize=16,
    [ValidateRange(0,3)][int]$Warmups=1,
    [ValidateRange(1,12)][int]$Repeats=3,
    [ValidateRange(1,3000)][int]$MaxIters=1500,
    [ValidateRange(1.0e-12,1.0e-3)][double]$Tolerance=1e-8,
    [string]$OutputDir='D:\Work',
    [string]$Prefix='k8d1_profile',
    [switch]$KeepCaseCsv
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if (!(Test-Path -LiteralPath $Matrix -PathType Leaf)) { throw "Matrix missing: $Matrix" }
if (!(Test-Path -LiteralPath $OutputDir -PathType Container)) { throw "OutputDir missing: $OutputDir" }
if ($Prefix -cnotmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid Prefix' }
foreach ($r in $Ranks) { if ($r -notin @(1,2,4,8)) { throw "Unsupported ranks: $r" } }
$root=[IO.Path]::GetFullPath($RepoRoot)
$exe=Join-Path $root 'target\release\examples\mpi_k8d1_phase_profile.exe'
$mpiexec='C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
if (!(Test-Path -LiteralPath $mpiexec -PathType Leaf)) { throw 'Microsoft MS-MPI missing' }
if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw "K8-D1 example not built: $exe" }
$summary=Join-Path $OutputDir "${Prefix}_summary.csv"
if (Test-Path -LiteralPath $summary) { throw "Summary exists: $summary" }
$caseDir=$OutputDir
$tempDir=$null
if (!$KeepCaseCsv) {
    $tempDir=Join-Path ([IO.Path]::GetTempPath()) ('hybit_k8d1_' + [Guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($tempDir) | Out-Null
    $caseDir=$tempDir
}
$results=[Collections.Generic.List[object]]::new()
$inv=[Globalization.CultureInfo]::InvariantCulture
try {
    Push-Location $root
    try {
        & { Set-StrictMode -Off; . .\tools\env\Enter-HybitMsMpi.ps1 }
        $tol=$Tolerance.ToString('G17',$inv)
        foreach ($r in $Ranks) { foreach ($p in $Policies) { foreach ($prec in $Preconditioners) {
            $out=Join-Path $caseDir "${Prefix}_${p}_${prec}_r${r}.csv"
            if (Test-Path -LiteralPath $out) { throw "Case CSV exists: $out" }
            Write-Host "=== K8-D1 PROFILE ranks=$r policy=$p preconditioner=$prec block=$BlockSize ==="
            & $mpiexec -n $r $exe $r $Matrix $p $Warmups $Repeats $MaxIters $tol $out $prec $BlockSize --experimental-large-spd
            if ($LASTEXITCODE -ne 0) { throw "Native MS-MPI failed ranks=$r policy=$p precond=$prec" }
            $rows=@(Import-Csv -LiteralPath $out)
            if ($rows.Count -ne 1) { throw 'Expected exactly one per-case CSV row' }
            $row=$rows[0]
            if ($row.exact_original_verified -cne 'true' -or $row.experimental_opt_in -cne 'true') {
                throw 'Root-original CSR verification or experimental opt-in missing'
            }
            if ($row.spd_certificate -cne 'experimental_symmetric_positive_diagonal_curvature_screen') {
                throw 'Did not reach K8-C experimental large SPD screen'
            }
            if ($row.phase_scope -cne 'rankwise_max_not_additive') { throw 'Missing profiled phase marker' }
            if ([int]$row.true_residual_checkpoints -lt 1) { throw 'No true residual checkpoints' }
            $rel=[double]::Parse($row.root_original_true_rel,$inv)
            if (![double]::IsFinite($rel) -or $rel -gt $Tolerance) { throw "Invalid root true residual: $rel" }
            foreach ($column in @('phase_spmv_halo_ms','phase_spmv_numeric_ms','phase_spmv_other_ms',
                 'phase_dot_local_ms','phase_dot_allreduce_ms','phase_precond_apply_ms',
                 'phase_vector_update_ms','phase_unattributed_ms')) {
                $v=[double]::Parse($row.$column,$inv)
                if (![double]::IsFinite($v) -or $v -lt 0) { throw "Invalid phase $column=$v" }
            }
            $results.Add($row)
            # Save completed cases even if a later case fails; only one retained summary.
            $results | Export-Csv -LiteralPath $summary -NoTypeInformation -Encoding utf8
        } } }
        Write-Host "=== K8-D1 PROFILE SUITE PASS: $($results.Count) / $summary ==="
        if (!$KeepCaseCsv) { Write-Host '=== Summary only; temporary case CSVs removed ===' }
    } finally { Pop-Location }
} finally {
    if ($null -ne $tempDir -and (Test-Path -LiteralPath $tempDir)) {
        Remove-Item -LiteralPath $tempDir -Recurse -Force
    }
}
