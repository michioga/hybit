[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$Matrix,
    [ValidateSet('equal','region','multisource')][string[]]$Policies=@('equal','region','multisource'),
    [int[]]$Ranks=@(4,8),
    [ValidateSet('jacobi','ic0','block')][string[]]$Preconditioners=@('jacobi'),
    [ValidateRange(2,512)][int]$BlockSize=16,
    [ValidateRange(0,3)][int]$Warmups=1,
    [ValidateRange(1,12)][int]$Repeats=5,
    [ValidateRange(1,3000)][int]$MaxIters=1500,
    [ValidateRange(1.0e-12,1.0e-3)][double]$Tolerance=1e-8,
    [string]$OutputDir='D:\Work',
    [string]$Prefix='k8d7_prepared_p2p',
    [switch]$KeepCaseCsv
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if (!(Test-Path -LiteralPath $Matrix -PathType Leaf)) { throw "Matrix missing: $Matrix" }
if (!(Test-Path -LiteralPath $OutputDir -PathType Container)) { throw "OutputDir missing: $OutputDir" }
if ($Prefix -cnotmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid Prefix' }
foreach ($r in $Ranks) { if ($r -notin @(1,2,4,8)) { throw "Unsupported MPI rank count: $r" } }
$root=[IO.Path]::GetFullPath($RepoRoot)
$exe=Join-Path $root 'target\release\examples\mpi_k8d7_prepared_p2p.exe'
$mpiexec='C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw "K8-D3 binary missing: $exe" }
if (!(Test-Path -LiteralPath $mpiexec -PathType Leaf)) { throw 'Microsoft MS-MPI missing' }
$summary=Join-Path $OutputDir "${Prefix}_summary.csv"
if (Test-Path -LiteralPath $summary) { throw "Summary already exists: $summary" }
$caseDir=$OutputDir
$tempDir=$null
if (!$KeepCaseCsv) {
    $tempDir=Join-Path ([IO.Path]::GetTempPath()) ('hybit_k8d7_' + [Guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($tempDir) | Out-Null
    $caseDir=$tempDir
}
$records=[Collections.Generic.List[object]]::new()
$inv=[Globalization.CultureInfo]::InvariantCulture
function ReadPositiveFloat([string]$value,[string]$label) {
    $n=[double]::Parse($value,$inv)
    if (![double]::IsFinite($n) -or $n -lt 0) { throw "Nonfinite/negative ${label}: $value" }
    return $n
}
function Middle2([double[]]$values) {
    if ($values.Length -ne 2) { throw 'Expected two values' }
    return 0.5*($values[0]+$values[1])
}
try {
    Push-Location $root
    try {
        & { Set-StrictMode -Off; . .\tools\env\Enter-HybitMsMpi.ps1 }
        $tol=$Tolerance.ToString('G17',$inv)
        foreach ($r in $Ranks) {
            foreach ($policy in $Policies) {
                foreach ($prec in $Preconditioners) {
                    $group=[Collections.Generic.List[object]]::new()
                    $order=@('alltoallv','p2p','p2p','alltoallv')
                    for ($slot=0; $slot -lt $order.Count; $slot++) {
                        $mode=$order[$slot]
                        $out=Join-Path $caseDir "${Prefix}_${policy}_${prec}_r${r}_slot${slot}_${mode}.csv"
                        if (Test-Path -LiteralPath $out) { throw "Case CSV exists: $out" }
                        Write-Host "=== K8-D7 ABBA: ranks=$r policy=$policy precond=$prec slot=$slot mode=$mode ==="
                        & $mpiexec -n $r $exe $r $Matrix $policy $Warmups $Repeats $MaxIters $tol $out $prec $BlockSize --experimental-large-spd $mode
                        if ($LASTEXITCODE -ne 0) { throw "MS-MPI error: ranks=$r policy=$policy slot=$slot mode=$mode" }
                        $rows=@(Import-Csv -LiteralPath $out)
                        if ($rows.Count -ne 1) { throw 'Expected one result row' }
                        $row=$rows[0]
                        if ($row.exact_original_verified -cne 'true' -or
                            $row.experimental_opt_in -cne 'true' -or $row.d2_mode -cne 'prevalidated' -or $row.d7_transport -cne $mode -or $row.d7_halo_bitwise -cne 'true' -or
                            $row.phase_scope -cne 'rankwise_max_not_additive' -or
                            $row.spd_certificate -cne 'experimental_symmetric_positive_diagonal_curvature_screen') {
                            throw 'D7 safety/provenance checks failed'
                        }
                        if ([int]$row.true_residual_checkpoints -lt 1) { throw 'True residual checkpoint missing' }
                        $rel=ReadPositiveFloat $row.root_original_true_rel 'root_original_true_rel'
                        if ($rel -gt $Tolerance) { throw "Root true residual $rel exceeds tolerance" }
                        $null=ReadPositiveFloat $row.median_solve_ms 'solve_ms'
                        foreach ($col in @('d3_rank_exchange_ms','d3_rank_dot_allreduce_ms',
                            'd3_rank_sent_values','d3_rank_received_values',
                            'd3_rank_sent_peers','d3_rank_received_peers')) {
                            $values=@($row.$col -split '\|')
                            if ($values.Count -ne $r) { throw "Expected $r rank entries in $col; got $($values.Count)" }
                            foreach ($value in $values) { $null=ReadPositiveFloat $value $col }
                        }
                        $send=@($row.d3_rank_sent_values -split '\|' | ForEach-Object { [long]$_ })
                        $recv=@($row.d3_rank_received_values -split '\|' | ForEach-Object { [long]$_ })
                        if ((($send | Measure-Object -Sum).Sum) -ne (($recv | Measure-Object -Sum).Sum)) {
                            throw 'D7 Halo total sent/received values mismatch'
                        }
                        if ([int]$row.d3_slowest_exchange_rank -ge $r) { throw 'Invalid slowest rank ID' }
                        # Check that the reported rank maximum matches the D2 reduction (rounding allowed).
                        $maxRank=(@($row.d3_rank_exchange_ms -split '\|' | ForEach-Object { [double]::Parse($_,$inv) }) | Measure-Object -Maximum).Maximum
                        $maxGlobal=ReadPositiveFloat $row.phase_spmv_exchange_ms 'phase_spmv_exchange_ms'
                        if ([Math]::Abs($maxRank-$maxGlobal) -gt 0.000003) { throw 'D7 rankwise/global Halo timer mismatch' }
                        if ($group.Count -gt 0) {
                            $ref=$group[0]
                            foreach ($field in @('iterations','spmv_calls','dot_reductions','root_original_true_rel',
                                'root_solution_error','distributed_true_rel','max_solution_error','ghosts')) {
                                if ($row.$field -cne $ref.$field) { throw "A/B numerical/topology mismatch in $field" }
                            }
                        }
                        $row | Add-Member -NotePropertyName 'd7_abba_slot' -NotePropertyValue $slot
                        $row | Add-Member -NotePropertyName 'd7_abba_position' -NotePropertyValue ($slot+1)
                        $row | Add-Member -NotePropertyName 'd7_abba_paired_speedup_pct' -NotePropertyValue ''
                        $row | Add-Member -NotePropertyName 'd7_abba_alltoallv_median_ms' -NotePropertyValue ''
                        $row | Add-Member -NotePropertyName 'd7_abba_p2p_median_ms' -NotePropertyValue ''
                        $group.Add($row)
                        $records.Add($row)
                        $records | Export-Csv -LiteralPath $summary -NoTypeInformation -Encoding utf8
                    }
                    [double[]]$alltoallvSamples=@(
                        (ReadPositiveFloat $group[0].median_solve_ms 'alltoallv_0'),
                        (ReadPositiveFloat $group[3].median_solve_ms 'alltoallv_3')
                    )
                    [double[]]$p2pSamples=@(
                        (ReadPositiveFloat $group[1].median_solve_ms 'p2p_1'),
                        (ReadPositiveFloat $group[2].median_solve_ms 'p2p_2')
                    )
                    $alltoallvMedian=Middle2 $alltoallvSamples
                    $p2pMedian=Middle2 $p2pSamples
                    if ($alltoallvMedian -le 0.0) { throw 'Invalid Alltoallv median' }
                    $speedup=100.0*($alltoallvMedian-$p2pMedian)/$alltoallvMedian
                    foreach ($row in $group) {
                        $row.d7_abba_paired_speedup_pct=$speedup.ToString('F6',$inv)
                        $row.d7_abba_alltoallv_median_ms=$alltoallvMedian.ToString('F6',$inv)
                        $row.d7_abba_p2p_median_ms=$p2pMedian.ToString('F6',$inv)
                    }
                    $records | Export-Csv -LiteralPath $summary -NoTypeInformation -Encoding utf8
                    Write-Host ("=== D7 ABBA PARITY PASS: ranks={0} policy={1} prec={2} alltoallv={3:F4}ms p2p={4:F4}ms improvement={5:F2}% ===" -f $r,$policy,$prec,$alltoallvMedian,$p2pMedian,$speedup)
                }
            }
        }
        Write-Host "=== K8-D7 ABBA SUITE PASS: $($records.Count) / $summary ==="
        if (!$KeepCaseCsv) { Write-Host '=== One summary CSV; temporary individual CSVs removed ===' }
    } finally { Pop-Location }
} finally {
    if ($null -ne $tempDir -and (Test-Path -LiteralPath $tempDir)) {
        Remove-Item -LiteralPath $tempDir -Recurse -Force
    }
}
