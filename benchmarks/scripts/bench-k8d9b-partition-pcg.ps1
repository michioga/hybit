# H1-C3K8-D9B. Same native MS-MPI PCG executable for six partitioners.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$Matrix,
    [Parameter(Mandatory)][string]$QualityCsv,
    [ValidateSet(4,8)][int[]]$Ranks=@(4,8),
    [ValidateSet('equal','region','multisource','multilevel','metis','scotch')][string[]]$Policies=@('equal','region','multisource','multilevel','metis','scotch'),
    [ValidateSet('jacobi','ic0','block')][string]$Preconditioner='jacobi',
    [ValidateRange(2,512)][int]$BlockSize=16,
    [ValidateRange(0,3)][int]$Warmups=1,
    [ValidateRange(1,12)][int]$Repeats=5,
    [ValidateRange(1,3000)][int]$MaxIters=1500,
    [ValidateRange(1.0e-12,1.0e-3)][double]$Tolerance=1e-8,
    [string]$OutputDir='D:\Work',
    [string]$Prefix='k8d9b_thermal1_pcg',
    [switch]$PreflightOnly,
    [switch]$KeepCaseCsv
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=[IO.Path]::GetFullPath($RepoRoot)
$matrixFull=[IO.Path]::GetFullPath($Matrix)
$exe=Join-Path $root 'target\release\examples\mpi_k8d7_prepared_p2p.exe'
$mpiexec='C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
$inv=[Globalization.CultureInfo]::InvariantCulture
if ($Prefix -cnotmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid prefix' }
if (!(Test-Path -LiteralPath $matrixFull -PathType Leaf)) { throw 'Matrix missing' }
if (!(Test-Path -LiteralPath $QualityCsv -PathType Leaf)) { throw 'D9-A quality CSV missing' }
if (!(Test-Path -LiteralPath $OutputDir -PathType Container)) { throw 'Output directory missing' }
if ($Ranks.Count -eq 0 -or $Policies.Count -eq 0) { throw 'Empty ranks or policies' }
if (@($Ranks | Select-Object -Unique).Count -ne $Ranks.Count -or @($Policies | Select-Object -Unique).Count -ne $Policies.Count) { throw 'Duplicate ranks/policies' }
$quality=@(Import-Csv -LiteralPath $QualityCsv)
if ($quality.Count -lt 12) { throw 'D9-A quality CSV appears incomplete (expected at least 12 records)' }
foreach($row in $quality) {
    if ([IO.Path]::GetFullPath($row.matrix) -ine $matrixFull) { throw 'D9-A quality CSV matrix mismatch' }
    if ($row.exact_telemetry -cne 'hybit_partition_quality_probe') { throw 'Unsupported quality CSV provenance' }
}
$external=@{}
foreach($r in $Ranks) {
    foreach($policy in @('metis','scotch')) {
        if ($policy -notin $Policies) { continue }
        $q=@($quality | Where-Object { [int]$_.ranks -eq $r -and $_.partitioner -ceq $policy.ToUpperInvariant() })
        if ($q.Count -ne 1) { throw "Expected one D9-A quality record for $policy ranks=$r" }
        $labels=$q[0].owner_labels
        if (!$labels -or !(Test-Path -LiteralPath $labels -PathType Leaf)) { throw "Missing $policy owner labels: $labels" }
        $hash=(Get-FileHash -LiteralPath $labels -Algorithm SHA256).Hash.ToLowerInvariant()
        $external["$r/$policy"]=[pscustomobject]@{Path=$labels;Hash=$hash;Quality=$q[0]}
        Write-Host "D9-B external source: ranks=$r method=$policy labels=$labels SHA256=$hash"
    }
}
$head=(& git -C $root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'git rev-parse failed' }
Write-Host "D9-B repo HEAD: $head"
if ($head -ne '3c3942e5b49005a9117b06f9e7da081de55060aa') { Write-Warning 'HEAD differs from D9-B pre-commit baseline; verify D9-B code before running' }
if ($PreflightOnly) {
    Write-Host '=== D9-B BENCH PREFLIGHT PASS: inputs present; no executions ==='
    return
}
if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw "D9-B binary missing: $exe" }
if (!(Test-Path -LiteralPath $mpiexec -PathType Leaf)) { throw 'Native Microsoft MS-MPI missing' }
$summary=Join-Path $OutputDir "${Prefix}_summary.csv"
if (Test-Path -LiteralPath $summary) { throw "Refuse to overwrite summary $summary" }
$caseDir=$OutputDir
$tempDir=$null
if (!$KeepCaseCsv) {
    $tempDir=Join-Path ([IO.Path]::GetTempPath()) ('hybit_d9b_'+[guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($tempDir) | Out-Null
    $caseDir=$tempDir
}
$records=[Collections.Generic.List[object]]::new()
function F([string]$s,[string]$field) {
    $value=[double]::Parse($s,$inv)
    if (![double]::IsFinite($value) -or $value -lt 0.0) { throw "Nonfinite/negative ${field}: $s" }
    return $value
}
$previousEnv=[Environment]::GetEnvironmentVariable('HYBIT_D9B_OWNER_LABELS','Process')
try {
    Push-Location $root
    try {
        & { Set-StrictMode -Off; . .\tools\env\Enter-HybitMsMpi.ps1 }
        if (!$env:MSMPI_LIB64) { throw 'Missing native MS-MPI environment' }
        $tol=$Tolerance.ToString('G17',$inv)
        foreach($r in $Ranks) {
            foreach($policy in $Policies) {
                $sourcePath=''
                $sourceHash=''
                $externalWall=''
                $sharedGraphWall=''
                $runPolicy=$policy
                if ($policy -in @('metis','scotch')) {
                    $data=$external["$r/$policy"]
                    $sourcePath=$data.Path
                    $sourceHash=$data.Hash
                    $externalWall=$data.Quality.external_wsl_process_wall_ms
                    $sharedGraphWall=$data.Quality.shared_graph_export_wall_ms
                    $runPolicy='external'
                    [Environment]::SetEnvironmentVariable('HYBIT_D9B_OWNER_LABELS',$sourcePath,'Process')
                } else {
                    [Environment]::SetEnvironmentVariable('HYBIT_D9B_OWNER_LABELS',$null,'Process')
                }
                $group=[Collections.Generic.List[object]]::new()
                $order=@('alltoallv','p2p','p2p','alltoallv')
                for($slot=0;$slot -lt 4;$slot++) {
                    $mode=$order[$slot]
                    $out=Join-Path $caseDir "${Prefix}_${policy}_${Preconditioner}_r${r}_slot${slot}_${mode}.csv"
                    if (Test-Path -LiteralPath $out) { throw "Refuse to overwrite $out" }
                    Write-Host "=== D9-B ranks=$r policy=$policy precond=$Preconditioner slot=$slot mode=$mode ==="
                    & $mpiexec -n $r $exe $r $matrixFull $runPolicy $Warmups $Repeats $MaxIters $tol $out $Preconditioner $BlockSize --experimental-large-spd $mode
                    if ($LASTEXITCODE -ne 0) { throw "D9-B MPI failed: $r $policy $slot $mode" }
                    $one=@(Import-Csv -LiteralPath $out)
                    if ($one.Count -ne 1) { throw 'Expected one CSV row per case' }
                    $item=$one[0]
                    if ($item.exact_original_verified -cne 'true' -or $item.d7_halo_bitwise -cne 'true' -or
                        $item.experimental_opt_in -cne 'true' -or $item.d2_mode -cne 'prevalidated' -or
                        $item.phase_scope -cne 'rankwise_max_not_additive' -or $item.d7_transport -cne $mode) {
                        throw 'D9-B exact/origin/transport gate failed'
                    }
                    if ($runPolicy -eq 'external' -and $item.method -cne 'external_owner_labels') { throw 'External method mismatch' }
                    $rel=F $item.root_original_true_rel 'root_original_true_rel'
                    if ($rel -gt $Tolerance -or [int]$item.true_residual_checkpoints -lt 1) { throw 'Root original CSR residual gate failed' }
                    $null=F $item.median_solve_ms 'median_solve_ms'
                    foreach($col in @('d3_rank_exchange_ms','d3_rank_sent_values','d3_rank_received_values','d3_rank_sent_peers','d3_rank_received_peers')) {
                        $vals=@($item.$col -split '\|')
                        if ($vals.Count -ne $r) { throw "Rankwise telemetry mismatch $col" }
                        foreach($v in $vals) { $null=F $v $col }
                    }
                    $send=@($item.d3_rank_sent_values -split '\|' | ForEach-Object { [long]$_ })
                    $recv=@($item.d3_rank_received_values -split '\|' | ForEach-Object { [long]$_ })
                    if (($send | Measure-Object -Sum).Sum -ne ($recv | Measure-Object -Sum).Sum) { throw 'Halo send/recv mismatch' }
                    if ($group.Count -gt 0) {
                        $prior=$group[0]
                        foreach($field in @('iterations','spmv_calls','dot_reductions','root_original_true_rel','root_solution_error','distributed_true_rel','max_solution_error','ghosts')) {
                            if ($item.$field -cne $prior.$field) { throw "ABBA numerical/topology mismatch: $field" }
                        }
                    }
                    $item | Add-Member -NotePropertyName 'd9b_partitioner' -NotePropertyValue $policy
                    $item | Add-Member -NotePropertyName 'd9b_owner_path' -NotePropertyValue $sourcePath
                    $item | Add-Member -NotePropertyName 'd9b_owner_sha256' -NotePropertyValue $sourceHash
                    $item | Add-Member -NotePropertyName 'd9b_external_wsl_process_wall_ms' -NotePropertyValue $externalWall
                    $item | Add-Member -NotePropertyName 'd9b_shared_graph_export_wall_ms' -NotePropertyValue $sharedGraphWall
                    $item | Add-Member -NotePropertyName 'd9b_abba_slot' -NotePropertyValue $slot
                    $item | Add-Member -NotePropertyName 'd9b_gain_pct' -NotePropertyValue ''
                    $item | Add-Member -NotePropertyName 'd9b_alltoallv_ms' -NotePropertyValue ''
                    $item | Add-Member -NotePropertyName 'd9b_p2p_ms' -NotePropertyValue ''
                    $group.Add($item)
                    $records.Add($item)
                    $records | Export-Csv -LiteralPath $summary -NoTypeInformation -Encoding utf8
                }
                $at=0.5*((F $group[0].median_solve_ms 'at0')+(F $group[3].median_solve_ms 'at3'))
                $pt=0.5*((F $group[1].median_solve_ms 'pt1')+(F $group[2].median_solve_ms 'pt2'))
                if ($at -le 0) { throw 'Invalid Alltoallv time' }
                $pct=100*($at-$pt)/$at
                foreach($item in $group) {
                    $item.d9b_gain_pct=$pct.ToString('F6',$inv)
                    $item.d9b_alltoallv_ms=$at.ToString('F6',$inv)
                    $item.d9b_p2p_ms=$pt.ToString('F6',$inv)
                }
                $records | Export-Csv -LiteralPath $summary -NoTypeInformation -Encoding utf8
                Write-Host ('=== D9-B PARITY PASS: ranks={0} method={1} Alltoallv={2:F4}ms P2P={3:F4}ms gain={4:F2}% ===' -f $r,$policy,$at,$pt,$pct)
            }
        }
        Write-Host "=== D9-B PCG SUITE PASS: $($records.Count) / $summary ==="
    } finally { Pop-Location }
} finally {
    [Environment]::SetEnvironmentVariable('HYBIT_D9B_OWNER_LABELS',$previousEnv,'Process')
    if ($null -ne $tempDir -and (Test-Path -LiteralPath $tempDir)) {
        Remove-Item -LiteralPath $tempDir -Recurse -Force
    }
}
