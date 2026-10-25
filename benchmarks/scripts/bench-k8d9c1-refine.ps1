# H1-C3K8-D9C1. Experimental ABTM incremental halo/cut refinement.
# No WSL/METIS/SCOTCH execution; consumes existing common D9-A quality CSV.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$Matrix,
    [Parameter(Mandatory)][string]$QualityCsv,
    [int[]]$Ranks=@(4,8),
    [ValidateRange(1,5)][int]$Passes=2,
    [ValidateRange(1,100000)][int]$MaxMoves=12000,
    [ValidateRange(1,9)][int]$Rounds=3,
    [string]$OutputDir='D:\Work',
    [string]$Prefix='k8d9c1_refine',
    [switch]$PreflightOnly
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=[IO.Path]::GetFullPath($RepoRoot)
$head='4e589997bcd248c976519c4518503e3beae168ee'
if ($Prefix -cnotmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid prefix' }
if (!(Test-Path -LiteralPath $Matrix -PathType Leaf)) { throw "Matrix missing: $Matrix" }
if (!(Test-Path -LiteralPath $QualityCsv -PathType Leaf)) { throw "Quality CSV missing: $QualityCsv" }
if (!(Test-Path -LiteralPath $OutputDir -PathType Container)) { throw "Output directory missing: $OutputDir" }
if ($Ranks.Count -eq 0 -or @($Ranks|Where-Object { $_ -notin @(4,8) }).Count) { throw 'Only 4/8 ranks supported' }
$exe=Join-Path $root 'target\release\examples\partition_d9c1_refine_probe.exe'
$summary=Join-Path $OutputDir "${Prefix}_summary.csv"
if (Test-Path -LiteralPath $summary) { throw "Refuse to overwrite: $summary" }
$quality=@(Import-Csv -LiteralPath $QualityCsv)
foreach($r in $Ranks) {
    foreach($method in @('METIS','SCOTCH')) {
        $matching=@($quality|Where-Object { [int]$_.ranks -eq $r -and $_.partitioner -ieq $method })
        if ($matching.Count -ne 1) { throw "Expected one quality row: ranks=$r method=$method" }
        if ($matching[0].exact_telemetry -cne 'hybit_partition_quality_probe') { throw 'Quality CSV provenance invalid' }
    }
}
Push-Location $root
try {
    if ((git rev-parse HEAD).Trim() -cne $head) { throw 'Unexpected HEAD; D9-C1 baseline changed' }
    & .\tools\gates\source-integrity-gate.ps1
    if ($LASTEXITCODE -ne 0) { throw 'Source Integrity failed' }
    if ($PreflightOnly) {
        Write-Host '=== D9-C1 BENCH PREFLIGHT PASS: NO OUTPUT FILES ==='
        return
    }
    if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw "D9C1 executable missing: $exe" }
    $records=[Collections.Generic.List[object]]::new()
    foreach ($r in $Ranks) {
        $hash0=$null
        for ($round=1; $round -le $Rounds; $round++) {
            $stem=Join-Path $OutputDir "${Prefix}_r${r}_round${round}"
            $labels="${stem}.labels"
            $csv="${stem}.csv"
            if ((Test-Path -LiteralPath $labels) -or (Test-Path -LiteralPath $csv)) {
                throw "D9-C1 output exists: $stem"
            }
            Write-Host "=== D9-C1 ranks=$r round=$round passes=$Passes budget=$MaxMoves ==="
            & $exe $Matrix $r $labels $csv $Passes $MaxMoves
            if ($LASTEXITCODE -ne 0) { throw "D9-C1 experiment failed: ranks=$r round=$round" }
            $rows=@(Import-Csv -LiteralPath $csv)
            if ($rows.Count -ne 1) { throw "Expected one row: $csv" }
            $row=$rows[0]
            if ($row.verified -cne 'true' -or [int]$row.ranks -ne $r) { throw 'D9-C1 result provenance invalid' }
            foreach ($metric in @('cut','ghosts','peers')) {
                $before=[long]($row."baseline_$metric")
                $after=[long]($row."refined_$metric")
                if ($after -gt $before) { throw "D9-C1 degraded ${metric}: $before -> $after" }
            }
            $sha=(Get-FileHash -LiteralPath $labels -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($null -ne $hash0 -and $sha -cne $hash0) { throw "Nondeterministic owner labels: ranks=$r" }
            $hash0=$sha
            $metis=@($quality | Where-Object { [int]$_.ranks -eq $r -and $_.partitioner -ieq 'METIS' })[0]
            $scotch=@($quality | Where-Object { [int]$_.ranks -eq $r -and $_.partitioner -ieq 'SCOTCH' })[0]
            $row | Add-Member -NotePropertyName round -NotePropertyValue $round
            $row | Add-Member -NotePropertyName owner_sha256 -NotePropertyValue $sha
            $row | Add-Member -NotePropertyName metis_ghosts -NotePropertyValue $metis.communication_volume
            $row | Add-Member -NotePropertyName scotch_ghosts -NotePropertyValue $scotch.communication_volume
            $row | Add-Member -NotePropertyName metis_cut -NotePropertyValue $metis.cut_nnz
            $row | Add-Member -NotePropertyName scotch_cut -NotePropertyValue $scotch.cut_nnz
            $row | Add-Member -NotePropertyName metis_peers -NotePropertyValue $metis.peer_relations
            $row | Add-Member -NotePropertyName scotch_peers -NotePropertyValue $scotch.peer_relations
            $records.Add($row)
            $records | Export-Csv -LiteralPath $summary -NoTypeInformation -Encoding utf8
            Write-Host ("D9-C1 QUALITY ranks={0} ABTM={1}/{2} -> {3}/{4} METIS={5}/{6} SCOTCH={7}/{8}" -f $r,$row.baseline_cut,$row.baseline_ghosts,$row.refined_cut,$row.refined_ghosts,$metis.cut_nnz,$metis.communication_volume,$scotch.cut_nnz,$scotch.communication_volume)
        }
    }
    Write-Host "=== D9-C1 SUITE PASS: $($records.Count) / $summary ==="
    Write-Host 'No source changes, no commit/push from benchmark runner'
} finally { Pop-Location }
