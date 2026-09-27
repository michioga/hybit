param(
    [string]$Root = 'D:\Work',
    [double]$Tolerance = 1.0e-8,
    [int]$MaxIterations = 3000,
    [string]$Checkpoints = '12,24,48,96,192,384,768,1536,3000',
    [int]$ReferenceWindow = 12,
    [int]$DefaultCoarseTarget = 1536,
    [int]$LAngleCoarseTarget = 1792,
    [string[]]$Include = @('nd3k','s3dkq4m2','x104','boneS01','L-angle'),
    [double]$PoorProgressRatio = 0.50,
    [string]$CsvPath = 'hybit-coarse-progress-trace.csv'
)

$ErrorActionPreference = 'Stop'
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path -LiteralPath $Root)) { throw "Root directory not found: $Root" }
if ($Tolerance -le 0.0 -or [double]::IsNaN($Tolerance) -or [double]::IsInfinity($Tolerance)) { throw 'Tolerance must be finite and > 0' }
if ($MaxIterations -le 0) { throw 'MaxIterations must be > 0' }
if ($ReferenceWindow -le 0) { throw 'ReferenceWindow must be > 0' }
if ($PoorProgressRatio -le 0.0 -or [double]::IsNaN($PoorProgressRatio) -or [double]::IsInfinity($PoorProgressRatio)) { throw 'PoorProgressRatio must be finite and > 0' }

function Join-RootPath {
    param([string]$RelativePath)
    return Join-Path -Path $Root -ChildPath $RelativePath
}

$cases = @(
    [PSCustomObject]@{ Name='nd3k';      Matrix=(Join-RootPath 'nd3k.mtx');                              Rhs=$null;                                      DofsPerNode=1; CoarseTarget=$DefaultCoarseTarget },
    [PSCustomObject]@{ Name='cant';      Matrix=(Join-RootPath 'cant.mtx');                              Rhs=$null;                                      DofsPerNode=3; CoarseTarget=$DefaultCoarseTarget },
    [PSCustomObject]@{ Name='cfd1';      Matrix=(Join-RootPath 'cfd1.mtx');                              Rhs=$null;                                      DofsPerNode=1; CoarseTarget=$DefaultCoarseTarget },
    [PSCustomObject]@{ Name='thermal1';  Matrix=(Join-RootPath 'thermal1.mtx');                          Rhs=$null;                                      DofsPerNode=1; CoarseTarget=$DefaultCoarseTarget },
    [PSCustomObject]@{ Name='s3dkq4m2';  Matrix=(Join-RootPath 's3dkq4m2.mtx');                          Rhs=$null;                                      DofsPerNode=1; CoarseTarget=$DefaultCoarseTarget },
    [PSCustomObject]@{ Name='x104';      Matrix=(Join-RootPath 'x104.mtx');                              Rhs=$null;                                      DofsPerNode=3; CoarseTarget=$DefaultCoarseTarget },
    [PSCustomObject]@{ Name='boneS01';   Matrix=(Join-RootPath 'boneS01.mtx');                           Rhs=$null;                                      DofsPerNode=3; CoarseTarget=$DefaultCoarseTarget },
    [PSCustomObject]@{ Name='L-angle';   Matrix=(Join-RootPath 'mf_solver-hybit-export\L-angle-K.mtx'); Rhs=(Join-RootPath 'mf_rhs\L-angle-b.txt'); DofsPerNode=3; CoarseTarget=$LAngleCoarseTarget }
)

if ($Include.Count -gt 0) {
    $requested = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($name in $Include) { [void]$requested.Add($name) }
    $cases = @($cases | Where-Object { $requested.Contains($_.Name) })
}

$available = @($cases | Where-Object {
    (Test-Path -LiteralPath $_.Matrix) -and ((-not $_.Rhs) -or (Test-Path -LiteralPath $_.Rhs))
})
if ($available.Count -eq 0) { throw 'No requested trace matrices are available.' }

Write-Host '== HyBIT coarse-PCG progress trace =='
Write-Host "Matrices: $($available.Name -join ', ')"
Write-Host "Checkpoints: $Checkpoints"
Write-Host "Reference window: $ReferenceWindow"
Write-Host ("Current poor-progress threshold on normalized reference window: {0}" -f $PoorProgressRatio.ToString('0.###', $Invariant))
Write-Host 'The PCG recurrence is preserved across every checkpoint; this script only observes progress.'

$results = [System.Collections.Generic.List[object]]::new()
foreach ($case in $available) {
    Write-Host ''
    Write-Host '============================================================'
    Write-Host " $($case.Name)"
    Write-Host '============================================================'

    $cargoArgs = @(
        'run', '--release', '-p', 'hybit', '--example', 'fem_coarse_progress_trace', '--',
        '--matrix', $case.Matrix,
        '--coarse-dofs', $case.DofsPerNode.ToString($Invariant),
        '--coarse-target', $case.CoarseTarget.ToString($Invariant),
        '--tol', $Tolerance.ToString('R', $Invariant),
        '--max-iters', $MaxIterations.ToString($Invariant),
        '--checkpoints', $Checkpoints,
        '--reference-window', $ReferenceWindow.ToString($Invariant)
    )
    if ($case.Rhs) { $cargoArgs += @('--rhs', $case.Rhs) }

    $captured = [System.Collections.Generic.List[string]]::new()
    & cargo @cargoArgs 2>&1 | ForEach-Object {
        $line = $_.ToString()
        $captured.Add($line)
        Write-Host $line

        if ($line -match '^TRACE checkpoint=([0-9]+) segment=([0-9]+) total=([0-9]+) segment_ratio=([0-9.Ee+-]+) equiv_ref_ratio=([0-9.Ee+-]+) per_iter_rho=([0-9.Ee+-]+) rel_res=([0-9.Ee+-]+) status=([A-Za-z]+)$') {
            $equiv = [double]::Parse($Matches[5], $Invariant)
            $results.Add([PSCustomObject]@{
                Matrix = $case.Name
                RhsMode = $(if ($case.Rhs) {'file'} else {'A*1'})
                DofsPerNode = $case.DofsPerNode
                CoarseTarget = $case.CoarseTarget
                Checkpoint = [int]$Matches[1]
                SegmentIterations = [int]$Matches[2]
                TotalIterations = [int]$Matches[3]
                SegmentRatio = [double]::Parse($Matches[4], $Invariant)
                EquivReferenceRatio = $equiv
                PerIterationRho = [double]::Parse($Matches[6], $Invariant)
                RelativeResidual = [double]::Parse($Matches[7], $Invariant)
                Status = $Matches[8]
                PoorAtReferenceRate = ($equiv -gt $PoorProgressRatio)
            })
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "$($case.Name) trace failed with exit code $LASTEXITCODE" }
}

Write-Host ''
Write-Host '== normalized progress summary =='
$results | ForEach-Object {
    [PSCustomObject]@{
        Matrix=$_.Matrix
        CP=$_.Checkpoint
        Seg=$_.SegmentIterations
        Eq12=$_.EquivReferenceRatio.ToString('0.000000', $Invariant)
        Rho=$_.PerIterationRho.ToString('0.000000', $Invariant)
        RelRes=$_.RelativeResidual.ToString('0.000000E+00', $Invariant)
        Poor=$_.PoorAtReferenceRate
        Status=$_.Status
    }
} | Format-Table -AutoSize

Write-Host ''
Write-Host '== first normalized poor-progress checkpoint =='
$poorSummary = foreach ($case in $available) {
    $rows = @($results | Where-Object { $_.Matrix -eq $case.Name })
    $firstPoor = $rows | Where-Object { $_.PoorAtReferenceRate } | Select-Object -First 1
    $last = $rows | Select-Object -Last 1
    [PSCustomObject]@{
        Matrix = $case.Name
        FirstPoorCheckpoint = $(if ($firstPoor) {$firstPoor.Checkpoint} else {'none'})
        FirstPoorEq12 = $(if ($firstPoor) {$firstPoor.EquivReferenceRatio.ToString('0.000000', $Invariant)} else {'-'})
        FinalIteration = $(if ($last) {$last.TotalIterations} else {-1})
        FinalResidual = $(if ($last) {$last.RelativeResidual.ToString('0.000000E+00', $Invariant)} else {'-'})
        FinalStatus = $(if ($last) {$last.Status} else {'-'})
    }
}
$poorSummary | Format-Table -AutoSize

$results | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $CsvPath
Write-Host "Trace CSV: $CsvPath"
Write-Host '== coarse progress trace complete =='
