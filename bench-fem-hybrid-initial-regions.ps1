param(
    [Parameter(Mandatory=$true)][string]$Matrix,
    [Parameter(Mandatory=$false)][string]$Rhs,
    [int]$CoarseTarget = 1792,
    [int]$Repeats = 3,
    [int]$FactorBudgetMiB = 64,
    [int]$MaxIterations = 3000,
    [double]$Tolerance = 1.0e-8,
    [int]$ProbeIterations = 12,
    [int]$MaxRegion = 128,
    [int]$Overlap = 1,
    [int]$StageIterations = 24,
    [int]$CoarseDofs = 3,
    [ValidateSet('auto','factor','inverse')][string]$CoarseApply = 'auto',
    [ValidateSet('auto','csr','abtm')][string]$Backend = 'auto',
    [string]$CsvPath = 'hybit-hybrid-initial-regions.csv',
    [string]$SummaryCsvPath = 'hybit-hybrid-initial-regions-summary.csv'
)

$ErrorActionPreference = 'Stop'
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path -LiteralPath $Matrix)) { throw "Matrix file not found: $Matrix" }
if ($Rhs -and -not (Test-Path -LiteralPath $Rhs)) { throw "RHS file not found: $Rhs" }
if ($Repeats -le 0) { throw 'Repeats must be > 0' }
if ($CoarseDofs -le 0) { throw 'CoarseDofs must be > 0' }
if ($CoarseTarget -lt $CoarseDofs) { throw 'CoarseTarget must be >= CoarseDofs' }
if ($StageIterations -le 0) { throw 'StageIterations must be > 0' }

$common = @(
    '--matrix', $Matrix,
    '--tol', $Tolerance.ToString('R', $Invariant),
    '--max-iters', $MaxIterations.ToString($Invariant),
    '--backend', $Backend,
    '--probe-iters', $ProbeIterations.ToString($Invariant),
    '--overlap', $Overlap.ToString($Invariant),
    '--max-region', $MaxRegion.ToString($Invariant),
    '--factor-budget-mib', $FactorBudgetMiB.ToString($Invariant),
    '--max-escalations', '1',
    '--stage-iters', $StageIterations.ToString($Invariant),
    '--selector', 'jacobi-byte',
    '--skip-plain',
    '--hybrid-coarse',
    '--coarse-dofs', $CoarseDofs.ToString($Invariant),
    '--coarse-target', $CoarseTarget.ToString($Invariant),
    '--coarse-apply', $CoarseApply,
    '--coarse-aggregation', 'graph',
    '--coarse-basis', 'smoothed',
    '--coarse-transfer', 'parallel',
    '--coarse-transfer-storage', 'wide',
    '--coarse-transfer-values', 'f32'
)
if ($Rhs) { $common += @('--rhs', $Rhs) }

$regionCaps = @(1, 2, 4, 8, 16)
$orders = @(
    @(0, 1, 2, 3, 4),
    @(4, 3, 2, 1, 0),
    @(2, 4, 1, 3, 0),
    @(3, 0, 4, 1, 2)
)

function Invoke-HybitBench {
    param(
        [Parameter(Mandatory=$true)][string]$Label,
        [Parameter(Mandatory=$true)][int]$RegionCap
    )
    Write-Host ''
    Write-Host '============================================================'
    Write-Host " $Label"
    Write-Host '============================================================'
    $cargoArgs = @('run', '--release', '-p', 'hybit', '--example', 'fem_bench', '--') + $common
    $cargoArgs += @('--max-regions', $RegionCap.ToString($Invariant))
    $captured = [System.Collections.Generic.List[string]]::new()
    & cargo @cargoArgs 2>&1 | ForEach-Object {
        $line = $_.ToString()
        $captured.Add($line)
        Write-Host $line
    }
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
    return $captured.ToArray()
}

function Get-HybitField {
    param([string[]]$Lines, [string]$Name)
    $pattern = '^' + [regex]::Escape($Name) + '\s*:\s*(.+?)\s*$'
    foreach ($line in $Lines) {
        if ($line -match $pattern) { return $Matches[1].Trim() }
    }
    return $null
}

function Get-HybitSection {
    param([string[]]$Lines)
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        if ($Lines[$i] -match '^\[1/1\] HyBIT Auto$') {
            return @($Lines[$i..($Lines.Count - 1)])
        }
    }
    throw 'HyBIT Auto section was not found in benchmark output'
}

function Parse-Milliseconds {
    param([string]$Value)
    if ($Value -match '^([0-9.Ee+-]+)') { return [double]::Parse($Matches[1], $Invariant) }
    return [double]::NaN
}

function Parse-MiB {
    param([string]$Value)
    if ($Value -match '^([0-9.Ee+-]+)') { return [double]::Parse($Matches[1], $Invariant) }
    return 0.0
}

function Convert-BenchResult {
    param([int]$RegionCap, [int]$Repeat, [string[]]$Lines)
    $section = Get-HybitSection -Lines $Lines
    $iterations = [int](Get-HybitField -Lines $section -Name 'iterations')
    $solverMs = Parse-Milliseconds (Get-HybitField -Lines $section -Name 'solver time')
    [PSCustomObject]@{
        RegionCap = $RegionCap
        Repeat = $Repeat
        Status = Get-HybitField -Lines $section -Name 'status'
        Iterations = $iterations
        VerifiedResidual = [double]::Parse((Get-HybitField -Lines $section -Name 'verified residual'), $Invariant)
        ActualEscalations = [int](Get-HybitField -Lines $section -Name 'escalations')
        LocalRegions = [int](Get-HybitField -Lines $section -Name 'local regions')
        FactorDofs = Get-HybitField -Lines $section -Name 'factor DOFs'
        FactorMiB = Parse-MiB (Get-HybitField -Lines $section -Name 'factor memory')
        CoarseMiB = Parse-MiB (Get-HybitField -Lines $section -Name 'coarse memory')
        CoarseSetupMs = Parse-Milliseconds (Get-HybitField -Lines $section -Name 'coarse setup')
        DiagnosticsMs = Parse-Milliseconds (Get-HybitField -Lines $section -Name 'diagnostics')
        LocalFactorMs = Parse-Milliseconds (Get-HybitField -Lines $section -Name 'local factor')
        SolverMs = $solverMs
        WallMs = Parse-Milliseconds (Get-HybitField -Lines $section -Name 'total wall')
        MsPerIter = if ($iterations -gt 0) { $solverMs / $iterations } else { [double]::NaN }
    }
}

function Get-Median {
    param([double[]]$Values)
    $sorted = @($Values | Sort-Object)
    $mid = [int][Math]::Floor($sorted.Count / 2.0)
    if (($sorted.Count % 2) -eq 1) { return [double]$sorted[$mid] }
    return ([double]$sorted[$mid - 1] + [double]$sorted[$mid]) / 2.0
}

Write-Host '== HyBIT smoothed+parallel initial-local-region repeated comparison =='
Write-Host "Matrix: $Matrix"
if ($Rhs) { Write-Host "RHS: $Rhs" }
Write-Host "Coarse target: $CoarseTarget"
Write-Host "Coarse apply: $CoarseApply"
Write-Host "Repeats: $Repeats"
Write-Host 'Max escalations: 1'
Write-Host 'Aggregation: graph'
Write-Host 'Basis: smoothed'
Write-Host 'Transfer: parallel / wide / f32'
Write-Host 'Max local-region caps: 1, 2, 4, 8, 16.'
Write-Host 'Run order rotates/reverses across repeats to reduce ordering bias.'

$results = [System.Collections.Generic.List[object]]::new()
for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
    $order = $orders[($repeat - 1) % $orders.Count]
    foreach ($index in $order) {
        $cap = $regionCaps[$index]
        $label = "repeat $repeat/$Repeats : max local regions = $cap"
        $lines = Invoke-HybitBench -Label $label -RegionCap $cap
        $results.Add((Convert-BenchResult -RegionCap $cap -Repeat $repeat -Lines $lines))
    }
}

Write-Host ''
Write-Host '== per-run summary =='
$results | ForEach-Object {
    [PSCustomObject]@{
        MaxRegions = $_.RegionCap
        Repeat = $_.Repeat
        Status = $_.Status
        Iters = $_.Iterations
        Residual = $_.VerifiedResidual.ToString('0.000000E+00', $Invariant)
        Regions = $_.LocalRegions
        FactorMiB = $_.FactorMiB.ToString('0.00', $Invariant)
        DiagMs = $_.DiagnosticsMs.ToString('0.00', $Invariant)
        FactorMs = $_.LocalFactorMs.ToString('0.00', $Invariant)
        MsPerIter = $_.MsPerIter.ToString('0.000', $Invariant)
        SolverMs = $_.SolverMs.ToString('0.00', $Invariant)
        WallMs = $_.WallMs.ToString('0.00', $Invariant)
    }
} | Format-Table -AutoSize

$summary = [System.Collections.Generic.List[object]]::new()
foreach ($cap in $regionCaps) {
    $runs = @($results | Where-Object { $_.RegionCap -eq $cap })
    $convergedCount = @($runs | Where-Object { $_.Status -eq 'Converged' }).Count
    $summary.Add([PSCustomObject]@{
        MaxRegions = $cap
        Converged = "$convergedCount/$($runs.Count)"
        MedianIterations = Get-Median -Values @($runs | ForEach-Object { [double]$_.Iterations })
        MaxResidual = ($runs | Measure-Object -Property VerifiedResidual -Maximum).Maximum
        MedianLocalRegions = Get-Median -Values @($runs | ForEach-Object { [double]$_.LocalRegions })
        MedianFactorMiB = Get-Median -Values @($runs | ForEach-Object { [double]$_.FactorMiB })
        MedianDiagnosticsMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.DiagnosticsMs })
        MedianLocalFactorMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.LocalFactorMs })
        MedianMsPerIter = Get-Median -Values @($runs | ForEach-Object { [double]$_.MsPerIter })
        MedianSolverMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.SolverMs })
        MedianWallMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.WallMs })
    })
}

Write-Host ''
Write-Host '== median summary =='
$summary | ForEach-Object {
    [PSCustomObject]@{
        MaxRegions = $_.MaxRegions
        Conv = $_.Converged
        Iters = [int]$_.MedianIterations
        Residual = $_.MaxResidual.ToString('0.000000E+00', $Invariant)
        Regions = $_.MedianLocalRegions.ToString('0', $Invariant)
        FactorMiB = $_.MedianFactorMiB.ToString('0.00', $Invariant)
        DiagMs = $_.MedianDiagnosticsMs.ToString('0.00', $Invariant)
        FactorMs = $_.MedianLocalFactorMs.ToString('0.00', $Invariant)
        MsPerIter = $_.MedianMsPerIter.ToString('0.000', $Invariant)
        SolverMs = $_.MedianSolverMs.ToString('0.00', $Invariant)
        WallMs = $_.MedianWallMs.ToString('0.00', $Invariant)
    }
} | Format-Table -AutoSize

$baseline = $summary | Where-Object { $_.MaxRegions -eq 8 } | Select-Object -First 1
Write-Host ''
Write-Host '== relative to max local regions 8 =='
$summary | ForEach-Object {
    $solverGain = if ($baseline.MedianSolverMs -gt 0.0) { 100.0 * ($baseline.MedianSolverMs - $_.MedianSolverMs) / $baseline.MedianSolverMs } else { 0.0 }
    $wallGain = if ($baseline.MedianWallMs -gt 0.0) { 100.0 * ($baseline.MedianWallMs - $_.MedianWallMs) / $baseline.MedianWallMs } else { 0.0 }
    [PSCustomObject]@{
        MaxRegions = $_.MaxRegions
        IterDelta = [int]($_.MedianIterations - $baseline.MedianIterations)
        SolverGainPct = $solverGain.ToString('0.00', $Invariant)
        WallGainPct = $wallGain.ToString('0.00', $Invariant)
        ActualRegionDelta = [int]($_.MedianLocalRegions - $baseline.MedianLocalRegions)
        FactorMiBDelta = ($_.MedianFactorMiB - $baseline.MedianFactorMiB).ToString('0.00', $Invariant)
    }
} | Format-Table -AutoSize

$allConverged = @($summary | Where-Object { $_.Converged -match '^([0-9]+)/\1$' })
if ($allConverged.Count -gt 0) {
    $best = $allConverged | Sort-Object MedianWallMs | Select-Object -First 1
    Write-Host ''
    Write-Host ("Fastest all-converged initial-region cap: {0}, median wall {1:F2} ms, solver {2:F2} ms, {3:F0} iters, {4:F0} actual regions, factor {5:F2} MiB." -f $best.MaxRegions, $best.MedianWallMs, $best.MedianSolverMs, $best.MedianIterations, $best.MedianLocalRegions, $best.MedianFactorMiB)
}

$results | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $CsvPath
$summary | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $SummaryCsvPath
Write-Host "Runs CSV: $CsvPath"
Write-Host "Summary CSV: $SummaryCsvPath"
Write-Host '== smoothed+parallel initial-local-region comparison complete =='
