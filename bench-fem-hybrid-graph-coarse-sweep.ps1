param(
    [Parameter(Mandatory=$true)][string]$Matrix,
    [Parameter(Mandatory=$false)][string]$Rhs,
    [int[]]$TargetCoarseDimensions = @(512, 768, 1024, 1152, 1280, 1536, 1792, 2048),
    [int]$Repeats = 3,
    [int]$FactorBudgetMiB = 64,
    [int]$MaxIterations = 3000,
    [double]$Tolerance = 1.0e-8,
    [int]$ProbeIterations = 12,
    [int]$MaxRegion = 128,
    [int]$MaxRegions = 8,
    [int]$Overlap = 1,
    [int]$MaxEscalations = 3,
    [int]$StageIterations = 24,
    [int]$CoarseDofs = 3,
    [ValidateSet('auto','factor','inverse')][string]$CoarseApply = 'auto',
    [ValidateSet('auto','csr','abtm')][string]$Backend = 'auto',
    [string]$CsvPath = 'hybit-hybrid-graph-coarse-sweep.csv',
    [string]$SummaryCsvPath = 'hybit-hybrid-graph-coarse-sweep-summary.csv'
)

$ErrorActionPreference = 'Stop'
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path -LiteralPath $Matrix)) { throw "Matrix file not found: $Matrix" }
if ($Rhs -and -not (Test-Path -LiteralPath $Rhs)) { throw "RHS file not found: $Rhs" }
if ($TargetCoarseDimensions.Count -eq 0) { throw 'TargetCoarseDimensions may not be empty' }
if ($Repeats -le 0) { throw 'Repeats must be > 0' }
if ($CoarseDofs -le 0) { throw 'CoarseDofs must be > 0' }
foreach ($target in $TargetCoarseDimensions) {
    if ($target -lt $CoarseDofs) {
        throw "Every target coarse dimension must be >= CoarseDofs ($CoarseDofs); got $target"
    }
}

$common = @(
    '--matrix', $Matrix,
    '--tol', $Tolerance.ToString('R', $Invariant),
    '--max-iters', $MaxIterations.ToString($Invariant),
    '--backend', $Backend,
    '--probe-iters', $ProbeIterations.ToString($Invariant),
    '--overlap', $Overlap.ToString($Invariant),
    '--max-region', $MaxRegion.ToString($Invariant),
    '--max-regions', $MaxRegions.ToString($Invariant),
    '--factor-budget-mib', $FactorBudgetMiB.ToString($Invariant),
    '--max-escalations', $MaxEscalations.ToString($Invariant),
    '--stage-iters', $StageIterations.ToString($Invariant),
    '--selector', 'jacobi-byte',
    '--skip-plain',
    '--hybrid-coarse',
    '--coarse-dofs', $CoarseDofs.ToString($Invariant),
    '--coarse-aggregation', 'graph',
    '--coarse-apply', $CoarseApply
)
if ($Rhs) { $common += @('--rhs', $Rhs) }

function Invoke-HybitBench {
    param(
        [Parameter(Mandatory=$true)][string]$Label,
        [Parameter(Mandatory=$true)][int]$Target
    )

    Write-Host ''
    Write-Host '============================================================'
    Write-Host " $Label"
    Write-Host '============================================================'

    $cargoArgs = @('run', '--release', '-p', 'hybit', '--example', 'fem_bench', '--') + $common
    $cargoArgs += @('--coarse-target', $Target.ToString($Invariant))

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

function Parse-FirstNumber {
    param([string]$Text)
    if ($Text -match '^([0-9.Ee+-]+)') {
        return [double]::Parse($Matches[1], $Invariant)
    }
    return [double]::NaN
}

function Convert-BenchResult {
    param([int]$Target, [int]$Repeat, [string[]]$Lines)

    $section = Get-HybitSection -Lines $Lines
    $iterations = [int](Get-HybitField -Lines $section -Name 'iterations')
    $solverMs = Parse-FirstNumber (Get-HybitField -Lines $section -Name 'solver time')
    $wallMs = Parse-FirstNumber (Get-HybitField -Lines $section -Name 'total wall')

    [PSCustomObject]@{
        Repeat = $Repeat
        TargetCoarseDim = $Target
        ActualCoarseDim = [int](Get-HybitField -Lines $section -Name 'coarse dimension')
        EffectiveApply = Get-HybitField -Lines $section -Name 'coarse apply eff.'
        Status = Get-HybitField -Lines $section -Name 'status'
        Iterations = $iterations
        VerifiedResidual = [double]::Parse((Get-HybitField -Lines $section -Name 'verified residual'), $Invariant)
        Escalations = [int](Get-HybitField -Lines $section -Name 'escalations')
        LocalRegions = [int](Get-HybitField -Lines $section -Name 'local regions')
        FactorMiB = Parse-FirstNumber (Get-HybitField -Lines $section -Name 'factor memory')
        CoarseMiB = Parse-FirstNumber (Get-HybitField -Lines $section -Name 'coarse memory')
        CoarseSetupMs = Parse-FirstNumber (Get-HybitField -Lines $section -Name 'coarse setup')
        SolverMs = $solverMs
        WallMs = $wallMs
        MsPerIter = if ($iterations -gt 0) { $solverMs / $iterations } else { [double]::NaN }
    }
}

function Get-Median {
    param([double[]]$Values)
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) { return [double]::NaN }
    $mid = [int][Math]::Floor($sorted.Count / 2.0)
    if (($sorted.Count % 2) -eq 1) { return [double]$sorted[$mid] }
    return ([double]$sorted[$mid - 1] + [double]$sorted[$mid]) / 2.0
}

Write-Host '== HyBIT graph coarse-dimension repeated sweep =='
Write-Host "Matrix: $Matrix"
if ($Rhs) { Write-Host "RHS: $Rhs" } else { Write-Host 'RHS: generated as b=A*1' }
Write-Host "Coarse DOFs/node: $CoarseDofs"
Write-Host 'Aggregation: graph'
Write-Host "Coarse apply: $CoarseApply"
Write-Host "Targets: $($TargetCoarseDimensions -join ', ')"
Write-Host "Repeats: $Repeats"
Write-Host "Tolerance: $Tolerance"
Write-Host 'Target order alternates ascending/descending between repeats.'

$results = [System.Collections.Generic.List[object]]::new()
$ascendingTargets = @($TargetCoarseDimensions | Sort-Object)
$descendingTargets = @($TargetCoarseDimensions | Sort-Object -Descending)

for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
    $orderedTargets = if (($repeat % 2) -eq 1) { $ascendingTargets } else { $descendingTargets }
    foreach ($target in $orderedTargets) {
        $lines = Invoke-HybitBench -Label "repeat $repeat/$Repeats : graph coarse target = $target" -Target $target
        $results.Add((Convert-BenchResult -Target $target -Repeat $repeat -Lines $lines))
    }
}

Write-Host ''
Write-Host '== per-run summary =='
$results | ForEach-Object {
    [PSCustomObject]@{
        Repeat = $_.Repeat
        Target = $_.TargetCoarseDim
        Actual = $_.ActualCoarseDim
        Apply = $_.EffectiveApply
        Status = $_.Status
        Iters = $_.Iterations
        Residual = $_.VerifiedResidual.ToString('0.000000E+00', $Invariant)
        Esc = $_.Escalations
        Regions = $_.LocalRegions
        FactorMiB = $_.FactorMiB.ToString('0.00', $Invariant)
        CoarseMiB = $_.CoarseMiB.ToString('0.00', $Invariant)
        SetupMs = $_.CoarseSetupMs.ToString('0.00', $Invariant)
        MsPerIter = $_.MsPerIter.ToString('0.000', $Invariant)
        SolverMs = $_.SolverMs.ToString('0.00', $Invariant)
        WallMs = $_.WallMs.ToString('0.00', $Invariant)
    }
} | Format-Table -AutoSize

$summary = [System.Collections.Generic.List[object]]::new()
foreach ($target in $ascendingTargets) {
    $runs = @($results | Where-Object { $_.TargetCoarseDim -eq $target })
    if ($runs.Count -eq 0) { continue }
    $convergedRuns = @($runs | Where-Object { $_.Status -eq 'Converged' })
    $applyValues = @($runs | ForEach-Object { $_.EffectiveApply } | Select-Object -Unique)

    $summary.Add([PSCustomObject]@{
        TargetCoarseDim = $target
        ActualCoarseDim = [int][Math]::Round((Get-Median -Values @($runs | ForEach-Object { [double]$_.ActualCoarseDim })))
        EffectiveApply = ($applyValues -join '/')
        Runs = $runs.Count
        ConvergedRuns = $convergedRuns.Count
        AllConverged = ($convergedRuns.Count -eq $runs.Count)
        MedianIterations = Get-Median -Values @($runs | ForEach-Object { [double]$_.Iterations })
        MaxVerifiedResidual = ($runs | Measure-Object -Property VerifiedResidual -Maximum).Maximum
        MedianEscalations = Get-Median -Values @($runs | ForEach-Object { [double]$_.Escalations })
        MedianLocalRegions = Get-Median -Values @($runs | ForEach-Object { [double]$_.LocalRegions })
        MedianFactorMiB = Get-Median -Values @($runs | ForEach-Object { [double]$_.FactorMiB })
        MedianCoarseMiB = Get-Median -Values @($runs | ForEach-Object { [double]$_.CoarseMiB })
        MedianCoarseSetupMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.CoarseSetupMs })
        MedianMsPerIteration = Get-Median -Values @($runs | ForEach-Object { [double]$_.MsPerIter })
        MedianSolverMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.SolverMs })
        MedianWallMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.WallMs })
        MinWallMs = ($runs | Measure-Object -Property WallMs -Minimum).Minimum
        MaxWallMs = ($runs | Measure-Object -Property WallMs -Maximum).Maximum
    })
}

Write-Host ''
Write-Host '== median summary =='
$summary | ForEach-Object {
    [PSCustomObject]@{
        Target = $_.TargetCoarseDim
        Actual = $_.ActualCoarseDim
        Apply = $_.EffectiveApply
        Conv = ("{0}/{1}" -f $_.ConvergedRuns, $_.Runs)
        Iters = $_.MedianIterations.ToString('0', $Invariant)
        Residual = ([double]$_.MaxVerifiedResidual).ToString('0.000000E+00', $Invariant)
        Esc = $_.MedianEscalations.ToString('0', $Invariant)
        Regions = $_.MedianLocalRegions.ToString('0', $Invariant)
        FactorMiB = $_.MedianFactorMiB.ToString('0.00', $Invariant)
        CoarseMiB = $_.MedianCoarseMiB.ToString('0.00', $Invariant)
        SetupMs = $_.MedianCoarseSetupMs.ToString('0.00', $Invariant)
        MsPerIter = $_.MedianMsPerIteration.ToString('0.000', $Invariant)
        SolverMs = $_.MedianSolverMs.ToString('0.00', $Invariant)
        WallMs = $_.MedianWallMs.ToString('0.00', $Invariant)
        WallRange = ("{0:F2}..{1:F2}" -f $_.MinWallMs, $_.MaxWallMs)
    }
} | Format-Table -AutoSize

$eligible = @($summary | Where-Object { $_.AllConverged } | Sort-Object MedianWallMs)
if ($eligible.Count -gt 0) {
    $best = $eligible[0]
    Write-Host ''
    Write-Host ('Best all-converged graph target: {0} (actual {1}, {2}), median wall {3:F2} ms, solver {4:F2} ms, {5:F0} iters, coarse {6:F2} MiB.' -f `
        $best.TargetCoarseDim, $best.ActualCoarseDim, $best.EffectiveApply, $best.MedianWallMs, $best.MedianSolverMs, $best.MedianIterations, $best.MedianCoarseMiB)
}

$baseline = $summary | Where-Object { $_.TargetCoarseDim -eq 1792 }
if ($baseline -and $eligible.Count -gt 0) {
    $best = $eligible[0]
    $wallGain = 100.0 * ($baseline.MedianWallMs - $best.MedianWallMs) / $baseline.MedianWallMs
    $memoryGain = 100.0 * ($baseline.MedianCoarseMiB - $best.MedianCoarseMiB) / $baseline.MedianCoarseMiB
    Write-Host ('Best vs graph target 1792: wall gain {0:F2}%, coarse-memory reduction {1:F2}%.' -f $wallGain, $memoryGain)
}

$results | Export-Csv -LiteralPath $CsvPath -NoTypeInformation -Encoding utf8
$summary | Export-Csv -LiteralPath $SummaryCsvPath -NoTypeInformation -Encoding utf8
Write-Host "Runs CSV: $CsvPath"
Write-Host "Summary CSV: $SummaryCsvPath"
Write-Host '== graph coarse-dimension sweep complete =='
