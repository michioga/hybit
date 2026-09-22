param(
    [Parameter(Mandatory=$true)][string]$Matrix,
    [Parameter(Mandatory=$false)][string]$Rhs,
    [int[]]$TargetCoarseDimensions = @(1280, 1536, 1792, 2048),
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
    [ValidateSet('auto','csr','abtm')][string]$Backend = 'auto',
    [string]$CsvPath = 'hybit-hybrid-coarse-sweep.csv',
    [string]$SummaryCsvPath = 'hybit-hybrid-coarse-sweep-summary.csv'
)

$ErrorActionPreference = 'Stop'
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path -LiteralPath $Matrix)) {
    throw "Matrix file not found: $Matrix"
}
if ($Rhs -and -not (Test-Path -LiteralPath $Rhs)) {
    throw "RHS file not found: $Rhs"
}
if ($TargetCoarseDimensions.Count -eq 0) {
    throw 'TargetCoarseDimensions may not be empty'
}
if ($Repeats -le 0) {
    throw 'Repeats must be > 0'
}
if ($CoarseDofs -le 0) {
    throw 'CoarseDofs must be > 0'
}
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
    '--selector', 'jacobi-byte'
)
if ($Rhs) {
    $common += @('--rhs', $Rhs)
}

function Invoke-HybitBench {
    param(
        [Parameter(Mandatory=$true)][string]$Label,
        [AllowEmptyCollection()][string[]]$ExtraArgs = @(),
        [switch]$SkipPlain
    )

    Write-Host ''
    Write-Host '============================================================'
    Write-Host " $Label"
    Write-Host '============================================================'

    $cargoArgs = @('run', '--release', '-p', 'hybit', '--example', 'fem_bench', '--') + $common
    if ($SkipPlain) {
        $cargoArgs += '--skip-plain'
    }
    $cargoArgs += $ExtraArgs

    $captured = [System.Collections.Generic.List[string]]::new()
    & cargo @cargoArgs 2>&1 | ForEach-Object {
        $line = $_.ToString()
        $captured.Add($line)
        Write-Host $line
    }
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode"
    }
    return $captured.ToArray()
}

function Get-HybitField {
    param(
        [string[]]$Lines,
        [string]$Name
    )
    $pattern = '^' + [regex]::Escape($Name) + '\s*:\s*(.+?)\s*$'
    foreach ($line in $Lines) {
        if ($line -match $pattern) {
            return $Matches[1].Trim()
        }
    }
    return $null
}

function Get-HybitSection {
    param([string[]]$Lines)
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        if ($Lines[$i] -match '^\[(?:1/1|2/2)\] HyBIT Auto$') {
            return @($Lines[$i..($Lines.Count - 1)])
        }
    }
    throw 'HyBIT Auto section was not found in benchmark output'
}

function Convert-BenchResult {
    param(
        [string]$Mode,
        [int]$Target,
        [int]$Repeat,
        [string[]]$Lines
    )

    $section = Get-HybitSection -Lines $Lines
    $stageRatios = @()
    foreach ($line in $section) {
        if ($line -match '^\s+stage\s+(\d+)\s+:.*residual ratio\s+([^,\s]+)') {
            $stageRatios += ("{0}:{1}" -f $Matches[1], $Matches[2])
        }
    }

    $coarseDim = Get-HybitField -Lines $section -Name 'coarse dimension'
    $coarseMemory = Get-HybitField -Lines $section -Name 'coarse memory'
    $coarseSetup = Get-HybitField -Lines $section -Name 'coarse setup'
    $factorMemory = Get-HybitField -Lines $section -Name 'factor memory'
    $solverTime = Get-HybitField -Lines $section -Name 'solver time'
    $totalWall = Get-HybitField -Lines $section -Name 'total wall'
    $iterations = [int](Get-HybitField -Lines $section -Name 'iterations')

    [PSCustomObject]@{
        Mode = $Mode
        Repeat = $Repeat
        TargetCoarseDim = $Target
        ActualCoarseDim = if ($coarseDim) { [int]$coarseDim } else { 0 }
        Status = (Get-HybitField -Lines $section -Name 'status')
        Iterations = $iterations
        VerifiedResidual = [double]::Parse((Get-HybitField -Lines $section -Name 'verified residual'), $Invariant)
        FactorMiB = if ($factorMemory -match '^([0-9.Ee+-]+)') { [double]::Parse($Matches[1], $Invariant) } else { 0.0 }
        CoarseMiB = if ($coarseMemory -match '^([0-9.Ee+-]+)') { [double]::Parse($Matches[1], $Invariant) } else { 0.0 }
        CoarseSetupMs = if ($coarseSetup -match '^([0-9.Ee+-]+)') { [double]::Parse($Matches[1], $Invariant) } else { 0.0 }
        SolverMs = if ($solverTime -match '^([0-9.Ee+-]+)') { [double]::Parse($Matches[1], $Invariant) } else { [double]::NaN }
        TotalWallMs = if ($totalWall -match '^([0-9.Ee+-]+)') { [double]::Parse($Matches[1], $Invariant) } else { [double]::NaN }
        SolverMsPerIteration = if ($solverTime -match '^([0-9.Ee+-]+)' -and $iterations -gt 0) { [double]::Parse($Matches[1], $Invariant) / $iterations } else { [double]::NaN }
        StageRatios = $stageRatios -join ';'
    }
}

function Get-Median {
    param([double[]]$Values)
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) {
        return [double]::NaN
    }
    $mid = [int][Math]::Floor($sorted.Count / 2.0)
    if (($sorted.Count % 2) -eq 1) {
        return [double]$sorted[$mid]
    }
    return ([double]$sorted[$mid - 1] + [double]$sorted[$mid]) / 2.0
}

Write-Host '== HyBIT hybrid coarse-dimension repeated sweep =='
Write-Host "Matrix: $Matrix"
if ($Rhs) { Write-Host "RHS: $Rhs" } else { Write-Host 'RHS: generated as b=A*1' }
Write-Host "Coarse DOFs/node: $CoarseDofs"
Write-Host "Targets: $($TargetCoarseDimensions -join ', ')"
Write-Host "Repeats: $Repeats"
Write-Host "Tolerance: $Tolerance"
Write-Host "Max iterations: $MaxIterations"
Write-Host 'Plain Jacobi-PCG is executed only in the local-only reference run.'
Write-Host 'Target order alternates ascending/descending between repeats.'

$results = [System.Collections.Generic.List[object]]::new()

$referenceOutput = Invoke-HybitBench -Label 'reference: selective-direct only' -ExtraArgs @()
$results.Add((Convert-BenchResult -Mode 'local-only' -Target 0 -Repeat 0 -Lines $referenceOutput))

$ascendingTargets = @($TargetCoarseDimensions | Sort-Object)
$descendingTargets = @($TargetCoarseDimensions | Sort-Object -Descending)

for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
    $orderedTargets = if (($repeat % 2) -eq 1) { $ascendingTargets } else { $descendingTargets }
    foreach ($target in $orderedTargets) {
        $output = Invoke-HybitBench `
            -Label "repeat $repeat/$Repeats : algebraic coarse target = $target" `
            -SkipPlain `
            -ExtraArgs @(
                '--hybrid-coarse',
                '--coarse-dofs', $CoarseDofs.ToString($Invariant),
                '--coarse-target', $target.ToString($Invariant)
            )
        $results.Add((Convert-BenchResult -Mode 'two-level' -Target $target -Repeat $repeat -Lines $output))
    }
}

Write-Host ''
Write-Host '== per-run summary =='
$runDisplay = $results | ForEach-Object {
    [PSCustomObject]@{
        Mode = $_.Mode
        Repeat = $_.Repeat
        Target = $_.TargetCoarseDim
        Actual = $_.ActualCoarseDim
        Status = $_.Status
        Iters = $_.Iterations
        Residual = $_.VerifiedResidual.ToString('0.000000E+00', $Invariant)
        CoarseMiB = $_.CoarseMiB.ToString('0.00', $Invariant)
        SetupMs = $_.CoarseSetupMs.ToString('0.00', $Invariant)
        MsPerIter = $_.SolverMsPerIteration.ToString('0.000', $Invariant)
        SolverMs = $_.SolverMs.ToString('0.00', $Invariant)
        WallMs = $_.TotalWallMs.ToString('0.00', $Invariant)
    }
}
$runDisplay | Format-Table -AutoSize

$summary = [System.Collections.Generic.List[object]]::new()
foreach ($target in $ascendingTargets) {
    $runs = @($results | Where-Object { $_.Mode -eq 'two-level' -and $_.TargetCoarseDim -eq $target })
    if ($runs.Count -eq 0) {
        continue
    }
    $convergedRuns = @($runs | Where-Object { $_.Status -eq 'Converged' })
    $actualDims = @($runs | ForEach-Object { [double]$_.ActualCoarseDim })
    $iterations = @($runs | ForEach-Object { [double]$_.Iterations })
    $residuals = @($runs | ForEach-Object { [double]$_.VerifiedResidual })
    $coarseMiB = @($runs | ForEach-Object { [double]$_.CoarseMiB })
    $setupMs = @($runs | ForEach-Object { [double]$_.CoarseSetupMs })
    $msPerIter = @($runs | ForEach-Object { [double]$_.SolverMsPerIteration })
    $solverMs = @($runs | ForEach-Object { [double]$_.SolverMs })
    $wallMs = @($runs | ForEach-Object { [double]$_.TotalWallMs })

    $summary.Add([PSCustomObject]@{
        TargetCoarseDim = $target
        ActualCoarseDim = [int][Math]::Round((Get-Median -Values $actualDims))
        Runs = $runs.Count
        ConvergedRuns = $convergedRuns.Count
        AllConverged = ($convergedRuns.Count -eq $runs.Count)
        MedianIterations = (Get-Median -Values $iterations)
        MaxVerifiedResidual = ($residuals | Measure-Object -Maximum).Maximum
        MedianCoarseMiB = (Get-Median -Values $coarseMiB)
        MedianCoarseSetupMs = (Get-Median -Values $setupMs)
        MedianMsPerIteration = (Get-Median -Values $msPerIter)
        MedianSolverMs = (Get-Median -Values $solverMs)
        MedianWallMs = (Get-Median -Values $wallMs)
        MinWallMs = ($wallMs | Measure-Object -Minimum).Minimum
        MaxWallMs = ($wallMs | Measure-Object -Maximum).Maximum
    })
}

Write-Host ''
Write-Host '== median summary =='
$summaryDisplay = $summary | ForEach-Object {
    [PSCustomObject]@{
        Target = $_.TargetCoarseDim
        Actual = $_.ActualCoarseDim
        Conv = ("{0}/{1}" -f $_.ConvergedRuns, $_.Runs)
        MedianIters = $_.MedianIterations.ToString('0', $Invariant)
        MaxResidual = $_.MaxVerifiedResidual.ToString('0.000000E+00', $Invariant)
        CoarseMiB = $_.MedianCoarseMiB.ToString('0.00', $Invariant)
        SetupMs = $_.MedianCoarseSetupMs.ToString('0.00', $Invariant)
        MsPerIter = $_.MedianMsPerIteration.ToString('0.000', $Invariant)
        SolverMedian = $_.MedianSolverMs.ToString('0.00', $Invariant)
        WallMedian = $_.MedianWallMs.ToString('0.00', $Invariant)
        WallRange = ("{0:F2}..{1:F2}" -f $_.MinWallMs, $_.MaxWallMs)
    }
}
$summaryDisplay | Format-Table -AutoSize

$eligible = @($summary | Where-Object { $_.AllConverged } | Sort-Object MedianWallMs)
if ($eligible.Count -gt 0) {
    $best = $eligible[0]
    Write-Host ''
    $bestMessage = 'Best all-converged median target: {0} (actual {1}), median wall {2:F2} ms, median solver {3:F2} ms, median {4:F0} iters, max residual {5}' -f $best.TargetCoarseDim, $best.ActualCoarseDim, $best.MedianWallMs, $best.MedianSolverMs, $best.MedianIterations, $best.MaxVerifiedResidual.ToString('0.000000E+00', $Invariant)
    Write-Host $bestMessage
} else {
    Write-Host ''
    Write-Host 'No target converged on every repeat.'
}

$results | Export-Csv -LiteralPath $CsvPath -NoTypeInformation -Encoding utf8
$summary | Export-Csv -LiteralPath $SummaryCsvPath -NoTypeInformation -Encoding utf8
Write-Host "Runs CSV: $CsvPath"
Write-Host "Summary CSV: $SummaryCsvPath"
Write-Host '== repeated coarse-dimension sweep complete =='
