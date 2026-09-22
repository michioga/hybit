param(
    [Parameter(Mandatory=$true)][string]$Matrix,
    [Parameter(Mandatory=$false)][string]$Rhs,
    [int[]]$CoarseTargets = @(768, 1280, 1792, 2048),
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
    [string]$RunsCsvPath = 'hybit-hybrid-coarse-apply-crossover.csv',
    [string]$SummaryCsvPath = 'hybit-hybrid-coarse-apply-crossover-summary.csv'
)

$ErrorActionPreference = 'Stop'
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path -LiteralPath $Matrix)) { throw "Matrix file not found: $Matrix" }
if ($Rhs -and -not (Test-Path -LiteralPath $Rhs)) { throw "RHS file not found: $Rhs" }
if ($Repeats -le 0) { throw 'Repeats must be > 0' }
if ($CoarseDofs -le 0) { throw 'CoarseDofs must be > 0' }
if (-not $CoarseTargets -or $CoarseTargets.Count -eq 0) { throw 'CoarseTargets must not be empty' }
foreach ($target in $CoarseTargets) {
    if ($target -lt $CoarseDofs) { throw "Coarse target $target must be >= CoarseDofs ($CoarseDofs)" }
}
$CoarseTargets = @($CoarseTargets | Sort-Object -Unique)

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
    '--coarse-dofs', $CoarseDofs.ToString($Invariant)
)
if ($Rhs) { $common += @('--rhs', $Rhs) }

function Invoke-HybitBench {
    param(
        [Parameter(Mandatory=$true)][string]$Label,
        [Parameter(Mandatory=$true)][int]$Target,
        [Parameter(Mandatory=$true)][ValidateSet('factor','inverse')][string]$Policy
    )

    Write-Host ''
    Write-Host '============================================================'
    Write-Host " $Label"
    Write-Host '============================================================'

    $cargoArgs = @('run', '--release', '-p', 'hybit', '--example', 'fem_bench', '--') + $common
    $cargoArgs += @('--coarse-target', $Target.ToString($Invariant), '--coarse-apply', $Policy)

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

function Convert-BenchResult {
    param([int]$Target, [string]$Policy, [int]$Repeat, [string[]]$Lines)
    $section = Get-HybitSection -Lines $Lines
    $solver = Get-HybitField -Lines $section -Name 'solver time'
    $wall = Get-HybitField -Lines $section -Name 'total wall'
    $setup = Get-HybitField -Lines $section -Name 'coarse setup'
    $memory = Get-HybitField -Lines $section -Name 'coarse memory'
    $iterations = [int](Get-HybitField -Lines $section -Name 'iterations')

    [PSCustomObject]@{
        Target = $Target
        Policy = $Policy
        Repeat = $Repeat
        Status = Get-HybitField -Lines $section -Name 'status'
        Iterations = $iterations
        VerifiedResidual = [double]::Parse((Get-HybitField -Lines $section -Name 'verified residual'), $Invariant)
        ActualCoarseDim = [int](Get-HybitField -Lines $section -Name 'coarse dimension')
        CoarseMiB = if ($memory -match '^([0-9.Ee+-]+)') { [double]::Parse($Matches[1], $Invariant) } else { 0.0 }
        CoarseSetupMs = if ($setup -match '^([0-9.Ee+-]+)') { [double]::Parse($Matches[1], $Invariant) } else { [double]::NaN }
        SolverMs = if ($solver -match '^([0-9.Ee+-]+)') { [double]::Parse($Matches[1], $Invariant) } else { [double]::NaN }
        WallMs = if ($wall -match '^([0-9.Ee+-]+)') { [double]::Parse($Matches[1], $Invariant) } else { [double]::NaN }
        MsPerIter = if ($solver -match '^([0-9.Ee+-]+)' -and $iterations -gt 0) { [double]::Parse($Matches[1], $Invariant) / $iterations } else { [double]::NaN }
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

Write-Host '== HyBIT coarse-apply crossover repeated benchmark =='
Write-Host "Matrix: $Matrix"
if ($Rhs) { Write-Host "RHS: $Rhs" }
Write-Host "Coarse targets: $($CoarseTargets -join ', ')"
Write-Host "Repeats: $Repeats"
Write-Host "Tolerance: $($Tolerance.ToString('R', $Invariant))"
Write-Host 'Target order alternates ascending/descending between repeats.'
Write-Host 'Policy order alternates factor/inverse for each target and repeat.'

$results = [System.Collections.Generic.List[object]]::new()
for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
    $targetsThisRepeat = if (($repeat % 2) -eq 1) {
        @($CoarseTargets)
    } else {
        @($CoarseTargets | Sort-Object -Descending)
    }

    foreach ($target in $targetsThisRepeat) {
        $factorFirst = ((($repeat + [Array]::IndexOf($CoarseTargets, $target)) % 2) -eq 1)
        $policies = if ($factorFirst) { @('factor','inverse') } else { @('inverse','factor') }
        foreach ($policy in $policies) {
            $label = "repeat $repeat/$Repeats : target = $target : coarse apply = $policy"
            $lines = Invoke-HybitBench -Label $label -Target $target -Policy $policy
            $results.Add((Convert-BenchResult -Target $target -Policy $policy -Repeat $repeat -Lines $lines))
        }
    }
}

Write-Host ''
Write-Host '== per-run summary =='
$results | ForEach-Object {
    [PSCustomObject]@{
        Target = $_.Target
        Policy = $_.Policy
        Repeat = $_.Repeat
        Status = $_.Status
        Iters = $_.Iterations
        Residual = $_.VerifiedResidual.ToString('0.000000E+00', $Invariant)
        Actual = $_.ActualCoarseDim
        CoarseMiB = $_.CoarseMiB.ToString('0.00', $Invariant)
        SetupMs = $_.CoarseSetupMs.ToString('0.00', $Invariant)
        MsPerIter = $_.MsPerIter.ToString('0.000', $Invariant)
        SolverMs = $_.SolverMs.ToString('0.00', $Invariant)
        WallMs = $_.WallMs.ToString('0.00', $Invariant)
    }
} | Format-Table -AutoSize

$summary = [System.Collections.Generic.List[object]]::new()
foreach ($target in $CoarseTargets) {
    $factorRuns = @($results | Where-Object { $_.Target -eq $target -and $_.Policy -eq 'factor' })
    $inverseRuns = @($results | Where-Object { $_.Target -eq $target -and $_.Policy -eq 'inverse' })

    foreach ($policy in @('factor','inverse')) {
        $runs = if ($policy -eq 'factor') { $factorRuns } else { $inverseRuns }
        $runs = @($runs)
        $convergedCount = @($runs | Where-Object { $_.Status -eq 'Converged' }).Count
        $summary.Add([PSCustomObject]@{
            Target = $target
            Policy = $policy
            Converged = "$convergedCount/$($runs.Count)"
            AllConverged = ($convergedCount -eq $runs.Count)
            ActualCoarseDim = [int](Get-Median -Values @($runs | ForEach-Object { [double]$_.ActualCoarseDim }))
            MedianIterations = Get-Median -Values @($runs | ForEach-Object { [double]$_.Iterations })
            MaxResidual = ($runs | Measure-Object -Property VerifiedResidual -Maximum).Maximum
            MedianCoarseMiB = Get-Median -Values @($runs | ForEach-Object { [double]$_.CoarseMiB })
            MedianSetupMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.CoarseSetupMs })
            MedianMsPerIter = Get-Median -Values @($runs | ForEach-Object { [double]$_.MsPerIter })
            MedianSolverMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.SolverMs })
            MedianWallMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.WallMs })
        })
    }
}

Write-Host ''
Write-Host '== policy median summary =='
$summary | ForEach-Object {
    [PSCustomObject]@{
        Target = $_.Target
        Policy = $_.Policy
        Conv = $_.Converged
        Actual = $_.ActualCoarseDim
        Iters = $_.MedianIterations.ToString('0', $Invariant)
        Residual = ([double]$_.MaxResidual).ToString('0.000000E+00', $Invariant)
        SetupMs = $_.MedianSetupMs.ToString('0.00', $Invariant)
        MsPerIter = $_.MedianMsPerIter.ToString('0.000', $Invariant)
        SolverMs = $_.MedianSolverMs.ToString('0.00', $Invariant)
        WallMs = $_.MedianWallMs.ToString('0.00', $Invariant)
    }
} | Format-Table -AutoSize

$comparison = [System.Collections.Generic.List[object]]::new()
foreach ($target in $CoarseTargets) {
    $factor = $summary | Where-Object { $_.Target -eq $target -and $_.Policy -eq 'factor' } | Select-Object -First 1
    $inverse = $summary | Where-Object { $_.Target -eq $target -and $_.Policy -eq 'inverse' } | Select-Object -First 1

    $solverGainPct = if ($factor.MedianSolverMs -gt 0.0) { 100.0 * ($factor.MedianSolverMs - $inverse.MedianSolverMs) / $factor.MedianSolverMs } else { [double]::NaN }
    $wallGainPct = if ($factor.MedianWallMs -gt 0.0) { 100.0 * ($factor.MedianWallMs - $inverse.MedianWallMs) / $factor.MedianWallMs } else { [double]::NaN }
    $setupPenaltyMs = $inverse.MedianSetupMs - $factor.MedianSetupMs
    $iterSavingMs = $factor.MedianMsPerIter - $inverse.MedianMsPerIter
    $breakEvenIterations = if ($setupPenaltyMs -gt 0.0 -and $iterSavingMs -gt 0.0) { $setupPenaltyMs / $iterSavingMs } else { [double]::NaN }

    $comparison.Add([PSCustomObject]@{
        Target = $target
        Actual = $inverse.ActualCoarseDim
        FactorConverged = $factor.AllConverged
        InverseConverged = $inverse.AllConverged
        FactorWallMs = $factor.MedianWallMs
        InverseWallMs = $inverse.MedianWallMs
        WallGainPct = $wallGainPct
        FactorSolverMs = $factor.MedianSolverMs
        InverseSolverMs = $inverse.MedianSolverMs
        SolverGainPct = $solverGainPct
        SetupPenaltyMs = $setupPenaltyMs
        IterSavingMs = $iterSavingMs
        BreakEvenIterations = $breakEvenIterations
        InverseWins = ($factor.AllConverged -and $inverse.AllConverged -and $inverse.MedianWallMs -lt $factor.MedianWallMs)
    })
}

Write-Host ''
Write-Host '== crossover summary =='
$comparison | ForEach-Object {
    [PSCustomObject]@{
        Target = $_.Target
        Actual = $_.Actual
        FactorWall = $_.FactorWallMs.ToString('0.00', $Invariant)
        InverseWall = $_.InverseWallMs.ToString('0.00', $Invariant)
        WallGainPct = $_.WallGainPct.ToString('0.00', $Invariant)
        SolverGainPct = $_.SolverGainPct.ToString('0.00', $Invariant)
        SetupPenalty = $_.SetupPenaltyMs.ToString('0.00', $Invariant)
        IterSaving = $_.IterSavingMs.ToString('0.000', $Invariant)
        BreakEvenIters = if ([double]::IsNaN($_.BreakEvenIterations)) { '-' } else { $_.BreakEvenIterations.ToString('0', $Invariant) }
        InverseWins = $_.InverseWins
    }
} | Format-Table -AutoSize

$firstInverseWin = $comparison | Where-Object { $_.InverseWins } | Sort-Object Target | Select-Object -First 1
if ($firstInverseWin) {
    $lowerFactorWin = $comparison | Where-Object { $_.Target -lt $firstInverseWin.Target -and -not $_.InverseWins } | Sort-Object Target -Descending | Select-Object -First 1
    if ($lowerFactorWin) {
        Write-Host ("Observed crossover lies between targets {0} and {1}; first measured inverse win is target {1} (actual {2})." -f $lowerFactorWin.Target, $firstInverseWin.Target, $firstInverseWin.Actual)
    } else {
        Write-Host ("ExplicitInverse already wins at the smallest tested target {0} (actual {1}); the true crossover is at or below this range." -f $firstInverseWin.Target, $firstInverseWin.Actual)
    }
} else {
    Write-Host 'FactorSolve wins or ties at every tested target; no ExplicitInverse crossover was observed in this range.'
}

$results | Export-Csv -Path $RunsCsvPath -NoTypeInformation -Encoding UTF8
$comparison | Export-Csv -Path $SummaryCsvPath -NoTypeInformation -Encoding UTF8
Write-Host "Runs CSV: $RunsCsvPath"
Write-Host "Summary CSV: $SummaryCsvPath"
Write-Host '== coarse-apply crossover benchmark complete =='
