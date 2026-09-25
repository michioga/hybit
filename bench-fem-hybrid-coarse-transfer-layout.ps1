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
    [int]$MaxRegions = 8,
    [int]$Overlap = 1,
    [int]$MaxEscalations = 3,
    [int]$StageIterations = 24,
    [int]$CoarseDofs = 3,
    [ValidateSet('auto','factor','inverse')][string]$CoarseApply = 'auto',
    [ValidateSet('auto','csr','abtm')][string]$Backend = 'auto',
    [string]$CsvPath = 'hybit-hybrid-coarse-transfer-layout.csv',
    [string]$SummaryCsvPath = 'hybit-hybrid-coarse-transfer-layout-summary.csv'
)

$ErrorActionPreference = 'Stop'
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path -LiteralPath $Matrix)) { throw "Matrix file not found: $Matrix" }
if ($Rhs -and -not (Test-Path -LiteralPath $Rhs)) { throw "RHS file not found: $Rhs" }
if ($Repeats -le 0) { throw 'Repeats must be > 0' }
if ($CoarseDofs -le 0) { throw 'CoarseDofs must be > 0' }
if ($CoarseTarget -lt $CoarseDofs) { throw 'CoarseTarget must be >= CoarseDofs' }

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
    '--coarse-target', $CoarseTarget.ToString($Invariant),
    '--coarse-apply', $CoarseApply,
    '--coarse-aggregation', 'graph',
    '--coarse-basis', 'smoothed',
    '--coarse-transfer', 'parallel'
)
if ($Rhs) { $common += @('--rhs', $Rhs) }

$configs = @(
    [PSCustomObject]@{ Name = 'wide-f64';    Storage = 'wide';    Values = 'f64' },
    [PSCustomObject]@{ Name = 'compact-f64'; Storage = 'compact'; Values = 'f64' },
    [PSCustomObject]@{ Name = 'wide-f32';    Storage = 'wide';    Values = 'f32' },
    [PSCustomObject]@{ Name = 'compact-f32'; Storage = 'compact'; Values = 'f32' }
)

function Invoke-HybitBench {
    param(
        [Parameter(Mandatory=$true)][string]$Label,
        [Parameter(Mandatory=$true)]$Config
    )
    Write-Host ''
    Write-Host '============================================================'
    Write-Host " $Label"
    Write-Host '============================================================'
    $cargoArgs = @('run', '--release', '-p', 'hybit', '--example', 'fem_bench', '--') + $common
    $cargoArgs += @(
        '--coarse-transfer-storage', $Config.Storage,
        '--coarse-transfer-values', $Config.Values
    )
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
    param($Config, [int]$Repeat, [string[]]$Lines)
    $section = Get-HybitSection -Lines $Lines
    $solver = Get-HybitField -Lines $section -Name 'solver time'
    $wall = Get-HybitField -Lines $section -Name 'total wall'
    $setup = Get-HybitField -Lines $section -Name 'coarse setup'
    $memory = Get-HybitField -Lines $section -Name 'coarse memory'
    $iterations = [int](Get-HybitField -Lines $section -Name 'iterations')
    [PSCustomObject]@{
        Layout = $Config.Name
        Storage = $Config.Storage
        Values = $Config.Values
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
    $mid = [int][Math]::Floor($sorted.Count / 2.0)
    if (($sorted.Count % 2) -eq 1) { return [double]$sorted[$mid] }
    return ([double]$sorted[$mid - 1] + [double]$sorted[$mid]) / 2.0
}

Write-Host '== HyBIT smoothed parallel transfer-layout repeated comparison =='
Write-Host "Matrix: $Matrix"
if ($Rhs) { Write-Host "RHS: $Rhs" }
Write-Host "Coarse target: $CoarseTarget"
Write-Host "Coarse apply: $CoarseApply"
Write-Host "Repeats: $Repeats"
Write-Host 'Aggregation: graph'
Write-Host 'Basis: smoothed'
Write-Host 'Transfer: parallel'
Write-Host 'Layouts: wide-f64, compact-f64, wide-f32, compact-f32.'
Write-Host 'Run order rotates/reverses across repeats to reduce ordering bias.'

$orders = @(
    @(0, 1, 2, 3),
    @(3, 2, 1, 0),
    @(2, 3, 0, 1),
    @(1, 0, 3, 2)
)

$results = [System.Collections.Generic.List[object]]::new()
for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
    $order = $orders[($repeat - 1) % $orders.Count]
    foreach ($index in $order) {
        $config = $configs[$index]
        $label = "repeat $repeat/$Repeats : coarse transfer layout = $($config.Name)"
        $lines = Invoke-HybitBench -Label $label -Config $config
        $results.Add((Convert-BenchResult -Config $config -Repeat $repeat -Lines $lines))
    }
}

Write-Host ''
Write-Host '== per-run summary =='
$results | ForEach-Object {
    [PSCustomObject]@{
        Layout = $_.Layout
        Repeat = $_.Repeat
        Status = $_.Status
        Actual = $_.ActualCoarseDim
        Iters = $_.Iterations
        Residual = $_.VerifiedResidual.ToString('0.000000E+00', $Invariant)
        CoarseMiB = $_.CoarseMiB.ToString('0.00', $Invariant)
        SetupMs = $_.CoarseSetupMs.ToString('0.00', $Invariant)
        MsPerIter = $_.MsPerIter.ToString('0.000', $Invariant)
        SolverMs = $_.SolverMs.ToString('0.00', $Invariant)
        WallMs = $_.WallMs.ToString('0.00', $Invariant)
    }
} | Format-Table -AutoSize

$summary = [System.Collections.Generic.List[object]]::new()
foreach ($config in $configs) {
    $runs = @($results | Where-Object { $_.Layout -eq $config.Name })
    $convergedCount = @($runs | Where-Object { $_.Status -eq 'Converged' }).Count
    $summary.Add([PSCustomObject]@{
        Layout = $config.Name
        Storage = $config.Storage
        Values = $config.Values
        Converged = "$convergedCount/$($runs.Count)"
        MedianActual = Get-Median -Values @($runs | ForEach-Object { [double]$_.ActualCoarseDim })
        MedianIterations = Get-Median -Values @($runs | ForEach-Object { [double]$_.Iterations })
        MaxResidual = ($runs | Measure-Object -Property VerifiedResidual -Maximum).Maximum
        MedianCoarseMiB = Get-Median -Values @($runs | ForEach-Object { [double]$_.CoarseMiB })
        MedianSetupMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.CoarseSetupMs })
        MedianMsPerIter = Get-Median -Values @($runs | ForEach-Object { [double]$_.MsPerIter })
        MedianSolverMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.SolverMs })
        MedianWallMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.WallMs })
    })
}

Write-Host ''
Write-Host '== median summary =='
$summary | ForEach-Object {
    [PSCustomObject]@{
        Layout = $_.Layout
        Conv = $_.Converged
        Actual = $_.MedianActual.ToString('0', $Invariant)
        MedianIters = $_.MedianIterations.ToString('0', $Invariant)
        MaxResidual = ([double]$_.MaxResidual).ToString('0.000000E+00', $Invariant)
        CoarseMiB = $_.MedianCoarseMiB.ToString('0.00', $Invariant)
        SetupMs = $_.MedianSetupMs.ToString('0.00', $Invariant)
        MsPerIter = $_.MedianMsPerIter.ToString('0.000', $Invariant)
        SolverMs = $_.MedianSolverMs.ToString('0.00', $Invariant)
        WallMs = $_.MedianWallMs.ToString('0.00', $Invariant)
    }
} | Format-Table -AutoSize

$baseline = $summary | Where-Object { $_.Layout -eq 'wide-f64' }
if ($baseline) {
    Write-Host ''
    Write-Host '== relative to wide-f64 =='
    $summary | ForEach-Object {
        $wallGain = 100.0 * ($baseline.MedianWallMs - $_.MedianWallMs) / $baseline.MedianWallMs
        $solverGain = 100.0 * ($baseline.MedianSolverMs - $_.MedianSolverMs) / $baseline.MedianSolverMs
        $memoryReduction = $baseline.MedianCoarseMiB - $_.MedianCoarseMiB
        [PSCustomObject]@{
            Layout = $_.Layout
            IterDelta = ($baseline.MedianIterations - $_.MedianIterations).ToString('0', $Invariant)
            SolverGainPct = $solverGain.ToString('0.00', $Invariant)
            WallGainPct = $wallGain.ToString('0.00', $Invariant)
            MemoryReductionMiB = $memoryReduction.ToString('0.00', $Invariant)
        }
    } | Format-Table -AutoSize
}

$allConverged = @($summary | Where-Object { $_.Converged -eq "$Repeats/$Repeats" })
if ($allConverged.Count -gt 0) {
    $fastest = $allConverged | Sort-Object -Property MedianWallMs | Select-Object -First 1
    $smallest = $allConverged | Sort-Object -Property MedianCoarseMiB | Select-Object -First 1
    Write-Host ('Fastest all-converged layout: {0}, median wall {1:F2} ms, solver {2:F2} ms, coarse {3:F2} MiB, {4:F0} iters.' -f $fastest.Layout, $fastest.MedianWallMs, $fastest.MedianSolverMs, $fastest.MedianCoarseMiB, $fastest.MedianIterations)
    Write-Host ('Smallest all-converged layout: {0}, coarse {1:F2} MiB, median wall {2:F2} ms, {3:F0} iters.' -f $smallest.Layout, $smallest.MedianCoarseMiB, $smallest.MedianWallMs, $smallest.MedianIterations)
}

$results | Export-Csv -Path $CsvPath -NoTypeInformation -Encoding UTF8
$summary | Export-Csv -Path $SummaryCsvPath -NoTypeInformation -Encoding UTF8
Write-Host "Runs CSV: $CsvPath"
Write-Host "Summary CSV: $SummaryCsvPath"
Write-Host '== smoothed parallel transfer-layout comparison complete =='
