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
    [int]$StageIterations = 24,
    [int]$CoarseDofs = 3,
    [ValidateSet('auto','factor','inverse')][string]$CoarseApply = 'auto',
    [ValidateSet('auto','csr','abtm')][string]$Backend = 'auto',
    [string]$CsvPath = 'hybit-hybrid-local-energy-gate.csv',
    [string]$SummaryCsvPath = 'hybit-hybrid-local-energy-gate-summary.csv'
)

$ErrorActionPreference = 'Stop'
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path -LiteralPath $Matrix)) { throw "Matrix file not found: $Matrix" }
if ($Rhs -and -not (Test-Path -LiteralPath $Rhs)) { throw "RHS file not found: $Rhs" }
if ($Repeats -le 0) { throw 'Repeats must be > 0' }
if ($CoarseDofs -le 0) { throw 'CoarseDofs must be > 0' }
if ($CoarseTarget -lt $CoarseDofs) { throw 'CoarseTarget must be >= CoarseDofs' }
if ($MaxRegions -le 0) { throw 'MaxRegions must be > 0' }

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

# 0.0 reproduces r32. 1.0 is an almost-coarse-only upper endpoint unless the
# proposed local regions capture all current Jacobi residual energy.
$thresholds = @(0.0, 0.01, 0.025, 0.05, 0.10, 0.25, 1.0)
$orders = @(
    @(0, 1, 2, 3, 4, 5, 6),
    @(6, 5, 4, 3, 2, 1, 0),
    @(3, 0, 5, 2, 6, 1, 4),
    @(4, 1, 6, 2, 5, 0, 3)
)

function Invoke-HybitBench {
    param([double]$Threshold, [string]$Label)
    Write-Host ''
    Write-Host '============================================================'
    Write-Host " $Label"
    Write-Host '============================================================'
    $cargoArgs = @('run', '--release', '-p', 'hybit', '--example', 'fem_bench', '--') + $common
    $cargoArgs += @('--local-energy-min', $Threshold.ToString('R', $Invariant))
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

function Parse-Number {
    param([string]$Value)
    if ($Value -match '^([0-9.Ee+-]+)') { return [double]::Parse($Matches[1], $Invariant) }
    return [double]::NaN
}

function Convert-BenchResult {
    param([double]$Threshold, [int]$Repeat, [string[]]$Lines)
    $section = Get-HybitSection -Lines $Lines
    $iterations = [int](Get-HybitField -Lines $section -Name 'iterations')
    $solverMs = Parse-Number (Get-HybitField -Lines $section -Name 'solver time')
    [PSCustomObject]@{
        Threshold = $Threshold
        Repeat = $Repeat
        Status = Get-HybitField -Lines $section -Name 'status'
        Iterations = $iterations
        VerifiedResidual = Parse-Number (Get-HybitField -Lines $section -Name 'verified residual')
        Escalations = [int](Get-HybitField -Lines $section -Name 'escalations')
        HardDofs = [int](Get-HybitField -Lines $section -Name 'hard core DOFs')
        LocalRegions = [int](Get-HybitField -Lines $section -Name 'local regions')
        FactorMiB = Parse-Number (Get-HybitField -Lines $section -Name 'factor memory')
        DiagnosticsMs = Parse-Number (Get-HybitField -Lines $section -Name 'diagnostics')
        LocalFactorMs = Parse-Number (Get-HybitField -Lines $section -Name 'local factor')
        SolverMs = $solverMs
        WallMs = Parse-Number (Get-HybitField -Lines $section -Name 'total wall')
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

Write-Host '== HyBIT local-direct Jacobi-energy gate sweep =='
Write-Host "Matrix: $Matrix"
if ($Rhs) { Write-Host "RHS: $Rhs" }
Write-Host "Coarse target: $CoarseTarget"
Write-Host "Repeats: $Repeats"
Write-Host "Max local regions: $MaxRegions"
Write-Host 'Max escalations: 1'
Write-Host 'Coarse path: graph / smoothed / parallel / wide / f32'
Write-Host 'Threshold 0.0 reproduces r32; higher values require stronger residual-energy localization.'

$results = [System.Collections.Generic.List[object]]::new()
for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
    $order = $orders[($repeat - 1) % $orders.Count]
    foreach ($index in $order) {
        $threshold = [double]$thresholds[$index]
        $label = "repeat $repeat/$Repeats : local-energy-min = $($threshold.ToString('0.###', $Invariant))"
        $lines = Invoke-HybitBench -Threshold $threshold -Label $label
        $results.Add((Convert-BenchResult -Threshold $threshold -Repeat $repeat -Lines $lines))
    }
}

$summary = [System.Collections.Generic.List[object]]::new()
foreach ($threshold in $thresholds) {
    $runs = @($results | Where-Object { $_.Threshold -eq [double]$threshold })
    $converged = @($runs | Where-Object { $_.Status -eq 'Converged' }).Count
    $summary.Add([PSCustomObject]@{
        Threshold = [double]$threshold
        Converged = "$converged/$($runs.Count)"
        MedianIterations = Get-Median -Values @($runs | ForEach-Object { [double]$_.Iterations })
        MedianEscalations = Get-Median -Values @($runs | ForEach-Object { [double]$_.Escalations })
        MedianHardDofs = Get-Median -Values @($runs | ForEach-Object { [double]$_.HardDofs })
        MedianLocalRegions = Get-Median -Values @($runs | ForEach-Object { [double]$_.LocalRegions })
        MedianFactorMiB = Get-Median -Values @($runs | ForEach-Object { [double]$_.FactorMiB })
        MedianDiagnosticsMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.DiagnosticsMs })
        MedianLocalFactorMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.LocalFactorMs })
        MedianMsPerIter = Get-Median -Values @($runs | ForEach-Object { [double]$_.MsPerIter })
        MedianSolverMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.SolverMs })
        MedianWallMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.WallMs })
        MaxResidual = ($runs | Measure-Object -Property VerifiedResidual -Maximum).Maximum
    })
}

Write-Host ''
Write-Host '== median summary =='
$summary | ForEach-Object {
    [PSCustomObject]@{
        EnergyMin = $_.Threshold.ToString('0.###', $Invariant)
        Conv = $_.Converged
        Iters = [int]$_.MedianIterations
        Esc = $_.MedianEscalations.ToString('0', $Invariant)
        Hard = $_.MedianHardDofs.ToString('0', $Invariant)
        Regions = $_.MedianLocalRegions.ToString('0', $Invariant)
        FactorMiB = $_.MedianFactorMiB.ToString('0.00', $Invariant)
        DiagMs = $_.MedianDiagnosticsMs.ToString('0.00', $Invariant)
        FactorMs = $_.MedianLocalFactorMs.ToString('0.00', $Invariant)
        MsPerIter = $_.MedianMsPerIter.ToString('0.000', $Invariant)
        SolverMs = $_.MedianSolverMs.ToString('0.00', $Invariant)
        WallMs = $_.MedianWallMs.ToString('0.00', $Invariant)
        Residual = $_.MaxResidual.ToString('0.000000E+00', $Invariant)
    }
} | Format-Table -AutoSize

$baseline = $summary | Where-Object { $_.Threshold -eq 0.0 } | Select-Object -First 1
Write-Host ''
Write-Host '== relative to r32-compatible threshold 0.0 =='
$summary | ForEach-Object {
    $wallGain = if ($baseline.MedianWallMs -gt 0.0) { 100.0 * ($baseline.MedianWallMs - $_.MedianWallMs) / $baseline.MedianWallMs } else { 0.0 }
    [PSCustomObject]@{
        EnergyMin = $_.Threshold.ToString('0.###', $Invariant)
        IterDelta = [int]($_.MedianIterations - $baseline.MedianIterations)
        EscDelta = ($_.MedianEscalations - $baseline.MedianEscalations).ToString('0', $Invariant)
        RegionDelta = ($_.MedianLocalRegions - $baseline.MedianLocalRegions).ToString('0', $Invariant)
        WallGainPct = $wallGain.ToString('0.00', $Invariant)
    }
} | Format-Table -AutoSize

$eligible = @($summary | Where-Object { $_.Converged -match '^([0-9]+)/\1$' })
if ($eligible.Count -gt 0) {
    $best = $eligible | Sort-Object MedianWallMs | Select-Object -First 1
    Write-Host ''
    Write-Host ("Fastest all-converged gate: {0}, median wall {1:F2} ms, {2:F0} iterations, {3:F0} escalations, {4:F0} local regions." -f $best.Threshold, $best.MedianWallMs, $best.MedianIterations, $best.MedianEscalations, $best.MedianLocalRegions)
}

$results | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $CsvPath
$summary | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $SummaryCsvPath
Write-Host "Runs CSV: $CsvPath"
Write-Host "Summary CSV: $SummaryCsvPath"
Write-Host '== local-energy gate sweep complete =='
