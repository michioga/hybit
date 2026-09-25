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
    [string]$CsvPath = 'hybit-hybrid-coarse-transfer-values.csv'
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
    '--coarse-aggregation', 'graph'
)
if ($Rhs) { $common += @('--rhs', $Rhs) }

function Invoke-HybitBench {
    param(
        [Parameter(Mandatory=$true)][string]$Label,
        [Parameter(Mandatory=$true)][ValidateSet('f64','f32')][string]$ValueMode
    )
    Write-Host ''
    Write-Host '============================================================'
    Write-Host " $Label"
    Write-Host '============================================================'
    $cargoArgs = @('run', '--release', '-p', 'hybit', '--example', 'fem_bench', '--') + $common
    $cargoArgs += @('--coarse-basis', 'smoothed', '--coarse-transfer', 'parallel', '--coarse-transfer-storage', 'wide', '--coarse-transfer-values', $ValueMode)
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
    param([string]$ValueMode, [int]$Repeat, [string[]]$Lines)
    $section = Get-HybitSection -Lines $Lines
    $solver = Get-HybitField -Lines $section -Name 'solver time'
    $wall = Get-HybitField -Lines $section -Name 'total wall'
    $setup = Get-HybitField -Lines $section -Name 'coarse setup'
    $memory = Get-HybitField -Lines $section -Name 'coarse memory'
    $iterations = [int](Get-HybitField -Lines $section -Name 'iterations')
    [PSCustomObject]@{
        Values = $ValueMode
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

Write-Host '== HyBIT smoothed parallel transfer-value repeated A/B =='
Write-Host "Matrix: $Matrix"
if ($Rhs) { Write-Host "RHS: $Rhs" }
Write-Host "Coarse target: $CoarseTarget"
Write-Host "Coarse apply: $CoarseApply"
Write-Host "Repeats: $Repeats"
Write-Host 'Aggregation: graph'
Write-Host 'Basis: smoothed'
Write-Host 'Transfer: parallel'
Write-Host 'Value storage order alternates f64/f32 between repeats.'

$results = [System.Collections.Generic.List[object]]::new()
for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
    $valueModes = if (($repeat % 2) -eq 1) { @('f64','f32') } else { @('f32','f64') }
    foreach ($valueMode in $valueModes) {
        $lines = Invoke-HybitBench -Label "repeat $repeat/$Repeats : coarse transfer values = $valueMode" -ValueMode $valueMode
        $results.Add((Convert-BenchResult -ValueMode $valueMode -Repeat $repeat -Lines $lines))
    }
}

Write-Host ''
Write-Host '== per-run summary =='
$results | ForEach-Object {
    [PSCustomObject]@{
        Values = $_.Values
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
foreach ($valueMode in @('f64','f32')) {
    $runs = @($results | Where-Object { $_.Values -eq $valueMode })
    $convergedCount = @($runs | Where-Object { $_.Status -eq 'Converged' }).Count
    $summary.Add([PSCustomObject]@{
        Values = $valueMode
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
        Values = $_.Values
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

$f64 = $summary | Where-Object { $_.Values -eq 'f64' }
$f32 = $summary | Where-Object { $_.Values -eq 'f32' }
if ($f64 -and $f32) {
    $iterDelta = 100.0 * ($f64.MedianIterations - $f32.MedianIterations) / $f64.MedianIterations
    $solverGain = 100.0 * ($f64.MedianSolverMs - $f32.MedianSolverMs) / $f64.MedianSolverMs
    $wallGain = 100.0 * ($f64.MedianWallMs - $f32.MedianWallMs) / $f64.MedianWallMs
    $memoryReduction = $f64.MedianCoarseMiB - $f32.MedianCoarseMiB
    Write-Host ('F32 vs F64 transfer values: iteration delta {0:F2}%, solver gain {1:F2}%, wall gain {2:F2}%, coarse-memory reduction {3:F2} MiB.' -f $iterDelta, $solverGain, $wallGain, $memoryReduction)
}

$results | Export-Csv -Path $CsvPath -NoTypeInformation -Encoding UTF8
Write-Host "CSV: $CsvPath"
Write-Host '== smoothed parallel transfer-value A/B complete =='
