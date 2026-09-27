param(
    [Parameter(Mandatory=$true)][string]$Matrix,
    [Parameter(Mandatory=$false)][string]$Rhs,
    [int]$CoarseDofs = 1,
    [int]$TargetCoarseDimension = 1536,
    [string[]]$Bases = @('smoothed','spectral1','spectral2','spectral3'),
    [int]$Repeats = 1,
    [int]$MaxIterations = 3000,
    [double]$Tolerance = 1.0e-8,
    [int]$ProbeIterations = 12,
    [ValidateSet('graph','strong-graph','contiguous')][string]$Aggregation = 'graph',
    [ValidateSet('auto','factor','inverse')][string]$CoarseApply = 'auto',
    [ValidateSet('f64','f32','auto')][string]$TransferValues = 'f32',
    [ValidateSet('wide','compact','auto')][string]$TransferStorage = 'wide',
    [ValidateSet('auto','csr','abtm')][string]$Backend = 'auto',
    [string]$CsvPath = 'hybit-spectral-coarse.csv',
    [string]$SummaryCsvPath = 'hybit-spectral-coarse-summary.csv'
)

$ErrorActionPreference = 'Stop'
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path -LiteralPath $Matrix)) { throw "Matrix file not found: $Matrix" }
if ($Rhs -and -not (Test-Path -LiteralPath $Rhs)) { throw "RHS file not found: $Rhs" }
if ($CoarseDofs -le 0) { throw 'CoarseDofs must be > 0' }
if ($TargetCoarseDimension -lt $CoarseDofs) { throw 'TargetCoarseDimension must be >= CoarseDofs' }
if ($Repeats -le 0) { throw 'Repeats must be > 0' }
if ($Bases.Count -eq 0) { throw 'Bases may not be empty' }
$allowed = @('smoothed','spectral1','spectral2','spectral3')
foreach ($basis in $Bases) {
    if ($basis -notin $allowed) { throw "Unsupported basis '$basis'" }
}

$common = @(
    '--matrix', $Matrix,
    '--tol', $Tolerance.ToString('R', $Invariant),
    '--max-iters', $MaxIterations.ToString($Invariant),
    '--backend', $Backend,
    '--probe-iters', $ProbeIterations.ToString($Invariant),
    '--max-escalations', '1',
    '--local-energy-min', '1',
    '--skip-plain',
    '--hybrid-coarse',
    '--coarse-dofs', $CoarseDofs.ToString($Invariant),
    '--coarse-target', $TargetCoarseDimension.ToString($Invariant),
    '--coarse-aggregation', $Aggregation,
    '--coarse-transfer', 'parallel',
    '--coarse-transfer-storage', $TransferStorage,
    '--coarse-transfer-values', $TransferValues,
    '--coarse-apply', $CoarseApply
)
if ($Rhs) { $common += @('--rhs', $Rhs) }

function Invoke-HybitBench {
    param([string]$Basis, [int]$Repeat)
    Write-Host ''
    Write-Host '============================================================'
    Write-Host " repeat $Repeat/$Repeats : basis = $Basis"
    Write-Host '============================================================'

    $cargoArgs = @('run', '--release', '-p', 'hybit', '--example', 'fem_bench', '--') + $common
    $cargoArgs += @('--coarse-basis', $Basis)
    $captured = [System.Collections.Generic.List[string]]::new()
    & cargo @cargoArgs 2>&1 | ForEach-Object {
        $line = $_.ToString()
        $captured.Add($line)
        Write-Host $line
    }
    if ($LASTEXITCODE -ne 0) { throw "basis $Basis failed with exit code $LASTEXITCODE" }
    return $captured.ToArray()
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

function Get-HybitField {
    param([string[]]$Lines, [string]$Name)
    $pattern = '^' + [regex]::Escape($Name) + '\s*:\s*(.+?)\s*$'
    foreach ($line in $Lines) {
        if ($line -match $pattern) { return $Matches[1].Trim() }
    }
    return $null
}

function Parse-FirstNumber {
    param([string]$Text)
    if ($Text -match '^([0-9.Ee+-]+)') {
        return [double]::Parse($Matches[1], $Invariant)
    }
    return [double]::NaN
}

function Get-Median {
    param([double[]]$Values)
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) { return [double]::NaN }
    $mid = [int][Math]::Floor($sorted.Count / 2.0)
    if (($sorted.Count % 2) -eq 1) { return [double]$sorted[$mid] }
    return ([double]$sorted[$mid - 1] + [double]$sorted[$mid]) / 2.0
}

Write-Host '== HyBIT experimental spectral-coarse sweep =='
Write-Host "Matrix: $Matrix"
if ($Rhs) { Write-Host "RHS: $Rhs" } else { Write-Host 'RHS: generated as b=A*1' }
Write-Host "Coarse DOFs/node: $CoarseDofs"
Write-Host "Target coarse dimension: $TargetCoarseDimension"
Write-Host "Aggregation: $Aggregation"
Write-Host "Bases: $($Bases -join ', ')"
Write-Host "Repeats: $Repeats"
Write-Host 'Local-direct is suppressed with --local-energy-min 1.'

$results = [System.Collections.Generic.List[object]]::new()
for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
    $orderedBases = if (($repeat % 2) -eq 1) { @($Bases) } else { @($Bases[($Bases.Count - 1)..0]) }
    foreach ($basis in $orderedBases) {
        $lines = Invoke-HybitBench -Basis $basis -Repeat $repeat
        $section = Get-HybitSection -Lines $lines
        $iterations = [int](Get-HybitField -Lines $section -Name 'iterations')
        $solverMs = Parse-FirstNumber (Get-HybitField -Lines $section -Name 'solver time')
        $wallMs = Parse-FirstNumber (Get-HybitField -Lines $section -Name 'total wall')
        $results.Add([PSCustomObject]@{
            Repeat = $repeat
            Basis = $basis
            Status = Get-HybitField -Lines $section -Name 'status'
            Iterations = $iterations
            VerifiedResidual = [double]::Parse((Get-HybitField -Lines $section -Name 'verified residual'), $Invariant)
            CoarseDimension = [int](Get-HybitField -Lines $section -Name 'coarse dimension')
            AggregateNodes = [int]((Get-HybitField -Lines $section -Name 'coarse aggregate') -replace '\s+nodes$','')
            TransferValues = Get-HybitField -Lines $section -Name 'coarse values'
            CoarseMiB = Parse-FirstNumber (Get-HybitField -Lines $section -Name 'coarse memory')
            CoarseSetupMs = Parse-FirstNumber (Get-HybitField -Lines $section -Name 'coarse setup')
            SolverMs = $solverMs
            WallMs = $wallMs
            MsPerIter = if ($iterations -gt 0) { $solverMs / $iterations } else { [double]::NaN }
        })
    }
}

Write-Host ''
Write-Host '== per-run summary =='
$runTable = $results | ForEach-Object {
    [PSCustomObject]@{
        Rep = $_.Repeat
        Basis = $_.Basis
        Status = $_.Status
        Iters = $_.Iterations
        Residual = $_.VerifiedResidual.ToString('0.000000E+00', $Invariant)
        Coarse = $_.CoarseDimension
        AggNodes = $_.AggregateNodes
        CoarseMiB = $_.CoarseMiB.ToString('0.00', $Invariant)
        SetupMs = $_.CoarseSetupMs.ToString('0.00', $Invariant)
        MsPerIter = $_.MsPerIter.ToString('0.000', $Invariant)
        SolverMs = $_.SolverMs.ToString('0.00', $Invariant)
        WallMs = $_.WallMs.ToString('0.00', $Invariant)
    }
}
$runTable | Format-Table -AutoSize

$summary = [System.Collections.Generic.List[object]]::new()
foreach ($basis in $Bases) {
    $runs = @($results | Where-Object { $_.Basis -eq $basis })
    if ($runs.Count -eq 0) { continue }
    $conv = @($runs | Where-Object { $_.Status -eq 'Converged' })
    $summary.Add([PSCustomObject]@{
        Basis = $basis
        Runs = $runs.Count
        ConvergedRuns = $conv.Count
        AllConverged = ($conv.Count -eq $runs.Count)
        MedianIterations = Get-Median -Values @($runs | ForEach-Object { [double]$_.Iterations })
        MaxVerifiedResidual = ($runs | Measure-Object -Property VerifiedResidual -Maximum).Maximum
        MedianCoarseDimension = Get-Median -Values @($runs | ForEach-Object { [double]$_.CoarseDimension })
        MedianAggregateNodes = Get-Median -Values @($runs | ForEach-Object { [double]$_.AggregateNodes })
        MedianCoarseMiB = Get-Median -Values @($runs | ForEach-Object { [double]$_.CoarseMiB })
        MedianCoarseSetupMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.CoarseSetupMs })
        MedianMsPerIteration = Get-Median -Values @($runs | ForEach-Object { [double]$_.MsPerIter })
        MedianSolverMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.SolverMs })
        MedianWallMs = Get-Median -Values @($runs | ForEach-Object { [double]$_.WallMs })
    })
}

Write-Host ''
Write-Host '== median summary =='
$summaryTable = $summary | ForEach-Object {
    [PSCustomObject]@{
        Basis = $_.Basis
        Conv = ("{0}/{1}" -f $_.ConvergedRuns, $_.Runs)
        Iters = $_.MedianIterations.ToString('0', $Invariant)
        Residual = ([double]$_.MaxVerifiedResidual).ToString('0.000000E+00', $Invariant)
        Coarse = $_.MedianCoarseDimension.ToString('0', $Invariant)
        AggNodes = $_.MedianAggregateNodes.ToString('0', $Invariant)
        CoarseMiB = $_.MedianCoarseMiB.ToString('0.00', $Invariant)
        SetupMs = $_.MedianCoarseSetupMs.ToString('0.00', $Invariant)
        MsPerIter = $_.MedianMsPerIteration.ToString('0.000', $Invariant)
        SolverMs = $_.MedianSolverMs.ToString('0.00', $Invariant)
        WallMs = $_.MedianWallMs.ToString('0.00', $Invariant)
    }
}
$summaryTable | Format-Table -AutoSize

$results | Export-Csv -LiteralPath $CsvPath -NoTypeInformation -Encoding utf8
$summary | Export-Csv -LiteralPath $SummaryCsvPath -NoTypeInformation -Encoding utf8
Write-Host "Runs CSV: $CsvPath"
Write-Host "Summary CSV: $SummaryCsvPath"
Write-Host '== spectral-coarse sweep complete =='
