param(
    [string]$Root = 'D:\Work',
    [int]$Repeats = 1,
    [double]$Tolerance = 1.0e-8,
    [int]$MaxIterations = 3000,
    [int]$FactorBudgetMiB = 64,
    [double]$EscalationRatio = 1.0e-4,
    [int]$MaxRegion = 128,
    [int]$MaxRegions = 8,
    [int]$Overlap = 1,
    [int]$StageIterations = 24,
    [int]$DefaultCoarseTarget = 1536,
    [int]$LAngleCoarseTarget = 1792,
    [ValidateSet('auto','factor','inverse')][string]$CoarseApply = 'auto',
    [ValidateSet('auto','csr','abtm')][string]$Backend = 'auto',
    [string[]]$Include = @(),
    [double]$NearTiePercent = 3.0,
    [string]$CsvPath = 'hybit-hybrid-watchdog-proxy.csv',
    [string]$SummaryCsvPath = 'hybit-hybrid-watchdog-proxy-summary.csv'
)

$ErrorActionPreference = 'Stop'
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path -LiteralPath $Root)) { throw "Root directory not found: $Root" }
if ($Repeats -le 0) { throw 'Repeats must be > 0' }
if ($Tolerance -le 0.0 -or [double]::IsNaN($Tolerance) -or [double]::IsInfinity($Tolerance)) { throw 'Tolerance must be finite and > 0' }
if ($MaxIterations -le 0) { throw 'MaxIterations must be > 0' }
if ($FactorBudgetMiB -le 0) { throw 'FactorBudgetMiB must be > 0' }
if ($EscalationRatio -le 0.0 -or [double]::IsNaN($EscalationRatio) -or [double]::IsInfinity($EscalationRatio)) { throw 'EscalationRatio must be finite and > 0' }
if ($MaxRegion -le 0) { throw 'MaxRegion must be > 0' }
if ($MaxRegions -le 0) { throw 'MaxRegions must be > 0' }
if ($Overlap -lt 0) { throw 'Overlap must be >= 0' }
if ($StageIterations -le 0) { throw 'StageIterations must be > 0' }
if ($DefaultCoarseTarget -le 0) { throw 'DefaultCoarseTarget must be > 0' }
if ($LAngleCoarseTarget -le 0) { throw 'LAngleCoarseTarget must be > 0' }
if ($NearTiePercent -lt 0.0 -or [double]::IsNaN($NearTiePercent) -or [double]::IsInfinity($NearTiePercent)) { throw 'NearTiePercent must be finite and >= 0' }

function Join-RootPath {
    param([string]$RelativePath)
    return Join-Path -Path $Root -ChildPath $RelativePath
}

# This is deliberately a screening manifest, not a claim that every matrix has
# the same physical DOF layout.  DofsPerNode=3 is used only for the structural
# matrices whose dimensions are compatible with the node-major 3-DOF hypothesis;
# scalar/unknown layouts use 1.  Follow-up runs can override/extend this script
# after the first pass identifies useful matrices.
$cases = @(
    [PSCustomObject]@{ Name='nd3k';     Matrix=(Join-RootPath 'nd3k.mtx');                              Rhs=$null;                                      DofsPerNode=1; CoarseTarget=$DefaultCoarseTarget; ProbeIterations=96; Layout='scalar/unknown' },
    [PSCustomObject]@{ Name='s3dkq4m2'; Matrix=(Join-RootPath 's3dkq4m2.mtx');                          Rhs=$null;                                      DofsPerNode=1; CoarseTarget=$DefaultCoarseTarget; ProbeIterations=48; Layout='unknown/non-3-divisible' },
    [PSCustomObject]@{ Name='x104';     Matrix=(Join-RootPath 'x104.mtx');                              Rhs=$null;                                      DofsPerNode=3; CoarseTarget=$DefaultCoarseTarget; ProbeIterations=96; Layout='structural-3dof-assumed' },
    [PSCustomObject]@{ Name='boneS01';  Matrix=(Join-RootPath 'boneS01.mtx');                           Rhs=$null;                                      DofsPerNode=3; CoarseTarget=$DefaultCoarseTarget; ProbeIterations=96; Layout='structural-3dof' },
    [PSCustomObject]@{ Name='L-angle';  Matrix=(Join-RootPath 'mf_solver-hybit-export\L-angle-K.mtx'); Rhs=(Join-RootPath 'mf_rhs\L-angle-b.txt'); DofsPerNode=3; CoarseTarget=$LAngleCoarseTarget; ProbeIterations=12; Layout='structural-3dof' }
)

if ($Include.Count -gt 0) {
    $requested = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($name in $Include) { [void]$requested.Add($name) }
    $known = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($case in $cases) { [void]$known.Add($case.Name) }
    foreach ($name in $Include) {
        if (-not $known.Contains($name)) {
            throw "Unknown matrix name '$name'. Known names: $($cases.Name -join ', ')"
        }
    }
    $cases = @($cases | Where-Object { $requested.Contains($_.Name) })
}

$available = [System.Collections.Generic.List[object]]::new()
foreach ($case in $cases) {
    if (-not (Test-Path -LiteralPath $case.Matrix)) {
        Write-Warning "Skipping $($case.Name): matrix not found: $($case.Matrix)"
        continue
    }
    if ($case.Rhs -and -not (Test-Path -LiteralPath $case.Rhs)) {
        Write-Warning "Skipping $($case.Name): RHS not found: $($case.Rhs)"
        continue
    }
    $available.Add($case)
}
if ($available.Count -eq 0) { throw 'No benchmark matrices are available.' }

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
    return @()
}

function Parse-DoublePrefix {
    param([string]$Value, [double]$Default = [double]::NaN)
    if ($null -ne $Value -and $Value -match '^([0-9.Ee+-]+)') {
        return [double]::Parse($Matches[1], $Invariant)
    }
    return $Default
}

function Parse-IntField {
    param([string]$Value, [int]$Default = -1)
    if ($null -ne $Value -and $Value -match '^([0-9]+)') {
        return [int]$Matches[1]
    }
    return $Default
}

function Test-Finite {
    param([double]$Value)
    return -not [double]::IsNaN($Value) -and -not [double]::IsInfinity($Value)
}

function Parse-ProbeMilliseconds {
    param([string]$Value)
    if ($null -ne $Value -and $Value -match '^[0-9]+\s+iters,\s+([0-9.Ee+-]+)') {
        return [double]::Parse($Matches[1], $Invariant)
    }
    return [double]::NaN
}

function Get-Median {
    param([double[]]$Values)
    $finite = @($Values | Where-Object { Test-Finite $_ } | Sort-Object)
    if ($finite.Count -eq 0) { return [double]::NaN }
    $mid = [int][Math]::Floor($finite.Count / 2.0)
    if (($finite.Count % 2) -eq 1) { return [double]$finite[$mid] }
    return ([double]$finite[$mid - 1] + [double]$finite[$mid]) / 2.0
}

function Invoke-ScreenRun {
    param(
        [Parameter(Mandatory=$true)]$Case,
        [Parameter(Mandatory=$true)][double]$EnergyMin,
        [Parameter(Mandatory=$true)][int]$Repeat
    )

    Write-Host ''
    Write-Host '============================================================'
    Write-Host (" {0} : repeat {1}/{2}, local-energy-min = {3}" -f $Case.Name, $Repeat, $Repeats, $EnergyMin.ToString('0.###', $Invariant))
    Write-Host '============================================================'

    $cargoArgs = @(
        'run', '--release', '-p', 'hybit', '--example', 'fem_bench', '--',
        '--matrix', $Case.Matrix,
        '--tol', $Tolerance.ToString('R', $Invariant),
        '--max-iters', $MaxIterations.ToString($Invariant),
        '--backend', $Backend,
        '--probe-iters', $Case.ProbeIterations.ToString($Invariant),
        '--overlap', $Overlap.ToString($Invariant),
        '--max-region', $MaxRegion.ToString($Invariant),
        '--max-regions', $MaxRegions.ToString($Invariant),
        '--factor-budget-mib', $FactorBudgetMiB.ToString($Invariant),
        '--max-escalations', '1',
        '--stage-iters', $StageIterations.ToString($Invariant),
        '--escalation-ratio', $EscalationRatio.ToString('R', $Invariant),
        '--selector', 'jacobi-byte',
        '--skip-plain',
        '--hybrid-coarse',
        '--coarse-dofs', $Case.DofsPerNode.ToString($Invariant),
        '--coarse-target', $Case.CoarseTarget.ToString($Invariant),
        '--coarse-apply', $CoarseApply,
        '--coarse-aggregation', 'graph',
        '--coarse-basis', 'smoothed',
        '--coarse-transfer', 'parallel',
        '--coarse-transfer-storage', 'wide',
        '--coarse-transfer-values', 'f32',
        '--local-energy-min', $EnergyMin.ToString('R', $Invariant)
    )
    if ($Case.Rhs) { $cargoArgs += @('--rhs', $Case.Rhs) }

    $captured = [System.Collections.Generic.List[string]]::new()
    & cargo @cargoArgs 2>&1 | ForEach-Object {
        $line = $_.ToString()
        $captured.Add($line)
        Write-Host $line
    }
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        $tail = @($captured | Select-Object -Last 8) -join ' | '
        return [PSCustomObject]@{
            Matrix=$Case.Name; Layout=$Case.Layout; RhsMode=$(if ($Case.Rhs) {'file'} else {'A*1'});
            DofsPerNode=$Case.DofsPerNode; CoarseTarget=$Case.CoarseTarget; ProbeIterations=$Case.ProbeIterations; Repeat=$Repeat; EnergyMin=$EnergyMin;
            Status='ERROR'; ExitCode=$exitCode; Iterations=-1; VerifiedResidual=[double]::NaN; Escalations=-1;
            HardDofs=-1; LocalRegions=-1; FactorMiB=[double]::NaN; CoarseMiB=[double]::NaN;
            ProbeMs=[double]::NaN; DiagnosticsMs=[double]::NaN; LocalFactorMs=[double]::NaN;
            SolverMs=[double]::NaN; WallMs=[double]::NaN; Error=$tail
        }
    }

    $section = Get-HybitSection -Lines $captured.ToArray()
    if ($section.Count -eq 0) {
        return [PSCustomObject]@{
            Matrix=$Case.Name; Layout=$Case.Layout; RhsMode=$(if ($Case.Rhs) {'file'} else {'A*1'});
            DofsPerNode=$Case.DofsPerNode; CoarseTarget=$Case.CoarseTarget; ProbeIterations=$Case.ProbeIterations; Repeat=$Repeat; EnergyMin=$EnergyMin;
            Status='PARSE_ERROR'; ExitCode=0; Iterations=-1; VerifiedResidual=[double]::NaN; Escalations=-1;
            HardDofs=-1; LocalRegions=-1; FactorMiB=[double]::NaN; CoarseMiB=[double]::NaN;
            ProbeMs=[double]::NaN; DiagnosticsMs=[double]::NaN; LocalFactorMs=[double]::NaN;
            SolverMs=[double]::NaN; WallMs=[double]::NaN; Error='HyBIT Auto section not found'
        }
    }

    [PSCustomObject]@{
        Matrix = $Case.Name
        Layout = $Case.Layout
        RhsMode = $(if ($Case.Rhs) {'file'} else {'A*1'})
        DofsPerNode = $Case.DofsPerNode
        CoarseTarget = $Case.CoarseTarget
        ProbeIterations = $Case.ProbeIterations
        Repeat = $Repeat
        EnergyMin = $EnergyMin
        Status = Get-HybitField -Lines $section -Name 'status'
        ExitCode = 0
        Iterations = Parse-IntField (Get-HybitField -Lines $section -Name 'iterations')
        VerifiedResidual = Parse-DoublePrefix (Get-HybitField -Lines $section -Name 'verified residual')
        Escalations = Parse-IntField (Get-HybitField -Lines $section -Name 'escalations')
        HardDofs = Parse-IntField (Get-HybitField -Lines $section -Name 'hard core DOFs')
        LocalRegions = Parse-IntField (Get-HybitField -Lines $section -Name 'local regions')
        FactorMiB = Parse-DoublePrefix (Get-HybitField -Lines $section -Name 'factor memory')
        CoarseMiB = Parse-DoublePrefix (Get-HybitField -Lines $section -Name 'coarse memory')
        ProbeMs = Parse-ProbeMilliseconds (Get-HybitField -Lines $section -Name 'probe')
        DiagnosticsMs = Parse-DoublePrefix (Get-HybitField -Lines $section -Name 'diagnostics')
        LocalFactorMs = Parse-DoublePrefix (Get-HybitField -Lines $section -Name 'local factor')
        SolverMs = Parse-DoublePrefix (Get-HybitField -Lines $section -Name 'solver time')
        WallMs = Parse-DoublePrefix (Get-HybitField -Lines $section -Name 'total wall')
        Error = ''
    }
}

Write-Host '== HyBIT watchdog-proxy local-direct 0-vs-1 screening =='
Write-Host "Root: $Root"
Write-Host "Matrices: $($available.Name -join ', ')"
Write-Host "Repeats: $Repeats"
Write-Host "Escalation ratio proxy: $($EscalationRatio.ToString('0.000000E+00', $Invariant))"
Write-Host 'Probe checkpoints come from r34 first normalized poor-progress observations.'
Write-Host 'Gate 0.0 admits local-direct after the proxy trigger; Gate 1.0 suppresses it unless all Jacobi energy is localized.'
Write-Host 'boneS01 is the false-positive control: its 96-iteration residual is already below the proxy threshold, so it should not escalate.'
Write-Host 'No-RHS matrices use deterministic b=A*1; treat those rows as controller screening, not a physical-load validation.'
Write-Host 'Coarse path: graph / smoothed / parallel / wide / f32; max escalations = 1.'

$results = [System.Collections.Generic.List[object]]::new()
for ($caseIndex = 0; $caseIndex -lt $available.Count; $caseIndex++) {
    $case = $available[$caseIndex]
    for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
        # Alternate order by case/repeat to reduce systematic thermal/order bias.
        $reverse = (($caseIndex + $repeat) % 2) -eq 0
        $thresholds = if ($reverse) { @(1.0, 0.0) } else { @(0.0, 1.0) }
        foreach ($threshold in $thresholds) {
            $results.Add((Invoke-ScreenRun -Case $case -EnergyMin $threshold -Repeat $repeat))
        }
    }
}

Write-Host ''
Write-Host '== per-run screening summary =='
$results | ForEach-Object {
    [PSCustomObject]@{
        Matrix=$_.Matrix; RHS=$_.RhsMode; DOFs=$_.DofsPerNode; Target=$_.CoarseTarget; Probe=$_.ProbeIterations;
        Rep=$_.Repeat; Gate=$_.EnergyMin.ToString('0.###', $Invariant); Status=$_.Status;
        Iters=$_.Iterations; Esc=$_.Escalations; Regions=$_.LocalRegions;
        FactorMiB=$(if (Test-Finite $_.FactorMiB) {$_.FactorMiB.ToString('0.00', $Invariant)} else {'NaN'});
        SolverMs=$(if (Test-Finite $_.SolverMs) {$_.SolverMs.ToString('0.0', $Invariant)} else {'NaN'});
        WallMs=$(if (Test-Finite $_.WallMs) {$_.WallMs.ToString('0.0', $Invariant)} else {'NaN'})
    }
} | Format-Table -AutoSize

$summary = [System.Collections.Generic.List[object]]::new()
foreach ($case in $available) {
    foreach ($threshold in @(0.0, 1.0)) {
        $runs = @($results | Where-Object { $_.Matrix -eq $case.Name -and $_.EnergyMin -eq $threshold })
        $valid = @($runs | Where-Object { $_.ExitCode -eq 0 -and $_.Status -ne 'PARSE_ERROR' })
        $converged = @($valid | Where-Object { $_.Status -eq 'Converged' })
        $summary.Add([PSCustomObject]@{
            Matrix = $case.Name
            Layout = $case.Layout
            RhsMode = $(if ($case.Rhs) {'file'} else {'A*1'})
            DofsPerNode = $case.DofsPerNode
            CoarseTarget = $case.CoarseTarget
            ProbeIterations = $case.ProbeIterations
            EnergyMin = $threshold
            Converged = "$($converged.Count)/$($runs.Count)"
            MedianIterations = Get-Median -Values @($valid | ForEach-Object { [double]$_.Iterations })
            MedianEscalations = Get-Median -Values @($valid | ForEach-Object { [double]$_.Escalations })
            MedianLocalRegions = Get-Median -Values @($valid | ForEach-Object { [double]$_.LocalRegions })
            MedianFactorMiB = Get-Median -Values @($valid | ForEach-Object { [double]$_.FactorMiB })
            MedianDiagnosticsMs = Get-Median -Values @($valid | ForEach-Object { [double]$_.DiagnosticsMs })
            MedianSolverMs = Get-Median -Values @($valid | ForEach-Object { [double]$_.SolverMs })
            MedianWallMs = Get-Median -Values @($valid | ForEach-Object { [double]$_.WallMs })
            MedianVerifiedResidual = Get-Median -Values @($valid | ForEach-Object { [double]$_.VerifiedResidual })
        })
    }
}

Write-Host ''
Write-Host '== median by matrix / gate =='
$summary | ForEach-Object {
    [PSCustomObject]@{
        Matrix=$_.Matrix; RHS=$_.RhsMode; Probe=$_.ProbeIterations; Gate=$_.EnergyMin.ToString('0.###', $Invariant);
        Conv=$_.Converged;
        Iters=$(if (Test-Finite $_.MedianIterations) {[int]$_.MedianIterations} else {-1});
        Esc=$(if (Test-Finite $_.MedianEscalations) {$_.MedianEscalations.ToString('0', $Invariant)} else {'NaN'});
        Regions=$(if (Test-Finite $_.MedianLocalRegions) {$_.MedianLocalRegions.ToString('0', $Invariant)} else {'NaN'});
        FactorMiB=$(if (Test-Finite $_.MedianFactorMiB) {$_.MedianFactorMiB.ToString('0.00', $Invariant)} else {'NaN'});
        SolverMs=$(if (Test-Finite $_.MedianSolverMs) {$_.MedianSolverMs.ToString('0.0', $Invariant)} else {'NaN'});
        WallMs=$(if (Test-Finite $_.MedianWallMs) {$_.MedianWallMs.ToString('0.0', $Invariant)} else {'NaN'})
    }
} | Format-Table -AutoSize

$comparisons = [System.Collections.Generic.List[object]]::new()
foreach ($case in $available) {
    $gate0 = $summary | Where-Object { $_.Matrix -eq $case.Name -and $_.EnergyMin -eq 0.0 } | Select-Object -First 1
    $gate1 = $summary | Where-Object { $_.Matrix -eq $case.Name -and $_.EnergyMin -eq 1.0 } | Select-Object -First 1

    $bothUsable = $null -ne $gate0 -and $null -ne $gate1 -and
                  (Test-Finite $gate0.MedianWallMs) -and (Test-Finite $gate1.MedianWallMs) -and
                  $gate0.MedianWallMs -gt 0.0
    if (-not $bothUsable) {
        $comparisons.Add([PSCustomObject]@{
            Matrix=$case.Name; RhsMode=$(if ($case.Rhs) {'file'} else {'A*1'}); ProbeIterations=$case.ProbeIterations;
            Gate0Conv=$gate0.Converged; Gate1Conv=$gate1.Converged; Gate0Residual=[double]::NaN; Gate1Residual=[double]::NaN;
            Gate0Iters=-1; Gate1Iters=-1; IterDelta=[double]::NaN;
            Gate0Esc=[double]::NaN; Gate1Esc=[double]::NaN; Gate0Regions=[double]::NaN; Gate1Regions=[double]::NaN;
            Gate0WallMs=[double]::NaN; Gate1WallMs=[double]::NaN; Gate1GainPct=[double]::NaN;
            Classification='INCOMPLETE'
        })
        continue
    }

    $gain = 100.0 * ($gate0.MedianWallMs - $gate1.MedianWallMs) / $gate0.MedianWallMs
    $gate0AllConverged = $gate0.Converged -eq "$Repeats/$Repeats"
    $gate1AllConverged = $gate1.Converged -eq "$Repeats/$Repeats"
    $classification = if ($gate0AllConverged -and -not $gate1AllConverged) {
        'Gate0Converges'
    } elseif ($gate1AllConverged -and -not $gate0AllConverged) {
        'Gate1Converges'
    } elseif ($gate0AllConverged -and $gate1AllConverged) {
        if ([Math]::Abs($gain) -lt $NearTiePercent) { 'NearTie' } elseif ($gain -gt 0.0) { 'Gate1Faster' } else { 'Gate0Faster' }
    } else {
        'BothUnconverged'
    }

    $comparisons.Add([PSCustomObject]@{
        Matrix=$case.Name
        RhsMode=$(if ($case.Rhs) {'file'} else {'A*1'})
        ProbeIterations=$case.ProbeIterations
        Gate0Conv=$gate0.Converged
        Gate1Conv=$gate1.Converged
        Gate0Residual=$gate0.MedianVerifiedResidual
        Gate1Residual=$gate1.MedianVerifiedResidual
        Gate0Iters=[int]$gate0.MedianIterations
        Gate1Iters=[int]$gate1.MedianIterations
        IterDelta=[int]($gate1.MedianIterations - $gate0.MedianIterations)
        Gate0Esc=$gate0.MedianEscalations
        Gate1Esc=$gate1.MedianEscalations
        Gate0Regions=$gate0.MedianLocalRegions
        Gate1Regions=$gate1.MedianLocalRegions
        Gate0WallMs=$gate0.MedianWallMs
        Gate1WallMs=$gate1.MedianWallMs
        Gate1GainPct=$gain
        Classification=$classification
    })
}

Write-Host ''
Write-Host '== watchdog-proxy local-vs-coarse-only comparison =='
Write-Host "NearTie means absolute wall difference < $($NearTiePercent.ToString('0.0', $Invariant))%."
$comparisons | ForEach-Object {
    [PSCustomObject]@{
        Matrix=$_.Matrix; RHS=$_.RhsMode; Probe=$_.ProbeIterations; Conv0=$_.Gate0Conv; Conv1=$_.Gate1Conv;
        Iters0=$_.Gate0Iters; Iters1=$_.Gate1Iters;
        Esc0=$(if (Test-Finite $_.Gate0Esc) {$_.Gate0Esc.ToString('0', $Invariant)} else {'NaN'});
        Esc1=$(if (Test-Finite $_.Gate1Esc) {$_.Gate1Esc.ToString('0', $Invariant)} else {'NaN'});
        Reg0=$(if (Test-Finite $_.Gate0Regions) {$_.Gate0Regions.ToString('0', $Invariant)} else {'NaN'});
        Reg1=$(if (Test-Finite $_.Gate1Regions) {$_.Gate1Regions.ToString('0', $Invariant)} else {'NaN'});
        Res0=$(if (Test-Finite $_.Gate0Residual) {$_.Gate0Residual.ToString('0.000E+00', $Invariant)} else {'NaN'});
        Res1=$(if (Test-Finite $_.Gate1Residual) {$_.Gate1Residual.ToString('0.000E+00', $Invariant)} else {'NaN'});
        Wall0=$(if (Test-Finite $_.Gate0WallMs) {$_.Gate0WallMs.ToString('0.0', $Invariant)} else {'NaN'});
        Wall1=$(if (Test-Finite $_.Gate1WallMs) {$_.Gate1WallMs.ToString('0.0', $Invariant)} else {'NaN'});
        Gate1GainPct=$(if (Test-Finite $_.Gate1GainPct) {$_.Gate1GainPct.ToString('0.00', $Invariant)} else {'NaN'});
        Class=$_.Classification
    }
} | Format-Table -AutoSize

$results | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $CsvPath
$summary | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $SummaryCsvPath
$comparisonPath = [System.IO.Path]::ChangeExtension($SummaryCsvPath, $null) + '-comparison.csv'
$comparisons | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $comparisonPath
Write-Host "Runs CSV      : $CsvPath"
Write-Host "Summary CSV   : $SummaryCsvPath"
Write-Host "Comparison CSV: $comparisonPath"
Write-Host '== watchdog-proxy screening complete =='
