[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,

    [int] $Repeat = 5,

    [int] $RhsCount = 5,

    [int] $ProbeIterations = 4,

    [double] $PromoteThreshold = 0.5,

    [int] $GuardMinRows = 10000,

    [int] $Restart = 30,

    [int] $MaxIterations = 5000,

    [double] $Tolerance = 1.0e-8
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

$culture = [Globalization.CultureInfo]::InvariantCulture

function Parse-Record([string] $Line, [string] $Prefix) {
    if (-not $Line.StartsWith($Prefix)) {
        throw "record does not start with $Prefix"
    }
    $record = @{}
    foreach ($piece in $Line.Substring($Prefix.Length) -split '\|') {
        $parts = $piece -split '=', 2
        if ($parts.Count -eq 2) {
            $record[$parts[0]] = $parts[1]
        }
    }
    return $record
}

function To-Double([string] $Value) {
    return [double]::Parse(
        $Value,
        [Globalization.NumberStyles]::Float,
        $culture
    )
}

function To-Int([string] $Value) {
    return [int]::Parse(
        $Value,
        [Globalization.NumberStyles]::Integer,
        $culture
    )
}

function Median([double[]] $Values) {
    if ($Values.Count -eq 0) {
        throw "cannot compute median of an empty set"
    }
    $sorted = @($Values | Sort-Object)
    $mid = [int][Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) {
        return [double]$sorted[$mid]
    }
    return ([double]$sorted[$mid - 1] + [double]$sorted[$mid]) / 2.0
}

function F9([double] $Value) {
    return $Value.ToString("0.000000000", $culture)
}

function F6([double] $Value) {
    return $Value.ToString("0.000000", $culture)
}

if ($Matrix.Count -eq 0) { throw "at least one -Matrix path is required" }
if ($Repeat -le 0) { throw "-Repeat must be > 0" }
if ($RhsCount -le 0) { throw "-RhsCount must be > 0" }
if ($ProbeIterations -notin @(4, 8, 16)) {
    throw "-ProbeIterations must be one of 4, 8, 16"
}
if (-not [double]::IsFinite($PromoteThreshold) -or $PromoteThreshold -le 0.0) {
    throw "-PromoteThreshold must be finite and > 0"
}
if ($GuardMinRows -lt 0) { throw "-GuardMinRows must be >= 0" }
if ($Restart -le 0) { throw "-Restart must be > 0" }
if ($MaxIterations -le 0) { throw "-MaxIterations must be > 0" }
if (-not [double]::IsFinite($Tolerance) -or $Tolerance -le 0.0) {
    throw "-Tolerance must be finite and > 0"
}

& cargo build --release -p hybit --example general_square_ordering_paired_probe
if ($LASTEXITCODE -ne 0) {
    throw "failed to build general_square_ordering_paired_probe"
}

$exe = Join-Path $root "target\release\examples\general_square_ordering_paired_probe.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "F6b executable not found: $exe"
}

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== F6c matrix: {0} ===" -f $resolved)

    $runSummaries = @()

    for ($repeatIndex = 1; $repeatIndex -le $Repeat; $repeatIndex++) {
        Write-Host ("-- repeat {0}/{1}" -f $repeatIndex, $Repeat)

        $lines = @(
            & $exe `
                --matrix $resolved `
                --rhs-count $RhsCount `
                --restart $Restart `
                --max-iters $MaxIterations `
                --tol $Tolerance |
            ForEach-Object { [string]$_ }
        )
        if ($LASTEXITCODE -ne 0) {
            throw "F6b executable failed for '$resolved' repeat $repeatIndex"
        }

        $dimensionLine = $lines | Where-Object { $_ -match '^dimensions\s+:' } | Select-Object -First 1
        if (-not $dimensionLine -or $dimensionLine -notmatch ':\s*(\d+)\s+x\s+(\d+)') {
            throw "could not parse dimensions for '$resolved'"
        }
        $nRows = [int]$Matches[1]
        $nCols = [int]$Matches[2]
        if ($nRows -ne $nCols) {
            throw "F6c expects square input"
        }

        $setupLine = $lines | Where-Object { $_.StartsWith("F6B_SETUP|") } | Select-Object -First 1
        if (-not $setupLine) { throw "missing F6B_SETUP record" }
        $setup = Parse-Record $setupLine "F6B_SETUP|"

        $probeLine = $null
        foreach ($line in $lines) {
            if (-not $line.StartsWith("F6B_PROBE|")) { continue }
            $candidate = Parse-Record $line "F6B_PROBE|"
            if ((To-Int $candidate["rhs"]) -eq 1 -and
                (To-Int $candidate["probe_iters"]) -eq $ProbeIterations) {
                $probeLine = $line
                break
            }
        }
        if (-not $probeLine) {
            throw "missing first-RHS F6B_PROBE for $ProbeIterations iterations"
        }
        $probe = Parse-Record $probeLine "F6B_PROBE|"

        $results = @{}
        foreach ($line in $lines) {
            if (-not $line.StartsWith("F6B_RESULT|")) { continue }
            $record = Parse-Record $line "F6B_RESULT|"
            $rhs = To-Int $record["rhs"]
            $results[$rhs] = $record
        }
        if ($results.Count -lt $RhsCount) {
            throw "expected $RhsCount F6B_RESULT records; got $($results.Count)"
        }

        $naturalPrepareMs = To-Double $setup["natural_prepare_ms"]
        $rcmPrepareMs = To-Double $setup["rcm_prepare_ms"]
        $orderingMs = To-Double $setup["ordering_ms"]
        $probeNaturalMs = To-Double $probe["natural_ms"]
        $probeRcmMs = To-Double $probe["rcm_ms"]
        $probeResidualRatio = To-Double $probe["rcm_over_natural_residual"]

        $rawChoice = if ($probeResidualRatio -le $PromoteThreshold) { "rcm" } else { "natural" }
        $guardedChoice = if ($nRows -lt $GuardMinRows) { "natural" } else { $rawChoice }
        $guardApplied = $nRows -lt $GuardMinRows

        $rawOverheadMs =
            $naturalPrepareMs +
            $orderingMs +
            $rcmPrepareMs +
            $probeNaturalMs +
            $probeRcmMs

        $guardedOverheadMs = if ($guardApplied) {
            $naturalPrepareMs
        } else {
            $rawOverheadMs
        }

        $naturalCumulativeSolveMs = 0.0
        $rawChosenCumulativeSolveMs = 0.0
        $guardedChosenCumulativeSolveMs = 0.0

        $horizons = @()

        for ($rhs = 1; $rhs -le $RhsCount; $rhs++) {
            $result = $results[$rhs]
            $naturalSolveMs = To-Double $result["natural_solve_ms"]
            $rcmSolveMs = To-Double $result["rcm_solve_ms"]

            $naturalCumulativeSolveMs += $naturalSolveMs
            $rawChosenCumulativeSolveMs += if ($rawChoice -eq "rcm") { $rcmSolveMs } else { $naturalSolveMs }
            $guardedChosenCumulativeSolveMs += if ($guardedChoice -eq "rcm") { $rcmSolveMs } else { $naturalSolveMs }

            $baselineMs = $naturalPrepareMs + $naturalCumulativeSolveMs
            $rawPolicyMs = $rawOverheadMs + $rawChosenCumulativeSolveMs
            $guardedPolicyMs = $guardedOverheadMs + $guardedChosenCumulativeSolveMs

            $rawRatio = $rawPolicyMs / [Math]::Max($baselineMs, [double]::Epsilon)
            $guardedRatio = $guardedPolicyMs / [Math]::Max($baselineMs, [double]::Epsilon)

            $horizons += [pscustomobject]@{
                Rhs = $rhs
                BaselineMs = $baselineMs
                RawPolicyMs = $rawPolicyMs
                RawRatio = $rawRatio
                GuardedPolicyMs = $guardedPolicyMs
                GuardedRatio = $guardedRatio
            }

            Write-Host (
                "F6C_POLICY_RUN|repeat={0}|n={1}|rhs_count={2}|probe_iters={3}|probe_ratio={4}|threshold={5}|raw_choice={6}|guard_min_rows={7}|guard_applied={8}|guarded_choice={9}|natural_baseline_ms={10}|raw_policy_ms={11}|raw_over_natural={12}|guarded_policy_ms={13}|guarded_over_natural={14}" -f
                $repeatIndex,
                $nRows,
                $rhs,
                $ProbeIterations,
                (F9 $probeResidualRatio),
                (F9 $PromoteThreshold),
                $rawChoice,
                $GuardMinRows,
                $guardApplied.ToString().ToLowerInvariant(),
                $guardedChoice,
                (F6 $baselineMs),
                (F6 $rawPolicyMs),
                (F9 $rawRatio),
                (F6 $guardedPolicyMs),
                (F9 $guardedRatio)
            )
        }

        $runSummaries += [pscustomobject]@{
            Repeat = $repeatIndex
            N = $nRows
            ProbeRatio = $probeResidualRatio
            RawChoice = $rawChoice
            GuardApplied = $guardApplied
            GuardedChoice = $guardedChoice
            Horizons = $horizons
        }
    }

    Write-Host ""
    Write-Host "== F6c median policy replay =="

    for ($rhs = 1; $rhs -le $RhsCount; $rhs++) {
        [double[]]$probeRatios = @($runSummaries | ForEach-Object { $_.ProbeRatio })
        [double[]]$rawRatios = @(
            $runSummaries | ForEach-Object { $_.Horizons[$rhs - 1].RawRatio }
        )
        [double[]]$guardedRatios = @(
            $runSummaries | ForEach-Object { $_.Horizons[$rhs - 1].GuardedRatio }
        )

        $medianProbe = Median $probeRatios
        $medianRaw = Median $rawRatios
        $medianGuarded = Median $guardedRatios
        $rawMin = ($rawRatios | Measure-Object -Minimum).Minimum
        $rawMax = ($rawRatios | Measure-Object -Maximum).Maximum
        $guardedMin = ($guardedRatios | Measure-Object -Minimum).Minimum
        $guardedMax = ($guardedRatios | Measure-Object -Maximum).Maximum

        $rawChoices = @($runSummaries | ForEach-Object { $_.RawChoice } | Sort-Object -Unique)
        $guardedChoices = @($runSummaries | ForEach-Object { $_.GuardedChoice } | Sort-Object -Unique)

        Write-Host (
            "F6C_POLICY_MEDIAN|n={0}|rhs_count={1}|repeats={2}|probe_iters={3}|median_probe_ratio={4}|threshold={5}|raw_choice={6}|median_raw_over_natural={7}|raw_ratio_min={8}|raw_ratio_max={9}|guard_min_rows={10}|guarded_choice={11}|median_guarded_over_natural={12}|guarded_ratio_min={13}|guarded_ratio_max={14}" -f
            $runSummaries[0].N,
            $rhs,
            $Repeat,
            $ProbeIterations,
            (F9 $medianProbe),
            (F9 $PromoteThreshold),
            ($rawChoices -join ","),
            (F9 $medianRaw),
            (F9 $rawMin),
            (F9 $rawMax),
            $GuardMinRows,
            ($guardedChoices -join ","),
            (F9 $medianGuarded),
            (F9 $guardedMin),
            (F9 $guardedMax)
        )
    }
}

Write-Host ""
Write-Host "=== GeneralSquare F6c amortized policy replay complete ==="
