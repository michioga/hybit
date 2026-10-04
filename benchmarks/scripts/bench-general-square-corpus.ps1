[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,

    [double] $Tolerance = 1.0e-8,

    [int] $MaxIterations = 5000,

    [int] $Restart = 30,

    [switch] $PreflightOnly,

    [string] $OutputCsv = ""
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

function Parse-TaggedLine([string] $Line) {
    $parts = $Line -split '\|'
    if ($parts.Count -lt 2) {
        return $null
    }

    $map = @{}
    $map["tag"] = $parts[0]
    for ($i = 1; $i -lt $parts.Count; $i++) {
        $pair = $parts[$i] -split '=', 2
        if ($pair.Count -eq 2) {
            $map[$pair[0]] = $pair[1]
        }
    }
    return $map
}

function Field($Map, [string] $Name, [string] $Default = "") {
    if ($null -ne $Map -and $Map.ContainsKey($Name)) {
        return $Map[$Name]
    }
    return $Default
}

if ($Matrix.Count -eq 0) {
    throw "at least one -Matrix path is required"
}
if ($Tolerance -le 0.0 -or [double]::IsNaN($Tolerance) -or [double]::IsInfinity($Tolerance)) {
    throw "-Tolerance must be finite and > 0"
}
if ($MaxIterations -le 0) {
    throw "-MaxIterations must be > 0"
}
if ($Restart -le 0) {
    throw "-Restart must be > 0"
}

if (-not $OutputCsv) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputCsv = Join-Path $root "benchmarks\results\general-square-corpus-$stamp.csv"
} elseif (-not [IO.Path]::IsPathRooted($OutputCsv)) {
    $OutputCsv = Join-Path $root $OutputCsv
}
$OutputCsv = [IO.Path]::GetFullPath($OutputCsv)
$csvDir = Split-Path -Parent $OutputCsv
if (-not (Test-Path -LiteralPath $csvDir)) {
    New-Item -ItemType Directory -Force -Path $csvDir | Out-Null
}

Write-Host "=== HyBIT GeneralSquare corpus benchmark ==="
Write-Host ("branch         : {0}" -f (git branch --show-current))
Write-Host ("HEAD           : {0}" -f (git rev-parse --short HEAD))
Write-Host ("matrices       : {0}" -f $Matrix.Count)
Write-Host ("tol            : {0:e3}" -f $Tolerance)
Write-Host ("max iterations : {0}" -f $MaxIterations)
Write-Host ("restart        : {0}" -f $Restart)
Write-Host ("preflight-only : {0}" -f [bool]$PreflightOnly)
Write-Host ("CSV            : {0}" -f $OutputCsv)
Write-Host ""

& cargo build --release -p hybit --example general_square_ordering
if ($LASTEXITCODE -ne 0) {
    throw "failed to build general_square_ordering"
}

$exe = Join-Path $root "target\release\examples\general_square_ordering.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "ordering benchmark executable was not found: $exe"
}

$rows = [Collections.Generic.List[object]]::new()

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== matrix: {0} ===" -f $resolved)

    $runArgs = @(
        "--matrix", $resolved,
        "--tol", $Tolerance.ToString("R", [Globalization.CultureInfo]::InvariantCulture),
        "--max-iters", $MaxIterations.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--restart", $Restart.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
    if ($PreflightOnly) {
        $runArgs += "--preflight-only"
    }

    $lines = [Collections.Generic.List[string]]::new()
    & $exe @runArgs 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Write-Host $line
        $lines.Add($line)
    }
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        $rows.Add([pscustomobject]@{
            Matrix = $resolved
            Supported = ""
            Reason = "process_error_$exitCode"
            NaturalBandwidth = ""
            RcmBandwidth = ""
            BandwidthRatio = ""
            OrderingMs = ""
            Ordering = ""
            Preconditioner = ""
            Status = "ERROR"
            Iterations = ""
            ReportedResidual = ""
            VerifiedResidual = ""
            AnalysisMs = ""
            PrepareMs = ""
            SolveMs = ""
            SolveWallMs = ""
            PreconditionerBytes = ""
            WorkspaceBytes = ""
            AdjustedPivots = ""
            Error = "benchmark process exited with $exitCode"
        })
        continue
    }

    $preflight = $null
    $ordering = $null
    $resultMaps = [Collections.Generic.List[object]]::new()

    foreach ($line in $lines) {
        if ($line.StartsWith("PREFLIGHT|")) {
            $preflight = Parse-TaggedLine $line
        } elseif ($line.StartsWith("ORDERING|")) {
            $ordering = Parse-TaggedLine $line
        } elseif ($line.StartsWith("RESULT|")) {
            $resultMaps.Add((Parse-TaggedLine $line))
        }
    }

    if ($null -eq $preflight) {
        throw "matrix '$resolved' produced no PREFLIGHT record"
    }

    $supported = Field $preflight "supported"
    if ($supported -ne "true") {
        $rows.Add([pscustomobject]@{
            Matrix = $resolved
            Supported = $supported
            Reason = Field $preflight "reason"
            NaturalBandwidth = ""
            RcmBandwidth = ""
            BandwidthRatio = ""
            OrderingMs = ""
            Ordering = ""
            Preconditioner = ""
            Status = "SKIP"
            Iterations = ""
            ReportedResidual = ""
            VerifiedResidual = ""
            AnalysisMs = ""
            PrepareMs = ""
            SolveMs = ""
            SolveWallMs = ""
            PreconditionerBytes = ""
            WorkspaceBytes = ""
            AdjustedPivots = ""
            Error = "missing_diagonal=$(Field $preflight 'missing_diagonal'); zero_diagonal=$(Field $preflight 'zero_diagonal')"
        })
        continue
    }

    if ($PreflightOnly) {
        $rows.Add([pscustomobject]@{
            Matrix = $resolved
            Supported = "true"
            Reason = "ok"
            NaturalBandwidth = ""
            RcmBandwidth = ""
            BandwidthRatio = ""
            OrderingMs = ""
            Ordering = ""
            Preconditioner = ""
            Status = "READY"
            Iterations = ""
            ReportedResidual = ""
            VerifiedResidual = ""
            AnalysisMs = ""
            PrepareMs = ""
            SolveMs = ""
            SolveWallMs = ""
            PreconditionerBytes = ""
            WorkspaceBytes = ""
            AdjustedPivots = ""
            Error = ""
        })
        continue
    }

    if ($resultMaps.Count -eq 0) {
        throw "matrix '$resolved' was supported but produced no RESULT records"
    }

    foreach ($result in $resultMaps) {
        $rows.Add([pscustomobject]@{
            Matrix = $resolved
            Supported = "true"
            Reason = "ok"
            NaturalBandwidth = Field $ordering "natural_bandwidth"
            RcmBandwidth = Field $ordering "rcm_bandwidth"
            BandwidthRatio = Field $ordering "bandwidth_ratio"
            OrderingMs = Field $ordering "ordering_ms"
            Ordering = Field $result "ordering"
            Preconditioner = Field $result "preconditioner"
            Status = Field $result "status"
            Iterations = Field $result "iterations"
            ReportedResidual = Field $result "reported_residual"
            VerifiedResidual = Field $result "verified_residual"
            AnalysisMs = Field $result "analysis_ms"
            PrepareMs = Field $result "prepare_ms"
            SolveMs = Field $result "solve_ms"
            SolveWallMs = Field $result "solve_wall_ms"
            PreconditionerBytes = Field $result "preconditioner_bytes"
            WorkspaceBytes = Field $result "workspace_bytes"
            AdjustedPivots = Field $result "adjusted_pivots"
            Error = Field $result "error"
        })
    }
}

$rows | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding utf8

Write-Host ""
Write-Host "=== corpus summary ==="
$rows | Format-Table Matrix, Supported, Ordering, Preconditioner, Status, Iterations, VerifiedResidual, SolveWallMs -AutoSize
Write-Host ""
Write-Host ("CSV: {0}" -f $OutputCsv)
Write-Host "=== GeneralSquare corpus benchmark complete ==="