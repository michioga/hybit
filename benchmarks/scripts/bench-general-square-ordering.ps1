[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,

    [string] $Rhs,

    [double] $Tolerance = 1.0e-8,

    [int] $MaxIterations = 1000,

    [int] $Restart = 30
)

$ErrorActionPreference = "Stop"

$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if ($Matrix.Count -eq 0) {
    throw "at least one -Matrix path is required"
}
if ($Rhs -and $Matrix.Count -ne 1) {
    throw "-Rhs can be used only when exactly one -Matrix is supplied"
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

Write-Host "=== HyBIT GeneralSquare ILU(0) ordering benchmark ==="
Write-Host ("branch : {0}" -f (git branch --show-current))
Write-Host ("HEAD   : {0}" -f (git rev-parse --short HEAD))
Write-Host ("tol    : {0:e3}" -f $Tolerance)
Write-Host ("iters  : {0}" -f $MaxIterations)
Write-Host ("restart: {0}" -f $Restart)

foreach ($inputPath in $Matrix) {
    $resolvedMatrix = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== matrix: {0} ===" -f $resolvedMatrix)

    $cargoArgs = @(
        "run", "--release", "-p", "hybit",
        "--example", "general_square_ordering", "--",
        "--matrix", $resolvedMatrix,
        "--tol", $Tolerance.ToString("R", [Globalization.CultureInfo]::InvariantCulture),
        "--max-iters", $MaxIterations.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--restart", $Restart.ToString([Globalization.CultureInfo]::InvariantCulture)
    )

    if ($Rhs) {
        $resolvedRhs = (Resolve-Path -LiteralPath $Rhs).Path
        $cargoArgs += @("--rhs", $resolvedRhs)
    }

    & cargo @cargoArgs
    if ($LASTEXITCODE -ne 0) {
        throw "ordering benchmark failed for '$resolvedMatrix' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== ordering benchmark complete ==="