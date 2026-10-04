[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,
    [int] $Restart = 30,
    [double] $Tolerance = 1.0e-8
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if ($Matrix.Count -eq 0) {
    throw "at least one -Matrix path is required"
}
if ($Restart -le 0) {
    throw "-Restart must be > 0"
}
if (-not [double]::IsFinite($Tolerance) -or $Tolerance -le 0.0) {
    throw "-Tolerance must be finite and > 0"
}

& cargo build --release -p hybit --example general_square_preconditioner_paired_probe
if ($LASTEXITCODE -ne 0) {
    throw "failed to build general_square_preconditioner_paired_probe"
}

$exe = Join-Path $root "target\release\examples\general_square_preconditioner_paired_probe.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "F7c executable not found: $exe"
}

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== F7c matrix: {0} ===" -f $resolved)

    & $exe `
        --matrix $resolved `
        --restart $Restart `
        --tol $Tolerance

    if ($LASTEXITCODE -ne 0) {
        throw "F7c paired probe failed for '$resolved' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== GeneralSquare F7c Jacobi-vs-ILU0 paired Krylov probe complete ==="
