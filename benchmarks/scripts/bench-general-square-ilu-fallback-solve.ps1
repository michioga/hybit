[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,

    [int] $Restart = 30,

    [int] $MaxIterations = 300,

    [double] $Tolerance = 1.0e-8
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if ($Matrix.Count -eq 0) { throw "at least one -Matrix path is required" }
if ($Restart -le 0) { throw "-Restart must be > 0" }
if ($MaxIterations -le 0) { throw "-MaxIterations must be > 0" }
if (-not [double]::IsFinite($Tolerance) -or $Tolerance -le 0.0) {
    throw "-Tolerance must be finite and > 0"
}

Write-Host "=== HyBIT GeneralSquare ILU(0)-fallback bounded solve probe ==="
Write-Host ("branch : {0}" -f (git branch --show-current))
Write-Host ("HEAD   : {0}" -f (git rev-parse --short HEAD))
Write-Host ("restart={0}, max-iters={1}, tol={2:e3}" -f $Restart, $MaxIterations, $Tolerance)

& cargo build --release -p hybit --example general_square_ilu_fallback_solve
if ($LASTEXITCODE -ne 0) {
    throw "failed to build general_square_ilu_fallback_solve"
}

$exe = Join-Path $root "target\release\examples\general_square_ilu_fallback_solve.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "bounded solve executable not found: $exe"
}

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== matrix: {0} ===" -f $resolved)

    & $exe `
        --matrix $resolved `
        --restart $Restart `
        --max-iters $MaxIterations `
        --tol $Tolerance

    if ($LASTEXITCODE -ne 0) {
        throw "bounded fallback solve failed for '$resolved' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== GeneralSquare ILU(0)-fallback bounded solve probe complete ==="
