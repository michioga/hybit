[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if ($Matrix.Count -eq 0) { throw "at least one -Matrix path is required" }

Write-Host "=== HyBIT GeneralSquare ILU(0) fallback preflight ==="
Write-Host ("branch : {0}" -f (git branch --show-current))
Write-Host ("HEAD   : {0}" -f (git rev-parse --short HEAD))

& cargo build --release -p hybit --example general_square_ilu_fallback
if ($LASTEXITCODE -ne 0) {
    throw "failed to build general_square_ilu_fallback"
}

$exe = Join-Path $root "target\release\examples\general_square_ilu_fallback.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "fallback preflight executable not found: $exe"
}

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== matrix: {0} ===" -f $resolved)

    & $exe --matrix $resolved

    if ($LASTEXITCODE -ne 0) {
        throw "fallback preflight failed for '$resolved' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== GeneralSquare ILU(0) fallback preflight complete ==="
