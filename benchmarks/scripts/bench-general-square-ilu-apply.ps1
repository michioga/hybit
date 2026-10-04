[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]] $Matrix,

    [int] $Samples = 9,

    [int] $Batch = 50,

    [int] $Warmup = 5
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if ($Matrix.Count -eq 0) { throw "at least one -Matrix path is required" }
if ($Samples -lt 3) { throw "-Samples must be >= 3" }
if ($Batch -le 0) { throw "-Batch must be > 0" }
if ($Warmup -lt 0) { throw "-Warmup must be >= 0" }

Write-Host "=== HyBIT GeneralSquare ILU(0) serial triangular-apply profile ==="
Write-Host ("branch  : {0}" -f (git branch --show-current))
Write-Host ("HEAD    : {0}" -f (git rev-parse --short HEAD))
Write-Host ("samples : {0}" -f $Samples)
Write-Host ("batch   : {0}" -f $Batch)
Write-Host ("warmup  : {0}" -f $Warmup)

& cargo build --release -p hybit --example general_square_ilu_apply_profile
if ($LASTEXITCODE -ne 0) {
    throw "failed to build general_square_ilu_apply_profile"
}

$exe = Join-Path $root "target\release\examples\general_square_ilu_apply_profile.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "profile executable not found: $exe"
}

foreach ($inputPath in $Matrix) {
    $resolved = (Resolve-Path -LiteralPath $inputPath).Path
    Write-Host ""
    Write-Host ("=== matrix: {0} ===" -f $resolved)

    & $exe `
        --matrix $resolved `
        --samples $Samples `
        --batch $Batch `
        --warmup $Warmup

    if ($LASTEXITCODE -ne 0) {
        throw "ILU apply profile failed for '$resolved' with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "=== ILU triangular-apply profile complete ==="
