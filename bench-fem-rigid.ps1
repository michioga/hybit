param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Matrix,
    [string]$Coordinates = "",
    [string]$Rhs = "",
    [double]$Tolerance = 1.0e-8,
    [int]$MaxIterations = 3000,
    [int]$AggregateNodes = 1024
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Coordinates)) {
    $Coordinates = [System.IO.Path]::ChangeExtension($Matrix, ".coords")
}
if (-not (Test-Path -LiteralPath $Coordinates)) {
    throw "coordinate sidecar not found: $Coordinates`nRe-run the mf_solver HyBIT exporter to generate it next to the .mtx file."
}
if (-not [string]::IsNullOrWhiteSpace($Rhs) -and -not (Test-Path -LiteralPath $Rhs)) {
    throw "RHS file not found: $Rhs"
}

Write-Host "== HyBIT 0.6.0 rigid-body two-level FEM benchmark =="
Write-Host "Matrix: $Matrix"
Write-Host "Coordinates: $Coordinates"
if (-not [string]::IsNullOrWhiteSpace($Rhs)) {
    Write-Host "RHS: $Rhs"
}

$benchArgs = @(
    "run", "--release", "-p", "hybit", "--example", "fem_rigid_bench", "--",
    "--matrix", $Matrix,
    "--coords", $Coordinates,
    "--tol", $Tolerance,
    "--max-iters", $MaxIterations,
    "--aggregate-nodes", $AggregateNodes
)
if (-not [string]::IsNullOrWhiteSpace($Rhs)) {
    $benchArgs += @("--rhs", $Rhs)
}

& cargo @benchArgs
if ($LASTEXITCODE -ne 0) {
    throw "rigid-body FEM benchmark failed with exit code $LASTEXITCODE"
}
