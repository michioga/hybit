param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Matrix,
    [string]$Rhs,
    [double]$Tolerance = 1.0e-8,
    [int]$MaxIterations = 3000,
    [ValidateSet("auto", "csr", "abtm")]
    [string]$Backend = "auto",
    [int]$ProbeIterations = 12,
    [int]$Overlap = 1,
    [int]$MaxRegion = 128,
    [int]$MaxRegions = 8
)

$ErrorActionPreference = "Stop"

$argsList = @(
    "run", "--release", "-p", "hybit", "--example", "fem_bench", "--",
    "--matrix", $Matrix,
    "--tol", $Tolerance,
    "--max-iters", $MaxIterations,
    "--backend", $Backend,
    "--probe-iters", $ProbeIterations,
    "--overlap", $Overlap,
    "--max-region", $MaxRegion,
    "--max-regions", $MaxRegions
)
if ($Rhs) {
    $argsList += @("--rhs", $Rhs)
}

Write-Host "== HyBIT 0.6.0 FEM benchmark =="
Write-Host "Matrix: $Matrix"
& cargo @argsList
if ($LASTEXITCODE -ne 0) {
    throw "FEM benchmark failed with exit code $LASTEXITCODE"
}
