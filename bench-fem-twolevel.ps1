param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Matrix,
    [double]$Tolerance = 1.0e-8,
    [int]$MaxIterations = 3000,
    [int]$DofsPerNode = 3,
    [int]$AggregateNodes = 1024
)

$ErrorActionPreference = "Stop"

Write-Host "== HyBIT 0.6.0 two-level FEM benchmark =="
Write-Host "Matrix: $Matrix"

& cargo run --release -p hybit --example fem_twolevel_bench -- `
    --matrix $Matrix `
    --tol $Tolerance `
    --max-iters $MaxIterations `
    --dofs-per-node $DofsPerNode `
    --aggregate-nodes $AggregateNodes

if ($LASTEXITCODE -ne 0) {
    throw "two-level FEM benchmark failed with exit code $LASTEXITCODE"
}
