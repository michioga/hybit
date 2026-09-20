param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Matrix,
    [double]$Tolerance = 1.0e-8,
    [int]$MaxIterations = 3000,
    [int]$BlockSize = 3
)

$ErrorActionPreference = "Stop"

Write-Host "== HyBIT 0.6.0 block-Jacobi FEM benchmark =="
Write-Host "Matrix: $Matrix"

& cargo run --release -p hybit --example fem_block_bench -- `
    --matrix $Matrix `
    --tol $Tolerance `
    --max-iters $MaxIterations `
    --block-size $BlockSize

if ($LASTEXITCODE -ne 0) {
    throw "block-Jacobi FEM benchmark failed with exit code $LASTEXITCODE"
}
