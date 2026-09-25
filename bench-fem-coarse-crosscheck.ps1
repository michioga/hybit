param(
    [Parameter(Mandatory=$true)][string]$Matrix,
    [int]$CoarseDofs = 3,
    [int]$CoarseTarget = 1536,
    [int]$MaxIterations = 3000,
    [double]$Tolerance = 1e-8,
    [int]$ProbeIterations = 12
)

$ErrorActionPreference = 'Stop'

Write-Host '== HyBIT algebraic-coarse cross-check =='
Write-Host "Matrix: $Matrix"
Write-Host "Coarse DOFs/node: $CoarseDofs"
Write-Host "Coarse target: $CoarseTarget"
Write-Host

foreach ($mode in @('direct', 'probe-coarse')) {
    Write-Host '============================================================'
    Write-Host " mode = $mode"
    Write-Host '============================================================'
    cargo run --release --example fem_coarse_crosscheck -- `
        --matrix $Matrix `
        --coarse-dofs $CoarseDofs `
        --coarse-target $CoarseTarget `
        --tol $Tolerance `
        --max-iters $MaxIterations `
        --probe-iters $ProbeIterations `
        --mode $mode
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host
}

Write-Host '== algebraic-coarse cross-check complete =='
