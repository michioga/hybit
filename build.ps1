$ErrorActionPreference = "Stop"

function Invoke-Checked([string]$Description, [scriptblock]$Command) {
    Write-Host "-- $Description"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

Write-Host "== HyBIT 0.6.0 build =="
Invoke-Checked "cargo test --release" { cargo test --release }
Invoke-Checked "build C ABI DLL" { cargo build --release -p hybit-ffi }
Invoke-Checked "run basic example" { cargo run --release -p hybit --example basic }
Invoke-Checked "run hybrid example" { cargo run --release -p hybit --example hybrid }
Invoke-Checked "run multiregion example" { cargo run --release -p hybit --example multiregion }
Invoke-Checked "run prepared solve-many example" { cargo run --release -p hybit --example prepared }
Invoke-Checked "run Matrix Market benchmark smoke" { cargo run --release -p hybit --example fem_bench -- --matrix benchmarks/data/poisson5.mtx }
Invoke-Checked "run block-Jacobi Matrix Market smoke" { cargo run --release -p hybit --example fem_block_bench -- --matrix benchmarks/data/poisson5.mtx --block-size 2 }
Invoke-Checked "run two-level Matrix Market smoke" { cargo run --release -p hybit --example fem_twolevel_bench -- --matrix benchmarks/data/poisson5.mtx --dofs-per-node 1 --aggregate-nodes 2 }
Invoke-Checked "run rigid-body two-level smoke" { cargo run --release -p hybit --example fem_rigid_bench -- --matrix benchmarks/data/cube8_identity.mtx --coords benchmarks/data/cube8_identity.coords --aggregate-nodes 8 --max-iters 50 }
Invoke-Checked "run structural-auto smoke" { cargo run --release -p hybit --example fem_structural_auto -- --matrix benchmarks/data/cube8_identity.mtx --coords benchmarks/data/cube8_identity.coords --target-coarse-dim 6 --max-iters 50 }
Invoke-Checked "run explicit parallel PCG-vector structural smoke" { cargo run --release -p hybit --example fem_structural_auto -- --matrix benchmarks/data/cube8_identity.mtx --coords benchmarks/data/cube8_identity.coords --target-coarse-dim 6 --aggregation contiguous --spmv serial --precond serial --pcg-vectors parallel --max-iters 50 }
Invoke-Checked "run prepared structural solve-many smoke" { cargo run --release -p hybit --example fem_structural_prepared -- --matrix benchmarks/data/cube8_identity.mtx --coords benchmarks/data/cube8_identity.coords --target-coarse-dim 6 --max-iters 50 --repeats 2 }
Invoke-Checked "run balanced structural smoke" { cargo run --release -p hybit --example fem_structural_balance -- --matrix benchmarks/data/cube8_identity.mtx --coords benchmarks/data/cube8_identity.coords --target-coarse-dim 6 --aggregation contiguous --max-iters 50 }
Invoke-Checked "run structural kernel profile smoke" { cargo run --release -p hybit --example fem_structural_profile -- --matrix benchmarks/data/cube8_identity.mtx --coords benchmarks/data/cube8_identity.coords --target-coarse-dim 6 --aggregation contiguous --max-iters 50 --kernel-repeats 2 }
Invoke-Checked "run parallel CSR structural smoke" { cargo run --release -p hybit --example fem_structural_spmv -- --matrix benchmarks/data/cube8_identity.mtx --coords benchmarks/data/cube8_identity.coords --target-coarse-dim 6 --aggregation contiguous --max-iters 50 --kernel-repeats 2 }
Invoke-Checked "run parallel structural preconditioner smoke" { cargo run --release -p hybit --example fem_structural_precond_parallel -- --matrix benchmarks/data/cube8_identity.mtx --coords benchmarks/data/cube8_identity.coords --target-coarse-dim 6 --aggregation contiguous --max-iters 50 --kernel-repeats 2 }
Invoke-Checked "run structural PCG vector profile smoke" { cargo run --release -p hybit --example fem_structural_pcg_profile -- --matrix benchmarks/data/cube8_identity.mtx --coords benchmarks/data/cube8_identity.coords --target-coarse-dim 6 --aggregation contiguous --max-iters 50 }
Invoke-Checked "run parallel/fused PCG vector smoke" { cargo run --release -p hybit --example fem_structural_pcg_parallel -- --matrix benchmarks/data/cube8_identity.mtx --coords benchmarks/data/cube8_identity.coords --target-coarse-dim 6 --aggregation contiguous --max-iters 50 }

Write-Host ""
Write-Host "Built C ABI DLL under target/release and headers under include/."
Write-Host "To build C/C++/Fortran examples (MSVC Rust + MinGW is supported):"
Write-Host "  .\build-examples.ps1"
