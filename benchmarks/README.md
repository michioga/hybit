# HyBIT 0.6 FEM benchmark input

`fem_bench` accepts a real or integer Matrix Market coordinate matrix (`.mtx`).
The current solver path expects the constrained linear system to be real SPD.
For symmetric matrices, prefer the Matrix Market `symmetric` header and store one
triangle only; the loader expands it to full CSR32 storage.

Run the bundled smoke matrix:

```powershell
cargo run --release -p hybit --example fem_bench -- --matrix benchmarks/data/poisson5.mtx
```

Run a real assembled FEM stiffness matrix:

```powershell
cargo run --release -p hybit --example fem_bench -- `
  --matrix D:\path\to\K.mtx `
  --tol 1e-8 `
  --max-iters 3000 `
  --overlap 1 `
  --max-region 128 `
  --max-regions 8
```

If `--rhs` is omitted, the benchmark constructs `x_exact = 1` and `b = A*x_exact`.
This isolates the linear solver and gives an independent solution-error check.
To use a physical load vector, pass a whitespace-separated vector with `--rhs`.

The benchmark prints both the solver-reported residual and an independently
recomputed `||Ax-b||/||b||`, plus setup, solve, hard-region and local-factor
metrics. Compare wall-clock results only on sufficiently large problems and use
multiple process runs for performance claims.

## Structural Graph coarse-dimension sweep

After a structural Matrix Market matrix, free-node coordinate sidecar, and optional
physical RHS have been exported, compare several dense coarse-space budgets while
holding Graph aggregation and the Krylov tolerance fixed:

```powershell
.\bench-fem-structural-coarse-sweep.ps1 `
  D:\Work\mf_solver-hybit-export\L-angle-K.mtx `
  -Coordinates D:\Work\mf_solver-hybit-export\L-angle-K.coords `
  -Rhs D:\Work\mf_rhs\L-angle-b.txt `
  -TargetCoarseDimensions 384,768,1536,3072 `
  -Tolerance 1e-8 `
  -MaxIterations 3000
```

The sweep deliberately uses strict `graph` aggregation by default. Compare the
actual coarse dimension, prepare time, iteration count, solve time, and total
setup+solve time. The target is a soft budget; power-of-two aggregate sizing and
graph remainder merging mean the actual coarse dimension can differ slightly.


## Structural r25 development checkpoint

The current 0.6 release-candidate baseline uses `target_coarse_dimension=1536` for the development L-angle case. With 8 Rayon workers, Structural Auto selected Graph aggregation, Parallel CSR SpMV, the Parallel rigid-body preconditioner, and Parallel/fused PCG vectors. The 358065-DOF / 28239653-nnz physical-load case used 233 aggregates, coarse dimension 1398, converged in 220 iterations, independently verified relative residual `9.378557e-9`, and measured 1.726 s solve / 2.797 s analysis+prepare+solve on the Ryzen 7 7800X3D development machine.

Treat these numbers as a regression reference for this matrix, RHS, machine, and revision. They are not a general performance guarantee. The structural production path is feature-frozen at r25 while the 0.6.0 release gate is prepared.
