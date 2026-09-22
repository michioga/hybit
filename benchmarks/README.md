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

## Hybrid local-factor selector A/B benchmark

The 0.7 development line can compare the diagnostic candidate order, raw
residual-energy-per-byte, and Jacobi-energy-per-byte selectors under the same
persistent local-factor memory budget.

```powershell
.\bench-fem-hybrid-selection.ps1 `
  -Matrix D:\Work\mf_solver-hybit-export\L-angle-K.mtx `
  -Rhs D:\Work\mf_rhs\L-angle-b.txt `
  -FactorBudgetMiB 64 `
  -Tolerance 1e-8 `
  -MaxIterations 3000 `
  -MaxEscalations 3 `
  -StageIterations 24
```

The script runs `candidate-order`, `benefit-byte`, and `jacobi-byte` in that
order with otherwise identical solver settings. Compare verified residual,
iterations, escalation count, selected local-region count, persistent factor
memory, budget-skipped regions, local-factor time, solver time, and total wall
time. Use repeated process runs before drawing performance conclusions.
## Hybrid coarse-apply crossover

`bench-fem-hybrid-coarse-apply-crossover.ps1` compares the packed triangular `FactorSolve` and Rayon-parallel `ExplicitInverse` coarse-apply paths across multiple coarse targets. It alternates target and policy order across repeats, reports median wall/solver/setup costs and an approximate break-even iteration count, and writes per-run plus comparison CSV files. The default targets are 768, 1280, 1792, and 2048.

## Hybrid coarse-apply Auto policy

`TwoLevelCoarseApplyPolicy::Auto` resolves from the actual coarse dimension after
aggregation. The current empirical threshold is 1024: smaller coarse systems use
packed `FactorSolve`, while dimensions of 1024 or larger use the Rayon-parallel
`ExplicitInverse` path. This threshold is based on the repeated L-angle crossover
benchmark and remains overrideable with `--coarse-apply factor` or
`--coarse-apply inverse`.


## Hybrid smoothed coarse-basis A/B

`bench-fem-hybrid-coarse-basis.ps1` holds Graph aggregation, the coarse target,
coarse-apply policy, and local-direct settings fixed while alternating the
original piecewise-constant tentative basis with a one-step Jacobi-smoothed
basis. The smoothed path uses a deterministic spectral-radius estimate for its
damping and forms the true Galerkin operator `P^T A P`. Compare iteration count,
coarse setup/memory, solver milliseconds per iteration, solver time, and total
wall time before changing the generic default.

