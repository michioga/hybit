# Structural Auto API (0.6 development)

HyBIT's generic `solve_csr32` path remains geometry-free and backward compatible.
For 3-D structural SPD systems, callers can provide node coordinates explicitly:

```rust
use hybit::{
    HybitSolver, RigidBodyAggregation, SolverOptions, StructuralOptions,
    StructuralPcgVectorPolicy, StructuralPreconditionerPolicy, StructuralSpmvPolicy,
};

let mut solver = HybitSolver::new();
solver.set_options(SolverOptions {
    relative_tolerance: 1.0e-8,
    absolute_tolerance: 0.0,
    max_iterations: 3000,
})?;
solver.set_structural_options(StructuralOptions {
    target_coarse_dimension: 1536,
    aggregation: RigidBodyAggregation::Auto,
    spmv_policy: StructuralSpmvPolicy::Auto,
    preconditioner_policy: StructuralPreconditionerPolicy::Auto,
    pcg_vector_policy: StructuralPcgVectorPolicy::Auto,
})?;

let report = solver.solve_structural_csr32(&a, &coordinates, &b, &mut x)?;
```

For solve-many workloads, prepare once and reuse the rigid-body coarse factor:

```rust
let analysis = solver.analyze_csr32(&a)?;
let mut prepared = solver.prepare_structural_csr32(&a, &analysis, &coordinates)?;
let first = prepared.solve(&a, &b1, &mut x1)?;
let second = prepared.solve(&a, &b2, &mut x2)?;
```

`HybitPreparedStructuralSystem` exposes the selected aggregate size, aggregate count,
coarse dimension, factor storage, geometry storage, and workspace size.

The structural path currently assumes exactly three displacement DOFs per node and
uses six rigid-body modes per aggregate. `RigidBodyAggregation::Auto` is the default: it
tries deterministic graph-connected aggregation first and falls back to the contiguous
RCM-order baseline if the graph coarse Cholesky is numerically singular or if the structural
node graph contains disconnected components too small to support a six-mode rigid-body aggregate.
`RigidBodyAggregation::Contiguous` preserves the fixed-width baseline, while
`RigidBodyAggregation::Graph` requests graph aggregation without fallback. Coordinate ordering must match the
CSR DOF ordering exactly: node `i` maps to rows `3*i`, `3*i+1`, `3*i+2`.

## Measuring prepared reuse

`bench-fem-structural-prepared.ps1` prepares the structural context once and then solves the same RHS repeatedly from a fresh zero initial guess. This deliberately prevents warm-start effects from being confused with preconditioner reuse. The first report charges analysis and preparation; later reports must show `preconditioner_reused = true` with zero analysis/prepare charge.

Example:

```powershell
.\bench-fem-structural-prepared.ps1 K.mtx `
  -Coordinates K.coords `
  -Rhs b.txt `
  -TargetCoarseDimension 1536 `
  -Tolerance 1e-8 `
  -MaxIterations 3000 `
  -Repeats 2
```

## Graph aggregation experiment (0.6-r10)

The graph mode keeps the same target coarse dimension and six rigid-body modes, but
forms connected node regions from 3x3 block sparsity instead of slicing the RCM node
order into fixed-width chunks. This is intended as an A/B experiment before changing
the default Structural Auto policy.

```powershell
.\bench-fem-structural-auto.ps1 K.mtx `
  -Coordinates K.coords `
  -Rhs b.txt `
  -TargetCoarseDimension 1536 `
  -Aggregation graph `
  -Tolerance 1e-8 `
  -MaxIterations 3000
```

The benchmark reports actual aggregate count and minimum/maximum aggregate sizes so
changes in coarse dimension are visible rather than hidden.


## Graph aggregation robustness (0.6-r11)

The initial capped-BFS graph experiment could leave tiny remainder islands after nearby nodes had already been assigned. A six-mode rigid-body basis can become rank deficient or severely ill-conditioned on such islands. r11 therefore merges graph aggregates smaller than one quarter of the requested aggregate size (never below three nodes) into the neighboring aggregate with the strongest edge connection before forming `Z^T A Z`. No diagonal regularization is added: failure of the coarse Cholesky still indicates a genuine basis/geometry problem rather than being hidden by a shift.


## Structural Auto graph policy (0.6-r12)

The L-angle physical-load development benchmark showed that graph-connected aggregation
was substantially more effective than contiguous RCM chunks at essentially the same
coarse dimension: 220 versus 1370 PCG iterations at a relative tolerance of `1e-8`,
with coarse dimensions 1398 and 1404 respectively. This is a single development
benchmark, not a general performance guarantee.

For that reason r12 makes `RigidBodyAggregation::Auto` the Structural Auto default.
Auto attempts the graph coarse space first. If coarse Cholesky reports a numerical
breakdown, or if graph topology leaves a disconnected component with fewer than three nodes,
Auto retries with contiguous aggregation. Explicit `Graph` remains strict so
A/B benchmarks still expose graph-basis failures rather than silently hiding them.

## Packed coarse Cholesky storage (0.6-r17)

The structural rigid-body path now stores the dense Cholesky lower factor in
row-wise packed triangular form. The Galerkin coarse operator, aggregation,
rigid-body basis, pivot test, and PCG algorithm are unchanged; only factor
storage and triangular-solve addressing change. A coarse dimension `m` uses
`m(m+1)/2` stored `f64` factor values instead of `m^2`, with row-start offsets
precomputed once during preparation.

This is intentionally an implementation optimization rather than a policy
change. Re-run the Graph target-coarse-dimension points at 1536 and 3072 to
measure whether the reduced factor footprint improves wall time as well as
memory use on the target CPU.


## Fine-grid SpMV policy (0.6-r20)

Structural PCG can now select the validated Rayon-parallel CSR operator through
`StructuralSpmvPolicy`. `Auto` keeps small matrices serial and switches CSR32
systems with at least 1,000,000 nonzeros to parallel row execution. `Serial`
and `Parallel` remain available for controlled benchmarking; explicit `Parallel`
requires the CSR32 backend. The parallel implementation preserves the same
CSR storage and arithmetic ordering within each row, so iteration counts should
remain unchanged.

The Rayon pool size is intentionally external to the solver policy in r20. Set
`RAYON_NUM_THREADS` (or use `-RayonThreads` in the supplied PowerShell wrappers)
to tune a particular CPU. On the development L-angle case on a Ryzen 7 7800X3D,
4 threads was the best tested point: CSR SpMV improved from about 13.24 ms to
5.09 ms and the 220-iteration structural PCG solve from about 4.44 s to 2.73 s.
This machine-specific result is not used as a hard-coded library thread count.


### 0.6 structural execution policy (development)
Structural Auto independently resolves `StructuralSpmvPolicy`, `StructuralPreconditionerPolicy`, and `StructuralPcgVectorPolicy`. Large CSR structural systems can therefore use parallel CSR SpMV, parallel 3x3 block-Jacobi/restriction/prolongation, and parallel/fused PCG vector kernels together. Small systems remain serial to avoid Rayon overhead. The packed dense coarse triangular solve remains serial. Rayon thread count is controlled externally (for example `RAYON_NUM_THREADS`).


## PCG dense-vector policy (0.6-r25)

Structural solves expose `StructuralPcgVectorPolicy::{Auto, Serial, Parallel}` independently from the sparse SpMV and rigid-body preconditioner policies. The parallel path uses the shared Rayon pool for dot/norm reductions, search-direction updates, and a fused `x += alpha*p`, `r -= alpha*Ap`, `||r||` traversal.

`Auto` is deliberately conservative: it selects the parallel/fused path only for at least 131072 unknowns and at least 4 Rayon workers. Explicit `Serial` remains available for low-thread-count execution and controlled A/B tests.

At the r25 production checkpoint, the physical-load 358065-DOF / 28239653-nnz L-angle case at 8 Rayon workers resolved to Graph aggregation, Parallel SpMV, Parallel preconditioning, and Parallel PCG vectors. It used 233 aggregates and a 1398-dimensional coarse space, converged in 220 iterations, and independently verified a relative residual of `9.378557e-9`. Measured solve time was 1.726 s; analysis+prepare+solve was 2.797 s on the Ryzen 7 7800X3D development machine. These timings are a reproducible development reference, not a cross-machine performance guarantee.
