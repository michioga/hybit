# HyBIT 0.6 architecture

HyBIT 0.6 retains the staged-PCG safety and reusable prepared execution model from 0.5, and adds an explicit geometry-aware structural path. The generic `solve_csr32` behavior remains separate from structural policy selection.

## Generic one-shot and prepared path

`HybitSolver::solve_csr32` remains available and internally follows analyze -> prepare -> solve. For repeated right-hand sides, the prepared path reuses matrix-dependent state and Krylov workspace.

1. **Analyze** CSR32 structure, SPD baseline requirements, backend policy, and exact structure/value signatures.
2. **Prepare** Jacobi and a reusable PCG workspace; ABTM is built eagerly only when required by the selected backend and otherwise lazily on hybrid escalation.
3. **Solve #1** with the generic adaptive PCG path.
4. If progress is poor, identify difficult DOFs, form bounded local regions, build local Cholesky corrections, and restart PCG with the strengthened preconditioner.
5. Cache learned local factors for later right-hand sides against the unchanged matrix.

ABTM remains internal; callers provide ordinary CSR32 matrices.

## Structural path

`solve_structural_csr32` / `prepare_structural_csr32` accept 3-D node coordinates matching the CSR DOF ordering and build a six-rigid-body-mode two-level preconditioner. At the r25 checkpoint Structural Auto resolves four independent choices:

- **Aggregation:** Graph-connected rigid-body aggregates by default, with conservative fallback to contiguous RCM-order aggregates when the graph coarse space cannot be formed safely.
- **Fine-grid operator:** serial CSR or Rayon-parallel CSR through `StructuralSpmvPolicy`.
- **Preconditioner kernels:** serial or parallel 3x3 block-Jacobi, restriction, and prolongation through `StructuralPreconditionerPolicy`; the packed dense coarse triangular solve remains serial.
- **PCG dense vectors:** serial or parallel/fused dot/norm/update kernels through `StructuralPcgVectorPolicy`.

The production structural pipeline is therefore:

```text
CSR32 stiffness + node coordinates
        |
        v
analyze
        |
        v
prepare structural state
  |-- graph/contiguous aggregation
  |-- six rigid-body modes per aggregate
  |-- E = Z^T A Z
  |-- packed Cholesky(E)
  |-- 3x3 block-Jacobi factors
  |-- cached aggregate->node index for parallel restriction
  |-- reusable PCG workspace
        |
        v
solve #1..N
  |-- selected serial/parallel CSR SpMV
  |-- additive rigid-body two-level preconditioner
  |-- selected serial/parallel-fused PCG vector kernels
  +-- reuse prepared factors/index/workspace
```

## Allocation policy

Repeated generic and structural PCG execution reuse preallocated work vectors. Local Schwarz factors own fixed scratch, and the structural parallel restriction index is created during preparation and reused by solve-many. No per-iteration `Vec` allocation is intended in the production Krylov loops.

## Reuse safety

Prepared contexts validate CSR structure and exact floating-point coefficient bits before reuse. This remains intentionally conservative: changing matrix values requires a new prepared context even when the sparsity pattern is unchanged.

## 0.6 r25 checkpoint

On the development 358065-DOF / 28239653-nnz L-angle physical-load case, 8 Rayon workers resolved Structural Auto to Graph aggregation + Parallel SpMV + Parallel preconditioner + Parallel/fused PCG vectors. The run used a 1398-dimensional coarse space, converged in 220 iterations, and independently verified a relative residual of `9.378557e-9`; solve time was 1.726 s and analysis+prepare+solve was 2.797 s. This is a machine-specific regression reference rather than a general performance guarantee.
