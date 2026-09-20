# HyBIT 0.6 development status

Last updated: 2026-09-20

## Branch and release state

- Active development branch: `develop/0.6.0`
- Published baseline: 0.5.0
- Current development checkpoint: 0.6.0-r26 release-gate hardening (solver code remains r25)
- Structural production path: feature-frozen for 0.6.0 release-candidate validation
- `main` should remain at the last validated public-release line until the 0.6.0 release gate passes.

## Current structural production path

For large 3-D structural SPD systems, Structural Auto can select:

1. Graph-connected rigid-body aggregation with conservative contiguous fallback.
2. Six rigid-body coarse modes per aggregate.
3. Packed lower-triangular dense coarse Cholesky storage.
4. Parallel CSR SpMV.
5. Parallel 3x3 block-Jacobi, restriction, and prolongation; coarse triangular solve remains serial.
6. Parallel/fused PCG vector reductions and updates when the system is large enough and the shared Rayon pool has at least four workers.
7. Prepared solve-many reuse of matrix-dependent factors, aggregate index, and Krylov workspace.

The generic `solve_csr32` path remains separate and unchanged by these structural execution policies.

## r25 L-angle reference

Physical-load reduced structural system:

- DOFs: 358065
- CSR nnz: 28239653
- Rayon workers: 8
- aggregation: Graph
- aggregate count: 233
- target coarse dimension: 1536
- actual coarse dimension: 1398
- SpMV policy: Parallel
- preconditioner policy: Parallel
- PCG vector policy: Parallel
- PCG iterations: 220
- solver-reported relative residual: `9.378495e-9`
- independently verified relative residual: `9.378557e-9`
- analysis: 326.212 ms
- prepare: 744.797 ms
- solve: 1725.863 ms
- analysis + prepare + solve: 2796.872 ms

These values are a regression reference for this matrix, RHS, machine, and revision. They are not a general performance guarantee.

## Release-candidate gate

`release-candidate-gate.ps1` is now the authoritative local RC gate. From a clean `develop/0.6.0` worktree, run:

```powershell
.\release-candidate-gate.ps1 `
  -Matrix D:\Work\mf_solver-hybit-export\L-angle-K.mtx `
  -Coordinates D:\Work\mf_solver-hybit-export\L-angle-K.coords `
  -Rhs D:\Work\mf_rhs\L-angle-b.txt `
  -RayonThreads 8
```

The gate checks source hashes, package metadata, formatting, Clippy, workspace release tests, Rust 1.73 MSRV, ABI/C/C++/Fortran runtime examples, crates.io package contents/dry-run, the real L-angle Auto path, independently verified residual, iteration guard, and prepared solve-many reuse. Runtime performance is reported but is not a pass/fail criterion.

Do not add new solver features before the 0.6.0 release candidate is stabilized. After the complete gate passes without skip switches, record the validated commit hash, merge that exact commit to `main`, rerun the complete gate on `main`, then tag `v0.6.0` and publish in dependency order.
