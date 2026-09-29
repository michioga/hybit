# HyBIT architecture

HyBIT 0.6 retains the staged-PCG safety and reusable prepared execution model from 0.5, and adds an explicit geometry-aware structural path. The generic `solve_csr32` behavior remains separate from structural policy selection.

## 0.8 execution-layer direction

The 0.8 development line separates three concepts that were previously easy to conflate:

- `MatrixBackend`: sparse/operator representation such as CSR32, ABTM, or matrix-free;
- `ExecutionTarget`: where Krylov work runs (`Cpu` today, `Gpu` as the resident-device target);
- `MatrixProblemClass`: the mathematical contract (`Spd`, `SymmetricIndefinite`, or `GeneralSquare`).

`MatrixProblemClass` is declarative. HyBIT does not infer positive definiteness from a positive diagonal or other weak structural checks.

`hybit-krylov` now contains an experimental `KrylovExecutionBackend` boundary plus `ResidentPcgWorkspace`. The established `pcg_with_workspace` implementation remains the production CPU path in 0.8-a1. A CPU reference execution backend runs the same PCG recurrence through resident vectors and is regression-tested against the established path.

The resident workspace owns solution, RHS, and Krylov scratch vectors. A future CUDA/CubeCL backend can therefore upload matrix/preconditioner state during prepare, upload one RHS and initial solution per solve, keep Krylov vectors resident for the complete iteration loop, and download only the final solution/report data. This avoids designing a GPU backend around per-kernel host/device transfers.

The same low-level execution boundary is intentionally broader than PCG so later MINRES and flexible GMRES work can reuse device vector operations. The automatic solver/controller remains real SPD + PCG in this checkpoint; no existing call is silently routed to GPU or to a non-PCG method.

### 0.8-a2 control plane

`HybitSolver` now exposes independent `ExecutionPolicy` and `MatrixProblemClass` controls. `ExecutionPolicy::Auto` and `ExecutionPolicy::Cpu` resolve to the established CPU implementation. `ExecutionPolicy::Gpu` is recognized but rejected before preparation until a resident GPU backend is implemented and validated.

Likewise, `MatrixProblemClass::Spd` remains the only executable automatic-solver class in this checkpoint. `SymmetricIndefinite` and `GeneralSquare` are represented explicitly so future MINRES and FGMRES/BiCGStab work does not require another API redesign, but neither class is silently routed through PCG.

Analyze records the resolved execution target and declared problem class, and prepare verifies that those contracts have not changed. This keeps prepared-state reuse deterministic as additional execution targets and Krylov methods are introduced.

### 0.8-a3 resident CPU cross-check

`ExecutionPolicy::CpuResident` connects the generic `HybitSolver` prepare/solve path to `CpuKrylovExecution` and `ResidentPcgWorkspace` for fixed-Jacobi PCG. This is an explicit validation policy rather than the default: `Auto` and `Cpu` continue to use the established staged PCG implementation.

The a3 resident path deliberately requires `HybridOptions.enabled = false`. This keeps the first production-level cross-check focused on the fixed-preconditioner recurrence that will also form the first CUDA PoC. Public tests compare legacy and resident CPU solves for convergence status, iteration count, residual, and solution, and verify solve-many reuse of the resident vectors.

For this validation checkpoint the prepared object retains both the established five-vector PCG workspace and the seven-vector resident workspace, so the reported Krylov workspace bytes include both. Once resident execution becomes the normal prepared representation, the duplicate legacy workspace can be removed.

### 0.8-b1 Rayon resident backend

`RayonKrylovExecution` adds a second CPU implementation of `KrylovExecutionBackend` without changing the PCG recurrence. It reuses the already-validated parallel/fused dense-vector kernels from the structural CPU path for dot/norm reductions, fused solution/residual updates, and search-direction updates. Generic axpy/scale operations use the same chunked shared Rayon pool.

Sparse operator and preconditioner application remain delegated through the existing traits. Rayon vector execution can therefore compose independently with serial or parallel CSR and preconditioner implementations; this checkpoint does not change `HybitSolver`, `ExecutionPolicy::CpuResident`, or production defaults.

The resident Rayon recommendation initially mirrors the validated Structural Auto vector crossover: at least 131072 unknowns and at least four shared Rayon workers. This is a CPU-vector scheduling heuristic, not a new `ExecutionTarget`.

GPU work is intentionally deferred until CubeCL 0.11 reaches a stable release. MPI remains a later distributed-memory layer after single-node CPU parallel execution is consolidated.

### 0.8-b2 resident Rayon CSR execution

`ExecutionPolicy::CpuResidentRayon` is an explicit validation policy that combines `RayonKrylovExecution` with `ParallelCsr32Operator` while retaining the existing serial Jacobi preconditioner. It therefore isolates the two already-validated Rayon layers: CSR row-parallel SpMV and resident dense-vector kernels.

`CpuResident` remains the serial resident comparison path, while `Auto` and `Cpu` remain unchanged. `CpuResidentRayon` requires `HybridOptions.enabled = false` and the CSR32 matrix backend; forced ABTM is rejected rather than silently dropping parallel CSR execution.

The resident workspace is still reused across right-hand sides. This checkpoint does not yet parallelize generic Jacobi application and does not route any default policy into the resident Rayon path.

### 0.8-b3 resident Rayon Jacobi execution

`ParallelJacobiPreconditioner` is a read-only Rayon view over the already prepared inverse diagonal. It adds no persistent numerical storage and leaves the ordinary `JacobiPreconditioner` trait implementation serial for low-overhead and A/B use.

`ExecutionPolicy::CpuResidentRayonJacobi` composes three explicit CPU layers: `ParallelCsr32Operator`, `RayonKrylovExecution`, and `ParallelJacobiPreconditioner`. The b2 `CpuResidentRayon` policy remains available with serial Jacobi, so serial-all, parallel-CSR+vectors, and parallel-CSR+vectors+Jacobi can be compared in one build.

No automatic policy is changed in this checkpoint.

### 0.8-b4 resident structural cross-check

`HybitPreparedStructuralSystem::solve_resident_rayon` adds an explicit validation path for the fully parallel structural configuration. It reuses the already prepared `RigidBodyTwoLevelBlockJacobiPreconditioner`, `ParallelCsr32Operator`, and `ParallelRigidBodyTwoLevelPreconditioner`, but runs PCG through `RayonKrylovExecution`.

The seven-vector resident workspace is allocated lazily on the first resident structural solve and reused for later right-hand sides. Ordinary `solve` remains the production path and pays no resident-workspace memory cost unless the validation method is called.

This checkpoint intentionally requires effective Parallel policies for SpMV, structural preconditioning, and PCG vectors. It is a same-preconditioner recurrence cross-check, not a new automatic policy.

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
