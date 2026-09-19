# HyBIT 0.6.0 release notes (draft)

> Status: release-candidate documentation draft. Development continues on `develop/0.6.0`; do not treat this file as a published release announcement until the final release gate passes and the release commit is merged to `main`.

HyBIT 0.6.0 extends the 0.5.0 SPD/PCG hybrid solver with a validated geometry-aware path for large 3-D structural FEM systems while keeping the generic `solve_csr32` API behavior unchanged.

## Major additions

- Matrix Market import/export and real-FEM benchmark tooling.
- `StructuralOptions`, `solve_structural_csr32`, and reusable `HybitPreparedStructuralSystem`.
- Six rigid-body coarse modes (Tx/Ty/Tz/Rx/Ry/Rz) per structural aggregate.
- Graph-connected aggregate construction with conservative Auto fallback to contiguous aggregation.
- Packed lower-triangular storage for the dense structural coarse Cholesky factor.
- Independent `StructuralSpmvPolicy`, `StructuralPreconditionerPolicy`, and `StructuralPcgVectorPolicy` controls.
- Rayon-parallel CSR SpMV for large structural systems.
- Parallel 3x3 block-Jacobi, rigid-body restriction, and prolongation while retaining a serial packed coarse triangular solve.
- Parallel/fused PCG dense-vector kernels for sufficiently large systems when the shared Rayon pool has at least four workers.
- Prepared solve-many reuse of coarse factors, geometry/index state, and Krylov workspace.
- Benchmark scripts for aggregation, coarse-dimension sweeps, kernel profiling, SpMV, preconditioner, and PCG-vector A/B studies.

## Structural development reference

The release-candidate reference problem is the physical-load L-angle reduced system:

- 358065 free DOFs;
- 28239653 CSR nonzeros;
- Graph aggregation with 233 aggregates;
- coarse dimension 1398 (`target_coarse_dimension=1536`);
- 8 Rayon workers;
- Parallel CSR SpMV;
- Parallel rigid-body preconditioner kernels;
- Parallel/fused PCG vector kernels;
- 220 PCG iterations;
- solver-reported relative residual `9.378495e-9`;
- independently verified relative residual `9.378557e-9`;
- analysis 326.212 ms;
- prepare 744.797 ms;
- solve 1725.863 ms;
- analysis+prepare+solve 2796.872 ms.

These timings were measured on the Ryzen 7 7800X3D development machine and are included as a reproducible regression reference, not as a universal speed claim.

## Compatibility and scope

- The automatic numerical scope remains real symmetric positive-definite systems with PCG.
- The new rigid-body coarse path assumes exactly three displacement DOFs per node and requires coordinates in the exact CSR node/DOF ordering.
- Generic `solve_csr32` behavior remains unchanged by the structural execution policies.
- The C ABI remains repository-built; `hybit-ffi` is not published to crates.io.
- HyBIT remains pre-1.0 experimental numerical software. Independently validate residuals and physical results for engineering use.

## Release gate remaining

Before publication, the exact candidate commit must pass the complete Rust, ABI/language-binding, structural regression, prepared-reuse, formatting/lint, package metadata, and crates.io dry-run gates. The exact validated source is then merged from `develop/0.6.0` to `main`, tagged `v0.6.0`, and published in dependency order.
