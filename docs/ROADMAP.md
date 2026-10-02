# HyBIT roadmap

This roadmap describes direction, not guaranteed dates.

## Published baseline

HyBIT 0.7.0 is published at immutable tag `v0.7.0`, commit
`1fdcd6a1b8127c84306c38c3fdbad42563538ad8`.

## Completed in 0.7

- generic algebraic two-level SPD/PCG correction;
- graph/contiguous aggregation and smoothed transfer;
- parallel coarse transfer and storage policies;
- coarse-first resumable PCG;
- prepared coarse reuse;
- validated structural Graph rigid-body parallel CPU path;
- release integrity/MSRV/package/ABI/real-FEM gates.

## Completed on develop/0.8.0 through E5

### Execution architecture

- problem/execution/backend/solver/preconditioner policy architecture;
- resident Krylov abstraction;
- serial/Rayon resident PCG validation;
- parallel CSR/Jacobi execution;
- structural resident cross-check.

### GeneralSquare

- restarted FGMRES with separate `V` and `Z`;
- reusable FGMRES workspace;
- prepared GeneralSquare routing;
- fixed, escalating, and budget-aware restart;
- canonical prepared ILU(0);
- selective row-relative pivot stabilization;
- prepared ILU reuse and memory telemetry.

## Immediate next work

### GeneralSquare robustness

1. Measure ILU(0) sensitivity to natural/RCM/other meaningful orderings.
2. Validate representative real nonsymmetric FEM/PDE matrices.
3. Repeat at larger dimensions and multiple RHS.
4. Profile serial triangular application separately.
5. Define fallback behavior for unsuitable ILU.
6. Decide whether automatic Jacobi -> ILU(0) promotion is justified.

Do not add ILUT/drop-tolerance fill until ILU(0)'s operating envelope is clear.

### Documentation/API completeness

- maintain separate user, numerical, architecture, and development docs;
- keep examples runnable under gates;
- add GeneralSquare selection to C/C++/Fortran only after the Rust policy is
  stable enough;
- document report fields by solver path.

## Subsequent numerical work

- MINRES for `SymmetricIndefinite`;
- genuine local spectral/GenEO-style SPD enrichment;
- symbolic/topological reuse separated from numeric refactorization;
- rectangular LSQR/LSMR later;
- complex arithmetic later.

## Parallel and large-scale work

- parallel setup where worthwhile;
- improve CPU sparse scheduling and NUMA/cache behavior;
- GPU-resident execution after the selected CubeCL interface is stable;
- GPU-suitable preconditioners rather than blindly porting serial ILU solves;
- distributed-memory domain decomposition;
- out-of-core state only when problem scale justifies it.

## Release discipline

Keep release tags immutable, preserve the C ABI where practical, update language
bindings together, and retain source-integrity, formatting, Clippy, MSRV,
package, ABI/language, and numerical regression gates.

## Non-goal

HyBIT does not claim universal sparse-solver coverage or universal performance
improvement. Automatic policy changes require evidence across representative
matrices.