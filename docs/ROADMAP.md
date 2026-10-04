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

## Completed on develop/0.8.0 through F2

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
- prepared ILU reuse and memory telemetry;
- F1 Natural-versus-RCM ordering harness with original-system residual verification;
- real nonsymmetric ordering evidence showing both a strong RCM benefit (`raefsky3`) and a no-benefit case (`sherman5`), ruling out unconditional RCM promotion;
- F2 multi-matrix preflight/corpus harness with machine-readable CSV evidence;
- larger real nonsymmetric `venkat25` evidence showing a moderate RCM ILU(0) benefit while Jacobi remains ordering-invariant;
- explicit corpus evidence that structurally missing diagonals are a common current-applicability boundary.

## Immediate next work

### GeneralSquare robustness

1. Measure prepared Natural/RCM ILU(0) reuse across multiple RHS, especially on `raefsky3` and `venkat25`, and quantify ordering/factor amortization.
2. Profile serial triangular application separately.
3. Define fallback behavior for unsuitable ILU, including the frequent structurally missing diagonals exposed by the F2 corpus.
4. Evaluate residual/progress signals for selective Natural -> reordered ILU escalation; bandwidth reduction alone is not sufficient.
5. Decide whether automatic Jacobi -> ILU(0) promotion is justified.

Do not add ILUT/drop-tolerance fill until ILU(0)'s operating envelope is clear.

### ABTM topology and metadata-first execution

After the current GeneralSquare robustness sequence, reintroduce ABTM as a
symbolic/topology layer rather than assuming it must replace CSR everywhere.

1. G1: bitmap topology algebra: AND/OR/AND-NOT, popcount, rank/select, invariants.
2. G2: metadata-first product pruning with candidate/executed work metrics.
3. G3: region growth, overlap/multiplicity, and local submatrix extraction.
4. G4: ABTM symbolic/numeric ILU(0) intersection versus canonical CSR ILU(0).
5. G5: 3x3/6x6 block-ABTM experiments for FEM node topology.
6. G6: ordinary and masked/restricted SpMV; pure SpMV is not the sole success criterion.
7. G7: Rayon prepared execution after scalar semantics and metrics stabilize.
8. G8: GPU/CubeCL-specific prepared ABTM rather than copying the CPU layout.
9. G9: partition/halo extraction before later MPI scheduling work.

Track occupancy, metadata bytes, rank/select cost, bitmap operations,
candidate/executed products, pruning ratio, preparation cost, and execution
time. Backend promotion remains evidence-driven.

### Documentation/API completeness

- maintain separate user, numerical, architecture, and development docs;
- keep examples runnable under gates;
- add GeneralSquare selection to C/C++/Fortran only after the Rust policy is
  stable enough;
- document report fields by solver path.

## Subsequent numerical work

- MINRES for `SymmetricIndefinite`;
- genuine local spectral/GenEO-style SPD enrichment;
- symbolic/topological reuse separated from numeric refactorization, with the ABTM topology track providing the structural representation experiments;
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