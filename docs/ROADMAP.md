# HyBIT roadmap

This roadmap describes direction, not guaranteed dates.

## Published baseline

HyBIT 0.8.0 is published at immutable tag `v0.8.0`, commit
`2eddfe1551f5a11aa0c07ee8d50d6cac4372e400`.

## Completed in 0.7

- generic algebraic two-level SPD/PCG correction;
- graph/contiguous aggregation and smoothed transfer;
- parallel coarse transfer and storage policies;
- coarse-first resumable PCG;
- prepared coarse reuse;
- validated structural Graph rigid-body parallel CPU path;
- release integrity/MSRV/package/ABI/real-FEM gates.

## Completed in 0.8.0 through G7

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
- explicit corpus evidence that structurally missing diagonals are a common current-applicability boundary;
- F3 prepared Natural/RCM ILU(0) multi-RHS harness with solver-only and end-to-end permutation accounting;
- repeated 5-RHS evidence showing robust cross-RHS RCM benefit on `raefsky3` but only small/timing-sensitive aggregate benefit on `venkat25`;
- F4 serial triangular-apply/dependency profiling, with per-level and width-threshold Rayon prototypes rejected because synchronization/scheduling overhead outweighed available level parallelism.

## Immediate next work

### GeneralSquare robustness

GeneralSquare robustness work is closed through F7 for this 0.8 checkpoint.

1. Keep Jacobi as the default GeneralSquare preconditioner.
2. Keep canonical ILU(0) explicit/opt-in.
3. Keep `Ilu0Fallback` explicit for the missing-diagonal safety path.
4. Keep Natural/RCM ordering explicit; F6/F7 evidence does not justify a
   universal automatic selector.
5. Revisit automatic selection only with broader held-out validation or
   application-provided solve horizon / cost budget.
6. Treat drop-tolerance/fill, pivoting, and multilevel ILU as separate future
   preconditioners rather than silently changing canonical ILU(0).
### ABTM topology and metadata-first execution

After the current GeneralSquare robustness sequence, reintroduce ABTM as a
symbolic/topology layer rather than assuming it must replace CSR everywhere.

1. G1 (validated): bitmap topology algebra: AND/OR/AND-NOT/XOR, popcount, rank/select, invariants.
2. G2 (validated): metadata-first product pruning with candidate/executed work metrics.
3. G3 (validated): region growth, overlap/multiplicity, local extraction, and symbolic/numeric reuse boundaries.
4. G4 (validated): ABTM ILU(0) symbolic intersection and explicit production ABTM factorization path; automatic routing deferred.
5. G5 (validated): 3x3/6x6 block-topology characterization and explicit fixed-size dense block-CSR numerical operator; automatic CSR/B3/B6 routing deferred pending broader positive held-out evidence.
6. G6 (validated): ordinary scalar ABTM SpMV remains diagnostic; fixed column restrictions and graph-local restrictions use explicit prepared compact CSR, with direct or ABTM-assisted preparation selected explicitly by workload.
7. G7 (validated): explicit full-Rayon and task-limited prepared CSR execution; serial gather retained; automatic hardware-specific size routing deferred.
8. G8 (active in 0.9): A1 reference partition/halo topology; A2 rank-local operator preparation and ABTM halo cross-check; A3 ABTM arbitrary-owner region partition prototype and external-label quality harness; A4 balanced multi-source ABTM growth after rejecting expensive pair-swap refinement; A5 external METIS/SCOTCH baseline export and identical HyBIT telemetry (boneS01 shows a substantial cut/halo gap while ABTM keeps competitive load balance); A6 multilevel ABTM coarsening/refinement, F12/F13 adjacency stream-merge equivalence and F14 merged production-default with explicit reference rollback; then B transport/MPI boundary, C distributed Krylov, and D real FEM MPI-host validation.
9. G9: GPU/CubeCL-specific prepared execution after the CubeCL API stabilizes.

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
