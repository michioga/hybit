# HyBIT roadmap

This roadmap describes direction, not guaranteed release dates.

## 0.7.0 release-candidate work

The r32 production line is feature-frozen on `develop/0.7.0`. Remaining 0.7 work is release engineering:

- keep the r32 solver/controller behavior fixed except for release-blocking correctness defects;
- run the full Rust workspace test, formatting, Clippy, MSRV, package, ABI, C/C++/Fortran, source-integrity, and physical L-angle gates;
- keep Cargo versions, C ABI version reporting, README/README.ja, changelog, release notes, publishing instructions, and benchmark documentation synchronized;
- publish exactly the commit that passes the complete gate.

## Completed for 0.7

- Generic algebraic two-level coarse correction for SPD/PCG solves.
- Graph aggregation in addition to contiguous aggregation.
- Jacobi-smoothed transfer basis and Galerkin `P^T A P` construction.
- Parallel coarse restriction/prolongation.
- Wide/compact transfer-index storage and F64/F32/Auto transfer-value storage.
- FactorSolve/ExplicitInverse/Auto coarse application.
- Coarse-first explicit-coarse controller sequencing.
- Resumable PCG continuation when the preconditioner is unchanged.
- Prepared coarse-only reuse.
- Retention of the validated 0.6 structural Graph rigid-body, parallel SpMV/preconditioner, and parallel/fused PCG path.

## Post-0.7 numerical research

- Improve long-tail convergence on difficult SPD matrices using genuine local spectral/GenEO-style coarse enrichment rather than the discarded r36 filtered proxy.
- Develop progress/watchdog logic that diagnoses late stagnation without automatically applying harmful local-direct restarts.
- Improve coarse-space selection using a fixed SuiteSparse representative/stress corpus.
- Add MINRES for symmetric indefinite systems.
- Add GMRES and/or BiCGStab for nonsymmetric systems.
- Generalize adaptive escalation beyond SPD/PCG.

## Parallel and large-scale execution

- Parallel independent local factorizations where setup cost warrants it.
- Improve CPU sparse-kernel scheduling and NUMA/cache behavior.
- GPU backends with persistent device residency.
- Distributed-memory domain decomposition and communication-aware topology handling.
- Separate symbolic/topological reuse from numerical refactorization when values change on fixed sparsity.
- Investigate out-of-core prepared state when problem scale requires it.

## Non-goals for 0.7.0

HyBIT 0.7.0 does not claim universal sparse-solver coverage or universal performance improvement. The automatic path remains real SPD/PCG. The release goal is a reproducible hybrid solver architecture with both validated structural-FEM and generic algebraic-coarse paths, stable C/C++/Fortran consumption through the existing C ABI, and conservative release gates.
