# HyBIT roadmap

This roadmap describes direction, not guaranteed release dates.

## Current baseline

HyBIT 0.7.0 is published. The immutable release source is tagged `v0.7.0` at commit `1fdcd6a1b8127c84306c38c3fdbad42563538ad8`.

The 0.7 production line is therefore closed except for fixes that would be delivered as a new version. Post-release documentation changes and future numerical research must not move or rewrite the `v0.7.0` tag.

## Completed in 0.7

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
- Rust 1.73 MSRV, source-integrity, package, ABI, C/C++/Fortran runtime, crates.io dry-run, and physical L-angle release gates.
- Publication of all six Rust crates as 0.7.0 plus the `v0.7.0` GitHub release.

## Post-0.7 / 0.8 numerical research

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

## Compatibility and release discipline

- Preserve the C ABI where practical; update C, C++, and Fortran consumers together when it changes.
- Keep release tags immutable.
- Keep experimental benchmark artifacts out of release source sets unless they are intentionally documented and manifested.
- Continue using source-integrity, MSRV, package, ABI/language-binding, and real-FEM regression gates for future releases.

## Non-goals inherited from 0.7.0

HyBIT 0.7.0 does not claim universal sparse-solver coverage or universal performance improvement. Its automatic path remains real SPD/PCG. Future work may broaden that scope, but any extension should preserve reproducible residual validation and conservative release gating.