# HyBIT roadmap

This roadmap describes direction, not guaranteed release dates.

## 0.6.0 release-candidate work

The r25 structural production path is feature-frozen on `develop/0.6.0`. Before merging the release candidate to `main`, the immediate work is validation rather than new solver functionality:

- run the full Rust workspace test suite in release mode;
- run C ABI, C, C++, and Fortran build/runtime checks;
- validate tiny-problem serial fallbacks and explicit parallel policy paths;
- rerun the physical-load L-angle structural case and independently verify `||Ax-b||/||b|| < 1e-8`;
- validate prepared structural solve-many reuse, including cached coarse factors, parallel aggregate index, and Krylov workspace;
- run `cargo fmt --check`, Clippy, package metadata checks, and `cargo package --list`/dry-run gates;
- reconcile README, API documentation, changelog, release notes, and benchmark documentation with the exact release commit.

## Completed in the 0.6 development line

- Matrix Market import/export and reproducible real-FEM benchmark paths.
- Geometry-aware six-mode rigid-body two-level correction for 3-D structural SPD systems.
- Graph-connected structural aggregation with deterministic fallback to contiguous aggregation in Auto mode.
- Packed lower-triangular coarse Cholesky storage.
- Reusable structural `analyze -> prepare -> solve-many` execution.
- Rayon-parallel CSR SpMV for large structural systems.
- Parallel 3x3 block-Jacobi, rigid-body restriction, and prolongation; the dense packed coarse triangular solve remains serial.
- Parallel/fused PCG vector reductions and updates for sufficiently large structural systems and Rayon pools with at least four workers.
- Independent structural execution policies for aggregation, SpMV, preconditioning, and PCG vector kernels.

## Post-0.6 solver coverage

- Add MINRES for symmetric indefinite systems.
- Add GMRES and/or BiCGStab for nonsymmetric systems.
- Generalize adaptive escalation rules beyond SPD/PCG.
- Generalize the current structural rigid-body coarse correction toward broader algebraic/multilevel coarse spaces and Schur/interface methods.

## Post-0.6 parallel execution

- Parallel independent local factorizations where setup cost warrants it.
- Improve CPU sparse-kernel scheduling and NUMA/cache behavior on larger multi-core systems.
- GPU backends for suitable matrix/vector kernels with persistent device residency.
- Distributed-memory domain decomposition and communication-aware topology handling.

## Storage and execution

- Separate symbolic/topological reuse from numerical refactorization when coefficient values change on fixed sparsity patterns.
- Refine adaptive tile geometry beyond the current row-oriented ABTM representation.
- Investigate out-of-core storage for cold local factors and prepared state when problem scale requires it.

## Non-goals for 0.6.0

HyBIT 0.6.0 does not attempt to support every sparse matrix class, replace established general-purpose sparse solvers, or claim universal performance improvements. The release goal is a transparent and reproducible SPD/PCG hybrid architecture with a validated structural-FEM path and conservative automatic policy selection.
