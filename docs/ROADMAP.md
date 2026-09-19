# HyBIT roadmap

This roadmap describes direction, not guaranteed release dates.

## Near term

- Validate HyBIT on real FEM stiffness matrices at substantially larger DOF counts.
- Separate symbolic/topological reuse from numerical refactorization when matrix values change.
- Improve hard-region diagnostics and make region selection less heuristic.
- Add reproducible benchmark harnesses with wall-clock, memory, and convergence reporting.
- Harden public Rust documentation and examples based on user feedback from 0.5.x.

## Solver coverage

- Add MINRES for symmetric indefinite systems.
- Add GMRES and/or BiCGStab for nonsymmetric systems.
- Generalize adaptive escalation rules beyond SPD/PCG.
- Investigate coarse corrections and Schur-type interfaces for large difficult regions.

## Parallel execution

- Parallel CPU SpMV and local-region application.
- Parallel independent local factorizations.
- GPU backends for suitable matrix and vector kernels.
- Distributed-memory domain decomposition and communication-aware topology handling.

## Storage and execution

- Refine adaptive tile geometry beyond the current row-oriented ABTM representation.
- Add explicit symbolic reuse for changing coefficient values on fixed sparsity patterns.
- Investigate out-of-core storage for cold local factors and prepared state when problem scale requires it.

## Non-goals for the immediate release

The 0.6 development line does not attempt to solve every sparse matrix class, replace every established solver, or claim universal performance improvements. The immediate goal is to establish a transparent adaptive hybrid architecture and validate it progressively on real workloads.
