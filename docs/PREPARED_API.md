# HyBIT 0.5 prepared execution

## Purpose

FEM and HPC applications often solve multiple right-hand sides against the same assembled matrix. HyBIT 0.5 separates reusable setup from RHS-dependent solving.

## Phases

### Analyze

`HybitSolver::analyze_csr32` validates the matrix, creates the matrix profile, chooses the backend policy, and records structure/value signatures.

### Prepare

`HybitSolver::prepare_csr32` constructs matrix-dependent reusable state:

- Jacobi inverse diagonal;
- ABTM storage when ABTM is explicitly selected as the SpMV backend; hybrid topology is otherwise built lazily only on escalation;
- five reusable PCG work vectors.

The prepare phase does not require a RHS.

### Solve-many

The first prepared solve performs the normal adaptive controller stage. If algebraic coarse is explicitly enabled, that coarse preconditioner is built before Krylov iteration zero and is used for the stage; otherwise the stage uses Jacobi. If poor progress is detected, HyBIT may build the selected local Cholesky/Schwarz correction and restart PCG only because the preconditioner has changed. If no strengthening occurs, the same PCG session continues across the controller boundary.

Subsequent RHS solves reuse:

- the PCG workspace;
- ABTM topology;
- Jacobi data;
- algebraic coarse state when enabled;
- learned local region mappings;
- local Cholesky factors and overlap weights.

A cached local Hybrid preconditioner skips probe, hard-region diagnostics, and factorization. A coarse-only prepared context reuses the already-built coarse state; its short controller boundary does not restart PCG when no local correction is added.

## Matrix immutability in 0.5

Local factors are numerical factors, not symbolic-only metadata. HyBIT therefore hashes both CSR structure and `f64` coefficient bit patterns. A prepared context rejects a different matrix.

Future work can support:

1. same topology + changed values -> numeric refactor only;
2. same topology + changed active regions -> selective plan rebuild;
3. topology change -> full analyze/prepare.

## Allocation behavior

The prepared PCG path owns its five `n`-length vectors once. Each local Schwarz region also owns two local scratch vectors allocated at factor construction. `Preconditioner::apply()` performs no `Vec` allocation.

Diagnostics during the first adaptive escalation may still allocate temporary masks/region vectors; those are outside the repeated Krylov inner loop.


## Structural prepared execution (0.6 development)

Three-dimensional structural systems may provide reduced/ordered node coordinates
through the geometry-aware API. `HybitSolver::prepare_structural_csr32` builds a
3x3 block-Jacobi fine level plus six rigid-body modes per aggregate, with the
aggregate size selected from `StructuralOptions::target_coarse_dimension`.

The returned `HybitPreparedStructuralSystem` owns the coarse Cholesky factor,
normalized geometry data, optional ABTM operator storage, and reusable PCG
workspace. Repeated `solve()` calls reuse all of this matrix/geometry-dependent
state across RHS vectors. The generic prepared path is intentionally unchanged.
