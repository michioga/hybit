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

The first prepared solve performs the normal adaptive probe. If poor progress is detected, it builds the selected local Cholesky/Schwarz preconditioner and caches it.

Subsequent RHS solves reuse:

- the PCG workspace;
- ABTM topology;
- Jacobi data;
- learned local region mappings;
- local Cholesky factors and overlap weights.

They therefore skip probe, hard-region diagnostics, and factorization when a cached Hybrid preconditioner exists.

## Matrix immutability in 0.5

Local factors are numerical factors, not symbolic-only metadata. HyBIT therefore hashes both CSR structure and `f64` coefficient bit patterns. A prepared context rejects a different matrix.

Future work can support:

1. same topology + changed values -> numeric refactor only;
2. same topology + changed active regions -> selective plan rebuild;
3. topology change -> full analyze/prepare.

## Allocation behavior

The prepared PCG path owns its five `n`-length vectors once. Each local Schwarz region also owns two local scratch vectors allocated at factor construction. `Preconditioner::apply()` performs no `Vec` allocation.

Diagnostics during the first adaptive escalation may still allocate temporary masks/region vectors; those are outside the repeated Krylov inner loop.
