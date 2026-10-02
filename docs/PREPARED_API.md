# HyBIT prepared execution

This document describes `analyze -> prepare -> solve-many`. HyBIT 0.7.0
provides the SPD and structural prepared paths; `develop/0.8.0` additionally
provides prepared GeneralSquare FGMRES with Jacobi or opt-in ILU(0).

## Analyze

`HybitSolver::analyze_csr32` validates the declared problem class, profiles the
matrix, resolves backend/execution policy, and records exact structure/value
signatures. Analyze does not require an RHS.

## Configuration snapshot

Configure the solver before `analyze`/`prepare`. Analysis binds the declared
problem/execution contract to the matrix signatures. Preparation then copies
the relevant solver and preconditioner options into the returned prepared
context.

An already-created prepared context is self-contained: later changes to the
`HybitSolver` object do not alter that prepared context. Build a new prepared
context when intentionally changing the matrix-dependent solver policy.
## Prepare

Preparation creates matrix-dependent reusable state.

### Generic SPD

The prepared state may include Jacobi, optional ABTM, reusable PCG workspace,
explicit algebraic coarse state, and later learned Hybrid local factors.

### GeneralSquare

The prepared state owns exactly one selected preconditioner: Jacobi or ILU(0).
It also owns reusable FGMRES workspace. Escalating/budget-aware policies
allocate workspace for the maximum permitted restart.

`general_square_preconditioner_bytes()` reports persistent preconditioner state.
For ILU(0), `general_square_ilu_adjusted_pivots()` reports factor-pivot
stabilization count.

### Structural SPD

`prepare_structural_csr32` additionally owns geometry-dependent aggregation,
rigid-body coarse state/factorization, and reusable PCG workspace.

## Solve-many

```text
analyze(matrix, policy)
        |
        v
prepare(matrix, analysis)
        |
        +--> solve(rhs_1)
        +--> solve(rhs_2)
        +--> ...
```

SPD Hybrid can learn local regions/factors on an early difficult RHS and reuse
them later.

GeneralSquare builds Jacobi or ILU(0) during prepare and reuses it across all
RHS vectors. Its report marks `preconditioner_reused = false` on solve sequence
1 and `true` from solve sequence 2 onward, meaning that the same prepared
preconditioner is being reused across RHS solves rather than rebuilt.

## Report accounting

The first solve in a prepared context charges the stored analysis/prepare time
to its `SolveReport`; later solves report zero for those two setup components.

`relative_residual` is `||b-Ax||_2 / ||b||_2` for nonzero RHS. This makes
solve-many comparisons meaningful even when RHS magnitudes differ.
## Exact matrix reuse rule

Prepared contexts validate both CSR structure and exact `f64` coefficient bit
patterns. Changed values require a new prepare even if sparsity is unchanged.

Future work may separate:

1. same topology + same values -> full reuse;
2. same topology + changed values -> symbolic reuse + numeric refactor;
3. changed topology -> full analyze/prepare.

Only the first case is supported by the high-level prepared context today.

## Allocation behavior

Repeated Krylov execution reuses preallocated vectors.

- PCG owns reusable full-size workspace.
- FGMRES owns `V`, `Z`, Hessenberg/Givens, and residual/update scratch.
- Local Schwarz owns local scratch.
- Structural preparation owns coarse/geometry state.

First-time diagnostics/factor construction may allocate outside the repeated
Krylov inner loop.

## Example

```rust
let analysis = solver.analyze_csr32(&a)?;
let mut prepared = solver.prepare_csr32(&a, &analysis)?;

let mut x1 = vec![0.0; a.nrows()];
let report1 = prepared.solve(&a, &b1, &mut x1)?;

let mut x2 = vec![0.0; a.nrows()];
let report2 = prepared.solve(&a, &b2, &mut x2)?;

assert_eq!(report1.solve_sequence, 1);
assert_eq!(report2.solve_sequence, 2);
assert!(report2.preconditioner_reused);
```

When benchmarking reuse, avoid warm-starting from the previous solution unless
warm-start behavior is itself the subject of the benchmark.