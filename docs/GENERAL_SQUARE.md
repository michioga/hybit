# GeneralSquare FGMRES and ILU(0)

> Development documentation for `develop/0.8.0`, current through E5.

## Purpose

`MatrixProblemClass::GeneralSquare` routes real nonsymmetric square systems
through restarted FGMRES. The prepared preconditioner is selected independently
from restart policy.

Current choices:

- Jacobi — default;
- ILU(0) — explicit opt-in.

GeneralSquare currently supports `ExecutionPolicy::Auto` and
`ExecutionPolicy::Cpu`.

## Why FGMRES

FGMRES stores the Arnoldi basis and the preconditioned basis separately. At
Arnoldi step `j`, conceptually:

```text
v_j                 Arnoldi basis vector
z_j = M_j^-1 v_j    preconditioned basis vector
w   = A z_j
```

`w` is orthogonalized against `V`, and the solution update uses `Z`:

```text
x = x_0 + Z y
```

The current high-level path uses a fixed Jacobi or fixed ILU(0), but the
separate `V`/`Z` representation keeps the recurrence valid for future variable
preconditioning.

The implementation uses two-pass modified Gram-Schmidt, Givens rotations, and
true residual recomputation at restart boundaries and before accepting
convergence.

## Restart and memory

Larger restart can reduce restart boundaries or iterations, but increases
Arnoldi `V`/`Z` storage and orthogonalization work. Larger is therefore not
automatically faster.

`GeneralSquareOptions` controls restart independently from the preconditioner.

### Fixed

Default. Default restart is 30.

### Escalating

Runs bounded stages and doubles restart up to `max_restart`. The final maximum
restart stage receives the remaining iteration budget.

### BudgetAware

Stays inside one FGMRES invocation, observes exact restart-boundary residuals,
estimates logarithmic decay over a small window, and grows restart only under
sustained projected-budget pressure with a non-improving trend, or stronger
emergency pressure.

This is a controller heuristic, not a convergence proof.

## Jacobi

GeneralSquare Jacobi permits negative diagonal values but requires a complete,
finite, nonzero diagonal.

Advantages are very low setup/storage cost and simple baseline behavior.
Its limitation is that it ignores off-diagonal coupling.

## ILU(0)

ILU(0) approximates

```text
A ~= L U
```

without adding fill outside the chosen sparse pattern.

Caller CSR rows need not be sorted or unique. HyBIT privately canonicalizes
the preconditioner input:

1. collect each row;
2. sort by column;
3. sum duplicate columns;
4. remove exact-zero off-diagonal entries;
5. verify a diagonal entry.

The original caller matrix remains unchanged and is still used by the sparse
operator. Canonicalization belongs only to ILU preparation.

The numeric factorization preserves the canonical sparsity pattern. It performs
no row pivoting and introduces no fill.

### Factor-pivot stabilization

No-pivot LU/ILU can encounter a zero factor pivot even for a nonsingular
matrix. HyBIT records each row scale from the canonical matrix before numeric
factor updates, then uses the selective floor

```text
row_scale_i = max_j |a_ij|
floor_i     = 1e-12 * row_scale_i
```

when a factor pivot magnitude is at or below the floor. Negative small pivots
keep negative sign; exact zero receives the positive floor.

The prepared context exposes:

- `general_square_ilu_adjusted_pivots()`;
- `general_square_preconditioner_bytes()`.

A nonzero adjusted-pivot count is diagnostic information. The floor is a
factorization safeguard, not a claim that arbitrary near-singular systems are
well-conditioned.

## Selecting ILU(0)

```rust
use hybit::{
    GeneralSquarePreconditionerPolicy, HybitSolver, MatrixProblemClass,
};

let mut solver = HybitSolver::new();
solver.set_problem_class(MatrixProblemClass::GeneralSquare);
solver.set_general_square_preconditioner_policy(
    GeneralSquarePreconditionerPolicy::Ilu0,
);
```

Selecting ILU(0) does not change restart.

## E5 evidence

Synthetic E5 screening compared Jacobi, Ruiz-scaled Jacobi, contiguous
dense-LU block Jacobi, and ILU(0). ILU(0) was the strongest tested candidate.

A restart sweep with prepared ILU(0) found restart 3 fastest on all three tested
hard families, even though larger restart sometimes used fewer iterations.

The production public-path hard-B cross-check at grid 256 measured:

```text
FGMRES restart          3
iterations              276
true relative residual  9.455610e-9
ILU state               4.238 MiB
FGMRES workspace        4.500 MiB
persistent total        8.738 MiB
adjusted ILU pivots     0
```

Prepared solves #2 and #3 reproduced the same iteration count and residual
while reusing the preconditioner.

These are development regression measurements, not universal performance
claims.

## Limitations

- ILU(0) is ordering-sensitive.
- Current `L`/`U` triangular application is serial.
- There is no ILUT, drop tolerance, level-of-fill, threshold pivoting, or
  internal reordering.
- E5 is based mainly on synthetic nonsymmetric PDE-like families; real
  nonsymmetric FEM/PDE validation is still required.
- GeneralSquare is not yet routed through resident Rayon or GPU execution.
- GeneralSquare selection is not yet exposed through the C ABI configuration
  surface.

## Recommended evaluation

For a new GeneralSquare family:

1. run Jacobi baseline;
2. verify true residual;
3. opt into ILU(0);
4. record factor bytes and adjusted pivot count;
5. sweep a small set of restart values;
6. compare natural and at least one meaningful reordering;
7. benchmark repeated RHS reuse;
8. validate representative real matrices before changing automatic policy.