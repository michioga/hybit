# HyBIT user guide

> Development documentation for `develop/0.8.0`.
>
> HyBIT 0.7.0 remains the current published crates.io release. The 0.8 APIs
> described here are development APIs and may change before the next release.

## 1. Choose the mathematical problem class

HyBIT does not infer the mathematical class of an arbitrary matrix. The caller
declares the contract with `MatrixProblemClass`.

| Problem class | Current route | Typical use |
| --- | --- | --- |
| `Spd` | PCG, with Jacobi or the SPD Hybrid/coarse policies | constrained structural stiffness, Poisson-like SPD systems |
| `GeneralSquare` | restarted FGMRES, with Jacobi or opt-in ILU(0) | nonsymmetric square systems |
| `SymmetricIndefinite` | recognized but not routed yet | future MINRES path |

`Spd` is the default. Do not label a matrix SPD merely to reach PCG: the
algorithm and SPD preconditioners rely on symmetry and positive definiteness.

Rectangular systems are not currently routed by the high-level solver.

### CSR32 input contract

`Csr32Matrix::new` validates the storage contract:

- `row_ptr.len() == nrows + 1`;
- `row_ptr[0] == 0`;
- `row_ptr` is monotonically nondecreasing;
- `row_ptr[nrows] == nnz`;
- `col_idx.len() == values.len()`;
- every column index is smaller than `ncols`;
- all matrix values are finite;
- `nnz <= u32::MAX`.

CSR rows do **not** have to be column-sorted and duplicate column entries are
allowed. Matrix-vector multiplication therefore follows the caller's stored
row order. `diagonal()` sums duplicate diagonal entries.

Problem-specific requirements are stricter than the storage contract. For
example, the SPD route requires a complete positive diagonal. GeneralSquare
Jacobi requires a complete finite nonzero diagonal. GeneralSquare ILU(0)
requires diagonal entries to exist, then privately canonicalizes rows before
factorization.

### Defaults at a glance

The most important high-level defaults on the 0.8 development branch are:

| Setting | Default |
| --- | --- |
| problem class | `MatrixProblemClass::Spd` |
| relative tolerance | `1e-8` |
| absolute tolerance | `0` |
| maximum Krylov iterations | `1000` |
| backend policy | `Auto` (currently resolves to CSR32) |
| execution policy | `Auto` (CPU) |
| generic SPD Hybrid controller | enabled |
| generic algebraic coarse correction | disabled |
| GeneralSquare preconditioner | Jacobi |
| GeneralSquare restart policy | `Fixed` |
| GeneralSquare restart | `30` |
| GeneralSquare `max_restart` | `70` |
| escalating stage budget | `300` iterations |

Defaults are conservative compatibility choices, not claims that they are
optimal for every matrix family.
## 2. Use the smallest correct interface

One-shot SPD:

```rust
use hybit::{Csr32Matrix, HybitSolver};

let a = Csr32Matrix::new(
    3,
    3,
    vec![0, 2, 5, 7],
    vec![0, 1, 0, 1, 2, 1, 2],
    vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
)?;
let b = vec![1.0, 0.0, 1.0];
let mut x = vec![0.0; 3];

let solver = HybitSolver::new();
let report = solver.solve_csr32(&a, &b, &mut x)?;
assert!(report.converged());
```

GeneralSquare with opt-in ILU(0):

```rust
use hybit::{
    GeneralSquareOptions, GeneralSquarePreconditionerPolicy, HybitSolver,
    MatrixProblemClass, SolverOptions,
};

let mut solver = HybitSolver::new();
solver.set_problem_class(MatrixProblemClass::GeneralSquare);
solver.set_options(SolverOptions {
    relative_tolerance: 1.0e-8,
    absolute_tolerance: 0.0,
    max_iterations: 1800,
})?;

solver.set_general_square_preconditioner_policy(
    GeneralSquarePreconditionerPolicy::Ilu0,
);

// Measured E5 setting, not an automatic default.
solver.set_general_square_options(GeneralSquareOptions {
    restart: 3,
    ..GeneralSquareOptions::default()
})?;

let mut x = vec![0.0; a.nrows()];
let report = solver.solve_csr32(&a, &b, &mut x)?;
```

Run the complete example with:

```text
cargo run --release -p hybit --example general_square
```

### Configure before analyze

For a reproducible lifecycle, set the problem class, solver tolerances,
backend/execution policy, GeneralSquare preconditioner, and restart policy
before calling `analyze_csr32`.

The analysis records the problem class, execution target/policy, backend
choice, and matrix signatures. `prepare_csr32` validates compatible analysis
state and then captures the current solver/preconditioner options into the
prepared context. Changing the `HybitSolver` object afterwards does not mutate
an already-created prepared context.
## 3. Prefer prepared execution for repeated right-hand sides

```rust
let analysis = solver.analyze_csr32(&a)?;
let mut prepared = solver.prepare_csr32(&a, &analysis)?;

let mut x1 = vec![0.0; a.nrows()];
let first = prepared.solve(&a, &b1, &mut x1)?;

let mut x2 = vec![0.0; a.nrows()];
let second = prepared.solve(&a, &b2, &mut x2)?;
```

Lifecycle:

```text
matrix + solver policy
        |
        v
     analyze
        |
        |  validate problem contract
        |  profile matrix
        |  resolve backend/execution policy
        |  record structure/value signatures
        v
     prepare
        |
        |  build matrix-dependent preconditioner state
        |  allocate reusable Krylov workspace
        v
     solve #1
        |
        v
     solve #2 ... N
        |
        +-- reuse matrix-dependent state
```

Prepared contexts currently require unchanged CSR structure and unchanged
coefficient bit patterns. See [PREPARED_API.md](PREPARED_API.md).

## 4. Validate the result

At minimum, inspect:

- `report.converged()`;
- `report.relative_residual`;
- `report.iterations`;
- `report.solver`;
- `report.preconditioner`;
- `report.backend`.

`report.final_residual` is the absolute Euclidean residual norm. For nonzero
RHS,

```text
report.relative_residual = ||b - A x||_2 / ||b||_2
```

For a zero RHS, `relative_residual` is reported as the absolute final residual
instead of dividing by zero.

For engineering/scientific work, independently recompute the same residual from
the original matrix and RHS. Linear-solver convergence does not validate the
physical model, mesh, units, boundary conditions, or material law.

## 5. GeneralSquare choices are independent

Preconditioner:

- `Jacobi` — default, low setup/storage, finite nonzero diagonal required.
- `Ilu0` — opt-in, stronger coupling model, canonical CSR factorization,
  currently serial triangular application.

Restart:

- `Fixed` — default; restart 30.
- `Escalating` — bounded staged restart growth.
- `BudgetAware` — restart-boundary residual controller.

E5 found FGMRES(3)+ILU(0) fastest among tested restart values on three synthetic
hard families. HyBIT does not make that combination the global default because
ordering and matrix family can materially change ILU(0) quality.

See [GENERAL_SQUARE.md](GENERAL_SQUARE.md).

## 6. SPD and structural paths

Generic SPD uses `solve_csr32` / `prepare_csr32` and PCG-based policies.

For 3-D structural SPD systems with node coordinates, use
`solve_structural_csr32` / `prepare_structural_csr32` to enable the rigid-body
coarse path. See [STRUCTURAL_AUTO_API.md](STRUCTURAL_AUTO_API.md).

## 7. Backend and execution scope

CSR32 is the public sparse input representation. ABTM is an internal
execution/topology representation.

GeneralSquare FGMRES currently accepts `ExecutionPolicy::Auto` or
`ExecutionPolicy::Cpu`. Resident Rayon and GPU routing are not active for this
path yet.

## 8. Rust versus C/C++/Fortran

The repository retains the stable 0.7 C ABI and language examples. The 0.8
Rust API can select GeneralSquare, FGMRES restart policy, and ILU(0).

The current C ABI can report FGMRES/ILU(0) enum codes, but does not yet expose
setters for selecting the GeneralSquare problem class and its preconditioner.
Treat the new GeneralSquare path as Rust-API-only until that surface is added
and validated.

## 9. Evaluation sequence for a new matrix family

1. confirm the mathematical problem class;
2. run a simple baseline;
3. independently verify the true residual;
4. record iterations, setup, solve time, and persistent state;
5. change one policy at a time;
6. for ILU(0), test ordering sensitivity;
7. test solve-many reuse separately from warm starts;
8. keep conclusions scoped to tested matrices and hardware.