# C1-H12-A8: HyBIT-native multilevel numerical AMG pilot

Status: **prototype only, owner WSL release/Rust/real-FEM acceptance pending**.
No dependency on PETSc or hypre. No production preconditioner, public API,
FFI, default PCG or distributed solver is modified by A8.

## Numerical hierarchy

A8 adds an opt-in Rust example that builds at least three numerical levels:

```text
A_0 = original symmetric positive definite CSR32 input
  | strength/graph/contiguous aggregates, nominal 4 nodes each
  | P_0 = tentative one-hot piecewise basis, optionally damped-Jacobi smoothed
A_1 = P_0^T A_0 P_0  (exact numerical Galerkin, ABTM Boolean candidate)
  | aggregate, construct P_1, apply Galerkin again
A_2 = P_1^T A_1 P_1
  ... repeat until n_bottom <= 512 (maximum 8 total levels)
  | dense Cholesky on the coarsest numerical operator
```

The interpolation candidates use the A2/A3 word topology idea on the *numeric*
`A_l P_l` product. Every fine row in `P_l` ORs the corresponding ABTM words
into each owning coarse row, generating the exact Boolean support of
`P_l^T (A_l P_l)`. Numerical coefficients are then accumulated exclusively
within these candidates. Exact-zero candidate entries remain explicitly
represented. No Boolean candidate is ever presented as a numerical AMG solve.

The input and each constructed level are checked for structural/value
symmetry. Numerical transposed entries in the Galerkin result are averaged
only when their discrepancy is inside a tight floating-point symmetry bound;
larger discrepancies are errors. Positive diagonals and coarsest Cholesky
factorization are mandatory. No silent identity or Jacobi fallback.

## Symmetric V-cycle

For each level other than the coarsest, an application performs:

1. One damped Jacobi pre-smoothing update from zero.
2. Restrict `r - A_l x` through `P_l^T`.
3. Recursively solve at level `l+1`, then prolongate through `P_l`.
4. One matching damped Jacobi post-smoothing update.

The damping factor uses a Gershgorin upper bound on
`lambda_max(D^-1 A)` and satisfies `omega < 2 / bound`, preserving the
SPD-compatible symmetry conditions under the tested SPD assumptions.
Per-application extra work is explicitly logged as
`2 * (levels - 1) * precond_calls` sparse `A_l` matvecs. All interpolation,
restriction, hierarchy storage, setup time and coarse factor bytes are
reported independently of the baseline two-level control.

The V-cycle is kept *fixed* within each PCG solve. A5's independently
verified true-residual restart wrapper remains the convergence authority.
All cases use `b=A*1`, `tol=1e-10`, at most 600 global PCG iterations,
one thread and original CSR residual verification.

## Paired owner experiment

Six coarse arm configurations from A7 (contiguous, graph, strong graph;
piecewise constant or Jacobi-smoothed basis) are run on each of the same
three real FEM matrices. Each arm runs:

- `two_level`: unchanged HyBIT A5/A7 additive preconditioner control,
  target coarse dimension 512 and F64 transfer values.
- `multilevel`: new A8 native numerical hierarchy and symmetric V-cycle.

Total: `3 matrices x 6 arms x 2 policies = 36 independent processes`.
The comparison gate requires the A8 two-level control to reproduce the
frozen A7 additive status, iteration count, factor sizes and true residual,
plus identical full Git HEAD and original matrix SHA-256 files.

```bash
cd ~/hybit
BASE="$HOME/c1h11_evidence/c1h12_a7_symmetric_Ox04Pe"
bash benchmarks/scripts/C1H12_A8_native_multilevel.sh --dry-run "$BASE"
bash benchmarks/scripts/C1H12_A8_native_multilevel.sh --run "$BASE"
```

`C1-H12-A8` records include the dimensions and nonzeros for every level,
ABTM word and symbolic candidate counts, numeric cancellation counts,
setup/solve times, coarse direct-factor bytes, V-cycle/SpMV counts, and
independent original-CSR residuals. The gate preserves honest MaxIterations,
Breakdown or failed construction and rejects any false convergence claim.

## Acceptance boundary

- Rust unit tests must verify ABTM symbolic candidate Galerkin numerical
  energy identity across the 64-bit boundary, a small symmetric/positive
  V-cycle, PCG true-residual convergence, and invalid hierarchy rejection.
- Python negative gates, rustfmt, Clippy, release build, source SHA checks,
  and original-CSR provenance must all pass.
- Compare convergence count **before** median solve/runtime costs. One
  process run per arm is pilot evidence, never a whole-solve speedup claim.
- Keep all failed cases and raw logs. If any gate fails, leave A8 uncommitted
  and do not claim AMG acceptance.

The prototype is serial, uses deterministic 4-node graph growth and
`BTreeMap` numeric accumulation, and deliberately prioritizes correctness
over tuned setup cost. In particular the new coarsening differs from the
existing A5 graph/strong-graph strategy; iteration differences cannot be
ascribed to level count alone. Future phases can compare adaptive
aggregation, interpolation truncation, numerical SpGEMM, ABTM-versus-CSR
break-even, and MPI hierarchy operations under separate gates.
