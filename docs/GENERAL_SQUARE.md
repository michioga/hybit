# GeneralSquare FGMRES and ILU(0)

> Development documentation for `develop/0.8.0`, current through F3.

## Purpose

`MatrixProblemClass::GeneralSquare` routes real nonsymmetric square systems
through restarted FGMRES. The prepared preconditioner is selected independently
from restart policy.

Current choices:

- Jacobi — default;
- ILU(0) — explicit opt-in.

GeneralSquare currently supports `ExecutionPolicy::Auto` and
`ExecutionPolicy::Cpu`.

For the cross-method rationale—PCG versus FGMRES, preconditioner compatibility,
restart complexity, and the future MINRES route—see
[SOLVER_SELECTION.md](SOLVER_SELECTION.md).
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

## F1 ordering evidence

F1 added a benchmark-only Natural-versus-RCM cross-check without changing the
production GeneralSquare policy. RCM is formed from the undirected sparsity
graph of `A + A^T`; the benchmark applies the same permutation to rows,
columns, and RHS, then maps the computed solution back to the original ordering
and recomputes the true residual against the original `A` and `b`.

Jacobi is run under both orderings as a control. Because a simultaneous
row/column permutation should not materially change diagonal scaling as a
mathematical preconditioner, matching Jacobi behavior is used to validate the
permutation/verification path before interpreting ILU(0) differences.

Measured fixed-restart-30 results at relative tolerance `1e-8`:

| Matrix | Matrix class | Bandwidth Natural -> RCM | ILU(0) Natural | ILU(0) RCM | Observation |
| --- | --- | ---: | ---: | ---: | --- |
| `cfd1` | symmetric | 6229 -> 3011 | 3363 iterations | 1932 iterations | strong RCM benefit |
| `sherman5` | nonsymmetric | 1106 -> 126 | 30 iterations | 30 iterations | no convergence benefit; ordering is overhead |
| `raefsky3` | nonsymmetric | 1263 -> 735 | 51 iterations | 16 iterations | strong RCM benefit |

For `cfd1`, ILU(0) solve wall time fell from about 18.0 s to 10.5 s; the
measured RCM graph/order/permutation cost was about 64 ms. For `raefsky3`,
solve wall time fell from about 141.6 ms to 51.0 ms, while ordering cost about
41.2 ms. `sherman5` demonstrates the opposite regime: Natural ILU(0) already
converged in one restart cycle, so RCM added cost without reducing iterations.

All three Natural/RCM ILU(0) comparisons reported zero adjusted pivots. The
observed convergence differences therefore cannot be attributed to the
row-relative pivot floor.

The corpus also shows that bandwidth reduction is not a sufficient automatic
selection signal: `sherman5` had the largest relative bandwidth reduction but
no ILU(0) iteration reduction. A future automatic policy should therefore
consider solve/progress behavior rather than promoting RCM solely from a graph
metric.

`Goodwin_010` was rejected by the current high-level GeneralSquare contract
because at least one structural diagonal entry is missing. F1 deliberately did
not insert artificial diagonal values; this matrix is retained as evidence for
the later unsuitable-ILU/fallback design.

These measurements use generated `b = A * 1` to provide a known solution and
an independent forward-error cross-check. They are development evidence for
the measured matrices, RHS construction, hardware, and revision rather than a
universal ordering rule.

## F2 real nonsymmetric corpus evidence

F2 adds preflight classification and a multi-matrix corpus wrapper. Unsupported
cases remain recorded evidence instead of aborting the sweep.

| Matrix | Dimension | Preflight | Detail |
| --- | ---: | --- | --- |
| `sherman5` | 3312 | supported | complete nonzero diagonal |
| `raefsky3` | 21200 | supported | complete nonzero diagonal |
| `venkat25` | 62424 | supported | complete nonzero diagonal |
| `Goodwin_010` | 1182 | unsupported | 299 missing diagonal rows |
| `Goodwin_023` | 6005 | unsupported | 1586 missing diagonal rows |
| `Goodwin_030` | 10142 | unsupported | 2699 missing diagonal rows |
| `goodwin` | 7320 | unsupported | 1079 missing diagonal rows |
| `rma10` | 46835 | unsupported | 5617 missing diagonal rows |

Unsupported inputs are not repaired by inserting artificial diagonal values.

For `venkat25`, generated `b=A*1`, restart 30, tolerance `1e-8`, maximum 5000
iterations produced:

| Metric | Natural | RCM |
| --- | ---: | ---: |
| structural bandwidth | 60323 | 2451 |
| Jacobi status | max iterations | max iterations |
| Jacobi iterations | 5000 | 5000 |
| Jacobi verified residual | `7.210955e-4` | `7.211180e-4` |
| ILU(0) status | converged | converged |
| ILU(0) iterations | 190 | 164 |
| ILU(0) solve wall | 883.959 ms | 771.268 ms |
| ILU(0) adjusted pivots | 0 | 0 |

RCM graph/order/permutation cost 52.990 ms. Charging analysis, preparation,
ordering, and solve once gives approximately 956.95 ms Natural versus 897.19 ms
RCM. Repeated RHS reuse should improve RCM amortization further if the prepared
ordering and factors are reused.

Together with F1, the supported real nonsymmetric corpus deliberately contains
different ordering regimes: no convergence benefit on `sherman5`, strong
benefit on `raefsky3`, and moderate benefit on the larger `venkat25`.
Bandwidth reduction is therefore telemetry, not an automatic selection rule.

## F3 prepared multi-RHS ordering evidence

F3 measures whether the F1/F2 ordering behavior survives prepared solve-many
reuse with different right-hand sides. Natural and RCM ILU(0) states are each
prepared once and reused for five deterministic RHS vectors (`ones`,
`alternating`, then deterministic hashed vectors). Every solve starts from a
zero initial guess. The RCM path includes per-RHS `P b` construction and
solution unpermutation in the end-to-end accounting.

The benchmark verifies the final residual in the original system after both
timed solve paths complete. From solve sequence 2 onward it also requires the
prepared context to report preconditioner reuse.

Five repeated 5-RHS runs produced:

| Matrix | 5-RHS Natural end-to-end median | 5-RHS RCM end-to-end median | Median RCM/Natural | Ratio range | Break-even |
| --- | ---: | ---: | ---: | ---: | --- |
| `raefsky3` | 744.271 ms | 393.005 ms | `0.527337` | `0.526056` - `0.528967` | RHS 1 in 5/5 repeats |
| `venkat25` | 3421.034 ms | 3363.304 ms | `0.984314` | `0.972316` - `0.991788` | RHS 1 in 5/5 repeats |

For `raefsky3`, the same ILU(0) ordering advantage seen with `b=A*1` remains
strong across all five RHS families. The deterministic iteration totals are
225 Natural versus 69 RCM over the five RHS vectors. The median cumulative
permutation/unpermutation cost is only about 0.257 ms, so end-to-end behavior
is dominated by the Krylov/triangular-apply savings.

`venkat25` behaves differently. Its deterministic five-RHS iteration total is
737 Natural versus 692 RCM, but most of that difference comes from the first
two RHS vectors. Later hashed RHS vectors differ by only a few iterations and
individual RCM wall times can match or exceed Natural. Across five repeated
process runs the median end-to-end advantage is only about 1.57%, despite RCM
being lower in all five paired totals. This is treated as near-parity,
timing-sensitive evidence rather than a robust performance win.

The result weakens two potential automatic-selection signals. Structural
bandwidth alone was already insufficient after F1/F2; F3 additionally shows
that a strong result on one RHS does not necessarily predict the ordering
benefit for later RHS vectors on the same matrix. A future ordering policy
should therefore use progress/solve evidence conservatively rather than
promoting RCM from bandwidth or one isolated RHS result.
## Limitations

- ILU(0) is ordering-sensitive.
- Current `L`/`U` triangular application is serial.
- There is no ILUT, drop tolerance, level-of-fill, threshold pivoting, or
  internal reordering.
- F2 validates three supported real nonsymmetric ordering cases up to 62424
  unknowns, but the corpus is not exhaustive; five additional screened
  matrices expose the current complete-diagonal contract as a major
  applicability boundary.
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