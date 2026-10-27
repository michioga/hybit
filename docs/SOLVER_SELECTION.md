# Solver selection and Krylov theory

> Numerical reference for the published HyBIT 0.9.0 API (SPD/GeneralSquare).
>
> This document describes the mathematical contracts behind HyBIT's current
> square-system routing. It distinguishes implemented paths from future routes.

## 1. Start from the matrix class, not the solver name

For a linear system

```text
A x = b
```

the first solver decision is a statement about `A`, not a performance tuning
choice.

```text
                         square system
                              |
             +----------------+----------------+
             |                                 |
        symmetric?                             no / unknown
             |                                 |
        +----+----+                            v
        |         |                    GeneralSquare
   positive     indefinite                    |
   definite        |                          v
        |          |                       FGMRES
        v          v                    + Jacobi / ILU(0)
       SPD      SymmetricIndefinite
        |          |
        v          v
       PCG       MINRES [future]
        |
        +-- Jacobi
        +-- Hybrid local direct
        +-- algebraic two-level
        +-- structural rigid-body two-level
```

Current high-level routing is:

| Declared problem class | Current route | Status |
| --- | --- | --- |
| `MatrixProblemClass::Spd` | PCG plus SPD-compatible HyBIT preconditioning | implemented |
| `MatrixProblemClass::GeneralSquare` | restarted right-preconditioned FGMRES | implemented |
| `MatrixProblemClass::SymmetricIndefinite` | intended MINRES route | recognized, not routed |
| rectangular | intended LSQR/LSMR-class route | not routed |

HyBIT does not infer positive definiteness from a positive diagonal, symmetry
flag, or successful factorization. `MatrixProblemClass` is a caller-declared
mathematical contract.

A singular, unconstrained, or nearly singular system is not made suitable for a
method merely by choosing one of these labels. For example, an unconstrained
structural stiffness matrix with rigid-body null modes is not SPD.

## 2. Why SPD systems use PCG

For PCG, the intended operator contract is

```text
A = A^T
x^T A x > 0   for every nonzero x
```

and the preconditioner must preserve the corresponding positive-definite
inner-product structure.

With an SPD preconditioner `M`, preconditioned CG can be interpreted through an
SPD transformed problem such as

```text
M^(-1/2) A M^(-1/2) y = M^(-1/2) b
x = M^(-1/2) y
```

in exact arithmetic. This structure is what supports the short recurrence and
the conjugacy/orthogonality relations on which CG depends.

The important consequence is operational:

> A preconditioner is not interchangeable merely because it approximately
> solves `A z = r`. PCG also needs the preconditioned recurrence to retain the
> required symmetry and positivity.

HyBIT's PCG implementation checks two quantities that directly expose loss of
those assumptions:

```text
r^T M^-1 r > 0
p^T A p    > 0
```

A non-finite or non-positive value produces numerical breakdown rather than
silently continuing a CG recurrence whose mathematical basis has failed.

### PCG workspace and per-iteration shape

The established `PcgWorkspace` owns five full-size `f64` vectors:

```text
A*x / scratch
r
z = M^-1 r
p
A*p
```

so its numeric payload is

```text
5 * n * sizeof(f64) = 40 n bytes
```

before counting matrix and preconditioner storage.

A normal PCG iteration has one operator application, one preconditioner
application, a small number of global reductions, and vector updates. Its
Krylov storage therefore stays `O(n)` rather than growing with iteration count.

## 3. Why HyBIT may continue or restart PCG

HyBIT's SPD controller divides a solve into observation and strengthening
stages. There are two mathematically different cases.

### Same preconditioner: continue the recurrence

If a probe or telemetry boundary does not change `A` or `M`, the
`PcgSession` retains the current search direction and `r^T M^-1 r` state.

```text
PCG iterations
      |
      +---- telemetry boundary ----+
      |                            |
      |       A and M unchanged    |
      +---------------------------> continue same recurrence
```

This is deliberately different from calling PCG again with the current `x`,
which would discard Krylov information.

### Changed preconditioner: restart PCG

If selective-direct Hybrid strengthening changes the preconditioner, the old
CG conjugacy relation no longer applies to the new operator/preconditioner
pair. HyBIT therefore starts a new PCG recurrence from the current solution.

```text
PCG with M0
    |
    +-- difficult region detected
    |
    v
build stronger M1
    |
    v
restart PCG from current x
```

This restart rule is a mathematical requirement of the current PCG design, not
just a controller implementation detail.

See [HYBRID_MATH.md](HYBRID_MATH.md) for the SPD construction used by the
Hybrid preconditioner.

## 4. SPD preconditioners in the current routes

The generic SPD and structural paths use preconditioners designed to preserve
PCG-compatible structure.

### Jacobi

The SPD Jacobi path requires a complete positive diagonal. Its approximation is
weak but cheap and positive definite when those requirements are satisfied.

### Hybrid local direct

For an SPD principal submatrix

```text
A_k = A[H_k, H_k]
```

HyBIT uses local Cholesky factors and symmetric overlap weighting. The local
terms are assembled in a form intended to remain symmetric positive
semidefinite, while uncovered DOFs retain positive Jacobi contribution.

### Algebraic two-level

The optional generic algebraic coarse correction is an SPD/PCG path. It is
disabled by default because an arbitrary generic matrix does not necessarily
carry the blocked/geometry semantics of a structural model.

### Structural rigid-body two-level

The explicit structural route combines 3x3 block-Jacobi with a six-rigid-body-
mode coarse space. It remains an SPD/PCG route; geometry is used to construct
the coarse correction, not to change the Krylov method.

## 5. Why GeneralSquare uses FGMRES

A general real square matrix may be nonsymmetric, nonnormal, or indefinite.
The SPD inner-product assumptions behind PCG therefore cannot be presumed.

HyBIT routes `MatrixProblemClass::GeneralSquare` through restarted,
right-preconditioned FGMRES.

At Arnoldi step `j`:

```text
v_j                  orthonormal Arnoldi basis
z_j = M_j^-1 v_j     independently stored preconditioned basis
w   = A z_j
```

After orthogonalization, the flexible Arnoldi relation is conceptually

```text
A Z_m = V_(m+1) Hbar_m
```

and the update is formed from `Z`, not directly from `V`:

```text
x_m = x_0 + Z_m y_m
```

This separation is the key distinction for flexible preconditioning. If
`M_j` changes with iteration, `z_j` cannot in general be reconstructed later
from `v_j` using one fixed inverse.

The current high-level GeneralSquare route still prepares a fixed Jacobi or
fixed ILU(0). FGMRES nevertheless keeps `V` and `Z` distinct, so the low-level
recurrence already supports a future iteration-dependent preconditioner through
`FlexiblePreconditioner`.

See [GENERAL_SQUARE.md](GENERAL_SQUARE.md) for the current Jacobi/ILU(0)
preconditioner policies.

## 6. Restart is a memory/work/convergence tradeoff

Unrestarted GMRES-style methods retain an expanding Krylov basis. Restart limits
that state to a maximum Arnoldi dimension `m`, but discards the old basis at a
cycle boundary after updating `x`.

For problem size `n` and restart `m`, the current `FgmresWorkspace` owns:

```text
V        (m + 1) * n
Z              m * n
r, w           2 * n
H        (m + 1) * m
cs, sn          2 * m
g              (m + 1)
y                    m
```

Therefore `FgmresWorkspace::bytes()` counts the numeric payload

```text
8 * [ (2m + 3)n + m^2 + 5m + 1 ] bytes
```

for `f64`, excluding Rust `Vec` headers and allocator overhead.

The basis part is `O(nm)`. Because the implementation uses two-pass modified
Gram-Schmidt, orthogonalization work over a full restart cycle grows roughly as
`O(n m^2)`.

Consequently:

- larger `m` retains a richer Krylov subspace and can reduce restart damage;
- larger `m` consumes more memory;
- larger `m` increases orthogonalization work;
- a stronger preconditioner can make a small restart faster even when it takes
  more iterations.

This is why E5's observed `FGMRES(3)+ILU(0)` result is treated as matrix-family
evidence, not as a universal default.

## 7. What the restart policies change

The GeneralSquare preconditioner choice and restart policy are independent.

### `Fixed`

Uses one restart dimension for every cycle. The current default is 30.

### `Escalating`

Starts from the configured restart and increases it in bounded stages up to
`max_restart`.

### `BudgetAware`

Observes exact residuals at restart boundaries and can increase the next cycle's
restart dimension when the remaining iteration budget is under sustained
pressure.

All policies reuse one prepared `FgmresWorkspace` whose capacity is large
enough for the maximum requested restart. Changing the next cycle's restart
does not require reallocating the Arnoldi basis.

## 8. True residual versus Hessenberg estimate

Inside a GMRES cycle, Givens rotations provide a cheap residual estimate from
the small Hessenberg least-squares problem.

HyBIT does not accept convergence solely from that estimate. The FGMRES
implementation recomputes

```text
r = b - A x
||r||_2
```

at restart boundaries and before reporting convergence.

For nonzero `b`, the public report uses

```text
relative_residual = ||b - A x||_2 / ||b||_2
```

This distinction matters for restarted methods, finite precision, and
ill-conditioned or strongly nonnormal problems.

## 9. Jacobi versus ILU(0) on GeneralSquare

The two current choices make different compromises.

| Property | GeneralSquare Jacobi | ILU(0) |
| --- | --- | --- |
| setup | very small | sparse factorization |
| coupling represented | diagonal only | original canonical sparsity pattern |
| fill | none | no added fill |
| pivoting | n/a | no row pivoting |
| application | simple diagonal scaling | forward/back triangular solves |
| current parallelism | simple | triangular application currently serial |
| ordering sensitivity | low | potentially strong |
| pivot diagnostics | nonzero diagonal requirement | adjusted-pivot count exposed |

ILU(0) is appropriate for FGMRES because FGMRES does not require the
preconditioner to define the SPD inner product required by PCG. That does not
mean ILU(0) guarantees good convergence: no-fill LU can be weak, ordering
sensitive, or numerically troublesome.

The selective row-relative pivot floor is a factorization safeguard. A nonzero
adjusted-pivot count should be treated as diagnostic evidence, not hidden as an
ordinary successful factorization.

## 10. Future SymmetricIndefinite route: MINRES

`MatrixProblemClass::SymmetricIndefinite` is represented in the API but is not
currently routed.

The intended Krylov family is MINRES because it can exploit symmetry without
requiring `A` itself to be positive definite. A future preconditioned MINRES
path must preserve the self-adjoint structure required by MINRES; in the
standard formulation this normally means using an SPD preconditioner.

A generic nonsymmetric ILU(0) should therefore not simply be reused as the
default MINRES preconditioner. Doing so would destroy the very symmetry that
makes MINRES attractive.

No current HyBIT call silently falls back from `SymmetricIndefinite` to PCG or
FGMRES.

## 11. Rectangular systems are a separate problem class

Least-squares systems

```text
min ||A x - b||_2
```

with rectangular `A` are not just another GeneralSquare case. Future
LSQR/LSMR-class support should have an operator contract that exposes the
actions required by the method, including transpose application.

Rectangular systems are currently outside the high-level HyBIT solver.

## 12. Current compatibility map

This table describes current high-level HyBIT routing, not every mathematically
possible combination of Krylov method and preconditioner.

| Path / preconditioner | PCG `Spd` | FGMRES `GeneralSquare` | future MINRES |
| --- | --- | --- | --- |
| positive-diagonal SPD Jacobi | yes | not the GeneralSquare object | potentially |
| SPD Hybrid local direct | yes | not routed | requires separate validation |
| generic algebraic two-level SPD | yes | not routed | requires separate validation |
| structural rigid-body two-level | yes | not routed | requires separate validation |
| GeneralSquare Jacobi | no PCG route | yes, default | not a MINRES policy |
| ILU(0) | no PCG route | yes, opt-in | not suitable as a generic symmetry-preserving default |

The presence of `Minres`, `Gmres`, or `Bicgstab` variants in reporting enums
does not by itself mean that the current high-level solver routes a problem
through those algorithms.

## 13. Failure modes: distinguish three different outcomes

### Converged

The solver reached its configured residual criterion. This says nothing by
itself about mesh quality, units, boundary conditions, material laws, or other
modeling correctness.

### MaxIterations

The recurrence remained numerically executable but did not reach tolerance
inside the iteration budget. Possible causes include a weak preconditioner,
poor restart choice, difficult spectrum/nonnormality, bad scaling, or a
problem that does not satisfy the declared model assumptions.

This is not the same as numerical breakdown.

### Breakdown

The recurrence encountered a condition that invalidates or prevents the next
mathematical step.

Examples in the current implementation include:

- PCG observes non-positive/non-finite `r^T M^-1 r`;
- PCG observes non-positive/non-finite `p^T A p`;
- FGMRES encounters a singular/non-finite Hessenberg factor;
- FGMRES Arnoldi can reach a happy breakdown; HyBIT recomputes the true
  residual and reports convergence only if the actual tolerance is met.

When breakdown occurs, increasing only `max_iterations` is generally not a
meaningful remedy. Recheck the declared matrix class, matrix validity,
preconditioner, scaling, and factorization diagnostics.

## 14. Practical selection workflow

For a new matrix family:

```text
1. Establish the mathematical contract of A.
              |
              v
2. Select the corresponding Krylov family.
              |
              v
3. Start with the simplest compatible preconditioner.
              |
              v
4. Independently verify ||b-Ax|| / ||b||.
              |
              v
5. Inspect iterations, breakdown/status, setup, solve time, and memory.
              |
              v
6. Strengthen one component at a time.
              |
              v
7. For repeated RHS, separate prepared reuse from warm-start effects.
```

For SPD problems, a PCG breakdown is evidence to investigate the SPD/operator
or preconditioner contract, not a reason to disguise the matrix as
`GeneralSquare`.

For GeneralSquare problems, compare Jacobi and ILU(0), inspect ILU pivot
adjustments, test ordering sensitivity, then tune restart. Do not infer a
universal restart from one matrix family.

For every path, independently recompute the true residual from the original
matrix and RHS before treating solver convergence as validated.

## G8-C12d-3: experimental MPI adaptive direct/Krylov hybrid (0.9)

The goal remains a memory-bounded **iterative + selective direct** solver, not
merely minimizing Krylov iteration counts. In this first MPI controller pilot,
rank-constant two-level Jacobi PCG begins with a fixed inexpensive SPD-compatible
preconditioner. Every configured iteration window, the globally reduced residual ratio is
measured relative to the **previous** checkpoint. A healthy early window is
not treated as evidence of indefinite future convergence: subsequent windows
are checked until convergence or one promotion occurs. The original PCG
recurrence continues unchanged while the light preconditioner is retained.
When a window stalls, every rank collectively prepares exact local
block-Cholesky factors (a specified uniform block size, max 512) and retains the
rank-constant Galerkin correction. **Only when the preconditioner changes** does
PCG restart, using a freshly recomputed `b - A*x` residual. Factorization is
performed once for this RHS, not on each iteration.

The promoted *fixed* additive preconditioner is
`M^-1 = sum_k R_k^T (A_kk)^-1 R_k + Z (Z^T A Z)^-1 Z^T`, subject to SPD assumptions
for the original operator and local/coarse factors. Positive pivots and PCG
convergence do not certify global SPD. Promotion rejects local Cholesky failures
collectively; there is no silent numerical fallback or claim of indefinite
support. `nd3k.mtx` remains **experimental, SPD unverified**.

This is a **guarded pilot**, not full serial `HybridOptions` parity. Local blocks
are contiguous, not dynamically selected from ABTM/residual risk. The controller
supports one promotion per RHS; no multi-stage adaptive region growth, general
MPI FGMRES, factor cache across RHS, or autonomous block-size selector is
claimed. Measurements must distinguish factor preparation cost, convergence
status, true residual, manufactured-solution error, communication, and memory.

## G8-C12d-6C4A: ABTM-style packed masks and globally pooled direct-factor budget

Experimental selector `replayg:HHHHH,HHHHH,HHHHH,HHHHH,HHHHH,HHHHH,HHHHH,HHHHH`
uses the same eight rank-local bitmasks as `replay:` but validates the **sum of
selected direct Cholesky bytes** across all eight MPI ranks, not an equal
per-rank quota. The total is checked with an MPI Allreduce **before** any
selected local Cholesky factorization. The existing `residual`, `boundary`,
`swap`, `replay`, and `replaync` selectors retain their original behavior.

An internal ABTM-inspired packed representation stores all 144 block flags in
three `u64` words and supports bitwise union/intersection/symmetric difference
and POPCNT. This is **not** the ABTM sparse matrix tile/value representation.
The numerical preconditioner is unchanged; only replay mask decoding and the
budget gate differ. The public result reports direct factor bytes separately
from the Jacobi fallback, rank-constant coarse factor payload, and process RSS.

G8-C12d-6C3 supplied a smooth-RHS mask at requested 4.250 MiB with exactly
4,303,736 selected direct bytes and 3,421 PCG iterations. These bytes fit a
4.125 MiB globally pooled cap of 4,325,376, although the final rank violates
4,325,376/8=540,672 bytes. Use the 4.250 MiB reference mask with `replayg`
under 4.125 MiB and verify identical iterations, factor bytes and true residual.
This does not constitute a globally optimized automatic block selector.
`nd3k` remains uncertified SPD; manufactured-solution convergence is not proof.

## G8-C12d-6C2: controlled rank-constant coarse ablation (experimental)

`replaync:HHHHH,HHHHH,HHHHH,HHHHH,HHHHH,HHHHH,HHHHH,HHHHH` is
identical to G8-C12d-6C1 `replay:` in its selected local Cholesky blocks,
rank-local mask validation, global direct-factor cap, and PCG restart.
During promotion only, `replaync:` installs `LocalBlockCholesky` (with the
same Jacobi fallback on unselected DOFs) **without** the additive
rank-constant Galerkin correction. The initial rank-constant Jacobi coarse
preconditioner is unchanged until promotion; the experiment forces
promotion at iteration 1 to minimize that common initial phase.

Compare the two early-promotion modes under the same manufactured RHS and
reference mask, never an automatic mask for another RHS. Both preserve the
same selected Cholesky factor bytes. Expected total factor payload differs:
with coarse it contains the selected Jacobi fallback plus replicated coarse
state; without coarse it contains only the local selected Cholesky / Jacobi
fallback. These are payload accounting figures, NOT process RSS. Coarse
Allreduce counts still include the common pre-promotion phase. Failed
convergence is a valid outcome for the experimental uncertified nd3k matrix.

This test isolates the existing eight rank-constant correction; it does
not establish the effectiveness of a richer GenEO/AMG coarse space.
The input nd3k remains unverified SPD, and no production selector is changed.

## G8-C12d-6C1: replayed block mask and promotion timing (experimental)

`replay:HHHHH,HHHHH,HHHHH,HHHHH,HHHHH,HHHHH,HHHHH,HHHHH` accepts
exactly eight hexadecimal masks, one per MPI rank. Bit `i` selects that
rank's local contiguous block `i`. The caller must obtain masks from the
already-verified G8-C12d-6B block traces; this mode is for controlled
experiments and is not a learned/general-purpose selector. Each rank
collectively validates nonempty selections and its **dense-Cholesky-only**
factor quota before factorizing. A malformed or overspending mask fails on
all ranks. The replayed preconditioner remains fixed after promotion, and
PCG performs the existing full restart. A production solver should not
accept external replay masks as an adaptive selection strategy.

For a fixed RHS, compare ordinary Residual at normal promotion with Replay
at normal promotion (validation of identical selected flags and numerical
results), then Replay with probe 1 and poor-progress ratio 0.01 to promote
near the first iteration. This isolates the timing effect subject to the
observed promotion iteration. All comparisons must account for the 4 MiB
payload bound *only for direct factors*, not full memory use.

Full Block and full Hybrid baselines can also indicate the benefit of the
rank-constant coarse correction with all blocks selected, but they do not
isolate a coarse correction effect for the selective replayed mask. That
requires a separate controlled experiment. nd3k remains uncertified SPD.

## G8-C12d-6B: recursive residual timeline and manufactured RHS families (experimental)

The existing 9/10/12/13/14-argument experiment calls remain unchanged.
A fixed-method 11-argument call appends one of `smooth`, `mixed`, or
`localized` after the relative tolerance. An adaptive 15-argument call
appends the same RHS kind after `residual|boundary|swap:R:FROM:TO`.
The exact solution `x_true` is generated by index; the distributed RHS is
`b=A*x_true`. `localized` refers to an index-localized exact solution,
NOT a physical concentrated force or certified FEM loading configuration.
The new calls opt into collecting the already global recursive residual
once per 64 iterations. The rank-zero `<run>.residual.csv` includes a
true-residual value at iteration 0, immediately after a promotion restart,
and at the terminal verification only. Intermediate samples have a blank
true-residual cell; these are NOT fresh SpMV verification values.
The 4 MiB selection budget covers dense Cholesky factor payload alone;
initial/coarse/Jacobi payload is reported separately and may push the total
above 4 MiB. Neither figure represents peak RSS or trace allocations.
Different RHS may select different block masks. Therefore the experimental
single-RHS `swap:1:13:10` must not be generalized to other RHS without
checking whether the swap remains valid. `nd3k` has no global SPD certificate.

## G8-C12d-6A: controlled single-block substitution (experimental)

`swap:R:FROM:TO` is an intervention for identifying which block mask decisions
materially influence the convergence of a given fixed test problem. It uses
exactly the G8-C12d-4 residual ranking on all MPI ranks before replacing one
selected block (`FROM`) with one previously unselected block (`TO`) on rank
`R`. It rejects invalid indices and swaps exceeding that rank's direct-factor
quota *collectively*. The prepared SPD-compatible preconditioner remains
constant after promotion, followed by a full PCG restart as in D3.

The selected mask is emitted to the ordinary rank trace CSV (mode `swap`).
One-factor pairwise swaps do **not** prove an automatically selected algorithm
or a universally better preconditioner. A different mask can converge more
slowly, fail to converge within the budget, or encounter invalid local factors;
all these are experimental outcomes, not reasons to loosen the accuracy gate.
No global SPD certificate exists for the `nd3k` experimental input.

## G8-C12d-5: diagnosable budgeted block selection (experimental)

G8-C12d-5 preserves the original `residual` selection mask and adds an opt-in
`boundary` ranking mode. Its score is the residual-energy-per-dense-factor-byte
scaled by `1/(1 + exterior_L1/(interior_L1 + exterior_L1))`, where interior and
exterior denote entries of the rank-owned CSR row block relative to that
contiguous block. Ghost columns count as exterior. This is a *heuristic*:
there is no proof it reduces the preconditioned condition number or iteration
count. The block mask is fixed after the collective adaptive escalation;
PCG is restarted as in G8-C12d-3, maintaining a fixed SPD-compatible
preconditioner for subsequent iterations under the declared SPD contract.

Each promoted MPI rank writes its own experiment-scoped `.rankN.blocks.csv`
containing rank/block ID, global first DOF, block size, dense factor bytes,
residual energy, interior/exterior coupling norms, score and selection flag.
Nonpromoted runs produce no block trace; downstream reports must explicitly
record this rather than interpreting a missing trace as a failure. Trace files
are created separately on each rank and I/O errors are reduced collectively
before the next MPI collective. Diagnostics are emitted *after* the solve and
are excluded from `report.elapsed_ns`; this avoids distorting the solver
comparison, but wall-clock experiment time includes diagnostic writing.

Neither ranking policy redistributes unused quotas between ranks, grows
ABTM regions or guarantees convergence for a prescribed budget. Both use the
same bounded dense-factor policy and same fixed additive coarse correction.
`nd3k` remains experimental and **not globally SPD-certified**. Compare the
masks first; treat any speedup as provisional until repeated, controlled runs.

## G8-C12d-4: bounded selective direct factors (pilot)

The MPI adaptive preconditioner uses a **strict global dense-factor byte cap**.
The cap is divided equally across MPI ranks; each rank ranks contiguous
principal blocks by residual squared energy per stored dense-factor byte,
with stable block-index ties, and factors only blocks fitting its quota.
Unselected DOFs retain positive Jacobi inverse-diagonal weights, while selected
DOFs receive exact principal-block Cholesky inverse; every DOF is covered
exactly once by the local base. The rank-constant Galerkin coarse correction
is additive and positive semidefinite when the global operator is SPD.
The overall preconditioner is therefore fixed SPD **after** promotion, provided
all global mathematical SPD assumptions are satisfied. PCG is restarted only
when direct factors actually change the preconditioner. An uncertified matrix
remains uncertified even if Cholesky preparation and manufactured-solution
residual checks pass.

This first budget allocator is rank-balanced, *not* globally optimal; unused
budget cannot be transferred from one rank to another. Memory telemetry must
separate dense-factor bytes from Jacobi/coarse storage; selection does not yet
use ABTM graph expansion or arbitrary disjoint region sets.

The G8-C12d-3 Fix4 controller specifically addresses the nd3k false negative:
the first 64 iterations reduced the residual to 0.0749759 times its initial
norm (below a 0.5 promotion threshold), yet convergence stalled later.
Only the last evaluated window ratio is reported by `adaptive_probe_ratio`.
No extra global reduction is introduced by the window check; the PCG norm is
already a global reduction. No additional preconditioner allocations occur
unless a window actually triggers promotion.

The new `DistributedPcg::solve_adaptive` and example `mpi_two_level_experiment`
`adaptive:<blocksize>` path are explicit opt-ins. Existing fixed solvers and
previous experiment CLI forms remain unchanged. See G8-C12d-3 package README
for validation protocol and command examples.
