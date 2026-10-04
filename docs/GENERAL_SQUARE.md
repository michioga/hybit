# GeneralSquare FGMRES and ILU(0)

> Development documentation for `develop/0.8.0`, current through F7.

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
| `raefsky3` | 744.271 ms | 392.196 ms | `0.527290` | `0.526349` - `0.528967` | RHS 1 in 5/5 repeats |
| `venkat25` | 3421.034 ms | 3363.304 ms | `0.984313` | `0.972316` - `0.991788` | RHS 1 in 5/5 repeats |

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
## F4 serial triangular-apply and dependency evidence

F4 separates ILU(0) triangular application cost from the surrounding FGMRES
solve and compares that kernel with serial CSR SpMV. The retained benchmark
also reconstructs the canonical ILU(0) pattern and profiles forward/backward
dependency levels without changing the production solver.

The measured serial kernel medians were:

| Matrix | Natural ILU apply | RCM ILU apply | RCM/Natural ILU | Natural SpMV | RCM SpMV |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sherman5` | 0.021418 ms | 0.015114 ms | `0.705668` | 0.007068 ms | 0.007222 ms |
| `raefsky3` | 0.950286 ms | 0.951054 ms | `1.000808` | 0.565418 ms | 0.565824 ms |
| `venkat25` | 1.062186 ms | 1.213100 ms | `1.142079` | 0.542808 ms | 0.515574 ms |

The three matrices therefore expose different mechanisms. `sherman5` gets a
faster triangular kernel from RCM but no measured F1 iteration reduction.
`raefsky3` gets essentially no triangular-kernel speedup, so its strong RCM
benefit is dominated by improved ILU(0) effectiveness. `venkat25` gets a
faster SpMV but a slower triangular application, partially offsetting its
modest RCM iteration reduction.

The dependency profile also shows why bandwidth alone is insufficient:

- `sherman5` RCM changes forward/backward levels from 39/66 to 52/70 while
  reducing dependency-distance p95 from 1105 to 117.
- `raefsky3` RCM changes 1176/1176 levels to 1592/1592 and reduces structural
  average parallelism from about 18.03 to 13.32.
- `venkat25` RCM changes 4176/4176 levels to 1700/1700 and raises structural
  average parallelism from about 14.95 to 36.72, but dependency-distance
  median grows from 117 to about 711/712 and p95 from 1948 to 2348.

A follow-up experimental level-scheduled Rayon prototype produced numerically
identical results, but one parallel dispatch/barrier per triangular level was
too fine grained. A width-threshold hybrid did not recover the overhead:
every measured case that actually invoked Rayon was slower than the canonical
serial row-order apply. That prototype is intentionally not retained in the
production code.

The current conclusion is to keep canonical ILU(0) triangular application
serial. Future parallel work, if justified by a larger workload, should use a
coarser persistent-worker, task-DAG, or superlevel design rather than the
rejected per-level Rayon schedule.
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

## F5 unsuitable-ILU fallback

F5 separates solver applicability from ILU(0) applicability.

The existing `GeneralSquarePreconditionerPolicy::Ilu0` remains strict. A
structurally missing diagonal still rejects explicit strict ILU(0). The new
explicit `GeneralSquarePreconditionerPolicy::Ilu0Fallback` first attempts the
same canonical ILU(0) preparation and falls back to a prepared Identity
preconditioner only for `HybitError::MissingDiagonal`. Other ILU preparation
errors remain errors and are not silently hidden.

The prepared context exposes the effective preconditioner kind and whether the
structural fallback was used. A solve using the Identity fallback reports
`PreconditionerKind::None`.

The real missing-diagonal corpus from F2 was rechecked with this policy. All
five matrices (`Goodwin_010`, `Goodwin_023`, `Goodwin_030`, `goodwin`, and
`rma10`) retained strict-ILU rejection while `Ilu0Fallback` reached prepared
state with `fallback_used=true`.

A retained bounded-solve harness then tested the Identity fallback for 300
FGMRES iterations at restart 30 and tolerance `1e-8` using deterministic
manufactured right-hand sides. The verified relative residuals were:

| Matrix | Identity fallback after 300 iterations |
| --- | ---: |
| `Goodwin_010` | `1.543836e-6` |
| `Goodwin_023` | `2.498295e-6` |
| `Goodwin_030` | `2.403120e-5` |
| `goodwin` | `7.393263e-1` |
| `rma10` | `4.815204e-2` |

Identity is therefore retained as a correctness/safety fallback, not promoted
as a claim of strong numerical preconditioning.

F5 also explored several benchmark-only stronger fallbacks and rejected them
from the retained implementation:

- inserting row-relative synthetic diagonal values before ILU(0) helped the
  three `Goodwin_0xx` matrices but degraded `goodwin` and especially `rma10`;
- arbitrary structural perfect matching existed for all five matrices but the
  resulting column-permuted ILU(0) was numerically unstable;
- row-relative bottleneck matching removed the weakest matched entries but all
  five cases still broke down;
- alternating max-norm row/column equilibration drove row and column norms near
  one after eight sweeps, yet the matching-ILU triangular solve still broke
  down. `goodwin` and `rma10` did so with zero adjusted pivots.

These negative experiments delimit the current canonical ILU(0) operating
envelope: missing-diagonal systems need more than structural completion,
matching, or simple equilibration for a robust stronger preconditioner. The
rejected F5c-F5f experiment programs are not retained in the repository.

The next GeneralSquare work is policy selection from residual/progress evidence,
not another ad-hoc missing-diagonal transformation. Drop-tolerance/fill or
pivoting methods may be considered later as separate preconditioners now that
the ILU(0) envelope is documented.

## F6 ordering-selection evidence

F6 tested whether Natural -> RCM ILU(0) ordering can be selected from cheap
runtime evidence without promoting RCM unconditionally.

### F6a: Natural-only progress is insufficient

F6a collected exact FGMRES restart-boundary residuals under Natural ILU(0) and
compared them with the eventual Natural-versus-RCM result.

Natural-only early progress did not separate the measured regimes. For example,
`sherman5` and `raefsky3` both showed strong first-cycle logarithmic residual
decay, but RCM provided essentially no convergence benefit on `sherman5` and a
large benefit on `raefsky3`. `venkat25` showed much slower Natural early decay
yet only modest, RHS-dependent RCM benefit.

A first-cycle projected-total-iteration extrapolation was also unreliable
because convergence rates changed after restart boundaries.

### F6b: paired short probes are more discriminating

F6b prepared both Natural and RCM ILU(0) states and compared equal-length
4-, 8-, and 16-iteration FGMRES probes from `x=0`.

The key benchmark signal is

```text
paired_ratio =
    (RCM residual / initial residual)
    /
    (Natural residual / initial residual)
```

after the same number of probe iterations.

The 4-iteration ratio separated several important measured cases:

| Matrix | RCM/Natural bandwidth | 4-iteration paired ratio | Full-run observation |
| --- | ---: | ---: | --- |
| `sherman5` | `0.1139` | `6.5546` | Natural-equivalent or better |
| `raefsky3` | `0.5819` | `0.2496` | strong RCM benefit |
| `venkat25` | `0.0406` | `2.1703` | only modest/RHS-sensitive RCM benefit |
| `cfd1` | `0.4834` | `0.9457` | RCM beneficial for the measured solve-many workload |
| `thermal1` | `0.00282` | `0.5800` | RCM beneficial across all five measured RHS vectors |
| `nd3k` | `0.3585` | `1.0474` | both hit the iteration cap; Natural residual better |
| `cant` | `2.3782` | `2.7272` | RCM increases bandwidth; both hit the cap |
| `s3dkq4m2` | `1.9609` | `1.00018` | RCM increases bandwidth; both hit the cap |
| `boneS01` | `2.1381` | `1.6857` | RCM increases bandwidth; both hit the cap |
| `x104` | `1.2057` | `1.000001` | RCM increases bandwidth; both nearly stagnate |

The additional capped cases are classification evidence only; they are not
claimed as converged performance wins.

`cfd1` also showed why the paired signal must not be interpreted as a
per-RHS winner predictor. RCM was much faster for RHS 1, 2, and 4 but slower
for RHS 3 and 5. The first-RHS paired probe is therefore best interpreted as a
candidate signal for a prepared solve-many workload, not as a guarantee for
every subsequent right-hand side.

### F6c: selector cost must be charged

F6c replayed the candidate decision while charging Natural preparation, RCM
ordering and preparation, both short probes, and the selected full solves.

With a four-iteration paired probe and a provisional threshold of `1.0`:

- `cfd1` selected RCM with paired ratio `0.945698`. Across five repeats, the
  median policy/Natural cost ratios were `0.591`, `0.606`, `0.733`, `0.731`,
  and `0.798` through RHS counts 1 through 5.
- `thermal1` selected RCM with paired ratio `0.579999`. The corresponding
  median ratios were `0.780`, `0.625`, `0.616`, `0.622`, and `0.609`.
- `raefsky3` had already shown paired ratio `0.249599`; with the earlier replay
  it became clearly profitable by the third RHS and reached about `0.668`
  through five RHS vectors.
- `venkat25` correctly retained Natural with paired ratio `2.170262`, but
  paying for the rejected alternate state still left about 3.8% overhead after
  five RHS vectors.
- a provisional small-row guard avoided this overhead on `sherman5`, but row
  count is a benchmark heuristic rather than a validated production criterion.

### F6 conclusion

The evidence does not justify an unconditional production Natural -> RCM
promotion.

A useful experimental solve-many selector now has two evidence-backed pieces:

1. skip RCM when it does not reduce structural bandwidth;
2. when RCM does reduce bandwidth, a four-iteration paired residual ratio below
   one is a promising numerical signal for an RCM candidate.

However, the paired probe requires constructing and probing the alternate RCM
state even when Natural is ultimately retained. `venkat25` demonstrates a real
selection-overhead regression, and `cfd1` demonstrates right-hand-side
sensitivity after the matrix-level decision.

Therefore F6 retains the progress, paired-probe, and policy-replay harnesses as
development evidence but does not change the production ordering/default
policy. A later explicit solve-many ordering policy may reuse these signals if
the caller can provide an expected reuse horizon or another cost-aware trigger.

## F7 Jacobi -> ILU(0) promotion evidence

F7 tested whether GeneralSquare can safely promote from the default Jacobi
preconditioner to canonical Natural-order ILU(0) automatically.

The answer for the current 0.8 development evidence is **no**: ILU(0) is
extremely valuable on several matrices, but no cheap signal tested in F7
separated those wins from the measured regressions without important false
positives or false negatives.

### F7a: complete prepared Jacobi-versus-ILU(0) solves

F7a charged analysis, preparation, and solve wall time for both prepared
preconditioners on deterministic right-hand sides.

Representative RHS1 results at restart 30 and tolerance `1e-8`:

| Matrix | Jacobi result | ILU(0) result | ILU/Jacobi total-time observation |
| --- | --- | --- | --- |
| `sherman5` | converged, 357 iterations | converged, 30 iterations | about `0.119` |
| `raefsky3` | converged, 3285 iterations | converged, 51 iterations | about `0.053` |
| `venkat25` | max 5000 | converged, 190 iterations | about `0.055` |
| `cfd1` | max 5000 | converged, 3363 iterations | about `0.900` |
| `thermal1` | max 5000 | converged, 3155 iterations | about `0.793` |
| `nd3k` | max 5000 | max 5000 | about `2.52` |
| `cant` | max 5000 | max 5000 | about `1.88` |
| `s3dkq4m2` | max 5000 | max 5000 | about `1.62` |
| `boneS01` | max 5000 | max 5000 | about `1.63` |
| `x104` | max 5000 | max 5000 | about `2.39` |

The same canonical ILU(0) implementation therefore ranges from a decisive
robustness/performance improvement to a large setup/apply penalty with worse
convergence.

Simple row density cannot classify these cases. For example `raefsky3` has
roughly 70 nonzeros per row and strongly benefits from ILU(0), whereas `cant`
and `x104` are of comparable row density and regress.

### F7b: one-apply approximate-inverse probes

F7b measured static structural/cost signals and a deterministic one-apply
quality probe

```text
defect(M) = ||M^-1 A x - x|| / ||x||
```

for Jacobi and ILU(0).

This signal is useful as a catastrophic-factor-quality diagnostic but is not a
positive promotion selector:

- `raefsky3` and `venkat25` strongly benefit from ILU(0) in complete FGMRES
  despite median ILU/Jacobi defect ratios greater than one.
- `boneS01` has a favorable median defect ratio below one yet remains slower
  with ILU(0) and fails to reach tolerance in the complete run.
- `nd3k`, `cant`, `s3dkq4m2`, and `x104` show strong negative signals, including
  very large defect ratios or ILU work/setup cost.

Accordingly F7b can veto obviously pathological factors in experiments, but no
production threshold is assigned from the current corpus.

### F7c: short paired Krylov probes

F7c compared actual Jacobi-FGMRES and ILU(0)-FGMRES progress from the same
`x=0` state after 4, 8, and 16 iterations.

The paired Krylov signal is more informative than the one-apply defect, but it
still is not sufficient for promotion:

- four iterations are too short: `raefsky3` initially gives an ILU/Jacobi
  residual ratio above one even though ILU(0) is a major complete-solve win;
- `venkat25`, `cfd1`, and `thermal1` show clear early ILU residual advantages;
- `nd3k`, `s3dkq4m2`, and `x104` are rejected clearly;
- `cant` and `boneS01` are false positives: ILU(0) has a smaller early residual
  while the complete wall time is substantially worse.

### F7d: restart-boundary replay

F7d replayed the complete prepared comparison with iteration budgets of 30, 60,
120, and 240, corresponding to one, two, four, and eight restart-30 cycles.

Restart-boundary telemetry explains several F7c false positives but does not
produce a universal selector.

`cant` is the clearest reversal. At 30 iterations ILU(0) has the smaller
residual, but by 60 iterations Jacobi is already better; the ILU residual then
remains worse through 120 and 240 iterations.

`boneS01` is the opposite difficulty. ILU(0) keeps the smaller residual through
240 iterations, but its higher setup and per-iteration cost keeps total wall
time substantially above Jacobi. At 240 iterations the ILU/Jacobi total-time
ratio is still about `1.81`.

`raefsky3` and `venkat25` show why a simple early cost-normalized rule is also
unsafe. Their ILU setup/apply cost can make short truncated comparisons look
unfavorable even though ILU(0) later reaches the requested tolerance far before
Jacobi, or succeeds when Jacobi reaches the 5000-iteration cap.

The convergence rate is therefore not stationary enough across restarts to
project time-to-tolerance safely from a small fixed observation window.

### F7 conclusion

HyBIT 0.8 keeps GeneralSquare Jacobi as the default and canonical ILU(0) as an
explicit opt-in policy.

F7 does **not** add automatic Jacobi -> ILU(0) promotion.

This is a deliberate robustness decision rather than a statement that Jacobi is
generally better. ILU(0) is dramatically superior on part of the measured
corpus and can turn a 5000-iteration Jacobi failure into convergence. The
problem is safe automatic classification: the tested static, factor-quality,
short-Krylov, and restart-boundary signals all leave meaningful counterexamples.

The retained F7a/F7b/F7c harnesses are development evidence for future
cost-aware or application-informed policy work. A future automatic policy
should require broader held-out validation and may need caller-provided solve
horizon, memory budget, or an explicit willingness to pay for exploratory
factorization/probing.
