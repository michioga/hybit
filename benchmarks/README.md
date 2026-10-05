# HyBIT 0.7 FEM benchmark input

`fem_bench` accepts a real or integer Matrix Market coordinate matrix (`.mtx`).
The current solver path expects the constrained linear system to be real SPD.
For symmetric matrices, prefer the Matrix Market `symmetric` header and store one
triangle only; the loader expands it to full CSR32 storage.

## Repository benchmark layout

PowerShell benchmark wrappers live under `benchmarks/scripts/` and are intended
to be invoked from the repository root. They also resolve the repository root
from `$PSScriptRoot`, so invoking them from another working directory is
supported while caller-relative matrix/RHS paths remain meaningful.

Generated comparison CSV files default to `benchmarks/results/`. That directory
is intentionally ignored by Git; benchmark measurements are local artifacts
unless a specific result is deliberately promoted into documented regression
evidence.

Example:

```powershell
.\benchmarks\scripts\bench-fem.ps1 benchmarks\data\poisson5.mtx
```
Run the bundled smoke matrix:

```powershell
cargo run --release -p hybit --example fem_bench -- --matrix benchmarks/data/poisson5.mtx
```

Run a real assembled FEM stiffness matrix:

```powershell
cargo run --release -p hybit --example fem_bench -- `
  --matrix D:\path\to\K.mtx `
  --tol 1e-8 `
  --max-iters 3000 `
  --overlap 1 `
  --max-region 128 `
  --max-regions 8
```

If `--rhs` is omitted, the benchmark constructs `x_exact = 1` and `b = A*x_exact`.
This isolates the linear solver and gives an independent solution-error check.
To use a physical load vector, pass a whitespace-separated vector with `--rhs`.

The benchmark prints both the solver-reported residual and an independently
recomputed `||Ax-b||/||b||`, plus setup, solve, hard-region and local-factor
metrics. Compare wall-clock results only on sufficiently large problems and use
multiple process runs for performance claims.

## GeneralSquare Natural/RCM ordering benchmark

`general_square_ordering` is an F1 development harness for measuring ILU(0)
ordering sensitivity without changing the production solver policy. The wrapper
runs Natural and deterministic RCM orderings with both Jacobi and ILU(0). RCM
uses the undirected sparsity graph of `A + A^T` and applies the simultaneous
row/column permutation `P A P^T`.

Example:

```powershell
.\benchmarks\scripts\bench-general-square-ordering.ps1 `
  -Matrix D:\Work\raefsky3.mtx `
  -MaxIterations 5000
```

The benchmark reports structural bandwidth, ordering cost, analysis/prepare
time, FGMRES iterations, solve wall time, ILU adjusted-pivot count, persistent
preconditioner/workspace bytes, and an independently recomputed residual after
mapping the RCM solution back to the original ordering.

The current high-level GeneralSquare path requires a structurally complete
diagonal. Matrices with missing diagonal entries are rejected rather than
silently modified for this benchmark.

## GeneralSquare corpus benchmark

`bench-general-square-corpus.ps1` runs the ordering harness across multiple
Matrix Market inputs and writes CSV output. `-PreflightOnly` screens eligibility
before expensive solves; missing/zero diagonal cases are recorded as skipped
instead of aborting the corpus.

```powershell
.\benchmarks\scripts\bench-general-square-corpus.ps1 `
  -Matrix @("D:\Work\sherman5.mtx","D:\Work\raefsky3.mtx","D:\Work\venkat25.mtx") `
  -PreflightOnly
```

Full example:

```powershell
.\benchmarks\scripts\bench-general-square-corpus.ps1 `
  -Matrix "D:\Work\venkat25.mtx" `
  -MaxIterations 5000
```

Generated CSV files default to ignored `benchmarks/results/`.

## GeneralSquare prepared multi-RHS benchmark

`bench-general-square-multi-rhs.ps1` measures prepared Natural and RCM ILU(0)
reuse across distinct deterministic right-hand sides.

```powershell
.\benchmarks\scripts\bench-general-square-multi-rhs.ps1 `
  -Matrix @(
    "D:\Work\raefsky3.mtx",
    "D:\Work\venkat25.mtx"
  ) `
  -RhsCount 5 `
  -MaxIterations 5000 `
  -Restart 30
```

The example prepares each ordering once, reuses the prepared ILU(0) and FGMRES
workspace, checks `solve_sequence` / `preconditioner_reused`, and independently
verifies the residual in the original ordering.

Machine-readable records distinguish:

- `REUSE_RESULT` for each RHS and ordering;
- `AMORTIZED_SOLVE_ONLY` for setup plus timed solver work;
- `AMORTIZED_END_TO_END` for setup, solver work, RHS permutation, and solution
  unpermutation;
- `BREAK_EVEN_SOLVE_ONLY` and `BREAK_EVEN_END_TO_END`.

Use repeated process runs before interpreting small timing differences. F3 used
five repeats for the documented `raefsky3` and `venkat25` evidence.
## GeneralSquare ILU(0) triangular-apply profiling

`bench-general-square-ilu-apply.ps1` measures canonical serial ILU(0)
triangular application separately from serial CSR SpMV for Natural and RCM
orderings. `bench-general-square-ilu-levels.ps1` profiles the same canonical
pattern's forward/backward dependency levels, widths, work imbalance, and
dependency distances.

```powershell
.\benchmarks\scripts\bench-general-square-ilu-apply.ps1 `
  -Matrix @(
    "D:\Work\sherman5.mtx",
    "D:\Work\raefsky3.mtx",
    "D:\Work\venkat25.mtx"
  ) `
  -Samples 9 `
  -Batch 50 `
  -Warmup 5

.\benchmarks\scripts\bench-general-square-ilu-levels.ps1 `
  -Matrix @(
    "D:\Work\sherman5.mtx",
    "D:\Work\raefsky3.mtx",
    "D:\Work\venkat25.mtx"
  )
```

These are diagnostic benchmarks. F4 also tested per-level and width-threshold
Rayon triangular schedules; those prototypes were slower whenever Rayon work
was actually dispatched, so only the serial/dependency profiling harnesses are
retained.
## Structural Graph coarse-dimension sweep

After a structural Matrix Market matrix, free-node coordinate sidecar, and optional
physical RHS have been exported, compare several dense coarse-space budgets while
holding Graph aggregation and the Krylov tolerance fixed:

```powershell
.\benchmarks\scripts\bench-fem-structural-coarse-sweep.ps1 `
  D:\Work\mf_solver-hybit-export\L-angle-K.mtx `
  -Coordinates D:\Work\mf_solver-hybit-export\L-angle-K.coords `
  -Rhs D:\Work\mf_rhs\L-angle-b.txt `
  -TargetCoarseDimensions 384,768,1536,3072 `
  -Tolerance 1e-8 `
  -MaxIterations 3000
```

The sweep deliberately uses strict `graph` aggregation by default. Compare the
actual coarse dimension, prepare time, iteration count, solve time, and total
setup+solve time. The target is a soft budget; power-of-two aggregate sizing and
graph remainder merging mean the actual coarse dimension can differ slightly.


## Structural r25 development checkpoint

Historical 0.6 r25 structural baseline used `target_coarse_dimension=1536` for the development L-angle case. With 8 Rayon workers, Structural Auto selected Graph aggregation, Parallel CSR SpMV, the Parallel rigid-body preconditioner, and Parallel/fused PCG vectors. The 358065-DOF / 28239653-nnz physical-load case used 233 aggregates, coarse dimension 1398, converged in 220 iterations, independently verified relative residual `9.378557e-9`, and measured 1.726 s solve / 2.797 s analysis+prepare+solve on the Ryzen 7 7800X3D development machine.

Treat these numbers as a regression reference for this matrix, RHS, machine, and revision. They are not a general performance guarantee. This r25 result remains the structural regression reference retained by the 0.7 release line.

## Hybrid local-factor selector A/B benchmark

The 0.7 development line can compare the diagnostic candidate order, raw
residual-energy-per-byte, and Jacobi-energy-per-byte selectors under the same
persistent local-factor memory budget.

```powershell
.\benchmarks\scripts\bench-fem-hybrid-selection.ps1 `
  -Matrix D:\Work\mf_solver-hybit-export\L-angle-K.mtx `
  -Rhs D:\Work\mf_rhs\L-angle-b.txt `
  -FactorBudgetMiB 64 `
  -Tolerance 1e-8 `
  -MaxIterations 3000 `
  -MaxEscalations 3 `
  -StageIterations 24
```

The script runs `candidate-order`, `benefit-byte`, and `jacobi-byte` in that
order with otherwise identical solver settings. Compare verified residual,
iterations, escalation count, selected local-region count, persistent factor
memory, budget-skipped regions, local-factor time, solver time, and total wall
time. Use repeated process runs before drawing performance conclusions.
## Hybrid coarse-apply crossover

`bench-fem-hybrid-coarse-apply-crossover.ps1` compares the packed triangular `FactorSolve` and Rayon-parallel `ExplicitInverse` coarse-apply paths across multiple coarse targets. It alternates target and policy order across repeats, reports median wall/solver/setup costs and an approximate break-even iteration count, and writes per-run plus comparison CSV files. The default targets are 768, 1280, 1792, and 2048.

## Hybrid coarse-apply Auto policy

`TwoLevelCoarseApplyPolicy::Auto` resolves from the actual coarse dimension after
aggregation. The current empirical threshold is 1024: smaller coarse systems use
packed `FactorSolve`, while dimensions of 1024 or larger use the Rayon-parallel
`ExplicitInverse` path. This threshold is based on the repeated L-angle crossover
benchmark and remains overrideable with `--coarse-apply factor` or
`--coarse-apply inverse`.


## Hybrid smoothed coarse-basis A/B

`bench-fem-hybrid-coarse-basis.ps1` holds Graph aggregation, the coarse target,
coarse-apply policy, and local-direct settings fixed while alternating the
original piecewise-constant tentative basis with a one-step Jacobi-smoothed
basis. The smoothed path uses a deterministic spectral-radius estimate for its
damping and forms the true Galerkin operator `P^T A P`. Compare iteration count,
coarse setup/memory, solver milliseconds per iteration, solver time, and total
wall time before changing the generic default.

## GeneralSquare missing-diagonal fallback validation

F5 retains two benchmark wrappers for the production `Ilu0Fallback` policy.

`bench-general-square-ilu-fallback.ps1` is a prepare-only preflight. It compares
strict `Ilu0` with explicit `Ilu0Fallback` and reports the effective prepared
preconditioner and whether structural fallback was used.

`bench-general-square-ilu-fallback-solve.ps1` performs a bounded FGMRES solve
using a deterministic manufactured right-hand side, reports independently
verified residual and forward error, and is intended to distinguish a safe
fallback from a numerically strong one.

Example:

```powershell
.\benchmarks\scripts\bench-general-square-ilu-fallback-solve.ps1 `
  -Matrix @(
    "D:\Work\Goodwin_010.mtx",
    "D:\Work\Goodwin_023.mtx",
    "D:\Work\Goodwin_030.mtx",
    "D:\Work\goodwin.mtx",
    "D:\Work\rma10.mtx"
  ) `
  -Restart 30 `
  -MaxIterations 300 `
  -Tolerance 1e-8
```

The stronger F5c-F5f missing-diagonal experiments were intentionally not
retained after they failed to provide a robust universal improvement over the
Identity safety fallback.

## GeneralSquare ordering-selection probes

F6 retains three benchmark-only harnesses. They do not change the production
GeneralSquare ordering policy.

`bench-general-square-ordering-progress.ps1` records exact Natural-ILU FGMRES
restart-boundary residuals. F6a showed that Natural-only progress is not enough
to decide whether RCM will help.

`bench-general-square-ordering-paired-probe.ps1` prepares Natural and RCM ILU(0)
states, runs equal-length 4-, 8-, and 16-iteration probes from `x=0`, and then
runs the complete solves. Its main signal is the RCM/Natural residual ratio
after the same probe length.

`bench-general-square-ordering-policy-replay.ps1` repeats the paired experiment
and charges both setup paths and both short probes before reusing the selected
ordering across a configurable RHS horizon.

The current development evidence favors a cheap structural prefilter
(`RCM bandwidth < Natural bandwidth`) followed by a four-iteration paired
residual comparison as a solve-many research signal. This is not a production
automatic-selection rule: alternate-state setup cost and right-hand-side
sensitivity remain material.

## GeneralSquare preconditioner-selection studies

F7 retains three benchmark-only preconditioner-selection harnesses. They do not
change the production GeneralSquare default.

`bench-general-square-preconditioner-compare.ps1` performs complete prepared
Jacobi-versus-ILU(0) solves, charges setup and solve time, reuses prepared
preconditioners across deterministic RHS vectors, and reports cumulative
break-even.

`bench-general-square-preconditioner-probe.ps1` measures static matrix/cost
signals plus deterministic one-apply approximate-inverse defects. It is useful
for diagnosing catastrophic ILU factors but is not a validated promotion rule.

`bench-general-square-preconditioner-paired-probe.ps1` compares actual
Jacobi-FGMRES and ILU(0)-FGMRES exact residuals after equal 4-, 8-, and
16-iteration horizons from `x=0`.

F7 also replayed the complete comparison at restart-boundary budgets 30, 60,
120, and 240. Those results show that early residual advantage, early wall
advantage, and eventual time-to-tolerance can disagree.

The F7 conclusion is intentionally conservative: retain explicit policies and
do not infer a production automatic threshold from this corpus.

## ABTM G1 topology benchmarks

`bench-abtm-topology-g1.ps1` validates the scalar logical topology, full
rank/select invariants, and Boolean self-identities on real matrices.

`bench-abtm-topology-g1b.ps1` adds:

- direct topology-versus-CSR metadata accounting;
- non-empty-word occupancy distribution;
- chunk-local word rank/select timing;
- two deterministic partially overlapping structural subsets;
- AND / OR / AND-NOT / XOR timing through the general merge path.

The ten-matrix G1 corpus shows three distinct occupancy regimes:

- low occupancy, where bitmap metadata is near or worse than CSR
  (`sherman5`, `thermal1`);
- intermediate occupancy, where bitmap metadata is already smaller but many
  words remain candidates for a sparse physical form (`venkat25`, `cfd1`,
  `boneS01`);
- high occupancy, where bitmap topology is strongly compact
  (`raefsky3`, `nd3k`, `cant`, `s3dkq4m2`, `x104`).

These benchmarks validate logical topology semantics. They do not by themselves
select a production physical layout or backend.

## ABTM G2 metadata-first sparse-dot pruning

`bench-abtm-metadata-pruning-g2.ps1` is the first G2 scalar reference
benchmark. It prepares a G1 topology plus a numerical value stream in topology
order and evaluates matrix-row dot products against deterministic sparse-vector
supports.

For each active-support density it reports:

- candidate structural products before pruning;
- products that survive bitmap support intersection;
- skipped products and pruning ratio;
- topology words visited and empty-word ratio;
- independently checked numerical error;
- median unpruned CSR and metadata-first scalar wall time.

The benchmark uses active supports of 100%, 75%, 50%, 25%, and 10%. The
100%-support case is intentionally an overhead baseline; lower support densities
measure whether metadata pruning can recover that overhead by avoiding matrix
value loads and multiplications.

G2a is a scalar reference experiment, not a production SpMV backend. The
`AbtmMetadataFirstMatrix` type remains in `hybit-matrix` and is deliberately not
re-exported through the top-level `hybit` facade at this checkpoint.

## ABTM G2c dual-topology sparse-dot support intersection

`bench-abtm-dual-topology-g2c.ps1` prepares both structural orientations of one
matrix:

```text
A row topology
A column topology == row topology of A^T
```

without duplicating numerical values. The validation workload treats the same
square input as both operands of `A * A` and samples deterministic `(row, col)`
dot-product supports.

For each corpus matrix it compares:

- explicit sorted-index support intersection;
- 64-bit topology-word intersection;
- total row+column metadata bytes for both representations;
- metadata comparisons;
- bitmap mask AND operations;
- exact overlap-product count;
- structurally empty dot-product pairs;
- scalar wall time for explicit-index versus bitmap support intersection.

This is the first direct validation of the G2 matrix-versus-matrix metadata
model. Numerical sparse-dot value loading remains a later G2 checkpoint.

## ABTM G2d dual-topology numerical sparse dot

`bench-abtm-dual-numeric-g2d.ps1` extends G2c from structural support
intersection to actual numerical row-by-column sparse dot products.

It compares three scalar layouts on the same deterministic `(row, col)` pairs:

1. explicit sorted-index row/column merge with a duplicated column value stream;
2. bitmap dual topology with one numerical value copy and a `u32` column-to-row
   source-value map;
3. bitmap dual topology with a duplicated column value stream.

The benchmark reports both wall time and total estimated storage. The mapped
variant directly tests the design goal of duplicating structural metadata
without duplicating `f64` numerical values. The duplicated-value variant shows
the performance ceiling available if a later backend decides that contiguous
column values justify the extra memory.

G2d remains a benchmark experiment. It does not promote either numerical layout
into production solver routing.

## ABTM G2e adaptive dual-numeric sparse dot

`bench-abtm-adaptive-dual-numeric-g2e.ps1` reuses the existing
`AbtmMatrix` `Sparse/Bitmap/Dense` tile classification for both matrix
orientations and executes numerical row-by-column sparse dots.

The kernel selects its local strategy from the matched tile kinds:

- Sparse/Sparse: local set-bit merge with ordinal value streams, avoiding
  per-product rank;
- Sparse/Bitmap or Sparse/Dense: enumerate the sparse side and probe the other
  mask;
- Bitmap/Bitmap and Dense-involved pairs: intersect word masks and use compact
  rank or direct dense offsets as required.

Both row and column numerical streams are prepared, so this experiment follows
the G2d evidence that hot transpose-oriented numerical work can justify
duplicated values. It compares adaptive ABTM directly with explicit CSR/CSC-like
index/value storage and reports tile-kind populations, value-slot expansion,
storage ratio, numerical agreement, and wall time.

The default thresholds remain `Sparse <= 8` and `Dense >= 40`; G2e validates
the mechanism before any threshold sweep.

## ABTM G2f packed adaptive dual-numeric sparse dot

G2e validates occupancy-aware execution, but its existing `AbtmMatrix`
`TileDesc` remains 16 bytes for every tile even when the tile is classified
`Sparse`. That means the execution path is adaptive while the sparse physical
metadata is not yet genuinely compact.

`bench-abtm-packed-adaptive-dual-g2f.ps1` therefore evaluates a benchmark-local
packed adaptive representation:

```text
per row:
    tile pointer     u32
    payload pointer  u32
    value pointer    u32

per tile:
    meta             u32  (word index + kind + sparse count)

payload:
    Sparse           k x u8 offsets
    Bitmap           u64 mask
    Dense            u64 mask

values:
    Sparse/Bitmap    k packed f64 values
    Dense            64 f64 slots
```

The high bits of `meta` are available because a `u32` matrix column index needs
at most 26 bits after division by the 64-column word width.

This representation makes the Sparse case physically compact instead of merely
selecting a sparse execution branch. G2f keeps the same default thresholds
(`Sparse <= 8`, `Dense >= 40`) so the experiment isolates physical packing from
threshold tuning.

## ABTM G2g typed-compact adaptive dual-numeric sparse dot

G2f proves that physically compact sparse metadata can reduce storage, but its
variable byte-payload decoder is too expensive in the numerical hot path.

G2g keeps the same adaptive execution policy but replaces byte-stream decoding
with typed arrays and an 8-byte descriptor:

```text
CompactTile:
    meta u32  = word index + kind + sparse count
    aux  u32  = sparse-offset start OR mask index

typed side streams:
    sparse_offsets Vec<u8>
    masks          Vec<u64>
    values         Vec<f64>

per row:
    tile_ptr  u32
    value_ptr u32
```

This deliberately trades some of G2f's maximum compression for constant-time,
typed descriptor access without `from_le_bytes`, `Result`, or variable-payload
parsing in the hot row/column merge.

Thresholds remain `Sparse <= 8`, `Dense >= 40`; G2g isolates decoder/layout
cost from threshold tuning.

## ABTM G3a undirected region growth

G3a starts the region-operations checkpoint with deterministic `k`-hop growth
over the structural graph `A union A^T`.

The ABTM path uses `AbtmDualTopology`: for every frontier node it ORs the row
(outgoing) and column (incoming) topology words into the next frontier before
removing nodes already in the accumulated region. Numerical matrix values are
never read.

The reference path uses explicit CSR plus transpose-CSR adjacency. Every tested
seed region is cross-checked for exact set equality before timing.

Machine-readable records:

- `G3A_PREPARE`: transpose and dual-topology preparation;
- `G3A_REGION`: region size, frontier work, topology-word work, explicit
  neighbor-entry work, exact-match count, and median scalar timing.

The first corpus should use modest independent single-node seed regions so that
growth behavior is measured rather than immediately saturating a connected
matrix.

## ABTM G3b overlap and multiplicity

G3b consumes deterministic G3a regions and builds an exact node multiplicity
map:

```text
m(v) = number of regions containing node v
```

A node is overlapped when `m(v) >= 2`. The implementation also records the
total membership count, extra memberships beyond first coverage, maximum
multiplicity, and the pair-overlap identity

```text
sum over region pairs |Ri intersect Rj| = sum over nodes C(m(v), 2)
```

as an independent correctness invariant.

The benchmark cross-checks every ABTM-grown region against CSR + transpose-CSR,
cross-checks multiplicity counts against a straightforward reference, validates
the pair-overlap identity, and times the complete
growth-plus-multiplicity pipeline.

The default G3b corpus uses 64 deterministic single-node seeds and two hops.
Two hops are deep enough to expose meaningful overlap while avoiding the strong
three-hop saturation observed for `nd3k`.

## ABTM G3c structural local-submatrix extraction

G3c extracts the structural pattern of `A[R,R]` for each deterministic G3
region. Local numbering follows ascending global node order.

The ABTM path intersects each visited row topology word with the region mask
before enumerating retained columns. The CSR reference scans every stored entry
of each selected global row and probes the region map.

Both paths construct the same temporary dense `global -> local` map so the
timing comparison focuses on row-structure traversal and pruning rather than on
different local-numbering semantics.

Machine-readable records report candidate versus retained structure, topology
words versus CSR entries, output pattern storage, mapping scratch storage, and
median extraction time. G3c is structural only; numerical value gathering is a
separate subsequent checkpoint.

## ABTM G3d prepared local numeric refresh

G3c showed that converting an already-known local topology into CSR-like local
column indices is not itself a universal ABTM speed win: output materialization
dominates once the local pattern is large.

G3d therefore tests the more important reuse case. A local numeric plan is
prepared once from stable topology and binds each local structural entry to its
source CSR value position(s). Subsequent numerical refreshes gather only values;
they do not rebuild region topology, local numbering, row pointers, or local
column indices.

The plan supports duplicate CSR entries by storing grouped source positions and
summing them on refresh. Canonical CSR gets a direct-source fast path.

The benchmark validates both original and deterministically perturbed numerical
values, compares repeated prepared refresh against full direct local numeric
extraction, and reports the refresh count required to amortize plan preparation.

## ABTM G4a ILU(0) symbolic intersection

G4a starts the ILU(0) checkpoint without changing the production
preconditioner. It compares the symbolic update lookup used by canonical
CSR ILU(0) with a prepared ABTM word-intersection path.

The canonical CSR reference follows the current ILU(0) structure:

```text
for lower entry (i,j):
    for upper entry (j,k), k > j:
        binary-search k in row i after j
```

The ABTM path uses the same canonicalized ILU(0) pattern, seeks to the pivot
word, and intersects row-word masks:

```text
support(row i, columns > j) AND support(row j, columns > j)
```

Every pivot's exact target-column list is cross-checked before timing.

Important: raw `AbtmTopology` retains explicit structural zeros, while the
canonical ILU(0) implementation drops off-diagonal entries whose duplicate sum
is zero. G4a therefore canonicalizes with the same ILU(0) structural rule
before building topology. This prevents a false symbolic comparison.

The prepared word-row view is benchmark-local. Promotion to a production API is
deferred until corpus evidence justifies it.

## ABTM G4b ILU(0) numeric intersection

G4b carries the G4a symbolic intersection into numeric factorization without
materializing one update-pair record per successful ILU(0) product.

A prepared word-row view stores, for every nonempty topology word, the number
of structural entries preceding that word in the row. During a row-word
intersection, a set bit therefore maps directly to the canonical CSR numerical
position using:

```text
row_ptr[row]
+ word_nnz_prefix
+ popcount(word_mask below target_bit)
```

The same mapping is used for the pivot row. This keeps preparation memory
proportional to nonempty topology words rather than successful ILU(0) updates;
that distinction is essential for dense-overlap cases such as `nd3k`.

The CSR reference mirrors the current canonical ILU(0) numeric loop:
upper-row candidate traversal followed by binary search in the current row.
Both paths use the same canonicalized values, diagonal positions, row-relative
pivot floor, update order, and triangular-apply validation.

G4b remains benchmark-only; production ILU(0) is not changed.

## ABTM G4c ILU(0) rank-LUT addressing

G4b established exact numeric equivalence but exposed an addressing cost:
every successful topology-intersection bit required two `popcount` rank
computations to recover the row and pivot CSR positions.

G4c is a diagnostic experiment that precomputes one 64-byte `u8` rank table for
every nonempty topology word. An intersection bit can then map to each local
word ordinal with a direct byte lookup.

The benchmark compares three otherwise identical factorizations:

- canonical CSR candidate traversal + binary search;
- G4b ABTM intersection + rank-by-popcount;
- G4c ABTM intersection + rank LUT.

This is deliberately an **all-word** LUT experiment. It measures whether rank
mapping is a material bottleneck before introducing any adaptive occupancy
threshold. The memory cost is reported explicitly and is not automatically
eligible for production promotion.

## ABTM G4d adaptive rank-LUT sweep

G4c showed that rank-by-popcount was a major part of the G4b numerical
addressing cost. An all-word `u8[64]` rank LUT made the ABTM factorization
faster than CSR on five of the six-matrix corpus, but added 64 bytes for every
nonempty topology word.

G4d tests whether word occupancy is a usable speed/memory selector. A rank LUT
is retained only when:

```text
popcount(word_mask) >= threshold
```

Other words fall back to G4b rank-by-popcount. The default sweep is:

```text
1,2,4,6,8,12,16,24,32
```

For every threshold the benchmark reports selected-word fraction, LUT bytes,
fraction of dynamic rank operations served by LUTs, numeric factorization time,
and exact factor/apply validation.

This remains diagnostic. A universal production threshold is not selected
unless the six-matrix sweep supports one.

## ABTM G4f explicit production ILU(0)

G4e held-out validation supported matrix-level routing evidence, but automatic
selection remains deliberately unpromoted. G4f therefore adds an explicit
production constructor and public GeneralSquare policy only.

`Ilu0Preconditioner::from_csr32_general_abtm` uses the G4c direct all-word
rank-LUT factorization path. The topology/word/rank metadata is constructor
scratch and is dropped before return. Persistent factor storage and triangular
apply are identical to canonical CSR ILU(0).

`GeneralSquarePreconditionerPolicy::Ilu0Abtm` exposes this path explicitly.
`Jacobi` remains the default; `Ilu0` and `Ilu0Fallback` retain their existing
behavior.
