# HyBIT development status

Last updated: 2026-10-06

## Published release

- Current published release: HyBIT 0.7.0.
- Release tag: `v0.7.0`.
- Immutable source:
  `1fdcd6a1b8127c84306c38c3fdbad42563538ad8`.
- Six Rust crates are published as 0.7.0 on crates.io.
- `hybit-ffi` remains repository-only.
- 0.7 production numerical freeze point: r32.

The `v0.7.0` tag is immutable. 0.8 development does not change the published
0.7 source.

## Current develop/0.8.0 checkpoint

Current validated development sequence is complete through G6. The F2 base
commit is `782ac38`; F3 adds prepared multi-RHS Natural/RCM ILU(0) reuse
measurement and documentation on top of that checkpoint.

### Execution architecture

Completed CPU checkpoints include common problem/execution/backend policy,
resident Krylov abstraction, serial resident PCG, Rayon vector execution,
parallel CSR SpMV, parallel resident Jacobi, and structural resident
cross-check.

GPU is an architectural target, not a production target here.

### GeneralSquare E1-F7

1. E1 — restarted FGMRES with reusable `V`/`Z` workspace.
2. E2 — prepared `GeneralSquare` routing.
3. E4a — staged restart escalation.
4. E4b — budget-aware restart controller.
5. E5 — prepared canonical ILU(0).
6. F1 — Natural/RCM ILU(0) ordering sensitivity.
7. F2 — real nonsymmetric corpus preflight and larger-case evidence.
8. F3 — repeated prepared multi-RHS ordering/reuse and end-to-end amortization.
9. F4 — ILU(0) application/topology profiling; per-level Rayon triangular apply rejected.
10. F5 — explicit unsuitable-ILU fallback with retained Identity safety path and real missing-diagonal validation.
11. F6 — Natural/RCM ordering-selection signals, paired short probes, and amortized policy replay; no automatic production promotion.
12. F7 — Jacobi -> ILU(0) promotion study; strong ILU wins and strong regressions observed, with no validated automatic production selector.

Jacobi + fixed restart 30 remains default.

E5 ILU(0) canonicalizes unsorted/duplicate CSR, introduces no fill, stores
diagonal positions as `u32`, uses a selective `1e-12` row-relative factor-pivot
floor, and is reused across solve-many.

The public hard-B cross-check reproduced 276 iterations, true relative residual
`9.455610e-9`, 4.238 MiB ILU state, 4.500 MiB FGMRES(3) workspace, 8.738 MiB
total persistent state, and zero adjusted pivots.

F1-F4 ordering/kernel work now shows three distinct real-nonsymmetric regimes:
`sherman5` with no measured RCM convergence benefit, `raefsky3` with robust
cross-RHS RCM benefit, and `venkat25` with RHS-dependent convergence changes
and only a small repeated 5-RHS end-to-end timing difference. Bandwidth and a
single RHS are therefore insufficient automatic-reordering signals.

These are regression measurements, not universal performance claims.

### Current routing limitations

- `Spd` -> PCG/Hybrid/structural.
- `GeneralSquare` -> FGMRES on `Auto`/`Cpu`.
- `SymmetricIndefinite` -> recognized; MINRES not implemented.
- no rectangular LSQR/LSMR;
- no complex arithmetic;
- ILU(0) triangular solve is serial;
- C ABI reports FGMRES/ILU0 codes but does not yet expose GeneralSquare
  configuration setters.

## Published 0.7 validation

The exact 0.7 release source passed source-integrity, metadata, formatting,
Clippy, workspace release tests, Rust 1.73 MSRV, C/C++/Fortran ABI/runtime,
package/dry-run, and physical L-angle gates.

The L-angle release regression used 358065 free DOFs and 28239653 CSR nonzeros,
a 1398-dimensional rigid-body coarse space, 220 iterations, and independently
verified relative residual `9.378557e-9`.

## Next validation focus

GeneralSquare robustness is now closed through F7 for the current 0.8
development checkpoint.

Production semantics remain explicit:

- Jacobi is the GeneralSquare default;
- canonical ILU(0) remains opt-in;
- `Ilu0Fallback` remains the explicit missing-diagonal safety policy;
- Natural/RCM ordering remains explicit rather than automatically selected.

ABTM G1-G6 are validated through topology algebra, metadata-first pruning,
region/local extraction, explicit ABTM ILU(0), fixed-size 3x3/6x6 block
execution, and explicit prepared restricted/local CSR operators. G5 retains an
explicit DenseBlockCsrOperator path without automatic CSR/B3/B6 routing. G6
retains ordinary CSR for scalar SpMV, adds explicit prepared A*M and R*A*R
operators, and adds no automatic CSR/ABTM preparation selector. The next
checkpoint is G7 Rayon prepared execution. GeneralSquare automatic-selection research can be
revisited later with a broader held-out corpus or application-provided
solve-horizon/cost information.

Do not add ILUT, fill, pivoting, or hidden automatic preconditioner changes as
part of the F7 conclusion.

### ABTM G1

G1 validates a numerical-value-independent sparse-of-64-bitmaps topology layer
with AND/OR/AND-NOT/XOR, popcount, rank/select, invariants, occupancy telemetry,
and general Boolean merge semantics.

The ten-matrix corpus shows that bitmap topology metadata is strongly compact
on several matrices but not universal: low-occupancy cases can be break-even or
worse than CSR metadata. Logical topology is therefore accepted while physical
prepared layout selection remains adaptive and workload-specific.

Word-local rank/select is the intended packed-value addressing primitive.
Row-wide rank/select remains a convenience/correctness API rather than a hot
numeric-kernel path.

G2-G6 are now complete. Next: G7 Rayon prepared execution.

### ABTM G2a

G2a adds a scalar reference prepared value stream aligned with the G1 logical
topology and a sparse-row-dot kernel that intersects topology with a
`DofMask` before loading numerical matrix values.

The first benchmark varies active support from 100% to 10% and records exact
candidate/executed/skipped products, empty-word ratio, numerical agreement, and
scalar timing versus unpruned CSR. This is an evidence checkpoint only; no
production solver/backend routing changes.

### ABTM G2c

G2c adds an experimental dual structural topology with row and column
orientations but no duplicated numerical values. The benchmark compares
bitmap-word row/column support intersection with an explicit sorted-index
baseline on deterministic sparse-dot pairs.

This checkpoint is intended to decide the metadata representation for later
matrix-product, ILU symbolic-intersection, and coarse/Galerkin experiments.

### ABTM G2d

G2d extends the dual-topology experiment to numerical row-by-column sparse dot
products. It compares an explicit CSR/CSC-like baseline with two bitmap
variants: one numerical copy plus a column source-index map, and duplicated
column values.

The goal is to determine whether dual structural metadata can remain
value-single-copy in practice or whether hot transpose-oriented kernels require
a second packed numerical stream.

### ABTM G2e

G2e evaluates numerical row/column products with the existing adaptive
Sparse/Bitmap/Dense tile classification in both orientations. Sparse tile pairs
avoid per-product rank, dense tiles use direct offsets, and bitmap tiles retain
mask/rank addressing.

The purpose is to determine whether the G2d regressions on low/intermediate
occupancy are a fixed-bitmap execution artifact before changing thresholds or
promoting a numerical backend.

### ABTM G2f

G2e confirms that adaptive Sparse/Bitmap/Dense execution is the correct
direction, but also exposes that the current fixed 16-byte tile descriptor is
not a compact Sparse physical encoding.

G2f benchmarks a packed variable-payload representation in which Sparse tiles
store byte offsets and Bitmap/Dense tiles store 64-bit masks. This is intended
to test the physical metadata model before threshold sweeps or production
backend promotion.

### ABTM G2g

G2g replaces the G2f byte-oriented payload decoder with typed compact streams
and an 8-byte descriptor while retaining adaptive Sparse/Bitmap/Dense execution.
It tests whether most of G2f's storage reduction can be retained without losing
the G2e numerical performance characteristics.

### ABTM G2 closed

G2 metadata-first product pruning is validated and closed. The final CPU
candidate is the G2e-style adaptive Sparse/Bitmap/Dense numerical execution
with dual row/column structural preparation and explicit fallback.

The ten-matrix G2e corpus contains both clear ABTM wins and clear explicit wins,
so no universal replacement or production selector threshold is promoted.
The complete evidence is in `ABTM_G2_CLOSEOUT.md`.

Next checkpoint: G3 region growth, overlap/multiplicity, and local submatrix
extraction.

### ABTM G3a region growth

G3 is active. G3a adds deterministic `k`-hop region growth over
`support(A) union support(A^T)` using the dual ABTM topology and an exact
CSR-plus-transpose reference cross-check.

The checkpoint measures frontier nodes, topology words, candidate neighbor
bits, explicit neighbor entries, preparation cost, and scalar execution time.
Overlap/multiplicity and local submatrix extraction remain subsequent G3 work.

### ABTM G3b overlap and multiplicity

G3b adds exact per-node multiplicity over G3a-grown regions, overlap-node
statistics, pairwise overlap accounting, and an end-to-end
growth-plus-multiplicity benchmark. Region equality is still verified against
CSR + transpose-CSR before overlap metrics are accepted.

G3 local submatrix extraction remains the next step after the overlap corpus is
validated.

### ABTM G3c structural local-submatrix extraction

G3c adds exact structural extraction of `A[R,R]` from a region mask and dual
ABTM topology. Extracted global-node lists, row pointers, and local column
indices are cross-checked against direct CSR extraction for every benchmark
region.

Numerical value gathering is intentionally deferred until structural pruning,
mapping scratch cost, and extraction performance are measured.

### ABTM G3d prepared local numeric refresh

G3d evaluates symbolic/numerical separation for local submatrices. A prepared
plan binds the G3c local structural pattern to source CSR value positions and
supports repeated numerical refresh without rebuilding the region or local
pattern.

The benchmark reports one-time plan preparation, refresh-only time, direct
full-extraction time, plan memory, numerical agreement under changed values,
and the measured refresh count needed to amortize preparation.

## ABTM G3 closeout

G3 region operations are validated and closed for this 0.8 checkpoint.

- G3a: deterministic undirected structural region growth;
- G3b: overlap and multiplicity with pair-overlap identity checks;
- G3c: exact structural `A[R,R]` extraction;
- G3d: prepared local numerical refresh under fixed topology.

The primary architectural result is a separation of concerns: ABTM is strongest
as a reusable symbolic/topology layer, while materialized local sparse/numeric
storage should remain evidence-driven. One-shot local pattern extraction did not
show a universal speed advantage, whereas repeated prepared value refresh
amortized its plan in roughly 4--8 refreshes on the six-matrix corpus when the
dual topology was already available.

`nd3k` remains a deliberate high-overlap memory stress case and should not drive
an unbounded all-regions-at-once policy.

Next work is G4 ABTM symbolic/numeric ILU(0) comparison against canonical CSR
ILU(0).

### ABTM G4a ILU(0) symbolic intersection

G4 is active. G4a compares canonical CSR ILU(0) symbolic target lookup against
prepared ABTM row-word intersection while preserving identical canonical
structure.

The first gate requires exact per-pivot target-list agreement. Numerical
factorization, pivot handling, triangular apply, and solver residual behavior
remain unchanged until symbolic evidence is established.

### ABTM G4b ILU(0) numeric intersection

G4b evaluates numeric ILU(0) updates driven directly by ABTM word
intersections. A per-word structural prefix maps each intersection bit to the
existing canonical CSR factor position, retaining conventional numerical
storage and avoiding per-product plan memory.

The gate requires factor and triangular-apply agreement against the canonical
CSR reference before timing. Production routing remains unchanged.

### ABTM G4c ILU(0) rank-LUT addressing

G4c isolates the G4b numerical-addressing overhead by replacing per-update
rank-by-popcount with an all-word `u8[64]` rank table. CSR, G4b popcount, and
G4c LUT factors are cross-validated before timing.

The experiment reports LUT preparation time and bytes separately. No production
selection rule is introduced at this stage.

### ABTM G4d adaptive rank-LUT sweep

G4d follows the successful G4c all-word LUT experiment with an
occupancy-threshold sweep. The goal is to determine whether most of the
numerical speedup can be retained with substantially less LUT memory.

Every threshold is cross-checked against canonical CSR factors and triangular
apply before timing. Production ILU(0) remains unchanged.

### ABTM G4f explicit production ILU(0)

G4f promotes the validated direct rank-LUT factorization algorithm into an
explicit production constructor and GeneralSquare policy. The factor object
retains only the canonical CSR factor arrays; ABTM topology and rank LUTs are
temporary preparation metadata.

Automatic matrix-level routing is intentionally deferred. G4e held-out results
are encouraging, but selector cost and a cheap production predictor still need
separate validation.

### ABTM G4 closeout

G4 is closed. The explicit production `Ilu0Abtm` path preserves canonical CSR
factor storage and triangular application while using temporary ABTM topology
and direct word-rank LUTs during factor construction.

The eleven-matrix production constructor corpus was exact on every supported
case and split 6 ABTM wins / 5 CSR wins. Strong wins include `nd3k` (~5.28x),
`x104` (~2.34x), `cant` (~2.33x), and `raefsky3` (~2.06x).

The production corpus also showed that the earlier permissive symbolic selector
must not be promoted: `venkat25` and `cfd1` have favorable symbolic compression
but lose after full ABTM preparation cost. Automatic matrix-level routing is
therefore deferred. G5 block-ABTM is next.

### ABTM G2-G5 closeout

G2 validates metadata-first support intersection/pruning and establishes that
logical row/column topology can drive candidate elimination before numerical
value access.

G3 validates topology-driven region growth, overlap/multiplicity, local
structural extraction, and repeated local numeric refresh. The architectural
boundary remains deliberate: ABTM discovers/prunes reusable structure while
conventional prepared local numerical layouts remain appropriate downstream.

G4 validates ABTM symbolic ILU(0) intersection and exposes explicit
Ilu0Preconditioner::from_csr32_general_abtm /
GeneralSquarePreconditionerPolicy::Ilu0Abtm. The returned factor remains the
canonical CSR ILU(0) representation; ABTM preparation metadata is scratch.
Automatic ILU route selection remains deferred.

G5 validates node-oriented dense block execution only where structure supports
it. Generic block traversal was rejected. Fixed-size 3x3/6x6 tail-specialized
kernels produced substantial CSR speedups on high-fill development cases, while
low-fill matrices remained better on CSR. A frozen held-out selector classified
five matrices correctly with zero false-positive block routes; inline_1
provided a positive B3 held-out case at about 1.78x CSR kernel speed and about
0.70x CSR storage.

The production G5 result is therefore explicit rather than automatic:
DenseBlockCsrOperator with DenseBlockSize::B3 / B6 is public and implements
LinearOperator, but existing scalar CSR and solver defaults are unchanged.
Positive held-out evidence for B6 is still missing, so automatic CSR/B3/B6
routing is not promoted.

Next: G6 ordinary and masked/restricted SpMV.