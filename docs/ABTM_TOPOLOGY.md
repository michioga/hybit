# ABTM topology algebra and metadata-first execution

> Development architecture direction for `develop/0.8.0`. This document
> describes a planned execution model; it does not claim that ABTM is already
> the production default or universally faster than CSR.

## Purpose

HyBIT began from the ABTM idea: encode sparse topology with bitmap-oriented
metadata while keeping numerical values in compact streams. The current solver
has since gained prepared execution, hybrid/local correction, algebraic coarse
spaces, resident CPU execution, FGMRES, ILU(0), and ordering experiments.

That broader solver architecture changes how ABTM should be positioned.

ABTM is not treated as a universal replacement for CSR. Its primary role is a
**symbolic/topology representation** from which CPU, GPU, block, local, and
distributed prepared layouts can be derived.

The central execution principle is:

> **Metadata-first numerical execution:** determine the set of useful numerical
> operations from topology metadata before loading and combining numerical
> values.

This separates structural algebra from numerical algebra.

## Structural and numerical layers

The logical model is:

```text
Sparse topology              Numerical values
      |                            |
      |-- adjacency                |
      |-- masks                    |
      |-- permutation              |
      |-- regions                  |
      |-- halo                     |
      |                            |
      +-------------+--------------+
                    |
                    v
              prepared layout
                    |
       +------------+-------------+
       |            |             |
     CSR32       CPU ABTM      GPU ABTM
```

`SparseTopology` is the reusable structural object. A concrete prepared backend
may use CSR32, bitmap chunks, block-ABTM, or another representation without
changing the mathematical problem class.

Ordering belongs before layout preparation:

```text
input structure
      |
      v
symbolic graph
      |
      v
ordering / partitioning
      |
      v
ABTM or CSR prepared layout
```

F1/F2 ordering measurements already show why this matters: the same matrix can
have materially different local structure after permutation, which affects
both ILU(0) quality and the locality/occupancy of a future bitmap layout.

## What ABTM removes, and what it does not

Bitmap topology can reduce or eliminate repeated loading and comparison of
explicit column-index metadata. A chunk can represent many possible structural
positions with one integer mask, and set bits can be enumerated using standard
integer operations.

ABTM does **not** remove every irregular access. In an SpMV,

```text
y[i] += a_ij * x[j]
```

the `x[j]` gather remains unless ordering, tiling, blocking, or local caching
also improves vector locality. Therefore the architecture does not claim that
ABTM guarantees fully streaming SpMV, branch-free execution, GPU coalescing, or
absence of divergence.

The narrower and testable claim is:

- matrix-side topology metadata can be compact and bit-oriented;
- packed numerical values can be streamed in topology order;
- structural filtering can happen before value loads;
- ordering/tiling can improve vector-side locality;
- specialized prepared layouts can map the same logical topology to different
  CPU and GPU execution strategies.

## Topology algebra

The topology layer should expose a small set of operations with clear semantics:

```text
AND       intersection / common support
OR        union / graph expansion
AND-NOT   exclusion / unvisited frontier
XOR       structural difference
popcount  cardinality / work estimate
rank      packed-value offset before a bit
select    bit position from packed ordinal
```

On Rust scalar backends, use stable integer methods such as
`u64::trailing_zeros()` and `u64::count_ones()`. Architecture-specific SIMD or
`std::arch` should be introduced only when measurement justifies it.

A statement such as "64 relationships in one clock" is intentionally avoided.
A 64-bit logical instruction can process up to 64 Boolean topology states at
once, but actual latency and throughput depend on the target processor,
dependencies, memory behavior, and compiler code generation.

## G1 scalar topology contract

The first implementation checkpoint uses a structural sparse-of-bitmaps object,
`AbtmTopology`, independent of numerical values.

The G1 contract is deliberately structural:

- every stored CSR column position becomes a topology bit, including an explicitly stored numerical zero;
- duplicate stored columns collapse to one topology bit;
- topology construction does not inspect the CSR value array;
- each row stores only non-empty 64-column words;
- row pointers and word indices are `u32`, while each topology mask is `u64`;
- the structure-of-arrays payload is therefore 12 bytes per non-empty word plus the row-pointer array;
- AND, OR, AND-NOT, and XOR preserve canonical increasing word order;
- popcount is exact structural cardinality;
- rank counts set bits strictly below a bit/column;
- select maps a zero-based packed ordinal back to a bit/column.

This structural definition is intentionally distinct from the current
`AbtmMatrix` numerical packing, which may discard explicit numerical zeros or
duplicate contributions that cancel. Later prepared numerical layouts must
state explicitly which topology they consume rather than silently conflating
stored structure with numerical activity.

G1 is scalar reference semantics. SIMD, Rayon, GPU preparation, packed numeric
updates, and backend promotion are later checkpoints and must not change these
Boolean invariants.

## Metadata-first product pruning

Consider a sparse dot product

```text
a^T b = sum_k a_k b_k
```

with chunk masks `M_a` and `M_b`. Before touching values:

```text
M_active = M_a AND M_b
```

If `M_active == 0`, the complete numerical contribution of that chunk is zero.
Otherwise `popcount(M_active)` gives the exact number of scalar products that
remain.

For packed values, the bitmap position must map to its packed ordinal. For a bit
position `p`, conceptually:

```text
rank(mask, p) = popcount(mask AND bits_below(p))
```

Together with the chunk''s value offset, rank maps a structural bit directly to
its packed numerical value without an index search.

The general execution pattern is:

```text
metadata intersection
        |
        +-- empty -> skip numerical work
        |
        v
     popcount
        |
     work size
        |
        v
   rank / select
        |
        v
 packed value loads
        |
        v
 numerical kernel
```

This is the key distinction between ABTM as a compressed storage format and
ABTM as an execution architecture.

## Candidate kernels

### Sparse dot and masked reductions

These are the simplest demonstrations of metadata-first pruning. The bitmap
intersection defines exactly which products are required.

### Masked and restricted SpMV

Ordinary SpMV has only the matrix support mask and therefore cannot use a second
mask to remove arbitrary products. Restricted operators do have another mask:

```text
effective = row_topology AND active_dof_mask
```

This applies naturally to local residuals, subdomain operators, boundary-only
operations, constrained/free DOF masks, and selected hard regions.

### Sparse matrix products

For

```text
C_ij = sum_k A_ik B_kj
```

a row-support mask of `A` and column-support mask of `B` can determine whether
the dot product is structurally empty before any numerical values are loaded.
A prepared matrix-product backend may therefore retain a transpose/column
topology in addition to row topology. Numerical values need not necessarily be
duplicated merely because topology is available in both orientations.

### Region growth

Hard-region expansion can use:

```text
next = neighbors(frontier) AND-NOT visited
```

where local adjacency masks are OR-combined before the exclusion step. This is
a topology operation and should not require numerical matrix values.

### Local submatrix extraction

For a region mask `R` and matrix row topology `A_i`:

```text
local_i = A_i AND R
```

`popcount(local_i)` gives the local row nnz before allocation; rank/select then
extract only the required packed values. This directly supports bounded local
direct factors used by HyBIT''s escalation path.

### Region overlap and multiplicity

Pairwise overlap is a bitmap intersection, but Schwarz weight multiplicity is
not merely a pairwise Boolean question when more than two regions share a DOF.
The implementation should build explicit multiplicity state, for example a
small integer counter per participating DOF or a bit-sliced counter, from the
region masks. Numerical matrix values are still unnecessary.

### ILU(0) symbolic/numeric intersection

For canonical no-fill ILU(0), an update can be restricted to structural
intersections between the current row and the relevant upper-factor row. A
future ABTM ILU(0) experiment should compare:

```text
CSR search / merge
```

against

```text
bitmap intersection -> rank -> packed update
```

rather than assuming the bitmap route is faster. This is especially relevant
to the current GeneralSquare work because ILU(0) is already a validated
prepared preconditioner.

### Coarse/Galerkin construction

The support of coarse couplings in `P^T A P` can be screened through topology
before accumulating numerical products. ABTM may therefore accelerate
symbolic coarse construction even when the final coarse matrix is stored in a
different format.

### MPI halo extraction

For a partition, topology can distinguish local/internal and remote/boundary
dependencies before numerical execution. Bitwise union/intersection is a
candidate for deriving halo DOFs and communication schedules; overlap of
nonblocking communication with internal computation remains a later
distributed-memory concern.

## Adaptive and block ABTM

A fixed bitmap representation is not expected to dominate every sparsity
pattern. Chunk occupancy determines whether bitmap metadata is cheaper than
explicit indices.

A prepared implementation should be able to choose among representations such
as:

```rust
enum TileKind {
    Dense,
    Bitmap,
    Sparse,
    Block3,
    Block6,
}
```

This is an architectural direction, not a frozen public API.

For FEM, node-level topology is especially important. A single node-adjacency
bit can guard an entire dense `3 x 3` or `6 x 6` coupling block. The metadata
cost is then amortized over 9 or 36 scalar coefficients and the numerical
kernel can use fixed-size block loads/FMA sequences.

## CPU and GPU preparation

A logical ABTM topology should not force one physical descriptor layout on all
targets.

CPU preparation may favor 64-bit chunks, compact descriptors, rank/popcount,
and Rayon row/tile partitioning.

GPU preparation may instead favor warp-sized work units, bounded bitmap-word
counts, cooperative decode, local/shared caching of vector tiles, and
device-specific descriptor packing. A future CubeCL backend should therefore
consume a GPU-prepared ABTM representation rather than copying the CPU layout
verbatim.

Coalesced access and low divergence are optimization targets to measure, not
properties guaranteed by the logical bitmap representation.

## Dynamic topology

When nonlinear analysis or contact changes sparsity, structural edits should be
buffered separately from packed numerical storage. A rebuild can merge topology
changes first and repack values in one preparation step instead of repeatedly
reallocating the complete numerical stream.

The prepared-state API should distinguish:

```text
topology unchanged, values changed
topology changed
```

so later symbolic/topological reuse can be separated from numeric
refactorization.

## Measurements required

ABTM evaluation should not be reduced to "SpMV faster or slower than CSR".
Each experiment should record at least:

- chunk/tile occupancy;
- metadata bytes per nnz;
- packed-value bytes;
- bitmap operations performed;
- candidate numerical products;
- executed numerical products;
- products skipped by metadata;
- rank/select cost;
- setup/preparation cost;
- solve/kernel wall time;
- cache/locality counters where practical.

Define a pruning ratio:

```text
pruning_ratio =
    1 - executed_numerical_products / candidate_numerical_products
```

The useful break-even condition is qualitative but fundamental:

```text
metadata cost < numerical and memory work avoided
```

A metadata intersection that avoids one cheap FMA may lose. The same
intersection that discards many scalar products, a `3 x 3`/`6 x 6` block, a
local-factor update, or a remote communication dependency can be highly
profitable.

## Development sequence

After the current GeneralSquare robustness checkpoints, the ABTM track should
proceed experimentally:

1. **G1 — topology algebra:** bitmap chunks, AND/OR/AND-NOT/XOR, popcount,
   rank/select, invariants, and scalar reference tests.
2. **G2 — metadata-first kernels:** sparse dot/support intersection and
   quantitative pruning metrics.
3. **G3 — HyBIT region operations:** region growth, overlap/multiplicity, and
   local submatrix extraction versus the current graph/index path.
4. **G4 — ABTM ILU(0):** symbolic intersection and packed numeric updates versus
   canonical CSR ILU(0).
5. **G5 — block ABTM (validated):** 3x3/6x6 FEM node topology and explicit
   fixed-size dense block-CSR numerical execution.
6. **G6 — prepared restricted SpMV (validated):** ordinary scalar ABTM remains
   diagnostic; fixed A*M and graph-local R*A*R restrictions materialize
   compact CSR, with preparation policy kept explicit and workload-specific.
7. **G7 — Rayon prepared execution (validated):** explicit full-Rayon and
   task-limited prepared CSR execution, serial gather, and no automatic
   hardware-specific size selector.
8. **G8 — distributed topology (next):** partition/halo extraction and MPI-host
   integration before distributed Krylov scheduling.
9. **G9 — GPU prepared execution:** CubeCL-oriented descriptors and resident
   execution after the CubeCL API is sufficiently stable.

No automatic backend promotion should occur until the relevant workload class
shows repeatable benefit including preparation cost.

## Architectural invariant

The long-term invariant is:

> Boolean topology algebra determines *where work can exist*; numerical algebra
> is invoked only for the surviving work, using a representation prepared for
> the current execution target.

This makes ABTM a symbolic computation layer for HyBIT rather than merely
another sparse matrix file format.

## G1 validation evidence and decision

G1 establishes scalar reference semantics for structural bitmap topology and
validates the representation on the same ten-matrix corpus used by the recent
GeneralSquare studies.

The measured `AbtmTopology` physical form is a sparse-of-64-bitmaps
structure-of-arrays:

```text
row pointer  : u32 per row boundary
word index   : u32 per non-empty word
word mask    : u64 per non-empty word
```

so one non-empty bitmap word costs 12 bytes before row-pointer amortization.

### Ten-matrix metadata/occupancy evidence

| Matrix | avg nnz/word | p50 nnz/word | topology / CSR metadata | word-local rank/select |
| --- | ---: | ---: | ---: | ---: |
| `sherman5` | 2.93 | 3 | 1.020 | 9.31 ns/nnz |
| `raefsky3` | 19.10 | 24 | 0.169 | 4.98 ns/nnz |
| `venkat25` | 5.18 | 4 | 0.594 | 3.49 ns/nnz |
| `cfd1` | 4.24 | 3 | 0.718 | 3.83 ns/nnz |
| `thermal1` | 2.11 | 1 | 1.371 | 5.85 ns/nnz |
| `nd3k` | 20.06 | 17 | 0.152 | 6.38 ns/nnz |
| `cant` | 11.55 | 10 | 0.271 | 4.07 ns/nnz |
| `s3dkq4m2` | 12.98 | 15 | 0.247 | 4.63 ns/nnz |
| `boneS01` | 5.90 | 6 | 0.519 | 4.06 ns/nnz |
| `x104` | 11.62 | 11 | 0.267 | 4.45 ns/nnz |

Eight of ten matrices use less topology metadata than CSR32 row-pointer plus
column-index metadata. `sherman5` is approximately break-even and `thermal1`
is materially worse because most 64-column words contain very few structural
entries.

The result confirms that the logical bitmap algebra is broadly useful, but a
single physical bitmap encoding is not an appropriate universal prepared
layout.

### Occupancy and adaptive representation

The observed occupancy distribution is highly matrix dependent.

- `thermal1` has median occupancy 1 and about 83% of words at three entries or
  fewer.
- `sherman5` has median occupancy 3 and about 81% of words at three entries or
  fewer.
- `raefsky3` has median occupancy 24.
- `nd3k` has median occupancy 17 and about 11% of words at 40 entries or more.
- `cant`, `s3dkq4m2`, and `x104` occupy an intermediate regime.

For a conceptual sparse word representation

```text
word_index: u32
offsets:    k x u8
```

the metadata model is roughly `4 + k` bytes, compared with 12 bytes for the
bitmap word. This gives a natural conceptual crossover near `k = 8`, consistent
with the existing ABTM sparse/bitmap design direction. Concrete prepared layouts
must still account for offsets, alignment, descriptor storage, and execution
cost rather than freezing this arithmetic as a universal threshold.

### Rank/select interpretation

G1a deliberately checked both chunk-local and row-wide convenience rank/select.
On dense rows this made the row-wide scan visible, most strongly on `nd3k`.

G1b isolates the operation required by packed numerical streams: chunk-local
word rank/select. Across the ten real matrices it measured approximately
3.5--9.3 ns per structural nonzero on the validation machine. Therefore future
metadata-first kernels should carry the matching word descriptor and use
word-local rank/select; row-wide rank/select remains a correctness/convenience
API rather than the inner numerical addressing path.

### Boolean merge validation

G1b also prepared two deterministic approximately 75%-dense structural subsets
of each source topology and exercised AND, OR, AND-NOT, and XOR through the
general merge path.

The measured intersection ratios stay close to the independent expectation

```text
0.75 * 0.75 = 0.5625
```

and union ratios stay close to

```text
1 - 0.25 * 0.25 = 0.9375
```

across all ten matrices. This validates missing-word and partial-mask merge
semantics in addition to the unit-test identities and G1a self-operations.

### G1 decision

G1 is accepted as the scalar logical topology layer.

The retained invariants are:

- numerical values are not part of topology;
- explicitly stored structural positions remain topology even when their
  numerical value is zero;
- duplicate structural columns collapse to one bit;
- row words remain canonical and strictly increasing;
- AND / OR / AND-NOT / XOR operate only on topology;
- popcount is exact structural cardinality;
- word-local rank/select is the packed-value addressing primitive.

G1 does **not** promote one bitmap physical layout as universal. Later prepared
execution may choose Sparse, Bitmap, Dense, block, CPU-specific, or GPU-specific
representations from the same logical topology.

G2-G7 are now validated. The next checkpoint is G8: distributed topology / MPI-host integration.

## G2a scalar metadata-first sparse-dot experiment

G2 begins with a deliberately narrow numerical kernel: one sparse matrix row
dotted with a vector whose structural support is represented by a `DofMask`.

The scalar prepared reference stores one numerical value per G1 structural
position in the same packed topology order. For each 64-column topology word:

```text
active = matrix_word_mask AND vector_support_word
```

If `active == 0`, no numerical value from that matrix word is loaded. Otherwise
word-local rank maps each active bit directly to its packed matrix value.

For this experiment the work metrics are defined precisely:

```text
candidate_products = structural matrix entries before support filtering
executed_products  = popcount(matrix topology AND vector support)
skipped_products   = candidate_products - executed_products
pruning_ratio      = skipped_products / candidate_products
```

This makes G2a a sparse-dot/support-intersection experiment rather than a claim
about a production ABTM SpMV implementation. Full masked/restricted SpMV policy,
parallel execution, and backend selection remain later checkpoints.

The 100%-active case measures metadata/rank overhead when no numerical work can
be pruned. Lower support densities measure whether avoided value loads and
multiplications can amortize that overhead.

## G2c dual row/column topology

Matrix-versus-matrix sparse products require metadata on both operands. For

```text
C_ij = sum_k A_ik B_kj
```

the structural intersection is between a row support of `A` and a column
support of `B`.

G2c introduces `AbtmDualTopology`, which retains:

```text
row topology       original matrix structure
column topology    structural transpose topology
```

while leaving numerical values unduplicated. A future numerical prepared
backend can therefore decide independently whether values also need a
transpose-oriented stream.

The G2c benchmark first isolates structural work. It compares a conventional
sorted explicit-index intersection with bitmap-word intersection for
deterministic row/column pairs. The key metrics are metadata footprint,
metadata comparisons, mask AND count, empty-dot ratio, exact overlap products,
and scalar intersection wall time.

This avoids conflating topology benefit with a particular numerical-value
layout before the dual-metadata economics are known.

## G2d numerical values for dual topology

G2c establishes that row/column bitmap metadata can greatly reduce structural
intersection work on high-occupancy matrices, while low-occupancy cases remain
neutral or unfavorable.

G2d asks the next architectural question: must numerical values also be
duplicated for the column orientation?

Two bitmap variants are compared:

```text
mapped-column values
    one packed row-value stream
    + u32 source-value index per column-topology entry

duplicated-column values
    packed row-value stream
    + packed column-value stream
```

Both use the same dual row/column topology. A conventional explicit
row/column-index merge with duplicated values is retained as the numerical
baseline.

This separates three costs that must not be conflated:

```text
topology compression
column-value addressing indirection
column-value duplication
```

The result will determine whether later G2 matrix-product kernels should keep a
single numerical copy, duplicate values for hot transpose-oriented work, or
select between the two as a prepared policy.

## G2e adaptive numerical dual layout

G2d shows that one fixed bitmap numerical addressing strategy is insufficient:
high-occupancy matrices can win, but low- and medium-occupancy matrices pay too
much rank/value-addressing overhead.

G2e therefore returns to the physical-layout rule established by G1. Both row
and column orientations use the existing adaptive tile classification:

```text
Sparse <= 8 entries / 64-column word
Bitmap intermediate occupancy
Dense  >= 40 entries / 64-column word
```

Execution is also adaptive. Sparse/Sparse tiles use a local ordered set-bit
merge and avoid rank. Dense tiles use direct bit offsets. Bitmap tiles retain
mask intersection and packed rank addressing.

This checkpoint tests whether representation and execution policy must both be
occupancy-aware. The thresholds are intentionally left unchanged until the
mechanism is measured on the corpus.

## G2f physically packed adaptive metadata

G2e demonstrates that occupancy-aware execution materially improves every
tested matrix compared with fixed bitmap numerical addressing. However, the
existing `TileDesc` is still a fixed 16-byte descriptor. In particular, a
`Sparse` tile still carries a full 64-bit mask and value offset.

G2f separates the representation question from the execution question. Sparse
tiles use variable-length byte offsets, while Bitmap and Dense tiles retain
64-bit masks. Row-local payload and value pointers make each row directly
addressable without storing a value offset in every tile.

The benchmark compares the resulting dual numerical layout directly with
explicit CSR plus transpose-CSR storage and numerical row/column dot time.
Threshold tuning is intentionally deferred until the packed representation
itself is validated.

## G2g typed compact descriptors

G2f establishes that compact Sparse storage is possible but also shows that a
byte-oriented variable payload should not be decoded directly in the numerical
hot loop.

G2g therefore evaluates an intermediate design:

```text
8-byte typed descriptor + typed auxiliary streams
```

Sparse offsets remain one byte each and Bitmap/Dense masks remain 64-bit, but
the descriptor directly identifies the relevant typed side stream. Value
offsets are advanced sequentially within each row, preserving compactness
without per-tile value offsets.

This checkpoint is intended to find the practical middle ground between G2e's
fast fixed 16-byte descriptors and G2f's compact but expensive variable decoder.

## G2 closeout

G2 is closed for the 0.8 development checkpoint. The full evidence and
ten-matrix G2e corpus are recorded in
[`ABTM_G2_CLOSEOUT.md`](ABTM_G2_CLOSEOUT.md).

The selected direction is dual structural topology plus adaptive
Sparse/Bitmap/Dense CPU execution, while retaining explicit CSR/CSC-like
fallback. G2 does not promote a universal bitmap layout or an automatic
ABTM/explicit threshold.

G3 is the next checkpoint and moves to region growth, overlap/multiplicity, and
local submatrix extraction.

## G3a region growth

G3 begins with structural region growth. For a seed set `R_0`, G3a defines
undirected adjacency from the matrix structure as

```text
G = support(A) union support(A^T)
```

and grows only the previous hop's frontier:

```text
F_0 = R_0
N_k = neighbors(F_k)
F_{k+1} = N_k \ R_k
R_{k+1} = R_k union F_{k+1}
```

Using `AbtmDualTopology` avoids constructing numerical transpose values for this
operation. Row topology supplies outgoing neighbors and column topology supplies
incoming neighbors. The result is deterministic and purely structural.

G3a intentionally does not yet assign multiplicity or extract local matrices;
those follow after region-growth semantics and cost are validated.

## G3b overlap and multiplicity

For a family of grown regions `R_i`, G3b defines the node multiplicity

```text
m(v) = sum_i [v in R_i].
```

This gives three directly useful domain-decomposition quantities:

- covered nodes: `m(v) >= 1`;
- overlap nodes: `m(v) >= 2`;
- multiplicity-weighted overlap work.

The implementation retains the full per-node multiplicity map and verifies the
identity

```text
sum_{i<j} |R_i intersect R_j| = sum_v m(v)(m(v)-1)/2.
```

This checkpoint establishes overlap semantics before G3 local submatrix
extraction. It does not yet define partition ownership or weighting policy.

## G3c structural local-submatrix extraction

For a region `R`, G3c constructs the CSR-like structural pattern of `A[R,R]`.
Rows and columns use deterministic ascending-global-index local numbering.

ABTM performs this as a metadata-first operation:

```text
for global row r in R:
    for topology word W in row(r):
        kept = W.mask AND R.word(W.index)
        enumerate only kept bits
```

The extracted pattern stores local row pointers and local column indices plus
the ordered global-node list. A temporary dense global-to-local map is currently
used and its scratch bytes are reported explicitly.

This checkpoint deliberately stops at structural extraction. Numerical value
gathering and local numeric factor preparation remain separate so topology
benefits are not hidden by a value-addressing policy chosen too early.

## G3d prepared local numeric refresh

G3d separates stable symbolic structure from changing numerical coefficients.

For each region, preparation stores:

```text
local pattern:
    global nodes
    row_ptr
    col_idx

numeric address plan:
    local structural entry -> CSR source value position(s)
```

If the input CSR is canonical, each local structural entry has one source
position and refresh becomes a direct indexed gather. If duplicate CSR entries
exist, the plan stores all source positions and sums them without changing the
local symbolic pattern.

This design targets repeated refactorization/nonlinear/time-stepping use cases:
region growth and local structure are paid once while numerical values can be
refreshed many times.

## G3 closeout decision

G3 confirms the topology/numerical boundary introduced in G2.

ABTM dual topology is retained for region discovery, frontier set algebra,
overlap metadata, and symbolic pruning. Local CSR-like structures are
materialized when downstream kernels need explicit row/column indexing.
Prepared source-address maps are appropriate when numerical coefficients change
while the local symbolic structure remains fixed.

The G3d refresh result is a symbolic-reuse result, not a claim that ABTM bitmap
value storage is universally faster than prepared CSR. See
`ABTM_G3_CLOSEOUT.md` for the corpus and interpretation limits.

## G4a ILU(0) symbolic intersection

G4 begins by isolating ILU(0) symbolic update lookup from numerical
factorization.

For each lower structural entry `(i,j)`, canonical CSR ILU(0) probes the upper
structure of row `j` and binary-searches candidates in the remainder of row
`i`. The G4a experiment replaces those per-entry probes with word-level
intersection of the same canonicalized pattern.

The benchmark does not use raw topology directly when explicit zeros or
duplicate cancellation would differ from canonical ILU(0). It first applies the
same canonical structural rule as the production preconditioner, then builds
the ABTM topology.

No production ILU(0) routing changes are made in G4a.

## G4b ILU(0) numeric intersection

G4b adds a compact row-word prefix to turn topology intersection bits into
canonical CSR numerical positions. This avoids a per-successful-update address
plan, whose storage could grow with the number of ILU(0) products rather than
with matrix topology.

The experiment keeps numerical factors in conventional contiguous `f64` CSR
order. ABTM supplies symbolic intersection and rank-within-word addressing; it
does not replace the numerical factor layout.

## G4c ILU(0) rank-LUT addressing

G4c tests a direct rank lookup for numerical addressing. Each nonempty topology
word receives a 64-byte table mapping a set bit position to its ordinal among
the word's structural bits.

This removes two hardware `popcount` rank operations per executed ILU(0)
update, at the cost of 64 bytes per nonempty word. G4c is diagnostic: if the
speed gain is small or inconsistent, the extra storage is rejected. If the gain
is significant only for dense words, a later adaptive threshold may retain
tables only where occupancy justifies them.

## G4d adaptive rank-LUT sweep

G4d evaluates selective rank LUT storage by topology-word occupancy. Dense
words may receive a direct 64-byte rank table while sparse words retain
rank-by-popcount.

The experiment explicitly measures both static selected-word fraction and
dynamic LUT-rank coverage. The distinction matters because words are not
accessed uniformly during ILU(0): a structurally modest word may still be hot
across many lower-pivot intersections.

No production threshold is implied by average word occupancy alone.

## G4f production lifetime boundary

The G4c rank LUT is required only while constructing ILU(0). G4f makes that
lifetime explicit: topology words, structural prefixes, and `u8[64]` rank
tables are preparation scratch and are released once the canonical `L/U`
values have been produced.

As a result, ABTM factorization can accelerate setup without increasing the
persistent ILU(0) factor footprint or changing triangular application.

## G5 block-topology validation and production decision

G5 tests the FEM-oriented hypothesis that scalar topology can be coarsened into
node-sized 3x3 or 6x6 couplings and that sufficiently full numerical blocks can
amortize index/gather overhead.

### G5a topology evidence

Across the 11-matrix development corpus, block topology compressed structural
metadata substantially:

| Block | metadata/scalar topology (geomean) | non-empty-word ratio (geomean) | dense-value inflation (geomean) | mean block fill |
| --- | ---: | ---: | ---: | ---: |
| B3 | 0.505 | 0.280 | 1.823 | 0.624 |
| B6 | 0.466 | 0.126 | 2.957 | 0.434 |

The B6 topology is especially compact, but low block fill can make dense value
storage prohibitively expensive. Therefore block topology and numerical
physical layout must remain separate decisions.

### G5b-G5d numerical kernel evidence

A generic runtime-sized block kernel lost to scalar CSR on all 11 development
matrices. G5 therefore does not promote a generic block traversal.

Fixed-size B3/B6 kernels changed the result. Tail-specialized execution keeps
full interior blocks on the fixed kernel and isolates only the final partial
row/column block. Representative kernel speedups versus scalar CSR were:

| Matrix | selected block | block fill | kernel speedup | storage / CSR | break-even SpMV |
| --- | ---: | ---: | ---: | ---: | ---: |
|
d3k | B6 | 0.741 | 2.57x | 0.912 | 62.8 |
| x104 | B6 | 0.857 | 2.31x | 0.786 | 66.8 |
| aefsky3 | B6 | 0.748 | 2.08x | 0.900 | 73.8 |
| cant | B3 | 0.930 | 1.88x | 0.755 | 77.6 |
| oneS01 | B3 | 0.822 | 1.47x | 0.853 | 116 |
| s3dkq4m2 | B6 | 0.549 | 1.35x | 1.224 | 171 |

Low-fill matrices such as 	hermal1, pache2, cfd1, and enkat25
remain better served by scalar CSR.

### G5e-G5f selector evidence

The development selector was deliberately simple:

`	ext
if B3 fill < 0.65:
    CSR
else if B6_fill / B3_fill >= 0.75:
    B6
else:
    B3
`

On the 11-matrix calibration corpus it produced no false-positive block routes.
Its one kernel-winner miss was sherman5, where B3 was only about 1% faster and
required roughly 2088 SpMV to repay preparation, so CSR remained the sensible
finite-horizon choice.

The thresholds were then frozen. Held-out validation on
Goodwin_010, G3_circuit, parabolic_fem, 	hermal2, and inline_1
classified all five correctly with:

`	ext
false_positive_block = 0
false_negative_csr   = 0
block_size_miss      = 0
`

inline_1 provides the positive held-out B3 case: B3 fill about 0.99998,
kernel speed about 1.78x CSR, storage about 0.70x CSR, and preparation break-even
about 73.5 SpMV.

No positive held-out B6 case is available yet. The structural rule is therefore
retained as development evidence, not promoted to an automatic production
selector.

### Explicit G5 production API

G5 exposes the validated numerical representation explicitly:

`ust
use hybit::{DenseBlockCsrOperator, DenseBlockSize};

let block =
    DenseBlockCsrOperator::from_csr32(&matrix, DenseBlockSize::B3)?;
`

The prepared numerical layout is conventional compact block CSR:

`	ext
block row_ptr
block col_idx
fixed dense BxB values
`

The operator supports B3/B6, partial final row/column blocks, duplicate scalar
contribution accumulation, storage/fill diagnostics, and the common
LinearOperator interface.

This preserves the ABTM architectural boundary: topology and measurements
identify profitable structure, while the hot numerical representation can be a
specialized conventional layout. Existing CSR paths and solver defaults are
unchanged.

G6 is the next checkpoint: ordinary and masked/restricted SpMV.

## G6 validation evidence and decision

G6 evaluated ordinary SpMV, masked column restriction, and graph-local
restriction as separate workload classes instead of forcing one ABTM numerical
layout onto all three.

### G6a ordinary and on-the-fly masked SpMV

The 11-matrix development corpus rejected scalar ABTM as a universal ordinary
SpMV replacement. Ordinary ABTM produced no wins and a geometric-mean
ABTM-versus-CSR speedup of about `0.43x`.

Metadata-first masking still demonstrated useful pruning semantics. Intersecting
tile topology with the active-DOF mask before loading numerical values strongly
beat a CSR implementation that branches on every stored entry. However, the
stronger baseline--a pre-zeroed input vector followed by ordinary CSR SpMV--was
harder to beat and did not justify promoting the on-the-fly ABTM masked kernel.

The retained conclusion is that metadata-first pruning is valuable during
preparation, but repeated scalar arithmetic should use a layout specialized for
the prepared workload.

### G6b fixed column restriction `A*M`

For a fixed active-column mask `M`, G6b materialized a compact CSR operator that
retains global dimensions but stores only active-column entries. Repeated apply
therefore consumes the changing global input vector directly without a
per-entry mask branch or a per-apply masked-vector refresh.

Development-corpus geometric-mean speedup of prepared execution versus the
dynamic full-CSR baseline was approximately:

| Active columns | Prepared / dynamic full CSR | Wins |
| ---: | ---: | ---: |
| 5% | 5.924x | 11/11 |
| 10% | 3.865x | 11/11 |
| 25% | 2.564x | 11/11 |
| 50% | 1.835x | 11/11 |
| 75% | 1.282x | 10/11 |

When an `AbtmMatrix` already existed, ABTM metadata intersection sometimes
accelerated preparation. Development-corpus preparation speedup versus direct
CSR scan was geometrically about `1.17x--1.35x` through 50% density. The
advantage was not universal, so it remains explicit rather than automatic.

### G6c graph-local `R*A*R`

G6c built deterministic graph-local regions and compared repeated scanning of
global CSR region rows against an explicitly materialized compact local CSR.

Development-corpus dynamic prepared speedup was:

| Region fraction | Prepared / repeated region-row scan | Wins |
| ---: | ---: | ---: |
| 1% | 3.613x | 11/11 |
| 5% | 3.089x | 11/11 |
| 10% | 2.934x | 11/11 |
| 25% | 2.747x | 11/11 |

Unlike the column-mask case, the existing ABTM local numeric-plan builder lost
to direct CSR region-row extraction on every measured development case.
The local plan performs general topology extraction and source binding work that
is unnecessary when a one-shot compact `R*A*R` operator can be built directly.

### G6d held-out validation

The held-out corpus was `Goodwin_010`, `G3_circuit`, `parabolic_fem`,
`thermal2`, and `inline_1`.

For fixed `A*M`, prepared dynamic execution won `5/5` held-out matrices at
5%, 10%, 25%, and 50% density and `4/5` at 75%. Geometric-mean speedups were
about `4.249x`, `2.798x`, `2.048x`, `1.745x`, and `1.229x` respectively.

The held-out ABTM-preparation advantage was mixed: geometric-mean
ABTM-versus-direct-CSR preparation speedup was about `1.03x`, `1.00x`, `0.97x`,
`1.07x`, and `0.87x` over the same densities. This rules out an unconditional
ABTM preparation selector.

For graph-local `R*A*R`, prepared dynamic execution won `5/5` held-out matrices
at every tested region fraction. Geometric-mean speedups at 1%, 5%, 10%, and
25% were about `3.727x`, `2.682x`, `2.516x`, and `2.381x`.

Direct CSR local preparation beat the existing ABTM local-plan path `5/5` at
every held-out fraction, by geometric means of about `6.84x`, `7.82x`, `8.09x`,
and `8.57x`.

### G6 production boundary

G6 therefore retains this workload-specific boundary:

```text
ordinary SpMV
    -> CSR remains the baseline numerical path

fixed column restriction A*M
    -> compact prepared CSR
    -> direct CSR preparation by default
    -> explicit ABTM preparation only when an ABTM layout already exists

graph-local R*A*R
    -> direct CSR region-row extraction
    -> compact local CSR

repeated arithmetic
    -> conventional compact CSR
```

The public production surface is explicit:

- `PreparedColumnRestrictedCsrOperator`
  - `from_csr32`
  - `from_abtm` as an explicit opt-in preparation path
- `PreparedLocalCsrOperator`
  - `from_csr32`
  - compact ascending local numbering
  - `gather_input` / `gather_input_vec` for changing global vectors

No automatic restriction policy, CSR/ABTM preparation selector, or solver
backend promotion is introduced by G6.

This reinforces the ABTM architectural role: topology metadata is used where it
can eliminate structural work profitably, while repeated numerical execution is
free to use a conventional representation better matched to the target
workload.

## G7 validation evidence and decision

G7 evaluates CPU parallel execution only after G6 has already materialized the
repeated numerical workload as compact prepared CSR. The question is therefore
not whether ABTM itself should become the parallel numerical storage format, but
how prepared `A*M` and `R*A*R` operators should expose Rayon execution without
hiding hardware-specific policy.

### G7a prepared execution crossover

With the default 16-worker Rayon pool, fixed-column `A*M` was strongly
parallel-friendly once the prepared matrix was beyond the smallest test case.
Across the 11-matrix development corpus the geometric-mean parallel speedups
versus serial prepared CSR were approximately:

| Active columns | Parallel / serial | Wins |
| ---: | ---: | ---: |
| 5% | 2.345x | 10/11 |
| 10% | 3.217x | 10/11 |
| 25% | 4.093x | 10/11 |
| 50% | 4.604x | 10/11 |
| 75% | 5.155x | 10/11 |

The five losses were the very small `sherman5` prepared operators. The
development envelope was `max_loss_nnz=15,330` and
`min_win_nnz=28,316`, which suggests a useful size signal but does not by
itself define a portable production threshold.

For graph-local `R*A*R`, full-pool parallelism was more size-sensitive.
At 16 workers the kernel geometric-mean speedups at 1%, 5%, 10%, and 25%
region fractions were about `0.298x`, `1.178x`, `1.814x`, and `3.118x`.
The corresponding changing-global-vector dynamic path, including serial
`gather_input`, measured about `0.331x`, `1.112x`, `1.580x`, and `2.015x`.

### G7b worker-count sensitivity

A 4/8/16-worker sweep showed that `A*M` continues to favor wider parallelism as
the prepared matrix grows. Across 55 development cases, the fastest measured
worker count was 4, 8, and 16 in 8, 22, and 25 cases respectively.

`R*A*R` behaves differently. Across the 44 development cases, the fastest
kernel worker count was 4/8/16 in 20/17/7 cases, and the fastest dynamic
gather-plus-SpMV path was 4/8/16 in 24/16/4 cases.

This demonstrates that one global worker count is not a sufficient portable
prepared-local execution policy.

### G7c task granularity and gather policy

G7c retained the default 16-worker global Rayon pool and varied only the number
of contiguous row tasks submitted per operation: 2, 4, 8, or 16.

The result rejected the idea that a small fixed task count can simply reproduce
the favorable 4-worker behavior. Sixteen tasks were the fastest full dynamic
path in 39 of 44 development cases; 8 tasks won 4 cases, 4 tasks won 1, and 2
tasks won none.

The experiment also isolates global-to-local gathering. Parallel gather was
usually slower than the existing serial `PreparedLocalCsrOperator::gather_input`.
Even at 16 tasks, the gather geometric-mean speedup versus serial gather was
only about `0.030x`, `0.126x`, `0.246x`, and `0.545x` at the tested 1%, 5%,
10%, and 25% region fractions. Therefore G7 retains serial gather.

With serial gather retained, 16-task local SpMV became profitable around the
low-tens-of-thousands nnz range on the development corpus, but this observation
is treated as evidence rather than API semantics.

### G7d held-out selector validation

G7d froze a candidate development selector and tested it on
`Goodwin_010`, `G3_circuit`, `parabolic_fem`, `thermal2`, and `inline_1`.

For fixed-column `A*M`, the candidate was:

```text
prepared nnz < 32k  -> serial CSR
otherwise           -> existing full Rayon CSR
```

This candidate produced no held-out regressions at any tested density. The
oracle-capture geometric mean was `1.0` at 5%, 10%, 25%, 50%, and 75% active
columns, meaning the selector chose the fastest measured serial/full-Rayon path
for every held-out `A*M` point.

For graph-local `R*A*R`, the candidate was:

```text
local nnz < 32k        -> serial gather + serial CSR
32k <= nnz < 512k      -> serial gather + capped contiguous-row tasks
local nnz >= 512k      -> serial gather + existing full Rayon CSR
```

The aggregate result was positive but not strong enough to freeze the policy.
At 5%, 10%, and 25% region fractions the candidate had no meaningful
held-out regression, but at 1% only 4/5 cases were non-regressions.

Two cases explain why an automatic selector remains deferred:

- `G3_circuit`, 1% region, 74,239 local nnz: the candidate chunked path ran at
  about `0.848x` the serial baseline.
- `inline_1`, 1% region, 348,237 local nnz: the candidate chunked path was still
  about `1.97x` faster than serial, but the existing full-Rayon path was much
  faster, reducing oracle capture to about `0.608`.

The held-out evidence therefore supports explicit Rayon capability but not a
portable hardware-independent `R*A*R` nnz selector.

### G7 production boundary

G7 exposes execution mechanisms without embedding the benchmark machine's
routing thresholds:

```text
Csr32Matrix
    apply_parallel(...)
    apply_parallel_with_tasks(...)

ParallelCsr32Operator
    apply(...)
    apply_with_tasks(...)

PreparedColumnRestrictedCsrOperator
    apply(...)                       # serial LinearOperator path
    apply_parallel(...)
    apply_parallel_with_tasks(...)

PreparedLocalCsrOperator
    gather_input(...)                # serial
    apply(...)                       # serial LinearOperator path
    apply_parallel(...)
    apply_parallel_with_tasks(...)
```

`target_tasks` bounds contiguous row-task granularity while retaining the
process-wide Rayon pool. It does not create a private pool and does not promise
that exactly that many workers run concurrently.

No automatic matrix-size threshold, hardware-specific worker policy, or
prepared-operator route is promoted in G7. The caller remains responsible for
selecting serial, full-Rayon, or task-limited execution from application and
hardware context.

This closes the CPU prepared-execution sequence while preserving the broader
ABTM architectural rule: topology discovers and prepares the workload; the
numerical representation and execution policy remain target-specific and
evidence-driven.
