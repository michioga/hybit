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

1. **G1 — topology algebra:** bitmap chunks, AND/OR/AND-NOT, popcount,
   rank/select, invariants, and scalar reference tests.
2. **G2 — metadata-first kernels:** sparse dot/support intersection and
   quantitative pruning metrics.
3. **G3 — HyBIT region operations:** region growth, overlap/multiplicity, and
   local submatrix extraction versus the current graph/index path.
4. **G4 — ABTM ILU(0):** symbolic intersection and packed numeric updates versus
   canonical CSR ILU(0).
5. **G5 — block ABTM:** 3x3/6x6 FEM node topology and fixed-size numerical
   blocks.
6. **G6 — SpMV evaluation:** CSR versus ABTM for ordinary and masked/restricted
   SpMV; do not use pure SpMV as the only ABTM success criterion.
7. **G7 — Rayon prepared execution:** work partitioning and NUMA/cache behavior.
8. **G8 — GPU prepared ABTM:** CubeCL-oriented descriptors and resident
   execution after the CPU semantics are stable.
9. **G9 — distributed topology:** partition/halo extraction and later MPI
   communication scheduling.

No automatic backend promotion should occur until the relevant workload class
shows repeatable benefit including preparation cost.

## Architectural invariant

The long-term invariant is:

> Boolean topology algebra determines *where work can exist*; numerical algebra
> is invoked only for the surviving work, using a representation prepared for
> the current execution target.

This makes ABTM a symbolic computation layer for HyBIT rather than merely
another sparse matrix file format.
