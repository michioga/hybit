# ABTM G3 closeout: regions, overlap, local extraction, and symbolic reuse

G3 validates ABTM as a structural/topology layer for region-oriented sparse
operations. The checkpoint covers deterministic region growth, overlap and
multiplicity, structural extraction of `A[R,R]`, and prepared numerical refresh
when topology remains fixed.

## Status

**G3 is validated and closed for the 0.8 development checkpoint.**

The results do not justify replacing CSR universally. They support a split
architecture:

- use ABTM dual topology for structural graph operations and metadata-first
  pruning;
- materialize ordinary local sparse structure when downstream numerical kernels
  require it;
- cache symbolic/numerical address plans when the same local structure is
  refreshed repeatedly.

## G3a: region growth

G3a grows seed regions over

```text
support(A) union support(A^T)
```

using a frontier formulation. Every ABTM region was cross-checked against an
explicit CSR + transpose-CSR reference.

Two-hop corpus:

| matrix | average region nodes | topology words / CSR neighbor entries | ABTM / CSR time |
| --- | ---: | ---: | ---: |
| sherman5 | 22.73 | 0.29035 | 1.1553 |
| raefsky3 | 191.88 | 0.05172 | 0.3089 |
| venkat25 | 74.38 | 0.19507 | 1.0994 |
| cfd1 | 116.44 | 0.24276 | 1.0021 |
| thermal1 | 19.48 | 0.50826 | 1.0825 |
| nd3k | 1979.48 | 0.04992 | 0.07476 |

The one/two/three-hop sweep showed that tiny frontiers are dominated by fixed
bitset and allocation costs. As frontier work grows, matrices with many
structural entries represented per topology word benefit strongly. `nd3k`
became a saturation stress case: at three hops the average region was about
4612 nodes out of a 9000-node universe.

This confirms that topology-word compression can translate directly into graph
traversal savings, but only when enough work exists to amortize setup and
frontier bookkeeping.

## G3b: overlap and multiplicity

For grown regions `R_i`, G3b defines

```text
m(v) = number of regions containing node v.
```

Correctness was checked three ways:

- every ABTM-grown region equals the CSR reference region;
- every per-node multiplicity equals a direct reference count;
- pair overlap obeys
  `sum_{i<j} |R_i intersect R_j| = sum_v C(m(v), 2)`.

Two-hop corpus:

| matrix | overlap fraction of covered nodes | average multiplicity | max multiplicity | ABTM / CSR pipeline |
| --- | ---: | ---: | ---: | ---: |
| sherman5 | 0.2821 | 1.3547 | 4 | 1.2222 |
| raefsky3 | 0.2537 | 1.3244 | 4 | 0.4200 |
| venkat25 | 0.0312 | 1.0312 | 2 | 1.1428 |
| cfd1 | 0.0430 | 1.0430 | 2 | 1.0808 |
| thermal1 | 0.0122 | 1.0122 | 2 | 1.3530 |
| nd3k | 1.0000 | 14.0763 | 31 | 0.08386 |

Multiplicity reduction itself was cheap compared with region growth, so the
pipeline largely inherited G3a behavior. `nd3k` is intentionally treated as a
high-overlap stress case: all 9000 nodes were covered and overlapped, 1594 of
2016 region pairs intersected, and maximum multiplicity reached 31.

## G3c: structural local-submatrix extraction

G3c materializes the structural CSR-like pattern of `A[R,R]`. ABTM intersects
row topology words with the region mask before enumerating retained columns.
The result was checked exactly against direct CSR extraction:

```text
global_nodes
row_ptr
col_idx
```

all matched for every tested region.

Extraction-time ratios:

| matrix | ABTM / CSR structural extraction |
| --- | ---: |
| sherman5 | 1.1065 |
| raefsky3 | 1.3075 |
| venkat25 | 1.0038 |
| cfd1 | 1.0476 |
| thermal1 | 0.9708 |
| nd3k | 1.1784 |

The main conclusion is negative but useful: after the local structure must be
fully materialized, output enumeration and global-to-local mapping dominate
enough that topology-word compression alone is not a general speed advantage.

`nd3k` produced 38,158,139 local structural entries across the 64 regions, so
the extraction benchmark was dominated by output construction despite a
topology-word / CSR-entry ratio near 0.05.

## G3d: prepared local numerical refresh

G3d tests the reuse case suggested by G3c. A plan is prepared once that binds
each local structural entry to source global CSR value position(s). Later value
refreshes reuse the local topology and address map without rebuilding regions,
local numbering, row pointers, or column indices.

All six corpus matrices had:

```text
mismatched_regions=0
mismatched_patterns=0
mismatched_values=0
max_scaled_error_original=0
max_scaled_error_refresh=0
```

for the tested values. All 64 plans on each corpus matrix used the direct
single-source path; the Matrix Market loader had already canonicalized
duplicates. Duplicate-source summation is covered separately by unit tests.

| matrix | plan prepare ms | prepared / direct refresh | refresh speedup | measured break-even refreshes | plan bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| sherman5 | 1.1398 | 0.06852 | 14.59x | 3.92 | 195,756 |
| raefsky3 | 43.5187 | 0.08969 | 11.15x | 5.11 | 8,151,104 |
| venkat25 | 12.6074 | 0.08992 | 11.12x | 8.16 | 1,232,032 |
| cfd1 | 15.0788 | 0.08652 | 11.56x | 6.76 | 1,605,956 |
| thermal1 | 5.9038 | 0.01440 | 69.44x | 5.07 | 91,304 |
| nd3k | 3118.8024 | 0.10073 | 9.93x | 3.86 | 521,182,904 |

The reported break-even count amortizes **plan preparation only**. If the dual
topology does not already exist, its preparation cost must also be included.
In the intended G3/G4 architecture the dual topology is a reusable global
symbolic object, so both views are relevant.

The `nd3k` stress case also exposes the memory boundary: 43,338,226 local
structural entries across 64 heavily overlapping regions require about
497 MiB for the current plans before local numerical values themselves are
materialized. Production code should avoid retaining unnecessary identity
source-pointer arrays for direct-source plans and should not build all
high-overlap region plans simultaneously without a memory policy.

## Interpretation boundary

G3d demonstrates the value of **prepared symbolic reuse**. It does not prove
that ABTM has an intrinsic advantage over every possible prepared CSR
implementation: a CSR-based implementation can also cache local source
positions once the structure is known.

The ABTM-specific evidence is strongest in G3a/G3b, where dual topology and
word-level set algebra reduce structural graph work. G3c/G3d then show the
appropriate interface boundary: use ABTM to discover and prune structure, but
allow conventional local sparse/numeric layouts downstream.

The G3 benchmark programs use independent deterministic seed streams. Therefore
absolute region totals from G3a, G3b, G3c, and G3d are reproducible within each
checkpoint but are not intended for direct stage-to-stage equality comparisons.

## G3 architectural decision

Promote the following direction:

1. dual ABTM topology for graph/region operations;
2. deterministic frontier growth over `support(A) union support(A^T)`;
3. explicit overlap/multiplicity metadata;
4. exact materialization of local sparse patterns when required;
5. prepared source-address maps for repeated numerical refresh;
6. evidence-driven memory policy for large or highly overlapping region sets.

Do not promote:

- universal ABTM replacement of CSR local matrices;
- one-shot local pattern extraction as a performance claim;
- all-region materialization without a memory budget;
- G3d refresh ratios as proof that bitmap value storage is superior.

## Next: G4

G4 should compare ABTM symbolic/numeric ILU(0) work against canonical CSR
ILU(0), while preserving the G2/G3 separation:

- symbolic candidate intersection and pruning from topology;
- conventional prepared numerical factors where that wins;
- exact factor/apply cross-checks and original-system residual verification;
- preparation, memory, factorization, and repeated-apply accounting;
- no automatic promotion from a single matrix or occupancy threshold.
