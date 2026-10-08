# G8 distributed topology and MPI-host integration

HyBIT 0.9 starts the G8 distributed-memory track.

The first checkpoint deliberately introduces **no MPI dependency**. G8-A1
freezes the ownership and halo semantics in a deterministic single-process
reference implementation so later MPI and ABTM-assisted paths have an exact
cross-check target.

## G8-A1 contract

The reference implementation lives in `hybit-distributed`.

### Index spaces

- global DOF: `u64`
- rank: `u32`
- rank-local owned/ghost index: `u32`

The current global `Csr32Matrix` reference input still has `u32` column indices.
Using `u64` for distributed global IDs avoids making that single-process
storage choice the long-term distributed identifier contract.

### Ownership

G8-A1 uses a balanced contiguous partition:

```text
rank 0 : [offset[0], offset[1])
rank 1 : [offset[1], offset[2])
...
```

This is a **reference ownership model**, not a claim that contiguous partitioning
is suitable for production FEM decomposition.

A graph/mesh partitioner will be a separate policy.

### Halo semantics

For each rank:

1. owned matrix rows are scanned;
2. a column owned by another rank becomes a receive ghost;
3. duplicate references collapse to one ghost;
4. receive requests are grouped by owner rank;
5. the owning rank receives the exact reciprocal send list;
6. local vector layout is:

```text
[ owned values | ghost values ]
```

Ghosts are deterministically ordered by `(owner rank, global DOF)`.

For every peer plan:

```text
send_globals
send_owned_indices
recv_globals
recv_extended_indices
```

are retained so an MPI transport can pack and unpack without rediscovering
topology during every SpMV.

## Why MPI is not in A1

The first failure mode to eliminate is semantic ambiguity, not communication
latency.

Before introducing `MPI_Isend`, `MPI_Irecv`, neighborhood collectives, or
global reductions, HyBIT needs a reference answer for:

- who owns a DOF;
- which remote values a rank requires;
- which local values a rank must send;
- local/ghost renumbering;
- peer reciprocity.

The A1 implementation is therefore testable on ordinary Windows CI without an
MPI installation.

## ABTM role

A1 uses CSR row scans as the correctness reference.

The next topology checkpoint will compare an ABTM-assisted boundary/halo
extraction path against the CSR reference. ABTM is useful here as a topology
engine; it is not assumed to be the numerical distributed SpMV storage.

## Planned G8 sequence

### G8-A1 — reference partition/halo topology

- `ContiguousPartition`
- deterministic halo extraction
- reciprocal send/receive plans
- global `u64` / local `u32` index contract
- no MPI runtime dependency

### G8-A2 — rank-local operator preparation

Implemented on the 0.9 development branch:

- owned-row extraction;
- global-to-local/ghost column renumbering;
- rank-local CSR over `[owned | ghosts]`;
- transport-free halo exchange using the exact reciprocal G8-A1 send/recv lists;
- distributed SpMV reference assembled from rank-local applies;
- bit-for-bit SpMV cross-check against the serial CSR reference;
- ABTM-topology halo extraction with exact plan equality against the CSR path;
- partition telemetry for stored cut references, unique communication volume,
  peer relations, neighbor count, owned-DOF balance, and local-nnz balance.

The repository also contains:

```text
cargo run --release -p hybit-distributed --example distributed_probe -- <matrix.mtx> <ranks>
```

This is a preparation/quality probe, not yet an MPI benchmark. Timing of CSR
versus ABTM halo preparation is diagnostic only and must not be interpreted as
a backend promotion result from one matrix or one machine.

### G8-B — transport boundary

- communicator/transport contract
- blocking reference halo exchange first
- explicit global sum reduction required by Krylov methods
- MPI adapter kept separate from topology

### G8-C — distributed Krylov

- distributed dot/norm
- PCG first for the established SPD path
- overlap communication with owned-row work only after the blocking reference
  is correct

### G8-D — FEM/MPI host validation

- multi-rank structural FEM input
- MPI + OpenMP/Rayon process/thread ownership guidance
- rank-local memory and communication telemetry
- numerical cross-check against the single-process result

The release path will not promote an MPI backend until residuals and physical
results match the established single-process reference.
## Partitioning direction after G8-A2

HyBIT will not treat balanced contiguous ownership as the production
partitioner. The next topology experiments will use ABTM as the structural
engine for partition/coarsening/refinement candidates and will retain external
graph partitioners as comparison baselines.

The comparison target is broader than edge cut alone. For distributed FEM,
HyBIT will record at least:

- stored cross-rank CSR references (`cut_nnz`);
- unique halo values communicated per refresh (`communication_volume`);
- neighbor-rank count;
- rank-local nnz balance;
- partition/preparation time and memory;
- repeated halo/SpMV cost.

The objective is to improve end-to-end distributed sparse execution rather than
claim a universal edge-cut advantage.

MPI transport remains a later layer. HyBIT will use the Rust `mpi` crate and
will not hard-code a particular MPI implementation into the distributed core.
MS-MPI is one development/test implementation; compatible MPI implementations
remain part of the intended portability boundary.
## G8-A3 ABTM partition prototype

G8-A3 moves from halo extraction to partition ownership itself.

The first prototype introduces `PartitionAssignment`, which stores an arbitrary
rank owner for every global DOF. This is intentionally separate from
`ContiguousPartition`, because production graph partitions do not preserve
contiguous global numbering.

`abtm_region_grow_partition()` uses `AbtmDualTopology` (`A union A^T` semantics
through row and column topology) to build deterministic, exactly balanced
regions. It is a baseline ABTM graph-growth prototype, **not yet a multilevel
replacement for METIS or Scotch**.

The quality harness evaluates arbitrary owner labels with the same metrics:

```text
cut_nnz
communication_volume
directional_peer_relations
max_neighbors
owned_dof_imbalance
local_nnz_imbalance
```

Run:

```text
cargo run --release -p hybit-distributed --example partition_quality_probe -- <matrix.mtx> <ranks>
```

An optional zero-based owner-label file can be supplied:

```text
cargo run --release -p hybit-distributed --example partition_quality_probe -- <matrix.mtx> <ranks> <owners.txt>
```

The owner file is one rank id per global DOF, matching the basic `gpmetis
*.part.N` convention. This gives HyBIT a dependency-free way to compare ABTM
quality against external partitioners before any automatic policy is promoted.

The next partition work should add refinement/coarsening and test real FEM
matrices. A claim that ABTM exceeds METIS/Scotch requires held-out evidence on
quality, setup/memory, halo preparation, and repeated distributed execution,
not one favorable cut result.
## G8-A4 balanced multi-source growth

An initial, uncommitted G8-A4 experiment tried exact-balance pair swaps on top
of G8-A3. On `boneS01` / 4 partitions it required about 7.16 seconds for only
384 candidate evaluations because every candidate recomputed full partition
telemetry. It also failed the intended communication objective:

```text
G8-A3 communication volume : 17695
pair-swap result            : 17698
peer relations              : 12 -> 12
max neighbors               : 3 -> 3
```

It improved `cut_nnz` further (`248866 -> 246898`), but that is not the problem
A4 was meant to solve. The pair-swap experiment was therefore discarded before
checkpointing.

The replacement G8-A4 attacks the observed root cause: G8-A3 filled one rank
completely before starting the next and required 11 disconnected restarts on
`boneS01`.

`abtm_balanced_multisource_partition()` instead:

1. chooses graph-spread initial seeds using repeated ABTM multi-source BFS;
2. grows all rank regions concurrently with independent FIFO frontiers;
3. permits at most one claimed DOF per rank per sweep;
4. preserves the same exact balanced DOF targets;
5. uses a restart only if a rank frontier is exhausted.

This remains a single-level prototype. The real-FEM probe decides whether
multi-source growth materially reduces communication volume / rank adjacency
before multilevel coarsening and refinement are introduced.
## G8-A5 external partitioner baselines

G8-A5 freezes the comparison boundary before multilevel ABTM work.

`partition_graph_export` converts the same structural graph used by ABTM
(`A union A^T`, with self-loops removed and duplicate adjacency collapsed) into
two unweighted graph files:

```text
<prefix>.metis.graph
<prefix>.scotch.grf
```

The METIS file uses 1-based adjacency identifiers and counts each undirected
edge once in the header. The SCOTCH file uses graph version 0, base 0, format
flag `000`, and stores the total number of adjacency arcs.

This intentionally compares all partitioners on the same unweighted structural
graph. HyBIT then evaluates the produced owner labels with its own matrix-aware
telemetry (`cut_nnz`, unique halo communication volume, rank adjacency, and
nnz balance). Therefore METIS/SCOTCH-reported edge cut is not substituted for
HyBIT's distributed execution metrics.

Examples:

```text
cargo run --release -p hybit-distributed --example partition_graph_export -- matrix.mtx out/prefix

gpmetis out/prefix.metis.graph 4

gpart 4 out/prefix.scotch.grf out/prefix.scotch.map
cargo run --release -p hybit-distributed --example scotch_map_to_labels -- \
  out/prefix.scotch.map out/prefix.scotch.labels 127224 4
```

Both resulting zero-based owner-label files can then be supplied to
`partition_quality_probe`.

G8-A5 does not add METIS or SCOTCH as HyBIT dependencies. They remain external
baseline tools. The next ABTM multilevel design is driven by the measured gap,
not by assumptions about partition quality.
### G8-A5 measured boneS01 baseline

The external baseline harness was run on WSL Ubuntu 26.04 using METIS
`5.1.0.dfsg-8` and SCOTCH `7.0.11`, on the exact common unweighted structural
graph exported by HyBIT:

```text
matrix       : boneS01.mtx
DOFs         : 127224
matrix nnz   : 5516602
graph edges  : 2694689
partitions   : 4
```

HyBIT re-evaluated every owner-label file with the same distributed telemetry:

| partitioner | cut_nnz | communication_volume | peer relations | max neighbors | owned DOF imbalance | local nnz imbalance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| contiguous | 288964 | 17256 | 6 | 2 | 1.000000 | 1.015148 |
| ABTM G8-A4 multi-source | 199002 | 16959 | 12 | 3 | 1.000000 | 1.011033 |
| METIS | 115730 | 10935 | 12 | 3 | 1.014840 | 1.017228 |
| SCOTCH | 112792 | 10762 | 12 | 3 | 1.009998 | 1.014705 |

The comparison changes the next ABTM priority:

- G8-A4 is materially better than contiguous ownership, but still has a large
  cut/halo-volume gap to mature graph partitioners.
- At four ranks, METIS and SCOTCH also produce 12 directional peer relations
  and a maximum of three neighbors. Therefore rank adjacency is not the main
  quality gap in this case.
- Exact equal-DOF ownership is stricter than the external baselines. METIS used
  an owned-DOF imbalance of about 1.015 and SCOTCH about 1.010. The next ABTM
  design should support an explicit balance tolerance rather than forcing exact
  cardinality when doing so harms separator quality.
- ABTM's local-nnz balance remains competitive, so multilevel work should retain
  nnz/load balance as a secondary constraint while aggressively reducing the
  separator and halo surface.

The METIS result also gives a useful consistency check: METIS reported an edge
cut of `57865`, while HyBIT measured `cut_nnz = 115730`, exactly twice that
value for this symmetric structural graph. METIS reported communication volume
`10935`, exactly matching HyBIT's independently computed value. This validates
the G8-A5 graph-export and telemetry interpretation for the measured case.

The next partition algorithm should therefore be **multilevel ABTM**:

1. topology-aware coarsening/aggregation;
2. coarse balanced partitioning;
3. uncoarsening;
4. boundary refinement with a configurable balance tolerance;
5. identical HyBIT telemetry against METIS/SCOTCH on held-out FEM matrices.