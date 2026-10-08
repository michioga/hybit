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