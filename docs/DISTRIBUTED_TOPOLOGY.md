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

- owned-row extraction
- global-to-local/ghost column renumbering
- rank-local CSR
- distributed SpMV reference assembled from simulated halo exchange
- ABTM-assisted halo extraction cross-check

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