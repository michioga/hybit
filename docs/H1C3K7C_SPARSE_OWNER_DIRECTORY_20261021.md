# H1-C3K7-C — Sparse Owner Directory

## Intent

The K6/K7-A/B noncontiguous MPI SpMV path broadcast an O(global DOF) owner vector to all ranks. K7-C adds a separate opt-in path retaining the global owner vector **only on root** during partition/scatter/verification. Every rank holds sorted `owned_globals` in the local CSR and `(remote_global, remote_owner)` pairs for its referenced off-rank columns. Halo setup no longer broadcasts/checks a full owner map.

## Algorithm

1. Root reads the original Matrix Market once and chooses Equal, ABTM Region Growth or Multi-source ownership. All ranks enter the same ordered collective sequence.
2. Root scans CSR with O(P) counts and announces rows/nnz. For each destination, Root materializes one CSR chunk and computes only its sorted unique remote referenced columns with owner labels. These are sent with the CSR; the temporary chunk is dropped before the next destination. Root's local chunk is last.
3. Every rank locally validates CSR dimensions, owned row sortedness, remote lookup coverage and owner ranges. The local `SparseOwnerDirectory` uses binary search. It does not allocate or broadcast O(global DOF) owners on nonroot processes.
4. An independent verifier on root generates original row IDs, `row_ptr`, columns, `f64::to_bits`, and remote owner identities from the original CSR/assignment. Remote ranks compare these reference frames against received data. Verification is outside SpMV timing.
5. `prepare_assignment_owned_rows_sparse` builds the same deduplicated reciprocal MPI_Alltoallv Halo plan as K3 via `directory.owner_of`, then calls the unchanged `spmv_assignment`.

## Caveats

- Root still owns all original CSR and global owner labels at setup; the new path is not a parallel graph partitioner or parallel file loader.
- Per-rank directory payload is O(number of unique referenced remote DOFs), potentially greater than a dense owner vector for pathological sparsity. Report the actual bytes rather than claiming unconditional memory savings.
- Row-ID and column-ID metadata remains 64-bit, local indices remain u32, MS-MPI Count is i32 bound, as in K6.
- No changes to production contiguous MPI-PCG, Krylov, preconditioners or automatic routing.
- Large problems require chunking, collective failure agreement and further memory/latency profiling; transient MPI and allocator overhead are not covered by the reported logical payload bytes.

## Acceptance

- Rustfmt, strict Clippy, distributed lib tests, example tests, release build and Source Integrity.
- 4-rank thermal1 and 4/8-rank boneS01 native MS-MPI on each of three ownership methods; original bit-exact verification PASS and `worst_rel <= 1e-11`.
- Compare `sparse_owner_bytes_sum` with `replicated_owner_baseline_bytes`; distinguish summed payload memory from per-rank maximum and root peak RSS.
- Preserve the legacy K6/K7-B path as a separate regression suite before commit.
