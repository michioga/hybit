# H1-C3K6 — Root-only CSR distribution with noncontiguous assignment

Scope: **experimental MPI input path**, no production solver routing, no ABI or solver changes. Builds on the committed `8a51d662` (K5-B/O1/C). This is a data-distribution milestone, not a claim of performance superiority over Equal or other partitions.

## Gap closed

K5-C reads a complete `Csr32Matrix` on each rank before extracting owned rows. K6 reads Matrix Market **only on rank 0**, selects Equal / ABTM Region Growth / ABTM Multi-source **only on rank 0**, broadcasts the common owner labels, and sends CSR rows and their original numerical coefficients to the owning ranks. After distribution, the root drops the full matrix. Non-root ranks never construct the full CSR. The established `mpi_assignment::prepare_assignment_owned_rows` and `spmv_assignment` remain unchanged.

- Owner labels are still **replicated O(global DOF)** on each rank; K6 does not yet implement fully distributed partitioning/owner directory.
- Root temporarily holds the complete input **plus all prepared rank chunks**, increasing its peak setup memory. Do not interpret this as parallel I/O or a memory-optimal root streamer.
- Each rank stores only its `AssignmentOwnedCsrRows`, prepared local CSR, and Halo buffers for numerical SpMV. The experiment reports logical allocated-payload estimates, **not process peak RSS**.
- MPI `Count` limits are checked before root sends, and pre-transfer errors are collectively checked. All ranks use the same executable and `mpiexec` launch.
- The example checks reference products computed directly from each rank's original received numerical rows with an independently generated global vector and checks all received ghost values. This validates the distributed SpMV path without requiring every rank to own a global reference matrix.
- Matrix Market parser supports formats accepted by `hybit_matrix::read_matrix_market`; this probe does **not** assume SPD or PCG convergence.

## Execution and data

```powershell
cd D:\Work\hybit\hybit
& .\benchmarks\scripts\bench-k6-root-scatter.ps1 `
  -RepoRoot D:\Work\hybit\hybit `
  -Matrix D:\Work\boneS01.mtx `
  -Ranks 4,8 -Repeats 64 -OutputDir D:\Work -Prefix k6_boneS01
```

Run `thermal1.mtx` first as a quick smoke check if desired. Each policy/rank pair uses a **fresh native Microsoft MS-MPI process**. `*_summary.csv` has one row per pair; no raw history is overwritten. Timing for SpMV includes the existing per-call validation collective in `spmv_assignment`, so the value is **not pure local numeric compute**. Report separate halo and numeric medians, too. The three timers and any stage-wise maxima should not be added as if they were one global critical path.

## Validation and acceptance

1. Native 2/4/8-rank runs exit 0; each supported method covers 64 repeated SpMV calls and all row products/ghosts pass the tolerance.
2. `sum(local_nnz) == global_nnz`, `sum(sent_f64) == sum(received_f64) == sum(ghosts)`.
3. `cargo fmt --all -- --check`; `cargo clippy --locked -p hybit-distributed --features mpi --all-targets -- -D warnings`; `cargo test --release --locked -p hybit-distributed --features mpi --lib`; example tests and release binary build all pass.
4. Baseline Source Integrity `329/327`; expected new `333/331`; no automatic commit/push.

Follow-ups: root streaming with bounded chunks, remove replicated owner labels with a distributed directory, gather full-vector-free residual verification, and then integrate with true distributed PCG/GMRES. Compare real numerical matrices and memory pressure, not any particular CPU cache.
