# HyBIT 0.9 H1-C3 — ABTM graph-aware, field-aligned *contiguous* partition

Base `bbe5cef` (`develop/0.9.0`). This is a proposed patch, NOT yet validated by Windows Rust or native MPI.

## Mathematical and MPI contract

- The public solver ABI, PCG/FGMRES routing, local factors and numerical SpMV remain unchanged. The new functions are explicit Rust `hybit-distributed` preparation APIs.
- Existing native MS-MPI `mpi_input::prepare_owned_rows` and `HaloPlan` assume contiguous physical global DOF ranges. Therefore H1-C3 refines **only contiguous boundaries aligned to complete field blocks**; it does not falsely convert arbitrary `PartitionAssignment` into such a HaloPlan.
- ABTM provides **structural** row masks. A stored zero remains an edge. Duplicate CSR coordinates are collapsed for symbolic boundary pressure. The pressure proxy counts one contribution per crossed *block boundary* for each unique directed structural edge, not actual MPI transferred values.
- The H1-C2 field-aligned partition is the *fallback*. Boundary movements are bounded by the explicit block radius and per-rank DOF upper limit `max(baseline max rank DOFs, ceil(1.15 * average DOFs))`. The heuristic is not a globally optimal partitioner and cannot promise that all block/rank configurations improve.
- The proposed candidate is accepted only after using the existing CSR reference `build_contiguous_halo_plans` and `partition_telemetry`: communication_volume (unique received f64), cut_nnz and directional_peer_relations must all be **no worse** than H1-C2, with at least one strictly lower. Accepted candidates may still have worse nnz imbalance or elapsed time; report those separately. This policy never promises a faster solve.
- Graph preparation requires a global CSR reference and ABTM topology in this checkpoint. This is a setup/reference experiment; **not** a production rank-owned distributed graph partitioner. Production distributed graph preparation must avoid requiring a global CSR on each rank.

## Gates

1. `cargo fmt --all -- --check`
2. `cargo test --release --locked -p hybit-distributed -p hybit-matrix -p hybit -p hybit-ffi`. Expected *32* tests in `hybit-distributed`, other suites unchanged.
3. `cargo clippy --locked -p hybit-distributed -p hybit-matrix -p hybit -p hybit-ffi --all-targets -- -D warnings`
4. `pwsh -NoProfile -File tools/gates/source-integrity-gate.ps1`
5. `git diff --cached --check`
6. Native MS-MPI `cargo clippy` and `cargo build --release` with `--features mpi --example mpi_field_partition_compare`, then `mpiexec -n 1/2/4/8`. Confirm full rank-dependent error, reciprocal payload counts, no increases in exact ghost/cut/peer metrics relative to H1-C2.

A successful MPI smoke proves only the given 200-DOF synthetic fixture. Follow with realistic mixed-DOF FEM meshes at 1/2/4/8 ranks, report setup costs, per-rank maximum SpMV time, memory, repeated trials and speed relative to exact baselines; do **not** infer production speedup from these tests.

## Next (after successful developer-machine gates)

- H1-C3B: quality/peak-memory benchmark at large size and realistic FEM matrices. Compare with H1-C2 and equal-DOF baselines; update heuristic only with evidence. Preserve fallback.
- Noncontiguous `PartitionAssignment` requires a distinct MPI `OwnedIndexMap` / `RankLocalCsr` translation and collective request ownership adapter, tested before it is used in the native MPI data path.
