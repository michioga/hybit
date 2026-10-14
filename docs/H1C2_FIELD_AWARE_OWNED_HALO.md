# H1-C2 — Field-aware owned/ghost and existing MPI Halo adapter (experimental)

Baseline: `bbfafc2` on `develop/0.9.0` (2026-10-14). This patch is an unvalidated candidate until developer-machine gates pass.

## Implementation and deliberately narrow scope

- `field_aligned_contiguous_partition(layout,ranks)` chooses deterministic contiguous rank cuts on **whole field-block boundaries**. It minimizes absolute distance to each uniform-DOF cumulative target while reserving at least one block per remaining rank; it does not globally optimize cuts, nnz, peer count, or compute time.
- `validate_field_assignment(layout,assignment)` checks arbitrary per-DOF ownership using H1-B's `field_block_owners`. It **does not** convert noncontiguous owner arrays into native MPI partitions; H1-C2 supports the existing contiguous transport only.
- `validate_field_contiguous_partition(layout,partition)` checks each cut without allocating a per-DOF owner vector. **Different fields on one entity may have different owners**, but one field block may never be split.
- `PreparedFieldAwareHalo::prepare` delegates to the existing `build_contiguous_halo_plans`, `prepare_rank_local_csrs`, and `partition_telemetry`. Ghost DOFs are unique per receiver, and send/recv peer lists are reciprocal, precisely as in G8-A1. It is a reference/in-memory preparation; the full CSR is available to all ranks only in the small native MPI validation fixture.
- **Real MPI**: `mpi_input::prepare_owned_rows(&mpi, prepared.partition(), &rank_owned_rows)` already works with arbitrary valid `ContiguousPartition::from_offsets`. H1-C2 does not reimplement MPI collectives, change `HaloPlan` representation, or automatically repartition PCG/FGMRES.
- `crates/hybit-distributed/examples/mpi_field_aware_halo.rs` compares rank-local native `prepare_owned_rows` against the reference plan on each rank, runs **actual MS-MPI SpMV**, checks numerical equivalence to global CSR, and prints payload, ghosts, peers, nnz/DOF imbalance, and timings against standard DOF-balanced contiguous ownership telemetry.
- Existing Rust `hybit::solve` and C/C++/Fortran ABI are untouched. Full mixed-field physics, contact nonlinear mechanics, block-aware noncontiguous halo transport, AMG and convergence benefits remain outside this patch.

## Validation gates (all pending until executed on Windows)

```powershell
cd D:\Work\hybit\hybit
cargo fmt --all -- --check
cargo test --release --locked -p hybit-distributed -p hybit-matrix -p hybit
cargo clippy --locked -p hybit-distributed -p hybit-matrix -p hybit --all-targets -- -D warnings
pwsh -NoProfile -File tools/gates/source-integrity-gate.ps1
```

Native MS-MPI must use the original `x86_64-pc-windows-msvc` target and supported MS-MPI, LLVM/libclang setup. Run `cargo clippy --locked -p hybit-distributed --features mpi --example mpi_field_aware_halo -- -D warnings`, then compile with `cargo build --release --locked -p hybit-distributed --features mpi --example mpi_field_aware_halo` and execute `mpiexec -n 1|2|4|8 target\release\examples\mpi_field_aware_halo.exe <ranks>` separately, checking each rank-count's `PASS` line. Native compile-only is **not** equivalent to runtime validation.

If the existing `C2` / `B3` gates are run, keep their actual numerical and performance evidence separate; this small native smoke test is not a production-scaling performance result. For a full MPI production integration, the remaining work is a block-aligned rank-owned matrix assembly path and partition policy selection validated against representative FEM matrices.

## Acceptance / rollback

PASS requires 4 new Rust unit tests, existing solver/FFI regression, strict Clippy, manifest/SHA, genuine MPI 1/2/4/8 ranks with expected reciprocal halo / true scalar SpMV agreement, and numerical + bandwidth telemetry. If any gate fails, leave the change uncommitted and diagnose. `git restore --staged` / `git restore` and remove added files from an uncommitted patch, or revert a later checkpoint normally. No repository push, tag, release or crates.io publication is performed by the installer.
