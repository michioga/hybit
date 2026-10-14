# HyBIT 0.9 H1-C1 — Prepared field-aware ABTM region plans (experimental)

**Base:** `develop/0.9.0` at `10b6eec`, H1-B previously passed Rust/Clippy/source-integrity and native C/C++/Fortran DLL smoke, and was pushed with a clean worktree.

## Purpose and contract

- `PreparedFieldAwareTopology::new(&csr, &layout)` validates matching square dimensions and constructs `AbtmDualTopology` once. It borrows immutable CSR storage and owns a copy of the mixed-field layout; no numerical value array is duplicated.
- `.prepare_region(&seeds, hops)` and `.prepare_regions(&[seeds], hops)` reuse that topology, call the exact H1-B field-block-closed undirected region algorithm and G3's existing `prepare_local_numeric_plan` to bind a local `A[R,R]` pattern to CSR value positions.
- Each `PreparedFieldAwareLocalRegion` owns its region, local symbolic pattern and value-address plan; `.gather_values(&new_values)` remaps changing numerical values without redoing graph/topology preparation. **Requires the same CSR row pointers and column indices, in identical entry order**. A same-length values slice is not proof of structural identity. Retire and re-prepare on any structural change; matrix structure is guarded by Rust borrowing for the source matrix.
- `.validate_block_owners(owners, rank_count)` delegates to H1-B block continuity validation; it **does not** partition the matrix or construct/execute an MPI HaloPlan. Real owned/ghost Halo integration and MPI cost metrics belong to H1-C2.
- This is **opt-in Rust-only experimental preparation**. Existing Rust `hybit::solve`, C DLL ABI, C++ wrappers and Fortran ISO_C_BINDING symbols remain **unchanged**. Portable prepared handles for C/C++/Fortran are required before exposing this optimization to those languages; do not confuse existing unprepared C ABI H1-B methods with this new prepared path.
- No local Cholesky/LU/LDLT is invoked; no Krylov routing changes, no nonlinear contact or Schur implementation, no numerical speedup claims.

## Correctness and cost acceptance gates

1. Rust tests: two different nonempty regions and empty region share one ABTM topology; local numeric values match a separate direct CSR `A[R,R]` reference, including directed edges and structurally stored numerical zero.
2. Repeated numeric refresh with unchanged CSR ordering; reject incorrect values length, layout mismatch, seed mask size mismatch, and split-owner assignment.
3. Preserve existing H1-B tests (50 matrix, 5 FFI), solver public API regressions, `cargo fmt` and `cargo clippy -D warnings`.
4. Run source-integrity and `git diff --cached --check`, review staged changes. Record `topology_stats().total_metadata_bytes()` and local `numeric_plan_stats().plan_bytes` before attributing any memory/speed benefit.
5. C/C++/Fortran H1-B runtime remains compatible. This patch changes no FFI files, exports or calling conventions. Full multi-language prepared API is deferred to a separately validated ABI checkpoint.

## Windows validation

```powershell
cd D:\Work\hybit\hybit
cargo fmt --all -- --check
cargo test --release --locked -p hybit-matrix -p hybit-ffi -p hybit
cargo clippy --locked -p hybit-matrix -p hybit-ffi -p hybit --all-targets -- -D warnings
pwsh -NoProfile -File tools/gates/source-integrity-gate.ps1
git diff --cached --check
git status --short
```

Do **not** call H1-C1 complete or commit before the developer machine executes these gates. Next checkpoint H1-C2: field-block ownership and actual existing MPI HaloPlan/partition contract, with 1/2/4/8-rank MS-MPI correctness and real communication volume accounting.
