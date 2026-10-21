# HyBIT 0.9 H1-C3K7-B — exact original-CSR integrity verification

Baseline: `develop/0.9.0` commit `bc04895d53c6a77a73f50b7a8a7c3944f0cd32ae` (K7-A).

## Scope

K7-A streams one CSR chunk at a time from root to noncontiguous MPI owners. K7-B adds a **collective, diagnostic-only, independent exact comparison** between the root's original `Csr32Matrix` and the already scattered `AssignmentOwnedCsrRows`. It does not switch solver algorithms, repartition, or introduce a production collective in the SpMV fast path.

The previous K6/K7-A SpMV check reconstructs a numerical reference from local Owned CSR, so it cannot alone prove that the owned matrix matches root's original coefficient data. K7-B verifies the missing invariant.

## Exactness contract

Rank 0 retains the **original** CSR until verification has completed; this happens after `scatter_assignment_from_root` and before `drop(matrix)`. It independently enumerates source row IDs selected by each owner, derives cumulative rank-local `row_ptr`, traverses original global columns, and emits `f64::to_bits()` for coefficients. It **does not use** `AssignmentOwnedCsrRows::from_global`, the scatter chunk builder, or an independent hash as the oracle.

Each destination compares the supplied reference frames against its *already received* Owned CSR. Checks cover:

- Global row ID and rank ownership, ascending order, missing and duplicate rows
- Exact local row-pointer boundaries, row widths, and total NNZ
- Global column IDs in the original within-row order, including duplicates
- Exact IEEE-754 `u64` coefficient bit patterns (including signed zero)

Every rank validates CSR shape before communication, collectively checks global owner-map lengths, and bitwise-compares its owner labels to root's in bounded broadcast frames. Root separately derives expected per-rank row/NNZ counts from the input. Counts are compared after an `all_gather` **before point-to-point frames**. On mismatch, ranks collectively fail, preventing the next target transfer. Root's own rank is directly compared without network traffic. Root sends independent reference frames to one destination at a time, each bounded to 16,384 elements (128 KiB for `u64`). No full global CSR is allocated on non-root ranks; root remains the only original-CSR reader.

This is a deterministic exact equality test of the data in this MPI verification epoch; no collision probability or tolerance is involved. Its trust boundary is root's parsed `Csr32Matrix` (it does not independently validate the source Matrix Market text or disk). Both source and received values must be finite according to the existing input contract; NaN payload comparison is outside the supported input set.

## Timings and memory

`verify_original_max_ms` / `verify_ms` measure diagnostic verification separately from `scatter_max_ms`, `halo_setup_max_ms` and SpMV. Do **not** claim the measured verification cost is part of the production solve timing or that the verification is a free operation. The root builds one target's row-index list during checking, and sends fixed-size frames; this is not a formal bound on root process RSS. K7-A's one-chunk scatter memory characterization is unchanged.

`verified_original_rows` and `verified_original_nnz` must equal the input dimensions, while `exact_original_verified=true` indicates the collective test completed. The original matrix is dropped only after verification passes.

## Native Windows verification

In `D:\Work\hybit\hybit` with native Microsoft MS-MPI, first use the packaged `APPLY_HYBIT_K7B_EXACT_CSR.ps1` `-PreflightOnly` and `-Apply`. Successful Apply includes baseline identity checks, Rustfmt, `-D warnings` Clippy, full distributed library regressions, example unit tests, release build, and Source Integrity gate.

Use the unchanged runner (`bench-k6-root-scatter.ps1`) on `thermal1.mtx` with 4 ranks, and `boneS01.mtx` with 4 and 8 ranks. Only one rank opens Matrix Market data; existing Equal, Region Growth, and Multi-source policies remain available. Check log fields `exact_original=true`, `verified_rows`, `verified_nnz`, `verify_ms`; verify that the new CSV columns match the printed values.

## Negative tests

The new Rust unit tests check an exact original-CSR pass, one-bit coefficient corruption, signed-zero bit corruption, changed column index, malformed row pointer, and wrong owned global row. The MPI integration checks root/non-root handshakes and positive full equality. Negative distributed fault injection and root parser independent checks can be added in a later diagnostic checkpoint if required.

## Limitations

- The O(n) owner map is still replicated; K7-C will address ownership scaling.
- Root's full CSR is retained until this optional check completes. K7-A streaming bounds only the scatter's temporary CSR chunks.
- Exact verification adds additional root-to-peer traffic and source-CSR traversal and should not be enabled silently in production runs.
- No change to PCG, GMRES, BiCGSTAB, factorization, or automatic solver routing.
