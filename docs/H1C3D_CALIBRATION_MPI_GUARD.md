# HyBIT 0.9 — H1-C3D / Calibration identity and native MPI consensus

**Experimental, explicit opt-in only.** Starting checkpoint `c711b99` on `develop/0.9.0`. No change to `hybit::solve`, production MPI PCG/FGMRES, C ABI, or existing automatic selector.

## Why this exists

H1-C3B confirmed that fewer ghosts do not guarantee faster MPI SpMV. H1-C3C added a cost-aware early exit, but accepted unscoped `PartitionCostCalibration` values from any timing run. During the Goodwin_010 probe, one supplied "baseline 8,000 ns" came from **Equal**, although the selected baseline was **Field** (7,900 ns); the supplied graph time came from radius **32**, while a radius **5** candidate was being selected. That is not valid evidence for a speedup.

## New APIs

- `capture_partition_calibration(...)` associates **caller-supplied, previously measured** maximum-rank SpMV values and graph preparation budget with a deterministic identity of CSR dimensions, full CSR sparsity pattern, numeric values, complete ordered physical DOF metadata, MPI rank count, graph search radius, actual Equal/Field baseline selection, and an **explicit caller-defined** machine/toolchain/MPI/thread environment tag. This API **does not benchmark** or validate the provenance of the numbers.
- `cost_aware_contiguous_partition_scoped(...)` refuses to execute ABTM graph construction if the scoped calibration is missing or any identity field differs. It returns a block-safe Equal/Field baseline with a separate `CalibrationGuardDecision::{Missing,ScopeMismatch,Matched}`. Matching evidence delegates to the previously validated H1-C3C selector, including its economic early rejection and exact Halo/load checks. The old unscoped API remains available for explicit experiments, but the new guarded API **ignores** unscoped timings in `options.calibration`.
- `verify_cost_partition_mpi_consensus(...)` (feature `mpi` only) is an opt-in collective gate to run **on every rank before Halo peer traffic**. It compares scope fingerprints, cost options, calibration values, selected policy and decision via fixed-size MPI Allreduce MIN/MAX. After confirming equal offset lengths, it compares **the entire actual offsets** using fixed-count MPI Allgather. If any differs, all participating ranks reject rather than entering a reciprocal Halo exchange with incompatible partitions.

Identity hashing is deterministic FNV-1a with domain-separation; it is **not cryptographically collision-resistant**, and does not verify the truth of externally supplied environment tags or performance measurements. It is a guard against accidental differences and wrong calibration reuse, not an adversarial integrity or security feature. All ranks must enter the same collective epoch; a rank that crashes, skips the collective or races another MPI call can still hang an MPI program. For production, a root-owned decision with a broadcasted partition and robust error coordination remains future work.

## Validation: run on Windows native MS-MPI

After `APPLY_HYBIT_H1C3D.ps1 -PreflightOnly` and `-Apply`, run the usual `cargo fmt --check`, `cargo test --release --locked -p hybit-distributed -p hybit-matrix -p hybit -p hybit-ffi`, strict Clippy, Source Integrity and `git diff --cached --check`.

Initialize the same PowerShell session for MS-MPI bindgen:

```powershell
$env:CLANG_PATH = 'C:\Program Files\LLVM\bin\clang.exe'
$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
$env:MSMPI_INC = 'C:\Program Files (x86)\Microsoft SDKs\MPI\Include\'
$env:MSMPI_LIB64 = 'C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\'

cargo clippy --locked -p hybit-distributed --features mpi --example mpi_cost_guard_smoke -- -D warnings
cargo build --release --locked -p hybit-distributed --features mpi --example mpi_cost_guard_smoke
$exe = (Resolve-Path '.\target\release\examples\mpi_cost_guard_smoke.exe').Path
foreach ($n in 1, 2, 4, 8) {
    & mpiexec -n $n $exe $n
    if ($LASTEXITCODE -ne 0) { throw "H1-C3D MPI $n-rank numerical gate failed" }
}
foreach ($n in 2, 4, 8) {
    & mpiexec -n $n $exe $n --mismatch-env
    if ($LASTEXITCODE -ne 0) { throw "H1-C3D MPI $n-rank rejection gate failed" }
}
```

The MPI positive smoke uses replicated small test CSR and 2-component field blocks, then calls the **existing** `mpi_input::prepare_owned_rows` / `spmv_local` and cross-checks actual results against serial CSR, peer data traffic and global ghosts. Negative runs deliberately modify the environment tag of rank 1 and must reject collectively before Halo. These are synthetic correctness smoke tests, **not MTX/mixed-physical-field validation, nor a distributed PCG benchmark**. Values in the synthetic smoke calibration are illustrative only.

## Acceptance criteria and limitations

- Matching calibration and identical inputs admit deterministic selection; missing/mismatched scope skips ABTM on all local calls.
- Exact per-rank partition offsets agree before any Halo traffic.
- No regression in baseline/graph numerical equivalence. Existing H1-C3C tests continue to pass.
- No performance improvement, MPI deadlock freedom under arbitrary error conditions, production distributed matrix ingest, full PCG speedup, or heterogeneous real FEM metadata is claimed from this checkpoint.
- No release, tag or `cargo publish` without explicit user approval.

**Status:** proposed patch; Windows/MS-MPI tests **pending**. Do not commit/push until the user-provided native logs PASS.

## Native Windows validation — 2026-10-15 (confirmed)

This entry **supersedes the earlier pending-testing status** for the H1-C3D patch only. The reference source checkpoint remains `c711b99` until the user commits the tested staged work.

### Windows acceptance gates

- `APPLY_HYBIT_H1C3D.ps1 -PreflightOnly` and `-Apply`: **PASS**. The install script staged the source and document changes, without committing or pushing.
- `cargo fmt --all -- --check`: **PASS**.
- `cargo test --release --locked -p hybit-distributed -p hybit-matrix -p hybit -p hybit-ffi`: **PASS**. `hybit-distributed`: **45/45** (including 6 new H1-C3D guard tests); `hybit-matrix`: **52/52**; `hybit-ffi`: **5/5**; existing HyBIT public API suites and doctest: **PASS**.
- Strict Clippy `--all-targets -- -D warnings` on the usual four packages, without `mpi`: **PASS**.
- Strict MPI-enabled Clippy `-p hybit-distributed --features mpi --example mpi_cost_guard_smoke -- -D warnings`: **PASS**.
- MPI-enabled release example build: **PASS** using explicit `CLANG_PATH`, `LIBCLANG_PATH`, `MSMPI_INC` and `MSMPI_LIB64` on the tested Windows MS-MPI environment.
- Source Integrity: **284 manifest entries / 282 verified SHA-256, PASS**; `git diff --cached --check`: **PASS**.

### Native MS-MPI positive smoke

| Ranks | Guard | Selector decision | Selected partition | Ghosts | Sent / received | Maximum absolute SpMV error |
|---:|---|---|---|---:|---:|---:|
| 1 | Matched | SingleRank | Equal | 0 | 0 / 0 | 0 |
| 2 | Matched | UnamortizedPreparation | Equal | 4 | 4 / 4 | 0 |
| 4 | Matched | UnamortizedPreparation | Equal | 8 | 8 / 8 | 0 |
| 8 | Matched | UnamortizedPreparation | Equal | 16 | 16 / 16 | 0 |

Every positive run reported `PASS H1-C3D`, exact global-CSR numerical agreement and consistent traffic. The exercised 2/4/8-rank path **rejects Graph at the cost gate**; Graph selection under actual MPI and root-coordinated broadcasting were **not** demonstrated.

### Native MS-MPI negative smoke

With rank 1 deliberately supplied a different calibration environment tag (`--mismatch-env`), all tested process counts (**2, 4, 8**) reported `PASS H1-C3D injected_environment_mismatch ... rejected_collectively=true`, returning success without entering incompatible Halo peer traffic.

**Not exercised by this user log:** deliberate partition-offset corruption, different option values on different ranks, rank failure or collective-order divergence, Graph-selected MPI SpMV, real heterogeneous physical FEM DOF metadata, production MPI-PCG/FGMRES behavior, or end-to-end performance improvements. The FNV-1a identity remains a noncryptographic accidental-mismatch guard, and caller-supplied environment and timing provenance cannot be independently authenticated.

**Checkpoint status:** functionality and exercised MPI guards **validated**; docs/hash changes should be checked and **committed/pushed only after user review**. No GitHub Release, tag or crates.io publication has been authorized.
