# HyBIT 0.9 — H1-C3E Root-coordinated MPI cost-aware partition

**Experimental/explicit opt-in**. Patch baseline `75fe1b1` on `develop/0.9.0`. No changes to production `hybit::solve`, MPI-PCG, FGMRES, C ABI, ABTM SpMV, MPI transport, or default partitioner.

## Problem / contract

H1-C3D verified that all ranks independently computing a cost-aware partition can **detect disagreement** (scope/decision/offsets) prior to Halo. H1-C3E instead computes the policy and offsets **on MPI rank 0 only** and distributes those offsets to all ranks. Workers require a matching global `FieldAwareDofLayout`, identical options and environment tag, but **do not need a global CSR matrix** for partition selection; afterwards each worker supplies only its own `OwnedCsrRows` to the existing `mpi_input::prepare_owned_rows`.

New MPI-feature APIs (exported from `hybit-distributed`):

- `select_and_broadcast_cost_partition_mpi(root_matrix, layout, options, environment_tag, root_calibration)` — all ranks enter collectively; `root_matrix` and `root_calibration` are provided only on rank 0. Rank 0 calls the H1-C3D scoped selector. No graph search or baseline CSR telemetry runs on workers.
- `broadcast_root_cost_selection_mpi(root_result, layout, options, environment_tag)` — lower-level collective distributing a root-owned **`Result<ScopedCostPartitionResult, DistributedTopologyError>`**; root must pass `Some(&result)` even if selection failed; others pass `None`. Do not use `?` on root before entering the collective, because other ranks could wait forever.
- `RootMpiCostSelection` — uniform decoded selected partition, selected/baseline kind, decision, calibration guard and root-generated identity on every rank. This is **not a timing or performance certification**.

## Collective sequence

1. Fixed-count Allreduce of locally valid inputs and root-result ownership. Before variable-length data, fixed-count MIN/MAX Allreduce compares full-field layout digest, tag digest and all relevant options (including old unscoped timing options). The digest is deterministic FNV-1a with the H1-C3D domains; it is **not** a cryptographic guarantee.
2. Rank 0 broadcasts a selection-success flag. Even a root selector error is returned on all ranks before Halo.
3. Rank 0 broadcasts a fixed 12-u64 header: DOFs, ranks, offset count, selected kind, guard, decision, matrix-pattern/value fingerprints, layout/environment fingerprints, radius and baseline kind. Header validation is reduced collectively.
4. Root broadcasts the **complete** `ranks + 1` ownership offsets, not a hash. Each rank reconstructs `ContiguousPartition` and checks `validate_field_contiguous_partition`, requiring every physical field block to have a single owner. Validation failure is reduced collectively; **no rank enters peer Halo**.
5. Only when every rank succeeds, existing MPI-owned CSR/Halo and `spmv_local` may execute. The root still owns a replicated global CSR; workers in the smoke test construct only rank-owned rows. A true distributed reader and memory scaling remain separate future work.

These collectives still require all MPI ranks to enter the **same epoch in the same order**. A crash, early return before the collective, inconsistent MPI world or unrelated concurrent MPI call can hang. The guard is an accidental-divergence safety aid, not full fault tolerance. H1-C3D scoped timings and caller-provided environment provenance are still not independently verified. Transmitted metadata digests are noncryptographic.

## Native Windows verification (PENDING)

From the ZIP folder:

```powershell
& .\APPLY_HYBIT_H1C3E.ps1 -RepoRoot 'D:\Work\hybit\hybit' -PreflightOnly
& .\APPLY_HYBIT_H1C3E.ps1 -RepoRoot 'D:\Work\hybit\hybit' -Apply
```

Then run Cargo formatting, release tests of `hybit-distributed`, `hybit-matrix`, `hybit`, `hybit-ffi`; strict Clippy for normal targets; Source Integrity; staged diff --check. Expect **45** existing `hybit-distributed` tests for non-MPI builds; 2 additional digest tests are compiled when the `mpi` feature is on. Expect **287 manifest files / 285 SHA-256** after applying.

Set the known-good native MS-MPI/LLVM build variables in the current PowerShell session:

```powershell
$env:CLANG_PATH = 'C:\Program Files\LLVM\bin\clang.exe'
$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
$env:MSMPI_INC = 'C:\Program Files (x86)\Microsoft SDKs\MPI\Include\'
$env:MSMPI_LIB64 = 'C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\'

cargo clippy --locked -p hybit-distributed --features mpi --all-targets -- -D warnings
cargo build --release --locked -p hybit-distributed --features mpi --example mpi_root_cost_partition_smoke
$exe = (Resolve-Path '.\target\release\examples\mpi_root_cost_partition_smoke.exe').Path
foreach ($ranks in 1, 2, 4, 8) {
    & mpiexec -n $ranks $exe $ranks
    if ($LASTEXITCODE -ne 0) { throw "H1-C3E $ranks-rank positive failure" }
}
foreach ($ranks in 2, 4, 8) {
    foreach ($fault in '--mismatch-env', '--mismatch-options', '--mismatch-layout', '--root-error', '--invalid-offset') {
        & mpiexec -n $ranks $exe $ranks $fault
        if ($LASTEXITCODE -ne 0) { throw "H1-C3E $ranks-rank $fault negative failure" }
    }
}
```

Smoke input: 64 mixed-width 1/2/3/4-component synthetic *physical field blocks*, a deterministic three-entry circulant CSR stencil, input vector and serial analytic SpMV cross-check. Rank 0 alone constructs a full CSR matrix and scoped calibration; workers generate **only their selected owned rows** after the broadcast. The positive suite checks rank-owned DOF totals, actual MPI Halo send/receive/ghost conservation, and SpMV absolute numerical error. Negatives deliberately inject an inconsistent environment tag, options, physical layout, root selector error and split-field ownership. They must return success only when all ranks reject *before* Halo. The synthetic calibration is intentionally unamortized; the test does **not** demonstrate Graph-selected MPI speedup.

## Limits / next phase

- Root-only **partition decision**, not root-less mesh metadata or matrix distribution; all ranks hold complete physical field layout for validation.
- Does not yet independently exercise a genuinely Graph-selected MPI run, full iterative FEM solve, real physical mixed-field `.msh`, large memory scaling, or root failure/disconnection during MPI communication.
- No claim of Graph performance benefit. H1-C3B had Graph fastest in only 1 of 8 confirmed cases.
- Future milestones: root-owned calibration capture with full error coordination, actual graph-selected MPI path; native mixed FEM owned-row ingest and PCG/FGMRES end-to-end comparisons. Release/tag/publish require explicit user approval.

**Gate status:** package prepared; Windows native Rust/MS-MPI results not yet provided. Do not commit/push until the user verifies the gates.
## Windows/MS-MPI validation — 2026-10-15 (confirmed)

This section **supersedes the initial `PENDING` gate status above** for the H1-C3E opt-in root-broadcast experiment. It records actual developer-machine results; no H1-C3E commit was made at the time of this validation. Reference base: `75fe1b1`, `develop/0.9.0`.

### Acceptance gates

- Patch `-PreflightOnly` and `-Apply`: **PASS**; staged without commit/push.
- `cargo fmt --all -- --check`: **PASS**.
- `cargo test --release --locked -p hybit-distributed -p hybit-matrix -p hybit -p hybit-ffi`: **PASS**. Distributed **45/45** (non-MPI feature), Matrix **52/52**, FFI **5/5**, existing HyBIT public API suites and doctest **PASS**.
- Strict normal `cargo clippy --locked ... --all-targets -- -D warnings`: **PASS**.
- Strict `cargo clippy --locked -p hybit-distributed --features mpi --all-targets -- -D warnings`: **PASS**.
- MPI-enabled release example build with known-good LLVM 23.1.3/MS-MPI environment: **PASS**.
- Source Integrity **287 manifest / 285 verified SHA-256: PASS**; `git diff --cached --check`: **PASS**.
- `cargo test --release --locked -p hybit-distributed --features mpi --lib`: **PASS, 72/72** (includes both new H1-C3E feature-gated digest tests plus existing MPI-feature tests). The first attempt failed while `mpi-sys`/`bindgen` rebuilt without the previously required LLVM/MS-MPI environment variables; after setting `CLANG_PATH`, `LIBCLANG_PATH`, `MSMPI_INC`, and `MSMPI_LIB64`, the build and all 72 tests succeeded. No HyBIT source change was needed.

### Native MS-MPI positive smoke — four cases PASS

| Ranks | Policy | Cost decision | Guard | Total owned DOFs | Ghosts | Sent/received | Workers holding global CSR | Abs. error |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | Equal | SingleRank | Matched | 160 | 0 | 0/0 | 0 | 0 |
| 2 | Equal | UnamortizedPreparation | Matched | 160 | 4 | 4/4 | 0 | 0 |
| 4 | Equal | UnamortizedPreparation | Matched | 160 | 8 | 8/8 | 0 | 0 |
| 8 | Equal | UnamortizedPreparation | Matched | 160 | 16 | 16/16 | 0 | 0 |

The smoke constructs the global synthetic CSR **only on root**; worker-owned rows are generated after receiving the selected offsets. The common `FieldAwareDofLayout` remains present on each rank. This validates the test execution path, not a production distributed matrix ingest or large memory scaling.

### Native MS-MPI negative smoke — fifteen cases PASS

At each of **2, 4, 8 ranks**, all five injected fault modes reported `rejected_collectively=true` and returned success **without proceeding to incompatible Halo**:

- `--mismatch-env`: divergent environment tag.
- `--mismatch-options`: divergent cost-selection options.
- `--mismatch-layout`: divergent physical field layout.
- `--root-error`: deliberately failed root-side selection.
- `--invalid-offset`: root-broadcast ownership boundary splitting a physical field block.

This demonstrates coordinated rejection for those deterministic injections, **not general MPI fault tolerance** (crashed/stalled rank or collective-order divergence remains out of scope).

### Interpretation and next gate

- **H1-C3E functionality validated** for the executed normal/negative MPI smoke and ordinary Rust test matrix. Production routing, existing MPI-PCG/FGMRES execution, and C ABI remain unchanged.
- All positive selections were **Equal**, so native **Graph-selected MPI** and cost-model-derived acceleration remain **unverified**. The numbers used for synthetic smoke calibration are illustrative, not performance measurements.
- Synthetic mixed-width (1/2/3/4-component) blocks do not substitute for **real heterogeneous FEM field metadata**. End-to-end iterative solver speed, real `.msh` owned-row input, root memory peak, distributed memory scalability, and concurrent/faulty collectives remain untested.
- Feature-gated MPI unit tests are now **72/72 PASS**. Complete the validation-document and SHA-256 gates, then checkpoint commit/push on `develop/0.9.0`. Do not create tags, GitHub Release or publish crates.io without user authorization.
