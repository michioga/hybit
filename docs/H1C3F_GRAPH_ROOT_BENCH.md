# HyBIT 0.9 — H1-C3F / Graph-selected root-owned MPI SpMV benchmark

**Experimental, opt-in only.** Starting checkpoint: `8601e93` on `develop/0.9.0`.

## Intent

H1-C3E validated root-owned decision/broadcast, but its native positive smoke
selected only Equal. H1-C3F adds a standalone MPI example specifically designed
so Graph is selected for 2, 4 and 8 ranks, while rank 1 follows the
`SingleRank` control path. No library, production solver, C ABI, or MPI transport
is modified.

## Test fixture and isolation

- 84 physical DOF blocks, two components per block, with different entity
  kinds and field IDs; 168 scalar DOFs, not a physically assembled FEM model.
- Eight internally dense structural regions with sizes in blocks
  `[12, 11, 10, 11, 10, 10, 10, 10]`; one directed bridge per region maintains
  genuine inter-rank Halo dependencies even at aligned Graph cuts.
- Rank 0 creates the **only global CSR** and computes both the no-calibration
  baseline and the Graph candidate. All ranks have the global *field metadata*
  and generate only their owned CSR rows from an identical deterministic
  synthetic row generator. This is a test harness, **not** distributed matrix
  ingestion or source-file scattering.
- Calls `select_and_broadcast_cost_partition_mpi` for the baseline and
  `cost_aware_contiguous_partition_scoped` on Rank 0 plus
  `broadcast_root_cost_selection_mpi` for the candidate. All-rank Bcast/field
  validation precedes `mpi_input::prepare_owned_rows` and `spmv_local`.
- Uses `search_radius_blocks=3`, 10,000 expected calls, DOF imbalance cap 1.5
  and nnz imbalance cap 1.8. The **synthetic cost calibration** (`20,000 ns`
  baseline, `10,000 ns` Graph, `100,000 ns` preparation budget) is strictly
  illustrative to exercise the matching-calibration and `GraphSelected` path.
  These are **not** independently measured performance data, and the
  `Matched` guard verifies identity, not the truth of the supplied timings.

## Correctness gates

For ranks 2/4/8, require GraphSelected, a distinct offset vector relative to
Equal/Field, fewer unique Ghost values than baseline, matching field-block
boundaries, positive nonroot communication, all-rank owned coverage, exact
sent/received ghost accounting and numerical CSR SpMV agreement (`abs_err <=
1e-10`). Rank 1 must select SingleRank.

The approximate structural expected values below were checked against the
known H1-C3 boundary-pressure heuristic **before** native MPI execution; they
are not measured native results and the executable itself remains definitive.

| MPI ranks | Baseline Ghosts | Graph Ghosts | Baseline blocks/rank cuts | Graph blocks/rank cuts |
|---:|---:|---:|---|---|
| 2 | 24 | 2 | 0,42,84 | 0,44,84 |
| 4 | 68 | 4 | 0,21,42,63,84 | 0,23,44,64,84 |
| 8 | 156 | 8 | 0,10,21,31,42,52,63,73,84 | 0,12,23,33,44,54,64,74,84 |

## Native Windows gate

First set the standard MS-MPI / LLVM environment in the **same PowerShell
session** (this is necessary for a freshly built `mpi-sys` on this machine):

```powershell
$env:CLANG_PATH = 'C:\Program Files\LLVM\bin\clang.exe'
$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
$env:MSMPI_INC = 'C:\Program Files (x86)\Microsoft SDKs\MPI\Include\'
$env:MSMPI_LIB64 = 'C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\'

cargo fmt --all -- --check
cargo clippy --locked -p hybit-distributed --features mpi `
    --example mpi_root_graph_selection_bench -- -D warnings
cargo build --release --locked -p hybit-distributed --features mpi `
    --example mpi_root_graph_selection_bench

$exe = (Resolve-Path '.\target\release\examples\mpi_root_graph_selection_bench.exe').Path
foreach ($ranks in @(1, 2, 4, 8)) {
    & mpiexec -n $ranks $exe $ranks
    if ($LASTEXITCODE -ne 0) { throw "H1-C3F $ranks-rank failed" }
}

pwsh -NoProfile -File tools/gates/source-integrity-gate.ps1
git diff --cached --check
```

The example alternates baseline/Graph execution order, performs 5 warmups and
20 timed measurements per policy, and records the **median of per-call maximum
rank latency** including blocking Halo/SpMV. It also reports max-rank native
setup time and Rank 0 selector time. Timings are descriptive and *are not
part of PASS criteria*; repeated independent trials on real matrices are
necessary before any speedup or break-even claim.

## Explicit limits and next work

- Does not demonstrate physical FEM element coupling, actual mixed-field
  assembly, MPI-PCG/FGMRES, production scaling, or lower resident memory in
  a complete distributed reader.
- Does not establish **real** calibration provenance: synthetic timing evidence
  is intentionally used only to force the experimental Graph decision.
- Rank 0 still requires a global CSR and shared global field metadata; workers
  retain only their owned-row CSR but generate it procedurally in this example.
- The timing comparison is a tiny synthetic benchmark; scheduling noise, MPI
  startup, thread affinity and real matrix structures can reverse the ranking.
- No tag, Release, automatic solver routing or crates.io publication.

**Status:** source patch prepared; Windows Rust and native MS-MPI validation
PENDING. Record actual results before committing this checkpoint.

## Native Windows/MS-MPI results — 2026-10-15 (confirmed; supersedes PENDING status)

The H1-C3F example was exercised on native Windows / MS-MPI after applying
H1-C3F FIX2, with Clang/LLVM and Microsoft MPI include/library environment
variables configured **in the same PowerShell session**. The original patch
and MSRV correction are staged but **not committed** at validation time.

### Acceptance gate evidence

- Initial package Preflight/Apply: **PASS**; Source Integrity **289 manifest / 287 SHA-256 PASS**.
- `cargo fmt --all -- --check`: **PASS**.
- Initial strict MPI Clippy flagged **two** uses of `is_multiple_of`, which are
  incompatible with project MSRV Rust 1.73 despite the installed rustc 1.99.
  FIX1 preflight rejected a mismatched *pre-rustfmt* source SHA without
  changing staged files. FIX2 matched the *post-rustfmt* SHA and changed
  `block.is_multiple_of(7)` to `block % 7 == 0`, and
  `trial.is_multiple_of(2)` to `trial % 2 == 0`: **Preflight/Apply PASS**.
- One later Clippy attempt stopped inside `mpi-sys`/bindgen (`libclang error`)
  because LLVM/MS-MPI environment was not established in that session;
  after restoring it, strict `cargo clippy --locked -p hybit-distributed
  --features mpi --all-targets -- -D warnings`: **PASS**.
- `cargo build --release --locked -p hybit-distributed --features mpi
  --example mpi_root_graph_selection_bench`: **PASS**.
- `mpiexec` with **1, 2, 4, 8 ranks: all four native benchmark gates PASS**.
  Baseline and selected solutions both had `abs_err=0.000e0` against the
  serial CSR reference, and `workers_global_csr=0` at all sizes.
- After FIX2, the user also ran Source Integrity **289/287 PASS** and
  `git diff --cached --check` **PASS**, before the final Clippy/MPI run.
- **Not reported in this H1-C3F run:** a fresh complete Rust workspace
  regression suite, MPI-PCG/FGMRES solve, physical FEM problem, confidence
  intervals/repeated independent runs, or an actual measured calibration
  being supplied to the Graph policy. Do not imply they passed.

### Actual Graph-selected MPI results

The matrix has **168 scalar DOFs, 3,552 nnz**, and 84 two-component blocks.
Timing values are **milliseconds**. Both SpMV times are medians of 20
samples of per-call maximum-rank latency, following 5 warmups. Ghost counts
are global sums. Displayed percentage is `(baseline - graph) / baseline`.

| MPI ranks | Policy / decision | Baseline ghosts | Selected ghosts | Ghost reduction | Baseline max-rank SpMV median (ms) | Selected max-rank SpMV median (ms) | Observed SpMV reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Equal / SingleRank | 0 | 0 | N/A | 0.001900 | 0.001900 | 0.0% |
| 2 | Graph / GraphSelected | 24 | 2 | 91.7% | 0.005400 | 0.005250 | 2.8% |
| 4 | Graph / GraphSelected | 68 | 4 | 94.1% | 0.004900 | 0.004250 | 13.3% |
| 8 | Graph / GraphSelected | 156 | 8 | 94.9% | 0.007900 | 0.006300 | 20.3% |

All ranks reported `guard=Matched`, zero numerical reference error, and
`workers_global_csr=0`. **Graph selection** is demonstrated in a real MPI
execution at 2/4/8 ranks, but it was *authorized by illustrative, externally
supplied synthetic calibration values*, not proven-profitable measured
calibration. `Matched` means the scope identity matched, not that its timings
were independently verified.

| Ranks | Rank-0 selection (ms) | Baseline native setup max (ms) | Graph native setup max (ms) | Baseline offsets | Graph offsets |
|---:|---:|---:|---:|---|---|
| 1 | 0.146800 | 0.040600 | 0.019600 | `[0,168]` | `[0,168]` |
| 2 | 0.237200 | 0.063300 | 0.027900 | `[0,84,168]` | `[0,88,168]` |
| 4 | 0.291400 | 0.158300 | 0.048700 | `[0,42,84,126,168]` | `[0,46,88,128,168]` |
| 8 | 0.333600 | 0.433100 | 0.073700 | `[0,20,42,62,84,104,126,146,168]` | `[0,24,46,66,88,108,128,148,168]` |

Measured setup timings are only illustrative. The rank-0 selection cost
(0.24–0.33 ms for 2/4/8 ranks) must be amortized over repeated SpMV calls;
comparing a few microseconds per SpMV without setup would overstate the
benefit. This one tiny synthetic workload is not an end-to-end MPI solver
speedup, and the measured numbers should **not** be substituted for the
fixture's synthetic calibration without an actual matching calibration run.

### Checkpoint decision and follow-on

The exercised H1-C3F acceptance criteria (true GraphSelected decision, root
broadcast, zero-error CSR/Halo equality, reduced ghost volume, and native
MPI execution) **passed**. Complete-regression recheck plus docs and
SOURCE_SHA256 validation are recommended before committing. Do **not**
release, tag, publish, or enable production auto-routing from these results.
Next, use actual consistently measured per-policy/max-rank timings on a
realistic matrix and evaluate whole-solve preparation-vs-iteration cost.
