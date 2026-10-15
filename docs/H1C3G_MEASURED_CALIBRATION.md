# HyBIT 0.9 — H1-C3G native measured calibration and economic gate

**Experimental benchmark, explicit opt-in, no production solver change.** Base `a118565` on `develop/0.9.0`. This checkpoint adds **one MPI example** and a validation document, but does not modify `hybit::solve`, MPI PCG, the distribution API, or the C ABI.

## Objective and experimental controls

H1-C3F demonstrated root-selected Graph ownership with an **illustrative, invented timing calibration**. H1-C3G removes that assumption from the benchmark. A root-generated ABTM/field-graph proposal is actually timed using native MPI owned-row CSR/Halo SpMV; baseline and candidate are measured under the same matrix, field layout, MPI rank count, search radius and environment tag. The observed maximum-rank median SpMV times and measured graph preparation budget are passed to the already implemented `capture_partition_calibration` and `select_and_broadcast_cost_partition_mpi`. A second, held-out MPI run checks the policy **after** selection.

The workload is still the synthetic 168-DOF, 3,552-nnz eight-cluster, 84 two-component-field fixture from H1-C3F. This is not a benchmark of actual FEM assembly, full Krylov solve, or production data distribution. Only Rank 0 holds the global CSR; workers build their owned rows directly from fixture metadata. Every rank retains global field-block metadata.

### Execution order (collectively on all ranks)

1. Rank 0 computes and broadcasts the valid **uncalibrated Equal/Field baseline**. All ranks prepare native MPI-owned CSR/halo.
2. Rank 0 constructs a Graph candidate via `field_graph_contiguous_partition` **without a cost calibration**; its actual local preparation duration is captured. Status and exact offsets are broadcast; every rank validates field boundaries collectively, before preparing Graph native MPI-owned CSR/halo. This is a **measurement candidate**, not a prematurely accepted policy.
3. After 5 warmups, alternate baseline/candidate and collect 40 per-call **maximum-rank** native SpMV times, including MPI communication. Verify global numerical reference errors, ghost traffic, send/receive balance, owned DOFs and no full CSR on workers. Native CSR/halo setup duration is also recorded; positive Graph-vs-baseline setup difference is added to the preparation budget. A negative setup difference does not reduce the root Graph-search budget.
4. Capture **measured** `PartitionCostCalibration` in nanoseconds. The H1-C3D scope fingerprints full numeric CSR, exact structure, physical layout, rank count, radius, actual baseline and explicit environment tag; rank 0 alone calls the cost-aware selector. H1-C3E broadcasts its actual accepted Equal/Field/Graph decision and validates offsets. Slow or unamortized Graph must fall back; Graph adoption is **not required** for PASS.
5. Independently prepare the selected policy and measure 5 warmups + **20 held-out samples**. Verify numerical and traffic integrity again. Output per-policy calibration medians, decision, holdout median, selected offsets, root Graph preparation, native setup deltas, break-even iteration estimate, and modeled total times for requested SpMV calls.

### Cost model and interpretation

For N SpMV calls:

`baseline_total_ms ≈ baseline_native_setup_ms + N * measured_baseline_spmv_ms`

`graph_total_ms ≈ measured_root_graph_preparation_ms + graph_native_setup_ms + N * measured_graph_spmv_ms`

Let ΔT = baseline SpMV latency minus Graph latency, and `P = max(0, graph_native_setup - baseline_native_setup) + root_graph_preparation`. If ΔT > 0, break-even is `ceil(P / ΔT)` calls; otherwise no finite break-even. The selector uses the measured nanosecond equivalents and supplied N (default 10,000), with its existing structural dominance and load-balance gates. It may reject Graph even when the simple time-based model favors it. The candidate and policy timings are not wall-time predictions for real MPI PCG/FGMRES.

**Important limitations:** the calibration experiment itself runs both paths and repeats 40 probes, so the real amortization threshold **including calibration overhead** is strictly higher. Setup samples are one-shot and sensitive to noise; timings are integer-nanosecond rounded up. The fixture is very small (microsecond timing noise is substantial), repeated independent runs and confidence intervals are not yet available. The held-out measurements reduce simple in-sample selection bias but do not prove stable speedup. The project MSRV is Rust 1.73: do not use `is_multiple_of` or newer stabilized APIs. Accidental scope mismatch guards are noncryptographic and caller-supplied tags are not independently authenticated.

## Native Windows validation instructions

```powershell
cd D:\Work\hybit_0_9_h1c3g_measured_calibration_20261015
& .\APPLY_HYBIT_H1C3G.ps1 -RepoRoot D:\Work\hybit\hybit -PreflightOnly
& .\APPLY_HYBIT_H1C3G.ps1 -RepoRoot D:\Work\hybit\hybit -Apply

cd D:\Work\hybit\hybit
$env:CLANG_PATH='C:\Program Files\LLVM\bin\clang.exe'
$env:LIBCLANG_PATH='C:\Program Files\LLVM\bin'
$env:MSMPI_INC='C:\Program Files (x86)\Microsoft SDKs\MPI\Include\'
$env:MSMPI_LIB64='C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\'
cargo fmt --all -- --check
cargo clippy --locked -p hybit-distributed --features mpi --all-targets -- -D warnings
cargo build --release --locked -p hybit-distributed --features mpi --example mpi_root_measured_calibration_bench
$exe=(Resolve-Path '.\target\release\examples\mpi_root_measured_calibration_bench.exe').Path
foreach($r in @(1,2,4,8)) { & mpiexec -n $r $exe $r 10000; if($LASTEXITCODE -ne 0){throw "MPI ranks=$r failed"} }
foreach($r in @(2,4,8)) { & mpiexec -n $r $exe $r 1; if($LASTEXITCODE -ne 0){throw "MPI no-amortization ranks=$r failed"} }
cargo test --release --locked -p hybit-distributed -p hybit-matrix -p hybit -p hybit-ffi
pwsh -NoProfile -File tools/gates/source-integrity-gate.ps1
git diff --cached --check
git status --short
```

Run MS-MPI environment variable setup in the **same PowerShell session** as Cargo. If `mpi-sys`/bindgen reports `libclang` errors, check `CLANG_PATH`, `LIBCLANG_PATH`, `MSMPI_INC`, and `MSMPI_LIB64` before changing source code.

### Acceptance

- Format, strict MPI-feature Clippy, release example build, Source Integrity **291 manifest / 289 SHA-256**, and staged whitespace checks PASS.
- Native 1/2/4/8 MPI calls with expected 10,000 and 2/4/8 MPI calls with expected 1 PASS numerical/traffic/ownership invariants. The 1-call run should never select Graph when preparation cost is nonzero; other decisions are subject to *actual measured performance*, not predetermined.
- Record actual per-rank time and decision, including any slower-Graph or no-amortization fallback. DO NOT misreport a rejected graph candidate as a failure.
- **Status: source package prepared, Windows compile/MPI runtime and actual calibrated decision PENDING.** Do not commit/push until reviewed. No release, tag, crates.io publish, or production auto-routing is authorized.
## Native Windows/MS-MPI validation — 2026-10-15 (confirmed; supersedes PENDING status)

The Windows operator ran the H1-C3G installer, strict MPI-enabled Clippy,
release example build, seven native MS-MPI jobs, complete non-MPI release
regression, Source Integrity, and staged diff checks. **All executed gates
passed.** This is a functional checkpoint for the **experimental benchmark**,
not a validated general-purpose automatic partition policy.

### Native acceptance gates

- Base `a118565` on `develop/0.9.0`; Preflight and Apply PASS, with no commit
  or push at time of validation.
- `cargo fmt --all -- --check`: PASS.
- `cargo clippy --locked -p hybit-distributed --features mpi --all-targets -- -D warnings`: PASS.
- Release example `mpi_root_measured_calibration_bench`: compile PASS.
- `cargo test --release --locked -p hybit-distributed -p hybit-matrix -p hybit -p hybit-ffi`: PASS.
  Distributed **45/45**, matrix **52/52**, FFI **5/5**, HyBIT public API
  suites and its doctest PASS.
- Source Integrity **291 manifest / 289 verified SHA-256 PASS**;
  `git diff --cached --check` PASS. Six files staged, no commit/push yet.
- Native MS-MPI: four `expected_calls=10000` runs on **1/2/4/8** ranks,
  three `expected_calls=1` runs on **2/4/8** ranks: **7/7 PASS**.
  For all seven: calibration guard `Matched`, zero reference SpMV error
  (baseline, Graph candidate and selected holdout), and `workers_global_csr=0`.

### Measured calibration (N = 10,000 requested SpMV calls)

The values below are **measured maximum-rank per-SpMV medians**, not invented
calibration values. The reported modeled total time is **calculated** using
these samples, not a measured 10,000-call execution.

| MPI ranks | Baseline / Graph calibration (us) | Ghosts baseline -> candidate | Root candidate preparation (ms) | Estimated break-even calls | Selected | Holdout selected (us) | Modeled total baseline / Graph (ms) |
|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 1.800 / 1.800 | 0 -> 0 | 0.0691 | None | Equal / SingleRank | 1.800 | 18.0280 / 18.0801 |
| 2 | 4.100 / 4.000 | 24 -> 2 | 0.0887 | 887 | Graph / GraphSelected | 2.250 | 41.0509 / 40.1276 |
| 4 | 12.200 / 11.400 | 68 -> 4 | 0.0878 | 110 | Graph / GraphSelected | 2.800 | 122.1589 / 114.1182 |
| 8 | 10.150 / 8.800 | 156 -> 8 | 0.1706 | 127 | Graph / GraphSelected | 5.200 | 101.9115 / 88.2250 |

The Graph candidate aligned the same cuts as H1-C3F (2-rank `[0,88,168]`,
4-rank `[0,46,88,128,168]`, and 8-rank
`[0,24,46,66,88,108,128,148,168]`). All seven runs reported
`extra_native_setup_ns=0`. This means the positive-increment budget did not
charge additional native setup; it does **not** mean absolute setup time is zero.

### One-call conservative fallback (N = 1)

| MPI ranks | Baseline / candidate calibration (us) | Root candidate preparation (ms) | Estimated break-even calls | Selector decision / selected | Holdout selected (us) | Modeled total baseline / Graph (ms) |
|---:|---:|---:|---:|---|---:|---:|
| 2 | 4.100 / 4.050 | 0.0794 | 1,588 | UnamortizedPreparation / Equal | 1.900 | 0.04690 / 0.10795 |
| 4 | 11.800 / 10.800 | 0.1432 | 144 | UnamortizedPreparation / Equal | 2.900 | 0.19370 / 0.18410 |
| 8 | 9.950 / 8.500 | 0.1472 | 102 | UnamortizedPreparation / Field | 8.450 | 0.42085 / 0.21430 |

All fallback decisions PASS by design: the original economic gate charges
Graph preparation conservatively and **does not credit a reduction in native
Halo setup**. Thus the reported total-time *model* can favor Graph at N=1
for 4/8 ranks while the existing gate correctly returns the current baseline
under its more conservative cost model. This is a policy-versus-model
inconsistency to quantify, not evidence of incorrect CSR arithmetic.

### What is NOT yet established

1. **Performance stability:** the calibration and held-out timing medians
   diverge strongly, especially at 4 ranks (Graph calibration 11.4 us vs
   selected holdout 2.8 us). The holdout compares *only the selected policy*;
   it is not a simultaneous matched holdout baseline-vs-Graph comparison.
   Repeat independent sessions/runs, rotate measurement order, report
   per-rank variance and confidence intervals before relying on decisions.
2. **Complete up-front economics:** the 40-sample calibration consumes
   time, and Graph candidate construction already happened before the final
   policy was selected. Furthermore, the guarded final selector **re-runs the
   Graph refinement**, whose incremental time is not separately charged in
   this example's `root_graph_prepare_ms` and modeled totals. These costs can
   raise the true operational break-even substantially. The printed
   `break_even_calls` includes only the positive native-setup delta plus the
   one measured root candidate-prepare budget.
3. **Realistic scaling:** the 168-DOF synthetic fixture is too small to prove
   faster production MPI-PCG/FGMRES or a benefit on real FEM mixed-field MTX.
   There is no proof of production-grade auto-routing, resilient arbitrary MPI
   failure recovery, or end-to-end solves.

**H1-C3G checkpoint status:** exercised implementation correctness and
native measured-calibration decision branches **validated**. Documentation,
source-hash update, and staging verification should precede an explicit
checkpoint commit/push. No tag, release, crates.io publish, or enabling
production auto-routing is authorized. Suggested follow-on: repeated paired
holdout across independent MPI runs, measured full preparation/selection
including calibration overhead, and real mixed-field FEM/PCG end-to-end runs.
