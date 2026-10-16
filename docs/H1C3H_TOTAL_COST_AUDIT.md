# HyBIT 0.9 H1-C3H — complete probe/decision cost ledger and paired holdout

**Status: Windows/MS-MPI validation PENDING.** Source baseline: `7ee08a4`
(`develop/0.9.0`, committed H1-C3G). This is an opt-in benchmark, not a
change to solver routing or the MPI runtime. Never claim production speedup
from this small synthetic test.

## Why

H1-C3G measured real MPI calibration instead of fabricated speedups, but its
reported break-even omitted the initial candidate probe (including benchmark
warmups/measurements) and the second graph preparation inside the guarded
selector. Its selected holdout was not paired with an independent baseline
holdout. This stage audits those costs without changing the selector.

## How the audit works

`mpi_root_cost_ledger_bench` uses **the identical 168-DOF/3552-nnz synthetic
problem** and scoped decision/options as H1-C3G. Rank 0 is the only rank with
full CSR and the only rank to construct the graph candidate; other ranks
construct owned-row CSR and Halo using the existing native MPI path.

1. **Probe wall time** (`probe_wall_ms`) starts just before the uncalibrated
   baseline selection. It includes the first MPI baseline broadcast, one root
   Graph candidate exploration and offsets broadcast, both native CSR/Halo
   preparations, the input vector construction, 5 warmups + 40 paired timed
   SpMVs per alternative and traffic/numerical checks. This phase includes
   incidental host and MPI synchronization costs, not merely graph search.
2. **Second selector wall time** (`second_selector_wall_ms`) measures the scoped
   calibration capture and final root selection/broadcast. When the cost gate
   permits the graph search, it includes a second ABTM refinement. This is a
   whole-collective time, not a separately instrumented root ABTM timer.
3. **Selected native setup** is timed separately after the final selection.
4. **Paired held-out measurements** run 3 epochs; each epoch has 5 fresh
   warmups + 20 samples for **baseline, Graph candidate, and selected** policy.
   Order rotates across epochs/trials and all three use the same input vector.
   Each sample is the **maximum** rank-local MPI SpMV time (not the mean of
   ranks). Medians are computed per epoch, followed by the median of epoch
   medians. These timings do *not* influence the guarded decision.
5. Numeric CSR-reference / ghost-value errors, reciprocal traffic, all owned
   row counts and `workers_global_csr=0` are checked collectively. Speedup is
   **not** a PASS criterion. Rank 1 is expected to stay SingleRank; N=1 on
   multi-rank should conservatively refuse nonamortized graph preparation.

## Two distinct economic models

For expected *future* SpMV reuse `N`, using held-out median times `s_b, s_s`:

```
T_baseline_only_estimate    = T_native_setup_baseline + N * s_b
T_selected_after_probe_est = T_second_selector + T_native_setup_selected + N * s_s
T_selected_from_probe_est  = T_probe + T_selected_after_probe_est
```

All durations are measured native MPI wall times **except** `N*s`, which is
extrapolated from short held-out measurements, **not** a real N-iteration
solve. The `baseline_only` alternative avoids the Graph probe entirely.
`selected_from_probe` starts *after* matrix/layout preparation; it does not
include input ingestion, MPI initialization, executable startup, or FEM-PCG.
`T_probe` already includes first root graph exploration and the first pair of
native preparations; do **not** add them again. The second selector's costs
are included once in `T_second_selector`.

A *fully accounted* break-even estimate is only displayed for a **selected
Graph** with strictly lower selected holdout SpMV time:

```
ceil((T_probe + T_second_selector + T_native_setup_selected
          - T_native_setup_baseline) / (s_b - s_s))
```

If selected is baseline, or savings are nonpositive / unresolvably small,
`fully_accounted_break_even_calls=None` is correct. The separate
`gate_break_even_calls` reflects the old conservative preflight using the
initial **40-sample** calibration and root candidate preparation budget and
is *not* a complete-run amortization point.

## Interpreting noisy microbenchmarks

The synthetic 168-DOF problem runs at the microsecond scale. Scheduling,
clock scaling, MPI runtime overhead and warmup differences may dominate.
Inspect the three epoch-median arrays and `all_epochs_graph_faster`; a
single low median is not sufficient evidence to select Graph in production.
Do not convert these three epochs to statistical confidence intervals without
repeated independent launches. Running 3+ completely separate MPI launches
per rank is recommended for assessing reproducibility.

No real mixed-field FEM/Matrix Market input is exercised. No end-to-end PCG
or FGMRES iteration count, residual convergence, re-partition of distributed
rows, memory scaling, or multi-RHS reuse policy is validated. These are
subsequent work items, not implicit H1-C3H success criteria.

## Windows/MS-MPI validation protocol

Run in the **same PowerShell 7 session** with:

```powershell
$env:CLANG_PATH='C:\Program Files\LLVM\bin\clang.exe'
$env:LIBCLANG_PATH='C:\Program Files\LLVM\bin'
$env:MSMPI_INC='C:\Program Files (x86)\Microsoft SDKs\MPI\Include\'
$env:MSMPI_LIB64='C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\'

cargo fmt --all -- --check
cargo clippy --locked -p hybit-distributed --features mpi --all-targets -- -D warnings
cargo build --release --locked -p hybit-distributed --features mpi --example mpi_root_cost_ledger_bench
$exe = (Resolve-Path .\target\release\examples\mpi_root_cost_ledger_bench.exe).Path
foreach ($r in @(1,2,4,8)) { & mpiexec -n $r $exe $r 10000; if ($LASTEXITCODE -ne 0) { throw "H1-C3H N=10000 $r failed" } }
foreach ($r in @(2,4,8)) { & mpiexec -n $r $exe $r 1; if ($LASTEXITCODE -ne 0) { throw "H1-C3H N=1 $r failed" } }
```

For independent launches, repeat the 2/4/8-rank N=10000 runs three times.
Only after reviewing logs run release regressions (`hybit-distributed`,
`hybit-matrix`, `hybit`, `hybit-ffi`), source integrity and staged-diff check.
Expected source integrity: **293 manifest entries / 291 SHA-256**. Do not
commit/push the checkpoint without real Windows logs. Never tag/release/publish
or enable default auto-routing based on this experiment.

## Native Windows/MS-MPI validation and independent-launch reproducibility — 2026-10-16 (confirmed; supersedes PENDING)

**Checkpoint scope:** based on the committed/pushed clean `7ee08a4` H1-C3G; the H1-C3H six-file experiment was staged, not committed, at validation time. No production selector, library, or solver path was modified. The 168-DOF, 3552-nnz, 84 two-component blocks/eight-region test remains **synthetic**; full-N totals are modeled, not measured full-N solves.

### Execution gates and correctness

- Package Preflight and Apply **PASS**; Rustfmt **PASS**; MPI-feature strict Clippy `--all-targets -- -D warnings` **PASS**; Release MPI benchmark build **PASS**. Source Integrity **293 manifest / 291 SHA-256 PASS**; staged diff integrity **PASS**.
- Initial native MS-MPI test **7/7 PASS**: N=10000 at 1/2/4/8 ranks, and N=1 at 2/4/8 ranks. All runs had `guard=Matched`, `workers_global_csr=0`, zero baseline/candidate/selected reference error, and checked reciprocal Halo/owned-row invariants. N=10000 chose Graph at 2/4/8 ranks and SingleRank/Equal at rank 1. All N=1 multi-rank tests correctly chose `UnamortizedPreparation` and the Equal/Field baseline.
- **15/15 separate-process native MS-MPI launches PASS**: five fresh launches each at 2/4/8 ranks, N=10000. All had zero reported reference errors and `workers_global_csr=0`. Release regressions **PASS** (`hybit-distributed` 45/45 non-MPI, `hybit-matrix` 52/52, `hybit-ffi` 5/5, HyBIT public API and doctest), plus MPI-feature `hybit-distributed` **72/72 PASS**.

### Five independent launches per rank count, N=10000

The following min/median/max and counts use the five fresh `mpiexec` launches **per rank count**; they are descriptive statistics, *not* confidence bounds. `fully_accounted_break_even_calls` is reported only when Graph was actually selected and the holdout savings were positive.

| MPI ranks | Graph selected | `all_epochs_graph_faster` | Fully accounted break-even, min / median / max calls | Lower modeled total *including probe*, launches |
|---:|---:|---:|---:|---:|
| 2 | **5/5** | **5/5** | **6,848 / 8,509 / 10,795** | **4/5** |
| 4 | **4/5** | **4/5** | **2,619 / 3,365 / 3,948** (4 selected runs) | **4/5** |
| 8 | **5/5** | **5/5** | **1,939 / 2,234 / 3,573** | **5/5** |

Here, `Lower modeled total` compares `modeled_selected_from_probe_start_ms` against `modeled_baseline_only_ms` **within the same launch**. The model extrapolates held-out microsecond SpMV medians over 10,000 calls; it is *not* observed 10,000-call execution time. Both selected non-Graph fallback and selected Graph count in the cost comparison.

Detailed exceptions:

- **2 ranks, launch 2:** Graph was selected with gate break-even **302** calls, but the fully accounted break-even was **10,795**, exceeding expected N=10,000. The modeled total worsened **22.0940 → 22.2132 ms** (+0.1192 ms, about 0.54%). The guard checks calibration scope, not true future payback; the gate is not a complete-cost policy.
- **4 ranks, launch 2:** the calibration said Graph was slower (baseline **9.55 µs**, Graph **9.75 µs**) and returned `NoCalibratedSpeedup`/Equal. Independent holdout medians later favored Graph (**5.50 → 5.00 µs**), but `all_epochs_graph_faster=false` due to a reversed epoch. The probe-inclusive selected fallback was **56.5777 ms** versus **55.2469 ms** baseline-only: a **1.3308 ms** modeled penalty from probing and selection. This is *not* an incorrect fail-closed decision on the original calibration, but it exposes calibration instability.
- **8 ranks:** five of five launches selected Graph and modeled a favorable total after including probe and second selector; this does **not** establish large-matrix speedup or statistical generality.

### Measurement instability and policy implications

Median-of-launch calibration-to-holdout Graph timing ratios were **2.294× / 2.120× / 1.400×** for 2/4/8 ranks (ranges respectively **1.971–3.235× / 1.950–2.280× / 1.070–1.683×**). Graph holdout medians across launches ranged **1.70–1.85 / 2.50–5.00 / 5.20–6.65 µs**. These gaps are too large to interpret a small calibrated speed difference as robust production evidence. Alternating candidate order and three held-out epochs help, but five independent launches are still a small sample and individual epoch medians are not raw independent timing samples.

The H1-C3H audit includes **whole-probe wall time** and **second selector/Broadcast wall time** exactly once in `modeled_selected_from_probe_start_ms`; initial graph proposal timing is a *component* of probe cost, not an extra independent charge. The comparison intentionally excludes executable startup, MPI initialization, matrix ingestion, distributed row redistribution and FEM solver convergence. Selector reuse and whether probes need repeating remain application-level design questions. The present Graph decision uses initial calibration and an estimated preparation budget, not the post-hoc fully accounted/holdout model. No default auto-routing change is justified by these observations.

**Checkpoint assessment:** experimental correctness and cost-ledger instrumentation **PASS**, with an explicitly recorded reproducibility/policy limitation. After this document update, validate Source Integrity **293/291**, staged diff and commit/push **only the H1-C3H development checkpoint** if all gates pass. Do **not** tag, release, publish crates, or assert general solver speedup. Next H1-C3I: study stable paired calibration / noise-aware conservative admission, then representative mixed-field Matrix Market/FEM + MPI-PCG end-to-end wall time.
