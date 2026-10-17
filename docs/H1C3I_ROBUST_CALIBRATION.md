# HyBIT 0.9 H1-C3I — paired-epoch robust cost admission (experimental)

**Baseline:** committed/pushed `dd1c9ad` (`develop/0.9.0`, clean) after H1-C3H.
**Validation:** Windows/MS-MPI initial and independent-launch tests PASS; non-MPI full release regressions still pending. See validated results below. This patch does NOT turn on automatic partition selection or change the production PCG/FGMRES path.

## Motivation and scope

H1-C3H's 15 independent MS-MPI launches showed a 2-rank Graph choice whose full probe-inclusive modeled break-even was **10,795** calls, above the requested 10,000, despite passing the prior setup-only cost gate. At 4 ranks, one trial rejected Graph during calibration even though subsequent holdout medians preferred it. Median calibrated graph/holdout ratios exceeded 2x at 2/4 ranks. A single calibration median is not enough to justify default Graph routing.

**H1-C3I adds an opt-in, deterministic *veto*, not a promotion policy**. The existing H1-C3D scoped-identity check, root-only selection/Broadcast, numerical Halo verification, field-safe offsets and graph structural/load guards remain the final authorities.

## Robust cost method

`evaluate_robust_cost` (MPI-independent `robust_cost.rs`) accepts **at least three paired calibration epochs**. Each pair is the baseline/Graph *max-rank* SpMV median measured in the same epoch. Reject if any timing is zero, any epoch shows no strict Graph advantage, or there are fewer than three epochs.

Let `g_min = min_epoch(baseline_ns - graph_ns)`, requiring `g_min > 0` in **every** epoch. Let

```
raw_extra = max(0, probe_wall + predeclared_second_selector_budget
                   + graph_native_setup - baseline_native_setup)
conservative_extra = raw_extra + ceil(raw_extra / 4)
predicted_gross = expected_calls * g_min
```

Admit only when `predicted_gross > conservative_extra + minimum_net_savings`. Report the strict integer threshold `floor((conservative_extra + minimum_net_savings)/g_min) + 1`. Use u128 for accounting to avoid overflow. **The 25% surcharge is an explicit policy contingency, NOT a statistical confidence bound or hardware performance guarantee.** Using the smallest epoch saving is deliberately pessimistic but can reject useful candidates. Conversely three/four favorable epochs do not prove future speedup.

A budget for a **second selector** is required *before the decision*. The MPI example preregisters `max(1 ms, 4 * measured_root_candidate_prepare)`; it reports `selector_budget_exceeded` if the realized selector takes longer. This is a hypothesis drawn from a tiny test, not a validated production upper bound. Its adequacy must be evaluated over independently launched MS-MPI processes and larger real FEM matrices.

## Native example and correctness

`mpi_robust_partition_bench` is based on H1-C3H's same synthetic `168 DOF / 3552 nnz`, 84 **two-component** blocks, 8 weak clusters. No full CSR on worker ranks. Root alone explores Graph; ownership and Halo are distributed by the existing MS-MPI path. The candidate is measured in **4 epochs × (5 warmups + 10 timed pairs)**; the experiment changes the warmup profile compared to H1-C3H and therefore must not be assumed to reproduce its raw medians. After cost veto:

- **Admit:** root captures the existing scoped identity and asks the existing guarded root selector to re-evaluate Graph; it may still reject for structural/balance reasons.
- **Veto:** root deliberately supplies *no scoped calibration*, and the selector stays at its field-safe Equal/Field baseline; this is a documented `CalibrationGuardDecision::Missing` rather than a scope error. The routine does not expose new production defaults.
- An *independent* **3 epochs × 20** held-out sample comparison is performed **after** selection. Holdout never influences the veto. The fully-accounted H1-C3H modeled baseline/selected times are still reported separately from the prespecified budget-based preflight.

The native benchmark checks reciprocal Halo send/receive, owned-row sum, max-rank timing reduction, CSR-reference SpMV errors, ghost values, identical policy/offsets, and `workers_global_csr=0`. A successful run is NOT proof of a full-N solve or FEM acceleration.

**Known limitation:** H1-C3I still builds the exploratory Graph candidate and may re-search Graph during the final root selector. H1-C3J will address reusable plans / duplicate work. The measured `probe_wall` already includes the first Graph exploration and baseline/candidate native setups: do not charge those components again. Graph and baseline timing remains at microsecond scale and is sensitive to scheduling and MPI overhead.

## Windows/MS-MPI validation protocol (executed)

Prerequisites: rustc x86_64-pc-windows-msvc, LLVM/libclang, Microsoft MPI SDK, PowerShell 7 and clean HEAD `dd1c9ad` before patching.

```powershell
cd D:\Work\hybit\hybit
$env:CLANG_PATH='C:\Program Files\LLVM\bin\clang.exe'
$env:LIBCLANG_PATH='C:\Program Files\LLVM\bin'
$env:MSMPI_INC='C:\Program Files (x86)\Microsoft SDKs\MPI\Include\'
$env:MSMPI_LIB64='C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\'
cargo fmt --all -- --check
cargo clippy --locked -p hybit-distributed --features mpi --all-targets -- -D warnings
cargo test --release --locked -p hybit-distributed --features mpi --lib
cargo test --release --locked -p hybit-distributed -p hybit-matrix -p hybit -p hybit-ffi
cargo build --release --locked -p hybit-distributed --features mpi --example mpi_robust_partition_bench
$exe = (Resolve-Path .\target\release\examples\mpi_robust_partition_bench.exe).Path
foreach ($r in @(1,2,4,8)) { & mpiexec -n $r $exe $r 10000; if ($LASTEXITCODE -ne 0) { throw "H1-C3I N=10000 ranks=$r failed" } }
foreach ($r in @(2,4,8)) { & mpiexec -n $r $exe $r 1; if ($LASTEXITCODE -ne 0) { throw "H1-C3I N=1 ranks=$r failed" } }
# Repeat independent mpiexec sessions after reviewing initial 7 cases.
# Then Source Integrity gate and git diff --cached --check.
```

**Expected Source Integrity after apply: 296 manifest / 294 SHA-256.** New non-MPI unit tests: nine; expected distributed unit test counts: 54 non-MPI / 81 MPI-feature, subject to the current workspace. Do not claim these pass before user execution. No commit/push/tag/release/publish until logs are inspected.

## Release boundary

H1-C3I is an evidence-collection phase. For HyBIT 0.9.0 production auto-promotion, require representative **real mixed-DOF FEM** matrices, genuine MPI-PCG end-to-end solves, true residual, peak memory, independent-run comparisons including a no-probe baseline, and forecast/actual budget agreement. Use CSR when ABTM's symbolic cost is not repaid. GPU/CubeCL and NN/Burn are distinct future tracks, not part of this gate.

## Native Windows/MS-MPI validation — 2026-10-17 (confirmed; supersedes the above PENDING notices)

**Tested source:** H1-C3I experimental stage applied on clean, pushed `dd1c9ad` (`develop/0.9.0`). The experiment was **staged but not committed** at the time of the user-provided Windows runs. The production solver, MPI transport, C ABI and default partition policy are unchanged. All tests below use the same 168-DOF/3552-nnz synthetic benchmark, not real FEM.

### Gates and collective correctness

- Package Preflight/Apply **PASS**; Source Integrity **296 manifest / 294 SHA-256 PASS**; `cargo fmt --all -- --check` **PASS**; MPI-feature strict Clippy `--all-targets -- -D warnings` **PASS**; Release MPI example build **PASS**.
- MPI-feature release unit tests **81/81 PASS** (including nine new `robust_cost` tests). **A full non-MPI workspace/package release regression has NOT yet been provided** for H1-C3I and must run before checkpoint commit.
- Initial native MS-MPI runs **7/7 PASS**: N=10000 at 1/2/4/8 ranks and N=1 at 2/4/8 ranks. Native numerical CSR/Halo reference errors all reported zero; `workers_global_csr=0`. Single-rank Equal and low-N fail-closed behavior retained.
- Additional **18/18 native MS-MPI launches PASS**: 15 independent processes (2/4/8 ranks × five, N=10000) plus 2/4/8 ranks × one (N=1,000,000). These are fresh launches, not extra in-process epochs; each also reports independent three-epoch holdout. All reported reference errors zero, no root/global CSR on workers, and `selector_budget_exceeded=false`. The count is **25/25 native MPI launches** combining initial and additional runs.

### Five independent launches per MPI rank count (N=10000)

| Ranks | Robust Graph admission / actual Graph selection | Veto reason(s) | `all_epochs_graph_faster` (holdout) | Selected path cheaper than baseline-only, probe-inclusive modeled N=10000 |
|---:|---:|---|---:|---:|
| 2 | **0/5** | 5 `NoConsistentPairedGain` | 3/5 | 0/5 |
| 4 | **0/5** | 3 `NoConsistentPairedGain`, 2 `UnamortizedTotalCost` | 5/5 | 0/5 |
| 8 | **4/5** | 1 `UnamortizedTotalCost` (robust required 12,109 calls) | 5/5 | **4/5** |

All robust vetoes propagated as intentionally missing scoped evidence: root selector returned `NoCalibration`, guard `Missing`, and block-safe Equal/Field offsets. These are *expected fail-closed results*, not `ScopeMismatch` or MPI failures. On rank 8, four successful `Admit` runs returned `GraphSelected`, `guard=Matched`, exact candidate offsets `[0,24,46,66,88,108,128,148,168]` and reduced total ghosts from 156 to 8, with zero reported numerical errors. That proves the positive collective execution path is reachable, not that Graph is universally faster.

Eight-rank probe-inclusive modeled cost comparison, matched *within each fresh launch*:

| Launch | Result | Baseline-only (ms) | Selected-from-probe (ms) | Conservative robust required calls |
|---:|---|---:|---:|---:|
| 1 | Field fallback | 76.1001 | 80.1440 | 12,109 |
| 2 | GraphSelected | 78.2322 | 65.0381 | 2,667 |
| 3 | GraphSelected | 56.0972 | 48.3017 | 5,113 |
| 4 | GraphSelected | 67.6215 | 62.6562 | 7,865 |
| 5 | GraphSelected | 80.1136 | 68.9302 | 5,139 |

The admitted 8-rank cases reduced the modeled total by **~7.3–16.9%** after including probe and selector time; this is an **extrapolation from short SpMV medians**, *not* real 10,000-step wall time or full-PCG speedup. The rank-8 fallback incurred a modeled ~4.04 ms wasted-probe penalty. On ranks 2 and 4 all five fallbacks incurred probe overhead even when holdout later favored Graph. Probe efficiency remains a first-class unresolved cost.

### N=1,000,000 *modeled* horizon tests

- 2 ranks: `NoConsistentPairedGain` → Equal; baseline/selected model **1800.0405/2301.7178 ms**. A larger nominal horizon cannot cure a failed every-epoch positive-gain prerequisite.
- 4 ranks: `NoConsistentPairedGain` → Equal; baseline/selected model **2800.1208/2901.8249 ms** despite favorable held-out Graph medians. This exemplifies the conservative veto's missed-opportunity trade-off.
- 8 ranks: `Admit` → `GraphSelected`, guard `Matched`, robust required **2,646** calls; baseline/selected model **8700.0990/7502.7391 ms**, all reported errors zero.

The million-call setting is an input to the **economic model only**; neither a million SpMVs nor a million PCG iterations were actually executed.

### Interpretation and next checkpoint

- The H1-C3I **fail-closed veto, safe fallback, root-owned scope verification, and admitted native MPI Graph path all pass** on the synthetic problem. Five fresh 8-rank N=10000 runs chose Graph only when their conservative preflight returned `Admit`; all four admitted runs also favored Graph in independently held-out *modeled total*.
- The every-epoch positive-gain rule can be **over-conservative** for tiny microsecond workloads: all five 4-rank N=10000 launches favored Graph throughout the independent holdout, yet none admitted Graph. That is an observed tradeoff, not a correctness defect; do not loosen the gate based on holdout from the same experiments without a predeclared out-of-sample validation plan.
- The fixed **1.0 ms second-selector budget** was not exceeded in the supplied launches, but this is not an established worst-case bound for larger matrices. The 25% additional preparation contingency is also a fixed policy assumption, not a confidence interval.
- **Required before commit:** full release regression for `hybit-distributed`, `hybit-matrix`, `hybit`, `hybit-ffi`; confirm updated Source Integrity **296/294** and staged diff. Next H1-C3J should prioritize reusing the exploratory ABTM Graph candidate and removing the second Graph search, reducing the economic penalty *before* trying to relax conservatism. Keep the solver's default route unchanged until real mixed-field FEM MPI-PCG, true residual and peak memory gates pass. Do not tag, release or publish.
