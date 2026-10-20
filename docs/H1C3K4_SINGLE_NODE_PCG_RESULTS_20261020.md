# H1-C3K4 native MPI-PCG experiment — single-node results (2026-10-20)

**Checkpoint scope:** `develop/0.9.0`, baseline commit `5240f1c`, after applying K4 Fix1. **Experimental only: single physical host, native Windows MSVC + MS-MPI, synthetic SPD surrogate.** Do not claim multi-node speedup, real-coefficient FEM convergence, production MPI-PCG integration, or a released default selector.

## Evidence and restrictions

- Matrix input: `D:\Work\thermal1.mtx` provides the **sparsity pattern**, not the numerical coefficients of the solved operator. K4 constructs a symmetric, strictly row diagonally dominant SPD surrogate (`shift=0.01`). The manufactured right-hand side comes from a known solution.
- Matrix dimensions: global `n=82,654`, surrogate CSR `nnz=574,458`; original Matrix Market `input_nnz=328,556`.
- Five partition methods: Equal, scalar-graph contiguous, ABTM region growth, ABTM multisource, ABTM multilevel. Each uses the same surrogate matrix, positive diagonal Jacobi and numerical MPI-PCG code path. Only ownership/halo layouts differ.
- Host: **one node only**; 1, 2, 4 and 8 MS-MPI ranks are processes on that node. There are no inter-node network measurements.
- Each invocation used graph radius 16, 2 warm-ups, 5 timed solves, max 3,000 iterations and tolerance `1e-10`. Console logs were supplied for all 20 method/rank combinations. **Raw CSV files for ranks 4 and 8 were inspected; CSV files for ranks 1 and 2 were not attached**, so those rows below come from the console logs only.
- The listed solve times are medians of each trial's **maximum rank-level wall time**. Exchange times are medians of the maximum rank's **cumulative halo exchange time per solve**, including local packing; neither is a pure network measurement. The CSV does not contain per-trial raw data or independent process-level repetitions, so significance claims are unsupported.
- `partition`, `extract`, `halo setup` and `RHS setup` are per-phase max-rank timings. **Do not sum independently reduced maxima/medians and label the result a measured end-to-end wall time.** Input read and SPD surrogate generation are excluded from the table.

## Full rank sweep

All times below are milliseconds. For ranks 1 and 2 RHS setup was not present in the supplied console logs. Columns `exchange` and `solve` are **not** additive independent phases.

| Ranks | Method | PCG solve median | Halo exchange cumulative median | Ghost DOFs | Cut nnz | Partition | Owned CSR extraction | Halo setup | RHS setup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `equal` | 102.4361 | 1.4344 | 0 | 0 | 0.177 | 4.986 | 24.183 | — |
| 1 | `graph_contiguous` | 102.3151 | 1.4330 | 0 | 0 | 39.392 | 6.729 | 31.581 | — |
| 1 | `abtm_region_growth` | 100.8248 | 1.4232 | 0 | 0 | 40.653 | 4.909 | 22.223 | — |
| 1 | `abtm_multisource` | 101.2377 | 1.4394 | 0 | 0 | 37.681 | 4.690 | 22.123 | — |
| 1 | `abtm_multilevel` | 101.0477 | 1.3992 | 0 | 0 | 194.450 | 4.862 | 22.289 | — |
| 2 | `equal` | 54.4462 | 1.8697 | 8,002 | 16,268 | 0.268 | 3.240 | 12.948 | — |
| 2 | `graph_contiguous` | 55.9406 | 1.9458 | 7,989 | 16,234 | 38.042 | 3.329 | 12.665 | — |
| 2 | `abtm_region_growth` | 53.9475 | 0.8788 | 300 | 600 | 33.724 | 2.816 | 11.557 | — |
| 2 | `abtm_multisource` | 52.5948 | 0.9320 | 600 | 1,196 | 42.787 | 3.426 | 12.251 | — |
| 2 | `abtm_multilevel` | 56.0098 | 1.1036 | 981 | 1,954 | 202.635 | 3.218 | 11.847 | — |
| 4 | `equal` | 29.0250 | 2.0054 | 14,035 | 28,372 | 0.293 | 1.783 | 7.298 | 0.471 |
| 4 | `graph_contiguous` | 28.5166 | 2.0402 | 14,008 | 28,296 | 37.537 | 1.896 | 7.042 | 0.461 |
| 4 | `abtm_region_growth` | 27.1426 | 0.6941 | 1,813 | 3,610 | 30.946 | 1.935 | 6.461 | 0.427 |
| 4 | `abtm_multisource` | 26.8145 | 0.7211 | 2,922 | 5,818 | 50.327 | 2.343 | 6.457 | 0.428 |
| 4 | `abtm_multilevel` | 26.2343 | 0.7619 | 2,053 | 4,094 | 185.016 | 1.961 | 6.056 | 0.454 |
| 8 | `equal` | 22.7069 | 4.1147 | 23,085 | 48,258 | 0.446 | 1.379 | 6.374 | 0.443 |
| 8 | `graph_contiguous` | 23.1112 | 4.2503 | 22,987 | 48,004 | 42.707 | 1.601 | 4.584 | 0.432 |
| 8 | `abtm_region_growth` | 22.9263 | 3.6421 | 3,227 | 6,422 | 32.493 | 1.224 | 3.010 | 0.252 |
| 8 | `abtm_multisource` | 19.5396 | 1.7521 | 5,841 | 11,602 | 72.479 | 1.291 | 3.457 | 0.459 |
| 8 | `abtm_multilevel` | 22.3883 | 3.6298 | 3,687 | 7,318 | 199.087 | 1.385 | 3.905 | 0.337 |

## Numerical validation

All **20/20** method/rank cases logged `K4 NATIVE PASS`. Every case reported:

- **125 PCG iterations** (same across layouts and runs).
- **126 SpMV calls** (includes final independent true-residual application).
- **377 global dot-product reductions**.
- Maximum logged true relative residual **4.423e-11**, below the `1e-10` tolerance.
- Maximum logged absolute error against the manufactured known solution **1.873e-9**.

These checks support correctness **for this synthetic SPD test only**. This is not a mathematical proof for arbitrary SPD matrices, indefinite/saddle-point systems, heterogeneous field layouts, or frictional contact systems. Runtime numerical results in each policy were validated by the experiment's true-residual guard.

## Performance observations

- **4 ranks:** ABTM multilevel is the lowest measured solve median, **26.2343 ms**, vs Equal **29.0250 ms** (**9.61%** reduction). Region growth's cumulative exchange median is lowest (**0.6941 ms**, vs Equal **2.0054 ms**). Multilevel's partition stage is **185.016 ms**, making one-shot cost substantially larger than Equal.
- **8 ranks:** ABTM multisource is the lowest measured solve median, **19.5396 ms**, vs Equal **22.7069 ms** (**13.95%** reduction). Cumulative exchange median is **1.7521 vs 4.1147 ms** (**57.42%** lower) and unique Ghost count **5,841 vs 23,085** (**74.70%** fewer).
- **8 ranks counterexample:** Region growth has only **3,227** Ghost DOFs (even fewer than multisource), but solve median **22.9263 ms**, **0.97% slower than Equal**. Communication-volume reduction alone does not establish a runtime advantage.
- **8-rank numeric sum:** `median_max_numeric_sum_ms` is **10.1089 ms** for Equal, **10.6377 ms** for multisource. Thus the overall solve improvement cannot be attributed to faster numeric SpMV. MPI reduction timing, process scheduling, synchronization and other solver work were **not instrumented separately** and are hypotheses, not established causes.
- **1 rank** has zero Ghost DOFs but nonzero reported exchange accumulation (~1.4 ms per solve). This is local halo routine overhead, not network traffic.
- Approximate 1-to-8 rank strong scaling of measured solve medians on this single host: Equal **4.51×**, multisource **5.18×**. This is **not** multi-node scaling.

## Illustrative setup amortization, not measured end-to-end

Using `P = partition_max_ms + owned_extract_max_ms + halo_setup_max_ms + rhs_setup_max_ms` and `S = median_max_solve_ms`, a hypothetical solve-many model is `P + N*S` for **one prepared operator and identical RHS/solver behavior**. Independent phase extrema and medians make it only a budgeting estimate.

- **4 ranks:** Equal `P≈9.845 ms`, region growth `P≈39.769 ms`, multisource `P≈59.554 ms`, multilevel `P≈193.486 ms`. If solve medians repeat unchanged, Equal vs region growth crosses near **16 solves**, Equal vs multisource near **23 solves**, Equal vs multilevel near **66 solves**.
- **8 ranks:** Equal `P≈8.642 ms`, multisource `P≈77.686 ms`; solve advantage `22.7069 - 19.5396 = 3.1673 ms`. The budget model crosses near **22 solves**. In a **single solve**, Equal's smaller preparation cost outweighs multisource's lower PCG time.
- CSV also reports one-time, full-rank input read and synthetic SPD conversion. Those are **replicated on every rank** in this bounded experiment, and are not included in `P`; rank-local production distribution and resident peak memory remain unproven.
- K4's RHS was manufactured and the same RHS reused; true solve-many with varying RHS and independently timed batch totals has **not** been demonstrated.

## Verification and next gates

1. Collect raw timed samples over **several independent mpiexec launches**, with alternating method order and variability (median, p10/p90, standard deviation or robust spread). Do not make default-policy changes from one 5-sample run.
2. Break out MPI `Allreduce` timing, per-rank halo packing/MPI transfer, local SpMV, Krylov vector updates and synchronization; record the slowest rank and rank imbalance.
3. Test more matrix topologies / SPD spectra, then a **certified genuine SPD FEM matrix**; avoid applying PCG to non-SPD inputs. Validate cross-rank true residual and known/reference solutions where available.
4. Implement true rank-owned input distribution and bounded owner-map storage; assess memory and setup separately. K4 presently replicates the global CSR to each rank.
5. Keep K4 explicitly experimental. Do not alter production MPI-PCG, auto selection, C/Fortran ABI or default solver routing without further numerical and end-to-end gates.
6. For multi-node performance, run with **two or more physical nodes** and record interconnect, process mapping and end-to-end timings.

**Checkpoint recommendation:** archive this experimental single-node numerical milestone on `develop/0.9.0` after checking the staged scope and Source Integrity. No claim of production deployment or multi-node acceleration.
