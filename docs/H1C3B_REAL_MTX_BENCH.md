# HyBIT 0.9 H1-C3B — Real Matrix Market MPI partition benchmark

Checkpoint baseline: `224b129` on `develop/0.9.0`. **Validated experimental benchmark (2026-10-14): Windows Rust/Clippy/Source Integrity and native MS-MPI numerical gates PASS.** Performance/production auto-selection is **not** certified; detailed measurements are below.

## Objective and interpretation

Compare (1) DOF-balanced contiguous ownership, (2) H1-C2 block-aligned ownership, and (3) H1-C3 ABTM-symbolic graph-refined *contiguous* ownership on the **same imported** sparse matrix. Run exactly the existing native MS-MPI `prepare_owned_rows` / rank-local CSR / `spmv_local` path for **every** policy; do not infer numerical speedup from only a change in Ghost counts.

Matrix Market contains no physical variable-DOF field layout. Every global scalar index is represented by a **synthetic one-component** block in this experiment; it is not a FEM node identity, not a real mixed-field partition, and not a validation of genuine thermal/beam/shell/contact coupling.

The benchmark supports symmetric and general Matrix Market real/integer coordinate files through HyBIT's existing reader; symmetric files are numerically expanded. The reader combines duplicates and drops numerical zeros, so the resulting CSR symbolic NNZ is the benchmark's topology, not necessarily the MTX header count. No SPD assumption, PCG convergence, or inferred physical field is claimed.

## Safety and scope

- Explicit memory protection for **replicated-global-CSR testing only**: input file <= 110 MiB, matrix n <= 200,000 and expanded CSR nnz <= 8,000,000. On 8 ranks each process independently reads and prepares the entire matrix; do **not** generalize its memory, file-read or graph-preparation timings to production distributed ingestion.
- Native partition ownership remains contiguous. No new noncontiguous Halo, MS-MPI transport, solver auto-selection or ABI changes.
- All three policies are applied with native MS-MPI at 1, 2, 4 and 8 ranks. Each apply is compared to the same global CSR reference and verifies total sends == receives == global unique Ghost count.
- Timing uses **median of the per-run maximum across MPI ranks** (warmups excluded). The executable alternates policy order each trial. Numeric relative error tolerance <=1e-11 after max(1, |reference|) scaling; all values must be finite.
- `partition_max_ms` is per-rank maximum symbolic/partition preparation. `native_halo_max_ms` is maximum of MPI-owned Halo setup across ranks. `estimated_rank_operator_halo_sum_bytes` is a partial operator/halo estimate **not** peak RSS. File read and full-matrix replication cost is separate.
- H1-C3 currently accepts candidate graph partitions only when Ghost count, cut nnz and peer relations are each no worse than H1-C2, with at least one strict gain. It does **not** reject candidates for expensive setup, excessive rank time or worse load balance beyond its current size cap. This benchmark tests whether stronger selection policy is justified.

## First test corpus

1. `Goodwin_010.mtx` (1,182, general) and `sherman5.mtx` (3,312, general) for small smoke.
2. `nd3k.mtx` (9,000, symmetric; potentially large after expansion).
3. `thermal1.mtx` (82,654, symmetric).
4. Next, `cfd1`, `cant`, `raefsky3` and `s3dkq4m2` subject to memory and elapsed time. Large `audikw_1`, `inline_1`, `thermal2`, `G3_circuit`, etc. are intentionally excluded.

Sample (after setting validated MS-MPI+libclang environment):

```powershell
cargo build --release --locked -p hybit-distributed --features mpi --example mpi_field_mtx_bench
$exe = (Resolve-Path '.\target\release\examples\mpi_field_mtx_bench.exe').Path
mpiexec -n 2 $exe 2 'D:\Work\Goodwin_010.mtx' Goodwin_010 32 2 5 'D:\Work\h1c3b_results.csv'
```

Invocation: `mpi_field_mtx_bench <ranks> <matrix.mtx> <label> <radius_blocks> <warmups> <repeats> <results.csv>`.

## Gates / limitations

Run `cargo fmt --all -- --check`, existing release regression tests, strict no-MPI Clippy and MPI-example Clippy, Source Integrity, actual 1/2/4/8-rank execution, CSV inspection, then commit/push after user approval. This checkpoint is an experimental benchmark, **not** a production partitioning/solve throughput improvement.

## 2026-10-14 actual validation and measurement results (supersedes pending status)

**Environment:** Windows 11, Rust MSVC target, Microsoft MPI, LLVM/Clang 23.1.3 and `libclang` (`CLANG_PATH`, `LIBCLANG_PATH`, `MSMPI_INC`, `MSMPI_LIB64` explicitly set). The initial MPI-enabled `mpi-sys v0.2.4`/`bindgen v0.72.1` build failed with `libclang error` when the variables were not supplied in that shell. After configuring those four variables, MPI-example Clippy with `-D warnings` and release compilation both **PASS**. Do not attribute this build failure to H1-C3B Rust source code.

**Gates reported by the Windows operator:**

- H1-C3B preflight and install **PASS**, source-integrity 277 manifest / 275 verified **PASS**, changes staged (not committed at this validation point).
- `cargo fmt --all -- --check` **PASS**; release Rust tests **PASS**: `hybit-distributed` 32/32, `hybit-matrix` 52/52, `hybit-ffi` 5/5, existing public API/regression suites and one hybit doc-test; no-MPI Clippy `-D warnings` **PASS**.
- Native Microsoft MPI real-MTX smoke tests for `Goodwin_010` at ranks 1/2/4/8 and `sherman5`, `nd3k`, `thermal1` at ranks 2/4/8: **PASS**, all three partition policies return `rel_err=0.000e0`.
- Confirmation: 4 MTX matrices × 2 rank counts (4, 8) × 3 policies = **24 policy/rank/matrix measurements**, 5 warmups and 20 timed repetitions each. All **PASS**, with reported `rel_err=0.000e0`. Times are medians of per-trial **maximum rank** time, not mean rank time.

### Confirmed MPI SpMV times (ms; lower is better)

| Matrix | Ranks | Equal | Field | Graph | Graph candidate accepted? |
|:--|--:|--:|--:|--:|:--|
| Goodwin_010 | 4 | 0.00800 | 0.00790 | **0.00760** | yes |
| Goodwin_010 | 8 | **0.00940** | 0.01045 | 0.00955 | yes |
| sherman5 | 4 | 0.00955 | **0.00920** | 0.00980 | yes |
| sherman5 | 8 | **0.01875** | 0.01955 | 0.01895 | yes |
| nd3k | 4 | 0.71745 | **0.66320** | 0.74120 | **no; fallback to Field partition** |
| nd3k | 8 | 0.78075 | **0.77120** | 0.78110 | **no; fallback to Field partition** |
| thermal1 | 4 | 0.18760 | **0.17525** | 0.18350 | yes |
| thermal1 | 8 | 0.18760 | **0.17155** | 0.20305 | yes |

Graph had the lowest measured SpMV median in only **one of eight** matched matrix/rank scenarios. At `nd3k` ranks 4 and 8, Graph was **not selected**; the apparent differences between Field and Graph timings there are variations while benchmarking the **same selected partition**, not evidence of different partition quality.

### Exact structural / preparation comparison (Field -> selected Graph)

| Matrix | Ranks | Ghost DOF | Cut nnz | Directed peers | Max owned imbalance | Max local nnz imbalance | Field prepare ms | Graph prepare ms |
|:--|--:|:--|:--|:--|:--|:--|--:|--:|
| Goodwin_010 | 4 | 542 -> 497 | 5,091 -> 4,780 | 6 -> 6 | 1.002 -> 1.151 | 1.119 -> 1.258 | 0.0101 | 1.5782 |
| Goodwin_010 | 8 | 1,244 -> 1,163 | 11,539 -> 10,814 | 14 -> 14 | 1.002 -> 1.144 | 1.126 -> 1.300 | 0.0102 | 1.9946 |
| sherman5 | 4 | 2,328 -> 2,324 | 4,301 -> 4,268 | 10 -> 10 | 1.000 -> 1.053 | 1.130 -> 1.110 | 0.0165 | 1.6055 |
| sherman5 | 8 | 2,505 -> 2,456 | 4,653 -> 4,510 | 36 -> 32 | 1.000 -> 1.065 | 1.410 -> 1.394 | 0.0136 | 2.4066 |
| nd3k | 4 | 24,147 -> 24,147 | 1,786,056 -> 1,786,056 | 12 -> 12 | 1.000 -> 1.000 | 1.184 -> 1.184 | 0.0504 | 124.3916 |
| nd3k | 8 | 44,213 -> 44,213 | 2,275,952 -> 2,275,952 | 56 -> 56 | 1.000 -> 1.000 | 1.195 -> 1.195 | 0.1968 | 195.9729 |
| thermal1 | 4 | 14,039 -> 13,999 | 28,374 -> 28,266 | 12 -> 12 | 1.000 -> 1.002 | 1.006 -> 1.005 | 0.1404 | 35.5296 |
| thermal1 | 8 | 23,083 -> 22,905 | 48,232 -> 47,774 | 46 -> 46 | 1.000 -> 1.006 | 1.007 -> 1.009 | 0.2575 | 51.6646 |

`Graph prepare ms` includes the H1-C3 symbolic graph refinement work, including a rejected proposal. For `nd3k` (4/8 ranks) this costly preparation does **not** change the chosen partition. The benchmark's `accepted` field describes the **graph proposal**; it does not mean the Equal or Field baseline was newly selected by the optimizer.

### Interpretation / next checkpoint

1. **Numerical/functional validation PASS**, but **end-to-end solver speedup NOT demonstrated**. These are SpMV-only results on one machine, with 20 repetitions in one confirmation run; small time differences need independently repeated trials and ordering analysis.
2. Ghost/cut/peer reductions do not reliably shorten SpMV. `thermal1` is the clearest counterexample: selected Graph has fewer ghosts, but higher measured SpMV time than Field at both 4 and 8 ranks.
3. Candidate acceptance based only on communication volume, cut nnz and peer relations is insufficient for automatic selection. A future explicit H1-C3C cost-aware policy should consider graph preparation, relative max-rank execution time, load balance, expected iteration count/solve-many reuse, and a cheap early-exit before full ABTM construction where possible. **No H1-C3C auto-selection is implemented in this checkpoint.**
4. `1 DOF = 1 synthetic block`: the run does **not** verify heterogeneous physical field ownership. The benchmark constructs global CSR on every rank, so its load/memory/preparation profile cannot be generalized to production distributed matrix ingestion. PCG convergence/SPD certification and full solver timings are out of scope.
5. Release/tag/publish are not authorized. After recording these validated results and rerunning Source Integrity, this benchmark can be committed as an experimental checkpoint on `develop/0.9.0`.
