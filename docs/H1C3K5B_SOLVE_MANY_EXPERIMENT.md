# H1-C3K5-B — Prepared noncontiguous MPI-PCG solve-many (experimental)

**Status:** new experiment; native runtime validation pending. Base: `develop/0.9.0` at `9efa616` (K5-A). This does **not** change the production MPI-PCG solver, its API, the K5-A example, or partition-policy defaults.

## Question

Can the extra ABTM partition/halo setup be amortized when **different** right-hand sides are solved against the **same** prepared matrix, partition, halo and fixed Jacobi preconditioner?

The experimental executable `mpi_k5b_assignment_solve_many` compares five K3/K4 policies: `equal`, `graph_contiguous`, `abtm_region_growth`, `abtm_multisource`, `abtm_multilevel`. It generates one SPD matrix from the **undirected graph** of the supplied Matrix Market file (not its original coefficients), on **one physical node** under MS-MPI.

## Experimental protocol

- All ranks load and hold the **complete global CSR**, as in K3–K5-A. This is bounded test scaffolding, not scalable distributed input.
- Rank 0 proposes each partition; every rank receives the noncontiguous owner map. Each policy prepares its own rank-owned CSR rows, reciprocal Halo plan, and Jacobi inverse **once**.
- For each policy, manufacture **64 distinct** global exact solutions using deterministic RHS-dependent phases and frequencies. Form each RHS as `b_k = A x*_k` using the prepared rank-local SpMV/Halo. Manufacturing is a separately timed validation fixture, **excluded from the solve-many comparison**. The exact solution is never used as the PCG starting guess.
- Run 64 consecutive, collective PCG solves per policy with a zero initial guess. Every solve recomputes the *true* residual, enforces `true_rel <= tol`, checks the known solution error, and rejects nonpositive curvature or nonconvergence. Store one raw CSV row **per RHS per policy**.
- Capture the **elapsed wall time across ranks** at checkpoints `1,8,16,32,64` RHS. Timed region includes PCG execution and per-RHS reporting/synchronization, but not matrix reading, surrogate generation, partition/halo preparation, or RHS manufacturing.
- CSV column `prepared_total_approx_ms = partition_max_ms + owned_extract_jacobi_max_ms + halo_setup_max_ms + solve_many_wall_max_ms`. This is a **stage-maxima sum**, *not* a directly measured end-to-end invocation wall time. `read_max_ms`, `surrogate_max_ms`, `rhs_manufacture_max_ms` are reported separately.
- `solve_individual_max_sum_ms` is the sum of the 1..N individual worst-rank PCG solve durations. It is different from the measured whole-prefix `solve_many_wall_max_ms`; do not conflate them.
- Allreduce and Halo columns are separate measured quantities; they include rank waiting and are not interchangeable with raw network latency.
- No automatic ABTM policy selector; do not infer an optimum from Ghost count alone.

## Numerical and engineering limits

- This is **single-node MPI**, using Windows x64 MSVC, LLVM/libclang, native MS-MPI. No inter-node network tests. The global input is replicated on each rank.
- Matrix Market input is limited to 110 MiB; up to 200,000 scalar DOFs / 8 million nonzeros before SPD conversion. The resulting surrogate is additionally bounded to 8 million nonzeros.
- Local index, ownership, MPI count, Halo correctness follow existing K3 checks. `true_rel` and max solution error must be finite; relative residual must not exceed `tol` and max absolute solution error must not exceed `1e-6` for this bounded synthetic experiment.
- The work is an experimental benchmark; a successful run does not demonstrate performance on real FEM coefficients, dynamic contact, non-SPD matrices, multi-node clusters, or production MPI-PCG.
- Policy order is fixed and each policy processes its own 64 RHS consecutively. For noise analysis, repeat in independent `mpiexec` process launches and consider order/randomization in a later phase.

## Build and run (after clean-head patch application)

```powershell
cd D:\Work\hybit\hybit
. .\tools\env\Enter-HybitMsMpi.ps1
$mpi = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
$exe = 'D:\Work\hybit\hybit\target\release\examples\mpi_k5b_assignment_solve_many.exe'
& $mpi -n 4 $exe 4 D:\Work\thermal1.mtx 16 3000 1e-10 0.01 D:\Work\k5b_thermal1_spd_r4.csv
if ($LASTEXITCODE -ne 0) { throw 'K5-B native MPI-PCG solve-many failed' }
```

Two new files are produced by the experiment: `k5b_thermal1_spd_r4.csv` (5 methods × 5 checkpoints = 25 rows) and `k5b_thermal1_spd_r4.rhs.csv` (5 methods × 64 distinct RHS = 320 rows). Existing filenames are never overwritten.

Run `benchmarks/scripts/bench-k5b-assignment-solve-many.ps1` for 4/8-rank checks after the first 4-rank manual smoke test passes.

## Closeout criteria

1. Source Integrity; strict `-D warnings` Clippy; library regressions; new example tests; release build all pass.
2. Native 4-rank benchmark reports all 25 checkpoints and 320 RHS records with finite validated residuals and solution errors.
3. Native 8-rank benchmark separately passes. Compare directly measured prefix wall times, plus preparation cost, for each checkpoint.
4. Document repeated independent launch evidence before any production auto-selection or claims of definitive setup amortization.
5. Commit/push **only after** numerical/runtime evidence is reviewed.
