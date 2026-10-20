# H1-C3K5 — One-node PCG Allreduce profiling and independent-launch repeatability

**Status:** Experimental, validation required on user's Windows/MS-MPI workstation. Baseline `develop/0.9.0` commit `047a2bb` (K4 complete). This is **not** a new production solver path.

## Motivation

K4 showed converged Jacobi-PCG with noncontiguous ownership at 1/2/4/8 MPI ranks (all on one physical host). The 8-rank ABTM multisource case was 19.5396 ms vs 22.7069 ms for equal, but the difference cannot be attributed solely to Halo exchange, and five inner timings in one launch do not establish reproducibility across independent starts. K4 has 377 global dot reductions for 125 iterations, so timed Allreduce is an important missing breakdown.

## K5-A changes

1. **Add** `mpi_k5_assignment_pcg_profile` as a standalone MPI-enabled example; do **not** edit or replace K4 example, production PCG, any existing numeric kernels, or the public API. Use the same synthetic SPD surrogate from the input's undirected sparsity graph, the same Jacobi-PCG operation sequence and convergence requirements, and the K3 assignment Halo/SpMV.
2. Measure the **local scalar dot evaluation** and the **MPI_Allreduce(sum)** portion of every PCG dot separately. Record the final **MPI_Allreduce(max)** for the known-solution error separately. Do not conflate reduction time with Halo exchange or numerical SpMV. Only the reductions inside the timed PCG solve count; preparation and benchmark-reporting reductions are excluded.
3. For every repeated solve, record each rank's elapsed component time, then use max-over-ranks to create a paired per-trial data row. This is a conservative critical-rank view. The median of independently max-reduced components is **not additive**, and `sum(dot_allreduce_ms)` is wall-clock time spent in these calls, **including rank waiting and synchronization**, not a pure network latency measure.
4. Write two files: `<stem>.csv` (policy summary) and `<stem>.samples.csv` (one row per policy per measured trial); refuse to overwrite either.
5. Add `benchmarks/scripts/bench-k5-assignment-pcg-repeatability.ps1`, invoking **a fresh mpiexec process for each run** and retaining all raw outputs. This script also emits a per-run, Equal-paired improvement CSV. It does not perform an automatic policy promotion.

## Input / output and test safeguards

- Input: Matrix Market `thermal1.mtx` or similar real sparse matrix, bounded to 200,000 DOFs and 8,000,000 input/surrogate nonzeros, each rank still loads the global CSR. The original matrix **coefficients are replaced** by a strictly diagonally dominant synthetic SPD system; no inference about real FEM coefficient solves is justified.
- Output `scope`: `synthetic_spd_single_node_mpi_experiment`. Do not rename it `multi_node` until running on at least two physical hosts and validating inter-node behavior.
- Expected `K5 NATIVE PASS`: all five methods converge with identical iteration count; true relative residual `<= tol`; no breakdown or nonpositive curvature; known-solution error is finite.
- Report: `median_max_solve_ms`, `median_max_exchange_sum_ms`, `median_max_numeric_sum_ms`, `median_max_dot_local_sum_ms`, `median_max_dot_allreduce_sum_ms`, `median_max_validation_allreduce_ms`. Dots are one MPI Allreduce(sum) each; the final error check is a separate max reduction.
- Numerical instrumentation is kept in the experimental copy, to avoid perturbing the committed K4 checkpoint or production routes. Allreduce timing does impose an additional clock-read overhead: compare K5 against K5 Equal under identical instrumentation; direct K4↔K5 timings need caution.
- The script's paired improvement is **within the same independent process launch**, after all 5 policies run in rotated trial order. To assess stability, compare the independent-launch CSV across at least 5 process starts and inspect per-trial sample CSVs.

## Commands on user's Windows workstation

```powershell
cd D:\Work\hybit\hybit
. .\tools\env\Enter-HybitMsMpi.ps1
$exe = 'D:\Work\hybit\hybit\target\release\examples\mpi_k5_assignment_pcg_profile.exe'
$mpi = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
& $mpi -n 4 $exe 4 D:\Work\thermal1.mtx 16 2 5 3000 1e-10 0.01 D:\Work\k5_thermal1_spd_r4_first.csv
if ($LASTEXITCODE -ne 0) { throw 'K5 native four-rank smoke failed' }
# When the first smoke has passed:
& .\benchmarks\scripts\bench-k5-assignment-pcg-repeatability.ps1 `
    -RepoRoot D:\Work\hybit\hybit -Matrix D:\Work\thermal1.mtx `
    -Ranks 4,8 -Launches 5 -OutputDir D:\Work -Prefix k5_thermal1_repeat
```

## Gates / scope

Apply requires exact branch + base HEAD, clean worktree/index, source-integrity PASS, verified package payload; must pass `cargo fmt --all -- --check`, strict Clippy with MPI/all-targets, full distributed library regression, K5 example unit tests, and release example build. Stage only new example/runner/doc plus `Cargo.toml`, `MANIFEST.txt`, `SOURCE_SHA256.txt`; no auto-commit or push.

## Next steps

- Interpret independent-start variance before accepting any statistically relevant advantage. If high, profile thread affinity, CPU oversubscription, NUMA/shared cache, frequency variation and timing design.
- Extend K5 with multiple independent RHS and amortized solve-many measurements, but do not conflate repeated identical-RHS timing with application-level solve-many. Profile Allreduce time per iteration and rank distributions before optimizing collective algorithms.
- Replace replicated global input with actual distributed CSR distribution before any multi-node or per-rank memory scalability claims.
