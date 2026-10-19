# HyBIT 0.9 H1-C3K2 — Real FEM MPI-PCG Graph Boundary Probe

Base checkpoint: `acf710f` on `develop/0.9.0`. This is an **experimental MPI-owned contiguous partition probe**; not an automatic production policy, physical FEM field layout or mathematically certified SPD result.

## Why this experiment

Third-party architectural review identified an overinvestment risk in topology microbenchmarks and heavy test gates before whole-solver evidence. The K1 MPI-PCG Equal baseline is already 16/16 numerically accepted (4 certified controls + 12 ExperimentalUnverified real FEM), 1/2/4/8 ranks, thermal1 82,654 DOFs, 574,458 nnz, rtol=1e-11 and max_iters=5000. 8-rank median solve-only time was 297.4493 ms. K2 directly compares actual rank-owned Jacobi MPI-PCG with Graph-adjusted **contiguous** ownership.

## Exactly what changed

- `MtxPartitionPolicy::Equal` remains the default of all legacy callers.
- New opt-in `MtxPartitionPolicy::ScalarGraph { search_radius_blocks }` performs rank-zero `field_graph_contiguous_partition` with **one synthetic scalar block per DOF**, since Matrix Market contains no physical field metadata. Do not describe it as physics-preserving field-aware partitioning.
- The graph selector returns contiguous offsets (not arbitrary `PartitionAssignment`). Root broadcasts and every rank validates offsets before CSR distribution and reciprocal MPI Halo preparation. Partition mode and radius agreement are checked collectively.
- `mpi_mtx_profile` takes a trailing optional `equal` or `graph:<radius>` only after the explicit PCG relative tolerance. Existing K1 CLI still uses Equal by default. CSV receives four appended fields: `partition_mode`, `graph_candidate_accepted`, `root_graph_ms`, `ownership_offsets`. Existing K1 columns remain unchanged.
- Graph preparation is root-only; numeric CSR copying and synthetic metadata allocation are charged to graph time. Current `MtxDistributionStats` reports root-only graph time, not peak RSS. No non-contiguous distribution, weak scaling, MatMarket physical DOF metadata, or production auto selection is provided.

## First Windows tests

From `D:\Work\hybit\hybit`:

```powershell
. .\tools\env\Enter-HybitMsMpi.ps1
cargo fmt --all -- --check
cargo clippy --locked -p hybit --features mpi --all-targets -- -D warnings
cargo test --release --locked -p hybit-distributed --features mpi --lib
cargo build --release --locked -p hybit --features mpi --example mpi_mtx_profile
```

Using the literal MS-MPI executable and separate output paths:

```powershell
$mpi = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
$exe = 'D:\Work\hybit\hybit\target\release\examples\mpi_mtx_profile.exe'
$mtx = 'D:\Work\thermal1.mtx'
& $mpi -n 4 $exe 4 'D:\Work\k2_equal_r4.csv' $mtx thermal1_equal experimental 1 2 5000 1e-11 equal
if ($LASTEXITCODE -ne 0) { throw 'Equal run failed' }
& $mpi -n 4 $exe 4 'D:\Work\k2_graph_r4.csv' $mtx thermal1_graph experimental 1 2 5000 1e-11 graph:16
if ($LASTEXITCODE -ne 0) { throw 'Graph run failed' }
Import-Csv D:\Work\k2_equal_r4.csv, D:\Work\k2_graph_r4.csv | Format-Table case,partition_mode,graph_candidate_accepted,ownership_offsets,iterations,status,true_relative_residual,max_abs_error,global_ghosts,root_graph_ms,median_max_solve_ms
```

Do not reuse nonempty output files: `mpi_mtx_profile` appends rows and writes a header only to new/empty files. Do not commit/push until the user's MSVC+MS-MPI tests pass.

## What success means

- Compare Equal/Graph using the **same matrix**, manufactured truth vector and right-hand side, Jacobi, PCG tolerance and rank count. Verify `Converged`, true relative residual <= 1.1e-11 and manufactured max abs error <= 2e-7 for all measured runs. `ExperimentalUnverified` is NEVER an SPD certificate.
- Require the Graph candidate **actually changes offsets** before claiming a Graph-vs-Equal solver timing difference. If `graph_candidate_accepted=false` or offsets are equal, record a *rejected candidate* not a Graph speedup.
- Compare solve-only medians separately from root graph build, source read, CSR scatter, MPI Halo, PCG prepare, process wall. Include rank NNZ imbalance, allreduce count and ghost traffic when available. A graph cut reduction alone does not prove better runtime.
- Initial K2-A smoke runs are not a performance verdict. Run 1/2/4/8 ranks, repeated fresh MPI launches, alternating Equal/Graph order after correctness and no regressions. Report total graph setup amortization against solve-many.
- Keep a cost model: `T_equal = T_read + T_scatter_equal + T_halo_equal + T_pcg_prepare_equal + N*T_solve_equal` and Graph adds `T_graph` and its own scatter/halo/prepare/solve terms. Preserve separate process wall; do not present a sum of median phases as an exact whole-run measurement.

## Strategic change from third-party review

Two lanes: **exploration** (opt-in implementations and quick A/B parity) and **promotion** (numerical acceptance, repeatability and end-to-end performance before production defaults). Compare ABTM with CSR, KaMinPar/KaHIP/METIS, and baseline external solvers without requiring ABTM to win. Keep MSVC+Clang+MS-MPI standard `tools/env/Enter-HybitMsMpi.ps1`.

Research: https://github.com/KaHIP/KaMinPar ; https://doi.org/10.4230/LIPIcs.ESA.2025.32 ; https://petsc.org/main/manual/ksp/ . External algorithms may allow noncontiguous rank ownership; quality metrics are not directly comparable without the same admissible partition class.
