# HyBIT 0.9 H1-C3K4 — experimental noncontiguous MPI-PCG

**Status: compile and native MS-MPI numerical/performance testing required. NOT production enabled.**

## Exact scope

- Parent: `develop/0.9.0` at `5240f1c` (K3 experimental single-node MPI SpMV and report already committed).
- Adds the opt-in `mpi_k4_assignment_pcg` executable. No changes to the production `mpi_pcg`, solver dispatch, C/Fortran ABI, or the K3 `mpi_assignment` module.
- Uses the existing K3 `AssignmentOwnedCsrRows`, `prepare_assignment_owned_rows`, `AssignmentHaloPlan`, and `spmv_assignment` to perform a rank-local **Jacobi-PCG** loop. Its dot products use native MPI Allreduce.
- **Not the original numerical FEM problem:** an input Matrix Market file supplies only the *undirected sparsity*. Every rank constructs the identical symmetric, strictly diagonally dominant positive-definite surrogate with off-diagonal weight `-1/sqrt((deg_i+1)(deg_j+1))`, and diagonal `positive_shift + sum(abs(off_diagonals))`. Values of the input matrix are discarded. This prevents unverified SPD assumptions about `thermal1.mtx`.
- Matrix Market rows are scalar only; no real mixed-field metadata, contact, nonsymmetric or indefinite support.
- The executable reads the **full input CSR separately on each rank** and keeps a replicated per-DOF owner map: memory, I/O and startup are intentionally not production distributed.
- The benchmark targets the user's existing **single physical node**. No multi-node claims are authorized by this experiment.

## PCG and correctness contract

The manufactured solution is `x*[g] = sin(0.017(g+1)) + 0.125 cos(0.0031(g+1))` and `b = A_surrogate x*` (formed by one native MPI SpMV per partition). The solver starts from zero and uses positive point Jacobi, ordinary PCG, global dot products, and an explicit `p^T A p > 0` curvature check. It stops only after its recursively updated residual reaches the specified tolerance, then **independently recomputes `||b - Ax||_2 / ||b||_2`** and rejects the solve if that true residual exceeds tolerance. A known-solution max error is also logged, but is not treated as a generic bound for arbitrary SPD systems.

The five methods are identical to K3: `equal`, `graph_contiguous`, `abtm_region_growth`, `abtm_multisource`, `abtm_multilevel`. Their partition and Halo preparation are excluded from `solve_ms` and reported separately. The `partition` topology is computed for the *surrogate*, which may have more nonzeros than the input if the original is not structurally symmetric.

## Timing and metrics

- `read_max_ms`: per-rank replicated input read, max rank.
- `surrogate_max_ms`: generate strictly SPD surrogate from sparsity on each rank, max rank.
- `partition_max_ms`: root graph/ABTM candidate plus owner-map broadcast, max rank.
- `owned_extract_max_ms`: construct noncontiguous owned CSR and extract Jacobi, max rank.
- `halo_setup_max_ms`: native rank-local Halo preparation, max rank.
- `rhs_setup_max_ms`: produce known-solution RHS by a native distributed SpMV, max rank.
- `median_max_solve_ms`: median across repeated solves of each solve's **maximum elapsed time across MPI ranks**. Solve includes PCG updates, MPI point-Jacobi, K3 Halo+numeric SpMV, dot-product Allreduces, and true-residual and solution-error checks.
- `median_max_exchange_sum_ms`: median of per-solve max-rank sums of Halo pack+Alltoallv+scatter durations. This is not equal to network transmission time and omits the dot-product Allreduces. `median_max_numeric_sum_ms` is the corresponding rank-local SpMV arithmetic time.
- `explicit_dot_reductions`: PCG's explicitly counted reductions. Internal validation collectives inside the K3 SpMV are **not** counted in this field.
- `rank_operator_halo_sum_bytes`: only the rank-local CSR/Halo estimate from K3; excludes replicated global input, owner-map copies and scratch/solver vectors. **Not process RSS or peak memory**.
- Timing stages are maxima/reductions collected independently: do not add medians and assert an independently measured end-to-end time. Surrogate conversion and global input read are separate costs, not silently amortized away.

No production automatic ABTM selection, contact support, multi-node scaling, or original-FEM PCG acceleration is established by this experiment.

## Windows validation (MSVC / MS-MPI only)

```powershell
cd D:\Work\hybit\hybit
. .\tools\env\Enter-HybitMsMpi.ps1
cargo fmt --all -- --check
cargo clippy --locked -p hybit-distributed --features mpi --all-targets -- -D warnings
cargo test --release --locked -p hybit-distributed --features mpi --lib
cargo test --release --locked -p hybit-distributed --features mpi --example mpi_k4_assignment_pcg
cargo build --release --locked -p hybit-distributed --features mpi --example mpi_k4_assignment_pcg

$exe = 'D:\Work\hybit\hybit\target\release\examples\mpi_k4_assignment_pcg.exe'
$mpi = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
& $mpi -n 4 $exe 4 D:\Work\thermal1.mtx 16 2 5 3000 1e-10 0.01 D:\Work\k4_thermal1_spd_r4.csv
if ($LASTEXITCODE -ne 0) { throw 'K4 4-rank native PCG failed' }
```

First confirm identical iteration counts and true residual convergence across the five policies and verify that no rank hangs. Compare raw CSV values with K3 only after noting the synthetic SPD and topology/symmetry difference. Then test 1/2/8 ranks; repeat independently on different runs before drawing conclusions about small time differences. Do not commit/push until the Windows results are reviewed.

## Next gates before production promotion

1. Native 1/2/4/8-rank numerical PASS, all 5 methods; inspect true residual and known-solution error.
2. Verify symmetric surrogate does not invalidate the intended partition-quality corpus comparisons; add genuinely SPD FEM matrices and strict input classification separately.
3. Replicate benchmark **process runs**, not just inner repetitions; include load-imbalance and clock/CPU placement sensitivity.
4. True distributed root-row redistribution instead of full CSR replication, compact owner metadata, and accurate resident/peak memory.
5. Stable end-to-end solve-many policy accounting (read, analyze, partition, assemble, prepare, RHS formation, solve and reuse); still no automatic routing until evidence supports it.
6. Multi-node MPI transport evaluation on physical network once hardware is available.
