# H1-C3K3: native noncontiguous ABTM ownership and MPI SpMV (experiment)

Baseline: `e26e081` (`develop/0.9.0`), clean worktree, K2 accepted and pushed.

K3 adds a **separate**, opt-in `mpi_assignment` module. Existing contiguous `OwnedCsrRows`, `HaloPlan`, `RankLocalCsr`, `mpi_mtx_profile`, MPI-PCG and automatic routing are untouched.

## Ownership and transport

- `PartitionAssignment` retains global DOF -> rank ownership; replicated owner map is agreed with a rank-zero broadcast at setup.
- `AssignmentOwnedCsrRows` stores increasing **noncontiguous** global row IDs; CSR columns remain global IDs until local renumbering.
- A unique sorted ghost request per owner is sent using `MPI_Alltoallv` (u64), then validated against locally owned IDs; send/receive counts are reciprocal.
- Subsequent SpMVs use a reusable `[owned | ghosts]` extended vector, with f64 `MPI_Alltoallv` numeric exchange and a rank-local CSR multiplication.
- Collective pre-validation checks row assignment agreement, buffer lengths, global ownership consistency and request reciprocity. There is no new PCG, solver selector, public FFI or ABI behavior.
- Experimental Matrix Market benchmark makes **one complete CSR copy on each rank**, bounded by input <= 110 MiB, n <= 200000 and expanded nnz <= 8000000. This is not production distributed I/O or a memory scaling claim. Matrix Market has no physical field-block information; the graph policy uses synthetic 1-DOF field blocks.

## Test workflow (native Windows MSVC + MS-MPI)

```powershell
cd D:\Work\hybit\hybit
. .\tools\env\Enter-HybitMsMpi.ps1
cargo fmt --all -- --check
cargo clippy --locked -p hybit-distributed --features mpi --all-targets -- -D warnings
cargo test --release --locked -p hybit-distributed --features mpi --lib
cargo build --release --locked -p hybit-distributed --features mpi --example mpi_k3_assignment_spmv
$mpi = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
$exe = 'D:\Work\hybit\hybit\target\release\examples\mpi_k3_assignment_spmv.exe'
& $mpi -n 4 $exe 4 D:\Work\thermal1.mtx 16 3 15 D:\Work\k3_thermal1_r4.csv
if ($LASTEXITCODE -ne 0) { throw 'K3 native MPI failure' }
pwsh -NoProfile -File tools/gates/source-integrity-gate.ps1
git diff --cached --check
```

Compare Equal, Graph, ABTM Region Growth, ABTM Multi-source and ABTM Multilevel on identical CSR and MPI calls. The CSV records rank-max timings and strict per-apply CSR/ghost numerical checks. Per-policy `partition_max_ms` includes root selection and owner-map broadcast; `halo_setup_max_ms` includes MPI request negotiation. `median_max_spmv_ms` includes numeric `Alltoallv` and multiplication, while `median_max_exchange_ms` excludes multiplication. The two separately reported max-rank substage medians must **not** be added to reconstruct the total.

Run rank 1/2/4/8 only after 4-rank PASS. Use output files that do not already exist. Review logs and raw CSV before committing or pushing.

### Important limitations

A lower Ghost count does not prove lower latency or PCG speedup. Value MPI collective startup, scatter/distribution cost, global owner map, index indirection, operator memory and partition amortization all matter. K3 does not deliver MPI-PCG on noncontiguous ownership; that would require a separate, reviewed change to the solver's global/local-index contract.

## 2026-10-19 native one-node checkpoint

**K3 FIX2 passed on Windows/MS-MPI:** 85/85 MPI-feature library tests; strict Clippy; release example build; source integrity 311/309. All five partition policies completed native MPI SpMV validation for **1, 2, 4, and 8 ranks on ONE physical node**, with each logged `rel_err=0.000e0` at printed precision. In the `thermal1.mtx` single-node experiment, 4-rank Region Growth and 8-rank Multi-source had the fastest observed median SpMV times among the five policies. These are not multi-node/network or full MPI-PCG results.

For the full 20-case table, timing scope, memory/input limitations, calculation of illustrative setup amortization and next-session K4 gates, see [H1C3K3_SINGLE_NODE_RESULTS_20261019.md](H1C3K3_SINGLE_NODE_RESULTS_20261019.md). The user supplied console logs; K3 raw result CSVs were not independently analyzed for this checkpoint. Retain the distinct opt-in MPI assignment path and do not enable production PCG/auto selection.
