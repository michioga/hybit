# H1-C3K8-D9B: external owner labels in Prepared P2P PCG

Base: `develop/0.9.0` HEAD `3c3942e5b49005a9117b06f9e7da081de55060aa` (D7). Native Windows MSVC / Microsoft MS-MPI only. WSL Ubuntu is used **only to construct METIS/SCOTCH labels in D9-A**, not to execute PCG.

## Goal and scope

Compare the **same** verified PCG implementation with the same matrix/RHS, convergence tolerance, preconditioner, MPI ranks, prepared halo, and P2P or Alltoallv transport. Vary **only** partition assignment: equal, ABTM region, ABTM multisource, ABTM multilevel, METIS labels, SCOTCH labels. The source of external labels is the D9-A `quality_summary.csv`, whose owner-label file paths are validated before execution.

The D7 experiment remains an opt-in experiment; production `spmv_assignment` and public MPI APIs are unchanged. Root rank reads the external owner labels via `HYBIT_D9B_OWNER_LABELS`; non-root ranks do not access the labels. The env variable is set by the D9-B benchmark driver and restored upon exit. Rank-zero assignment validates **exactly n unsigned rank IDs**, each `< ranks`, all ranks nonempty, bounded file size; standard collective preflight checks, sparse scatter, coefficient-bit equality and halo-bit equality remain intact.

## Methodology

- Inputs: `thermal1.mtx`; D9-A `k8d9a_thermal1_fix1_quality_summary.csv`; four files `...metis.graph.part.4/.8` and `...scotch_r4/r8.labels` that D9-A produced.
- Methods: Equal, Region, Multi-source, Multilevel, METIS, SCOTCH, at 4 and 8 ranks.
- Each method uses ABBA: Alltoallv, P2P, P2P, Alltoallv. Default `2 ranks x 6 methods x 4 slots = 48 jobs` with 5 timed solves per job.
- `phase_spmv_exchange_ms` etc. are rankwise maxima; these **must not be summed** for additive attribution.
- Require exact invariants within each ABBA group: root original CSR true residual, iteration count, SpMV/dot counts, solution-error fields, ghost totals, and bitwise Halo correctness.
- Summarize the five/optional modes into a single persisted CSV with source-path SHA256 and ABBA speedups. No archive/input files are overwritten.
- D9-A `external_wsl_process_wall_ms` **includes WSL launch, file I/O and graph partition**; the in-solver `partition_ms` for external mode measures **label loading/validation only**. Do **not** compare these times as equal definitions. D9-A common graph export is also separately recorded. Report solve-only rankings and end-to-end rankings distinctly.
- D9-A graph is unweighted `A union A^T`, shared by METIS and SCOTCH. The graph is disconnected (249 original components in thermal1), so quality conclusions should be validated on additional matrices.

## Apply

```powershell
$apply='D:\Work\hybit_0_9_h1c3k8d9b_external_labels_pcg_20261024\APPLY_HYBIT_K8D9B_EXTERNAL_PCG.ps1'
& $apply -RepoRoot D:\Work\hybit\hybit -PreflightOnly
& $apply -RepoRoot D:\Work\hybit\hybit -Apply
```

Only if Apply passes, run:

```powershell
& 'D:\Work\hybit\hybit\benchmarks\scripts\bench-k8d9b-partition-pcg.ps1' `
  -RepoRoot D:\Work\hybit\hybit `
  -Matrix D:\Work\thermal1.mtx `
  -QualityCsv D:\Work\k8d9a_thermal1_fix1_quality_summary.csv `
  -Ranks @(4,8) `
  -Policies @('equal','region','multisource','multilevel','metis','scotch') `
  -Preconditioner jacobi `
  -Warmups 1 -Repeats 5 -MaxIters 1500 -Tolerance 1e-8 `
  -OutputDir D:\Work -Prefix k8d9b_thermal1_pcg
```

Expected: `D:\Work\k8d9b_thermal1_pcg_summary.csv` with 48 rows. Review correctness and performance **before commit/push**. Apply stages exactly five paths, with SHA-256 manifest gate 369 / 367. No commit/push done automatically.

## Limits

- SPD experimental opt-in is **not a general mathematical SPD certificate**.
- A single ABBA group (2 per method/transport) cannot establish all-platform gains. Repeat any close comparisons.
- No WSL execution, MS-MPI run, Rust Clippy or Windows build has been performed when creating this package; those gates run on the local workstation.
- D2/D3's prevalidated flag propagation deficiency remains out of scope and should be corrected separately. D7/D9B uses the correct prevalidated propagation.
