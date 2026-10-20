# HyBIT 0.9 H1-C3K5-C — Large-matrix original-coefficient SpMV diagnostic

## Scope

- Base: `develop/0.9.0`, HEAD `9efa616`, with the **nine already staged** K5-B/O1 files; this is a further experimental addition, **not** a branch rewrite.
- Adds an example and benchmark scripts without modifying production MPI-PCG, K5-B, or O1 files.
- MS-MPI, one physical Windows node only, ranks 4 or 8 by default.
- Three partition methods: Equal, ABTM Region Growth, ABTM Multi-source. Expensive Graph/Multilevel intentionally excluded from this cache-scaling probe; preserved in K5-B/O1.
- SpMV executes **original Matrix Market numerical coefficients**; no surrogate and no PCG. Numerical results are cross-checked against serial CSR SpMV on the same original matrix, and exchanged ghost values are checked.

## Why K5-C

The Ryzen 7 7800X3D has eight physical cores and 96 MiB shared L3 cache. `thermal1` with ~574k nonzeros can be L3-resident, yielding misleading conclusions about DRAM-bound SpMV and the value of ABTM partitioning. A larger sparse matrix may have a different balance among local numeric multiply, memory access, and MPI Halo exchange. **It does not follow automatically that ABTM is faster**: larger matrices may cause numerical SpMV to dominate and make reducing Halo traffic less impactful.

`boneS01.mtx` (~127k rows, ~5.5m nonzeros) exercises a much larger CSR. The included streaming Python generator can create a deterministic SPD 7-point stencil Matrix Market input with `120 x 100 x 100` nodes (1.2m DOF; ~8.3m expanded CSR nonzeros). The full CSR data bytes per rank (4-byte rowptr/index, 8-byte values) exceed the 96 MiB L3 capacity for that grid; actual cache residency also depends on additional vectors and memory allocation. The generated matrix is **not** the original `boneS01` nor a mechanical FEM stiffness matrix.

## Build / execution

Apply via `APPLY_HYBIT_K5C_LARGE_SPMV.ps1` PreflightOnly then Apply. It checks the baseline staged state, builds the new experimental MPI example, runs tests and Clippy, verifies Source Integrity, and leaves **13** files staged, with no commit/push.

```powershell
cd D:\Work\hybit\hybit

& .\benchmarks\scripts\bench-k5c-large-spmv.ps1 `
  -RepoRoot D:\Work\hybit\hybit `
  -Matrix D:\Work\boneS01.mtx `
  -Ranks 4,8 -Rotations 0,1,2 -Warmups 3 -Repeats 64 `
  -ProjectedSpmvCalls 8000 -OutputDir D:\Work -Prefix k5c_boneS01
```

Optional larger structured stencil input:

```powershell
python .\tools\bench\generate-k5c-grid7-mtx.py `
  --nx 120 --ny 100 --nz 100 `
  --output D:\Work\k5c_grid7_120x100x100.mtx

& .\benchmarks\scripts\bench-k5c-large-spmv.ps1 `
  -RepoRoot D:\Work\hybit\hybit `
  -Matrix D:\Work\k5c_grid7_120x100x100.mtx `
  -Ranks 4,8 -Rotations 0,1,2 -Warmups 3 -Repeats 64 `
  -ProjectedSpmvCalls 8000 -OutputDir D:\Work -Prefix k5c_grid7
```

If the target matrix is not supported or the local system becomes memory-constrained, **do not increase limits blindly**. K5-C loads a global CSR **on every MPI rank**. Bounds are 384 MiB Matrix Market input file, 1.2m DOF, and 16m nnz in loaded CSR. Peak parser, partition, and Halo allocation overhead is not captured by CSR bytes. The test is not multi-node scalability or a memory-efficient distributed loader.

## CSV semantics

One file per `(rank count, rotation)`, each with 3 policy rows. Aggregated `*_summary.csv` includes a row per experiment and `*_platform.txt` records CPU model / physical/logical core counts / installed RAM. These are **static metadata**, not live temperature, clock, power, or affinity measurements.

- `partition_max_ms`, `extraction_max_ms`, `halo_setup_max_ms`: rank-wise maximum of stage wall time. Their sum is an **approximate preparation cost**, not a synchronized single wall clock.
- `median_max_spmv_ms`: median across measured calls of per-call maximum MPI-rank time (Halo plus local numeric SpMV; the profiling allreduce is outside that timed call).
- `median_max_halo_ms`, `median_max_numeric_ms`: separately measured per-call maxima. Independent maxima do not add algebraically to the total.
- `measured_spmv_sum_ms`: **actually measured** sum of repeat-call max times. Includes neither setup nor serial cross-check; all three policies use identical vector sequences alternating between two manufactured vectors.
- `projected_prepared_total_ms = partition + extraction + halo + median_max_spmv_ms * ProjectedSpmvCalls` is a **projection**, not measured end-to-end solve-many. Default 8000 calls approximates prior 64 RHS PCG SpMV order of magnitude and deliberately excludes PCG Allreduce, vector updates, RHS generation, and validation costs.
- `global_csr_bytes_per_rank`: lower-bound representation bytes, **per replicated rank**. `operator_halo_sum_bytes`: rank-local operators and Halo across all ranks, not additional peak process RSS.
- Numerical validation occurs on early and last trial (including both alternating vector cases early). The serial CSR reference multiplication is computed **outside** the timed repeated multiply and is not included in setup. Using two reused vectors may still favour cache reuse compared with PCG's changing direction vectors.

## Interpretation and release gate

1. Require all 4-/8-rank methods and rotations to pass native MS-MPI numerical tests (`worst_rel_error <= 1e-11`).
2. Compare **both** measured local numeric SpMV, measured Halo, and measured complete SpMV time, not just ghost counts.
3. Compare early and later order slots to recognize frequency/OS scheduling/cache effects on the 7800X3D.
4. Compare `thermal1`, `boneS01`, and generated grid separately; do not extrapolate the graph-stencil result to mechanical FEM.
5. If large-matrix runs fail due to memory, reduce input and proceed to root-scatter / genuinely distributed input work.
6. Commit only after Windows/MSVC/MS-MPI compile, native numerical checks, staged Source Integrity, and a documented results review.
