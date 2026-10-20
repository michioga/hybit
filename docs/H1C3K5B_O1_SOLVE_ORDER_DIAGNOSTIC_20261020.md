# H1-C3K5-B-O1 — Solve-order rotation diagnostic (2026-10-20)

Base: `9efa616` with **K5-B Fix1 already staged** (6 files; Source Integrity 322/320). The existing K5-B source and numerical/experimental algorithm are unchanged. A separate experimental example is added. This diagnostic is not a production optimizer.

## Trigger: first five independent launches are not reproducibly faster

The user-supplied `k5b_repeat_summary.csv` has 250 unique rows: five independent MPI invocations for each of 4 and 8 ranks, five methods, five checkpoints (1,8,16,32,64 RHS). 64-RHS stage-maxima-inclusive total medians (ms):

| Method | 4 ranks | 8 ranks |
| --- | ---: | ---: |
| Equal | 1903.0481 | 1338.2701 |
| Graph | 2406.98 | 1396.28 |
| ABTM Region Growth | 2479.8154 | 1466.1126 |
| ABTM Multi-source | 2482.5809 | 1350.0874 |
| ABTM Multilevel | 2673.6923 | 1629.2817 |

Against Equal at 64 RHS, 4-rank Region wins **1/5**, 8-rank Multi-source wins **2/5**. The favorable first launch was not a stable effect. The worst true-relative residual in the uploaded combined CSV is `4.977166114614e-11` (criterion `<= 1e-10`), all rows unique.

## Confound observed in source

K5-B prepares all five partitions first and then solves **64 RHS per method, in fixed sequence** Equal, Graph, Region, Multi-source, Multilevel. Since each method occupies the same time slot, a time-dependent execution effect (CPU frequency/thermal state, scheduling, memory/cache pressure or background workload) can be confounded with a policy effect. These are *hypotheses*, not established root causes.

## Diagnostic intervention

The new standalone example `mpi_k5b_order_rotation` supports optional 8th CLI argument `rotate0` (default), `rotate1`, `rotate2`, `rotate3`, `rotate4`.

- Rotate0 runs indices `0,1,2,3,4`; rotate1 runs `1,2,3,4,0`; etc.
- Preparation is still in the original fixed order, once per method. Each method still solves the **same deterministic 64 distinct RHS** with a zero initial iterate. Numerical validation and PCG mathematics are unchanged.
- CSV policy rows remain in the **original method order** even when runtime execution order rotates. A new runner records `solve_slot=1..5` explicitly in its combined CSV.
- Five independent `mpiexec` invocations per rank count place each method in **every solve slot exactly once**. Do not compare a single rotation run against the old baseline as if it had equal operating conditions.
- The sweep checks 25 checkpoints and 320 RHS per invocation; expects 250 combined records for ranks `(4,8)`.

## Execution after preflight / apply

```powershell
& .\benchmarks\scripts\bench-k5b-order-rotation.ps1 `
    -RepoRoot D:\Work\hybit\hybit `
    -Matrix D:\Work\thermal1.mtx `
    -Ranks 4,8 `
    -OutputDir D:\Work `
    -Prefix k5b_o1_rotation
```

The combined output is `D:\Work\k5b_o1_rotation_summary.csv`, plus 10 aggregate CSV and 10 RHS-level CSV files. The runner refuses to overwrite existing output files. Compare 64-RHS times for each policy by slot and rank. If differences track slots rather than policies, investigate time-dependent performance. Repeat randomized/balanced order or CPU-affinity-controlled testing before any positive release claim.

Limitations: one physical node, Windows MS-MPI, synthetic SPD surrogate, full CSR replicated on every rank; stage-maxima approximate prep, not true end-to-end wall time; 5 rotations do not eliminate workload drift across independent launches. **Do not commit/push K5-B based solely on original favorable first-run speedups.**
