# H1-C3K8-D1 — Instrumented PCG phase profile (research only)

Base: committed, clean `develop/0.9.0` at `dfd6020e9318379ef5d1c395eb8905238baf3042`.
The production code and original K8-C `mpi_k8b_fem_pcg` example are **not modified**.
The new `mpi_k8d1_phase_profile` example is an instrumented clone of K8-C,
with the same CSR, MPI SpMV, PCG recurrences, preconditioners and root-original verification.
The original scientific conclusion remains provisional: curvature probing **does not prove SPD**.

## Phases

- `phase_spmv_halo_ms`: pre-existing `AssignmentSpmvTiming.halo_ns`; pack + native Alltoallv + unpack, excluding validation Allreduce.
- `phase_spmv_numeric_ms`: pre-existing `AssignmentSpmvTiming.numeric_ns`; owned-row numeric sparse multiply.
- `phase_spmv_other_ms`: full SpMV call minus halo/numeric; includes repeated per-call Allreduce input validation and routine overhead. **Not entirely MPI time**.
- `phase_dot_local_ms`: local inner-product floating-point accumulation.
- `phase_dot_allreduce_ms`: native MPI Allreduce for dot products only; includes synchronization waiting.
- `phase_precond_apply_ms`: local application of Jacobi, IC(0) or block Cholesky.
- `phase_vector_update_ms`: PCG `x/r` and search-direction `p` update loops.
- `phase_unattributed_ms`: per-rank solve elapsed minus above; includes the terminal `world_max`, norms, residual-vector setup, loop overhead, etc.

Every phase is aggregated with a **rank-wise maximum**. Maxima may belong to *different* ranks: DO NOT SUM COLUMNS or derive percentages by summing them. Values are selected from the trial at the median (sorted by max-rank solve duration). For even repeats median solve time averages two trial values while phases belong to the upper trial. `median_solve_ms` includes timers' instrumentation overhead, so DO NOT directly compare it as a zero-overhead speed measurement to the old K8-C benchmark. Use old K8-D results as an external reference with that caveat.

Every PCG dot Allreduce is timed separately. The per-SpMV validation Allreduce belongs to `phase_spmv_other_ms`, *not* `phase_dot_allreduce_ms`. SpMV's final 1 or periodic checkpoint calls are included.

## Gate

Strict native Microsoft MS-MPI / Windows MSVC Clippy and unit tests. Release builds both original and profiled examples. K8-A and original K8-C regression checks. Bitwise original matrix scatter checks and final independent root-original true residual remain required. Source Integrity expected 358 manifest / 356 hashes; stage 7 files, no commit/push. No MPI library replacement.

## Suggested matrix and run

`make-k8d1-fem-spring.ps1 -OutputPath D:\Work\k8d1_fem64_shift002.mtx -Side 64 -Shift 0.02`

`bench-k8d1-phase-profile.ps1 -RepoRoot D:\Work\hybit\hybit -Matrix D:\Work\k8d1_fem64_shift002.mtx -Ranks @(1,4,8) -Policies @('equal','region','multisource') -Preconditioners @('jacobi','ic0','block') -OutputDir D:\Work -Prefix k8d1_fem64_shift002`

27 cases, default output **one** `k8d1_fem64_shift002_summary.csv` and no per-case CSV. Optional `-KeepCaseCsv` preserves debugging outputs. The matrix file is intentionally retained for reproducibility.
