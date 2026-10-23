# H1-C3K8-D2: MS-MPI SpMV collective-guard A/B and Halo breakdown

Baseline: `develop/0.9.0` `d463eed30ed840d39caec73a3659427ab23370ce` (K8-D1).

## Scope

- Preserve production `spmv_assignment` and existing D1 example unchanged.
- Introduce a separate experimental `spmv_assignment_d2_profile` and `mpi_k8d2_spmv_ab` example.
- `checked`: each SpMV validates fixed operand lengths via one MPI Allreduce, as in K8-C/D1.
- `prevalidated`: each *solve* collectively validates the immutable Halo plan, local operator, owned-vector and workspace lengths before the timed region; SpMV repeats local checks but omits the per-call collective guard. All MPI ranks select the same mode. A mutated layout after preflight violates the contract and can cause MPI deadlock; this is not yet a production API.
- The zero-ghost single-rank case is still exercised via MPI (and is a useful control).

## Timings

D1 fields plus: `phase_spmv_validation_ms`, `phase_spmv_pack_ms`, `phase_spmv_exchange_ms`, `phase_spmv_unpack_ms`, `d2_mode`.
`phase_spmv_exchange_ms` measures MPI `Alltoallv` including waiting for peers. It is not pure network transit.
The rank-wise maximum per phase is **not additive**; sample corresponds to middle ranked total solve time for 5 repeats.
`phase_spmv_halo_ms` equals rank-local pack + exchange + unpack, before max reduction.

## Numerical acceptance

Both modes must pass independent Root original CSR true residual verification and preserve the manufactured solution. The PowerShell suite compares iteration count, SpMV/dot counts and CSV-rounded root and distributed residuals/solution errors **exactly** across A/B modes; any mismatch is a failure. Input must be independent SPD-justified: the K8-C Rayleigh screen is not an SPD proof.

## Reporting

Default one durable `${Prefix}_summary.csv` for all modes and cases; case CSVs temporary unless `-KeepCaseCsv` is explicitly supplied. On failure, retain completed rows only. Start with 4 and 8 ranks, three partition policies, Jacobi (12 runs). Optional IC(0)/Block expand the matrix. Native Microsoft MS-MPI only.

## Status

Git-controlled experiment; Apply stages exactly 7 paths and never commits or pushes. Rustfmt / strict Clippy / regressions and native build run on the user's Windows MSVC machine. Correctness and performance on target hardware are not claimed before logs are returned.
