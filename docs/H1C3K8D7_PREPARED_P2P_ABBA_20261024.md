# H1-C3K8-D7: experimental prepared P2P Halo (MS-MPI)

Baseline: `develop/0.9.0`, `4ecb34581c72673aee4219cd8cc9f4a06bbf8d37` (D3).

## Scope and safety

- Keep production `spmv_assignment` and existing D1/D2/D3 executables unchanged.
- Add an opt-in experiment-only native `mpi::request::scope` P2P transport, implemented with nonblocking Irecv then Isend then Wait.
- Both trial modes (`alltoallv`, `p2p`) use the same prevalidated D2 length/layout semantics. No per-SpMV collective validation. Buffers/plan and rank mode must remain fixed. Violating this protocol can deadlock MPI; do not promote this API yet.
- Reuse the prepared send/receive buffers, peer counts and displacements. Skip count-zero peers. Constant tag=18307 reserved for the single sequential WORLD Halo traffic; no concurrent Halo calls.
- Prior to solve, compute manufactured RHS by original Alltoallv and independently by P2P, requiring collective **bitwise** equality of output across all ranks. This guard is outside the timed solve.
- The solve retains independently gathered original ROOT CSR true-residual and manufactured-solution checks; experimental SPD curvature screen is not a general SPD proof.

## A/B

- Run `alltoallv,p2p,p2p,alltoallv` for each group of 4/8 ranks, `equal,region,multisource`, Jacobi first.
- Preserve existing rankwise MPI phase breakdown and halo/peer counts. Compare exact CSV-root numerical fields, iteration counts, SpMV/dot counts and ghost totals across A/B.
- Each group carries 4 samples; default 24 jobs, with 5 solve repeats per job.
- One `${Prefix}_summary.csv`. Temporary per-job files removed after successful/failed runs, while completed summary is preserved on failure.
- Native Microsoft MS-MPI only; no MinGW/Open MPI. Benchmark does not commit/push.
- Do not equate rankwise-max non-additive phase totals with an additive decomposition.

## Status

Apply runs rustfmt, strict release Clippy, library and example regressions, builds the D7 executable, verifies source integrity, and stages exactly seven paths. Baseline and payload SHA-256 are pinned. Tested performance / correctness **on Windows MS-MPI remains pending** until the user's logs and CSV are returned.
