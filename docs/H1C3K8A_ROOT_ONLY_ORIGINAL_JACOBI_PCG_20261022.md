# HyBIT 0.9 H1-C3K8-A: root-only original-coefficient Jacobi-PCG (experimental)

Base `f70cf8fcbfe5b9363ead3e5ca14ae554b04f0585`, branch `develop/0.9.0`.

## Scope

- Root rank alone reads the original Matrix Market file and selects Equal, ABTM region-growth, or multi-source assignment.
- Uses **K7-D** `scatter_sparse_owners_from_root`, exact root-vs-owned-CSR bitwise check, `prepare_assignment_owned_rows_sparse`, and `spmv_assignment`. No global owner vector is replicated to nonroot ranks.
- Operates on **original numeric coefficients**. No synthetic SPD surrogate. This is not a general solver or a production routing change.
- Strict original-input SPD **sufficient certificate**: sorted unique CSR rows, exact numerical symmetry of opposing coefficients, positive diagonal and strict row diagonal dominance. This intentionally excludes many valid SPD FEM matrices, such as general stiffness matrices; a failed certificate must not be construed as proof of indefinite input.
- Manufactured known solution `sin(0.017*(i+1))+0.125*cos(0.0031*(i+1))` for controlled solve accuracy testing. The right-hand side `A*x_exact` is computed by distributed SpMV, using bytewise-verified original coefficients.
- Classic PCG with local Jacobi, global MPI reductions and guards against zero/negative curvature, NaN and nonconvergence. Recomputes the true residual after each solve.
- Independent **root-original** final residual: collects noncontiguous solution values by global row ID, re-evaluates `b=A_original*x_exact` and `A_original*x_solution` on rank 0, and validates residual norm and solution error without using scattered coefficients. Root retains original matrix until this stage, intentionally increasing diagnostic memory occupancy.
- Reports separate root read, partition, scatter, exact CSR verification, sparse Halo setup, RHS construction, median PCG and root reference verification durations. Operator and sparse owner telemetry are separately reported. Runs are not compared as end-to-end solve gains without paired baselines.

## Tested input envelope and limitations

The included SPD grid producer emits a symmetrical, strictly diagonally dominant, original Matrix Market coefficient set; `--Shift` ensures strict SPD certification. It is a mathematical test fixture, not a real FEM benchmark. The solver rejects nonsymmetric and uncertified SPD inputs rather than attempting PCG unsafely. General FEM SPD certification and GMRES/minres/automatic method routing are later stages. Native **Microsoft MS-MPI**, Windows MSVC, is the sole specified implementation target; multi-host scaling is not yet validated.

`mpi_k8a_sparse_pcg` positional CLI:
`<ranks> <matrix.mtx> <equal|region|multisource> <warmups> <repeats> <max_iters> <tol> <output.csv>`.

Preflight and apply use clean base HEAD, per-file SHA-256, manifest gate, fmt, strict MS-MPI Clippy, release lib+example tests, and release build. They do not automatically commit or push. Numerical MPI validation requires user runs via `bench-k8a-original-pcg.ps1`.


## K8-A benchmark reporting: summary-only default (2026-10-22)

`benchmarks/scripts/bench-k8a-original-pcg.ps1` now retains **only**
`<Prefix>_summary.csv` by default. Rank/policy case CSVs are generated in a
unique temporary directory (the OS temp path) for MPI and imported immediately;
the temporary directory is deleted on success or failure. An updated aggregate
summary is written after every successfully verified case, so a stopped suite
keeps a **partial** summary with completed rows; only the `SUITE PASS` banner
confirms all requested cases ran. To retain all individual case CSVs for
troubleshooting, pass `-KeepCaseCsv`. The script refuses to overwrite an
existing summary, so use a fresh Prefix for each new suite.

The result-file schema and numerical verification remain unchanged. No Rust,
MPI or solver logic is modified by this reporting-only patch.
