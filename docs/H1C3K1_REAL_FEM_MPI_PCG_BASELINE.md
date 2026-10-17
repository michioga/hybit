# H1-C3K1 — Real FEM / Matrix Market MPI-PCG baseline

Status: **experimental Equal/Jacobi baseline NUMERICAL PASS on Windows/MS-MPI; K1 source remains staged, not committed**.
Baseline: `develop/0.9.0` HEAD `b28dd0c` (H1-C3J checkpoint).

## Purpose

The H1-C3J evidence is from a 168-DOF synthetic matrix and *modeled* 10,000
SpMV calls. K1 establishes a genuine end-to-end MPI-PCG numerical and timing
baseline using an actual user-supplied Matrix Market stiffness matrix. **It
only benchmarks Equal contiguous ownership and the existing Jacobi MPI-PCG.**
K2 will compare Field and ABTM Graph partitions against this exact reference.

This checkpoint deliberately reuses the existing `mpi_mtx_profile` example,
`mpi_mtx` root-only parser/distributor, `prepare_owned_rows` reciprocal MPI
halo and `DistributedPcg` true-residual verification. No solver routing,
public library API, MPI transport, field metadata or GPU path changes.

## Scope and numerical evidence

- Root reads `.mtx`, checks positive diagonal and numerical symmetry. Under
  `certified`, the current parser also demands strict row diagonal dominance;
  only then is positive definiteness certified by this sufficient condition.
  **Many legitimate SPD FEM matrices do not satisfy this sufficient test** and
  must not be claimed certified by the existing parser.
- `ExperimentalUnverified` is explicit opt-in and **does not certify SPD**.
  Successful manufactured-solution PCG runs are evidence of numerical
  consistency at the chosen RHS and tolerance, not mathematical SPD proof.
- A deterministic global truth vector is applied through *rank-owned rows*
  to form `b=A*x_true`; each solve returns a fresh true relative residual and
  a max-norm solution error. No full CSR is replicated on MPI workers.
- This K1 patch tightens the `mpi_mtx_profile` gate: **every measured repetition**
  must converge and meet both tolerances; it records worst measured true
  residual and maximum solution error, rather than validating only the last
  repetition. CSV schema is unchanged.
- A strict 24-DOF identity-matrix control runs at 1/2/4/8 ranks before the
  user matrix to verify the MPI input and solve path.

## Observability

The new PowerShell harness `benchmarks/scripts/bench-h1c3k1-real-fem-mpi.ps1`
repeats independent MPI processes for the actual matrix at 1/2/4/8 ranks,
retains each native CSV, records `mpiexec` process wall clock time, and
produces a merged CSV and full log. Source Integrity and compiler gates are
required before interpreting data. Reported fields include:

- root read, rank-owned distribution, Halo setup and PCG prepare time;
- median/min/max of max-rank **real solver** execution time over repeated
  solves, iterations, SpMV and Allreduce counts;
- max-rank solve imbalance and unique global ghost counts;
- true relative residual and manufactured-solution max absolute error;
- process wall time inclusive of MPI startup/IO (not comparable to internal
  per-solve time or a production steady-state run).

**Not measured here:** physical variable-DOF layout, Field/Graph partition
comparison, ABTM structural construction and reuse, factor memory, allocator
high-water mark or process peak RSS, nonlinear contact, independent physical
reference displacement, full-N synthetic-model costs. Do not report these as
K1 achievements. Raw `.mtx` does not supply field-block metadata.

## Required Windows commands

```powershell
cd D:\Work\hybit\hybit
$env:CLANG_PATH = 'C:\Program Files\LLVM\bin\clang.exe'
$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
$env:MSMPI_INC = 'C:\Program Files (x86)\Microsoft SDKs\MPI\Include\'
$env:MSMPI_LIB64 = 'C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\'

cargo fmt --all -- --check
cargo clippy --locked -p hybit --features mpi --all-targets -- -D warnings
cargo test --release --locked -p hybit -p hybit-distributed -p hybit-matrix -p hybit-ffi
cargo test --release --locked -p hybit-distributed --features mpi --lib
pwsh -NoProfile -File tools/gates/source-integrity-gate.ps1
```

Use a real matrix the user currently has (for example `thermal1.mtx`). Choose
`-Policy ExperimentalUnverified -PermitUnverifiedSpd` **only** if the input
cannot meet the strict sufficient SPD certificate and is independently
plausible as SPD. With unknown/indefinite matrices, a numerical failure is
valid evidence, not a parser regression. Do not silently shift the diagonal.

```powershell
$script = '.\benchmarks\scripts\bench-h1c3k1-real-fem-mpi.ps1'
& $script -Matrix 'D:\Work\thermal1.mtx' -Label thermal1 `
    -Policy ExperimentalUnverified -PermitUnverifiedSpd -PreflightOnly
& $script -Matrix 'D:\Work\thermal1.mtx' -Label thermal1 `
    -Policy ExperimentalUnverified -PermitUnverifiedSpd -Run
```

The preflight does not build or run MPI. The real matrix can be tested with
`-Policy Certified` (omitting `-PermitUnverifiedSpd`) when the input meets
the strict sufficient test. `-MaxEstimatedNnz` bounds a conservative
symmetric-storage expansion estimate **before** the potentially expensive
root Matrix Market parser; it does not measure actual in-memory peak usage.

## Acceptance and next step

- All gates, strict control and actual FEM 1/2/4/8 runs numerically PASS;
  failed cases remain visible, and experimental SPD status stays unverified.
- At least 3 independently launched FEM runs per rank, `Warmups=3`,
  `InnerRepeats=5` by default; retain CSV and logs.
- Compare solve and total process time without mixing the two scopes.
- Next H1-C3K2: create a **separate** root matrix/field layout policy and
  genuine MPI-owned redistribution bridge for Equal/Field/ABTM Graph, reusing
  the H1-C3J scoped candidate without replicating full CSR on workers.

No tag, release, `cargo publish`, or production auto-routing from K1.


## K1-F2 diagnostic extension and native validation — 2026-10-17

**Source/baseline:** branch `develop/0.9.0`, still uncommitted on HEAD
`b28dd0c` (H1-C3J). Source Integrity **301 manifest / 299 SHA-256** PASS.
Rustfmt, strict `cargo clippy --locked -p hybit --features mpi --all-targets
-- -D warnings`, non-MPI release regressions and 81/81 MPI-feature library tests
passed. One later Clippy rerun initially failed in `mpi-sys`/`bindgen` because
the PowerShell session lacked a full MSVC/LLVM environment; `Launch-VsDevShell.ps1`
and LLVM `bin` on `PATH`, with MS-MPI SDK variables, restored a successful
Clippy/release build. That environment failure was not a Rust source regression.

### Diagnostic reason for the changed stopping tolerance

The actual `thermal1.mtx` input has **82,654 DOF** and **574,458 expanded
nonzeros** (declared Matrix Market entries 328,556). At the original/default
`relative_tolerance=1e-9`, each 1/2/4/8-rank run ended with PCG status
`Converged` after 1304 iterations and true relative residual of approximately
`9.76e-10` to `9.95e-10`; however, the manufactured-solution max absolute
error was `7.90e-7` to `7.94e-7`. All **12/12** real-matrix runs correctly
FAILED the unchanged `2.0e-7` forward-error gate. The 24-DOF certified
control passed 4/4. This failure is valid benchmark evidence, not a solver
or MPI crash and not an SPD certificate.

K1-F2 adds an optional `relative_tolerance` argument to `mpi_mtx_profile`
and `-RelativeTolerance` to the harness. The legacy default `1e-9` is
preserved. The acceptance threshold for *true residual* tracks the requested
solver tolerance (`<= 1.1 * rtol`); the independent maximum absolute error
gate remains **`2.0e-7`**, unchanged. All measured repeats are checked and a
residual-only success is labeled `CONVERGED_RESIDUAL_ONLY` rather than
`NOT_CONVERGED`.

### Completed strict numerical and reproducibility gates

- Actual native Windows/MS-MPI runs used `ExperimentalUnverified`,
  `rtol=1e-11`, `max_iterations=5000`, manufactured RHS and Jacobi PCG;
  the existing Equal contiguous distribution was unchanged. This is an
  opt-in *numerical* test; **SPD is not mathematically certified**.
- Diagnostic pilot: 1/2/4/8 ranks × 1 independent MPI launch, 1 warmup and
  2 measured solves; certified 24-DOF controls **4/4 PASS**, actual FEM
  **4/4 PASS**. Worst manufactured error approximately `5.7e-9`.
- Reproducibility gate: certified 24-DOF controls **4/4 PASS**; actual FEM
  **12/12 PASS** = 1/2/4/8 ranks × 3 independent MPI launches,
  `Warmups=3`, `InnerRepeats=5` per launch. **0 failures / 16 cases**.
  `Source Integrity 301/299 PASS`. Each reported solve number below is the
  **median of max-rank solve durations** across five measured solves in one
  launch, then **median across three independent launches**; no synthetic
  modeled-iteration extrapolation.

| MPI ranks | Median solve (ms) | Speedup vs 1 rank | Parallel efficiency | PCG iters | True relative residual | Max manufactured error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1485.4827 | 1.0000× | 100.0% | 1564 | 9.808674132905e-12 | 5.695293930907e-9 |
| 2 | 834.7466 | 1.7796× | 89.0% | 1564 | 9.765736375804e-12 | 5.671638936278e-9 |
| 4 | 448.6422 | 3.3111× | 82.8% | 1563 | 9.937682221780e-12 | 5.733512553618e-9 |
| 8 | 297.4493 | 4.9941× | 62.4% | 1563 | 9.924406343422e-12 | 5.720240614515e-9 |

The per-launch solve medians (ms) are: 1r `[1485.4827, 1497.2555,
1445.3616]`, 2r `[834.7466, 845.8364, 827.6126]`, 4r `[448.6422,
456.2315, 436.8800]`, 8r `[297.4493, 297.2800, 300.3551]`.
Measured PCG iteration count and accuracy were reproducible for a given rank
count; small changes in floating-point reduction order can lead to 1563 vs
1564 iterations. These values are **steady-state solver time**, not total
process elapsed time. `mpiexec`/MPI startup, input parsing/distribution,
setup, warmups, output, and repeated solves contribute separately to process
wall time. Full peak process RSS was **not measured**.

Raw artifacts on the testing machine:

```
D:\Work\h1c3k1_results\k1_summary_20261017T100505439Z.csv
D:\Work\h1c3k1_results\k1_runs_20261017T100505439Z.log
```

### Acceptance boundary / next K2 task

**Accept K1 as a numerical and baseline-timing checkpoint**, conditional on
final staged-source regression and Git source-integrity gates before commit.
This benchmark does **not** show ABTM-vs-CSR or Equal-vs-Field/Graph
performance because all runs used the **same Equal CSR/Jacobi path**.
`thermal1.mtx` in `ExperimentalUnverified` mode is not SPD-certified, the
manufactured solution does not validate FEM physics, and neither peak memory
nor a real production full application run was established. K2 should preserve
identical matrix, stopping tolerance, preconditioner, RHS, and MS-MPI
conditions across Equal, Field and ABTM Graph paths and account for graph
preparation, distribution, Halo, PCG setup and solves separately. Real
field-aware layout must come from an external physical DOF description;
Matrix Market alone has no element/field metadata.

No production auto routing, tag, release, or publish is authorized by K1.
