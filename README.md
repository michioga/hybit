# HyBIT

[![CI](https://github.com/michioga/hybit/actions/workflows/ci.yml/badge.svg)](https://github.com/michioga/hybit/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/hybit.svg)](https://crates.io/crates/hybit)
[![docs.rs](https://docs.rs/hybit/badge.svg)](https://docs.rs/hybit)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**HyBIT — Autonomous Hybrid Sparse Solver** is a Rust-first sparse linear-solver framework for large sparse systems arising in FEM and HPC workloads.

HyBIT starts from a low-cost iterative path, observes convergence, identifies numerically difficult degrees of freedom when progress is poor, and can promote bounded local regions to direct Cholesky corrections. ABTM bitmap topology metadata is used internally to expand and organize selected regions. Applications continue to provide ordinary CSR32 matrices.

> **Project status:** HyBIT 0.7.0 is the current published release. It is available from crates.io and tagged as `v0.7.0` in this repository. The 0.7 release adds a generic algebraic two-level coarse path and a coarse-first resumable PCG controller while retaining the validated 0.6 structural path. The automatic solver path remains restricted to real symmetric positive-definite (SPD) systems and PCG. APIs may evolve before 1.0.

日本語の説明は [README.ja.md](README.ja.md) を参照してください。

> **Development branch:** `develop/0.8.0` adds the resident execution
> architecture and a prepared `GeneralSquare` FGMRES path with Jacobi or
> opt-in ILU(0). These APIs are not part of the published 0.7.0 release.
> Start with [docs/README.md](docs/README.md) and
> [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for development-branch usage.
## Highlights

- Rust-first implementation with a stable C ABI for C, C++, and Fortran consumers.
- CSR32 public matrix input; ABTM stays an internal execution/topology backend.
- Matrix Market coordinate import/export for real FEM benchmark interchange.
- PCG with reusable Krylov workspaces and resumable sessions across unchanged controller stages.
- Optional generic algebraic two-level coarse correction with graph aggregation, Jacobi-smoothed transfer, configurable transfer storage, and coarse-apply policy.
- Automatic poor-progress probing and selective local direct escalation; PCG restarts only when the preconditioner actually changes.
- Multiple hard regions with weighted overlapping Schwarz correction.
- Bounded dense Cholesky factors for selected SPD principal submatrices.
- `analyze -> prepare -> solve-many` execution with reusable workspaces and learned local factors.
- Detailed `SolveReport` diagnostics for convergence, timings, selected regions, factor memory, and reuse.
- MIT licensed.

## Install

The HyBIT 0.7.0 release uses the `hybit` facade crate together with its internal Rust crates:

```bash
cargo add hybit@0.7.0
```

HyBIT 0.7.0 is published on crates.io; the matching release source is tagged `v0.7.0`.

Rust 1.73 or newer is required.

## Quick start

```rust
use hybit::{Csr32Matrix, HybitSolver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // SPD tridiagonal matrix:
    // [ 2 -1  0 ]
    // [-1  2 -1 ]
    // [ 0 -1  2 ]
    let a = Csr32Matrix::new(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
    )?;

    let b = vec![1.0, 0.0, 1.0];
    let mut x = vec![0.0; 3];

    let solver = HybitSolver::new();
    let report = solver.solve_csr32(&a, &b, &mut x)?;

    println!("x = {x:?}");
    println!(
        "status={:?}, iterations={}, relative_residual={:.3e}",
        report.status, report.iterations, report.relative_residual
    );
    Ok(())
}
```

A convenience one-shot API is also available:

```rust
let (x, report) = hybit::solve(&a, &b)?;
```

## Analyze, prepare, solve many

For repeated right-hand sides against an unchanged matrix, use a prepared context:

```rust
use hybit::HybitSolver;

let solver = HybitSolver::new();
let analysis = solver.analyze_csr32(&a)?;
let mut prepared = solver.prepare_csr32(&a, &analysis)?;

let mut x1 = vec![0.0; a.nrows()];
let report1 = prepared.solve(&a, &b1, &mut x1)?;

let mut x2 = vec![0.0; a.nrows()];
let report2 = prepared.solve(&a, &b2, &mut x2)?;
```

The first difficult RHS may trigger adaptive region detection and local Cholesky construction. Later RHS vectors reuse the learned Hybrid preconditioner and the PCG workspace while the matrix remains bitwise unchanged.

## Numerical pipeline

```text
CSR32 matrix
    |
    v
 analyze
    |-- MatrixProfile
    |-- SPD baseline checks
    |-- backend policy
    |-- structure/value signatures
    v
 prepare
    |-- Jacobi preconditioner
    |-- reusable PCG workspace
    |-- optional ABTM backend
    v
 prepared solve #1
    |-- choose base preconditioner
    |       |-- algebraic coarse when explicitly enabled
    |       +-- Jacobi otherwise
    |-- short controller PCG stage
    |       |
    |       +-- good progress ------> continue the same PCG session
    |       |
    |       +-- poor progress
    |             |-- residual/risk masks
    |             |-- hard-region components
    |             |-- ABTM halo expansion
    |             |-- local Cholesky factors
    |             +-- restart only if the preconditioner is strengthened
    v
 cache learned local factors / coarse state
    v
 prepared solve #2..N
    |-- reuse coarse/local factors
    |-- reuse Krylov workspace
    +-- cached local Hybrid skips adaptive diagnostics/factor build
```

## Hybrid preconditioner

For local restriction operators `R_k`, local SPD principal matrices `A_k`, and symmetric overlap weights `W_k`, HyBIT uses the conceptual form

```text
M^-1 = J_uncovered + sum_k R_k^T W_k A_k^-1 W_k R_k
```

For a DOF contained in `m_i` local regions, each local term uses weight `1/sqrt(m_i)`. Jacobi acts on DOFs not covered by any local direct factor. When the preconditioner changes, HyBIT restarts PCG rather than mutating the preconditioner inside an active PCG recurrence.

See [docs/HYBRID_MATH.md](docs/HYBRID_MATH.md) for details.

## Real FEM / Matrix Market benchmark

HyBIT 0.6 introduced a Matrix Market path, retained in 0.7, so an assembled, constrained SPD stiffness matrix can be benchmarked without adopting a HyBIT-specific file format.

```powershell
cargo run --release -p hybit --example fem_bench -- `
  --matrix D:\path\to\K.mtx `
  --tol 1e-8 `
  --max-iters 3000
```

If `--rhs` is omitted, the benchmark sets `x_exact = 1` and constructs `b = A*x_exact`. It then compares plain Jacobi-PCG with HyBIT Auto and independently recomputes `||Ax-b||/||b||` for both results. See [benchmarks/README.md](benchmarks/README.md).

## C, C++, and Fortran

The repository contains a C ABI and thin language bindings under `include/` and `fortran/`. On Windows the Rust core builds `hybit.dll`; MinGW consumers use a generated GNU import library.

```powershell
.\tools\build\build.ps1
.\tools\build\build-examples.ps1

.\build\hybit_c.exe
.\build\hybit_cpp.exe
.\build\hybit_fortran.exe
```

Prepared execution is available through the C ABI functions `hybit_prepare`, `hybit_solve_prepared`, and `hybit_prepared_destroy`. The C++ wrapper provides an RAII `Prepared` object and the Fortran module exposes matching `ISO_C_BINDING` declarations.

## 0.7.0 release focus

HyBIT 0.7.0 keeps the 0.6 structural FEM execution path and adds the generic algebraic coarse work validated through the r23-r32 development checkpoints. The release includes:

- algebraic two-level coarse correction for generic SPD/PCG solves;
- Graph aggregation and one-step Jacobi-smoothed transfer basis;
- serial/parallel transfer application and wide/compact transfer-index storage;
- F64/F32/Auto persistent transfer-value storage;
- factor-solve / explicit-inverse / Auto coarse application;
- coarse-first controller sequencing for explicitly enabled algebraic coarse correction;
- resumable `PcgSession` continuation whenever the preconditioner is unchanged;
- prepared coarse-only reuse across solve-many RHS vectors.

The 0.7 release does **not** include the later r33-r36 watchdog, energy-gate, or filtered spectral-enrichment experiments. Those remain post-0.7 research work. Performance measurements in the benchmark scripts are regression evidence, not universal speedup claims.

## Current scope and limitations

HyBIT 0.7.0 intentionally has a narrow numerical scope:

- real `f64` matrices;
- square SPD systems on the automatic path;
- PCG as the automatic Krylov method;
- CSR32 public storage and an internal ABTM backend;
- dense local Cholesky factors with bounded region sizes;
- up to 8 local regions by default, each limited to 128 DOFs by default;
- prepared-factor reuse only when matrix structure and coefficient bits are unchanged;
- prepared contexts are intended for single-threaded use;
- no MINRES, GMRES, BiCGStab, distributed memory, GPU, or out-of-core execution yet;
- the geometry-aware rigid-body coarse correction is currently specific to the explicit 3-D structural path and is not a general algebraic multigrid implementation;
- adaptive region selection is currently heuristic rather than spectral.

The project should therefore be treated as experimental numerical software. Validate residuals and physical results independently before using it in engineering decisions.

## Repository layout

```text
crates/
  hybit-core      common traits, errors, options, reports
  hybit-matrix    CSR32, ABTM, Matrix Market I/O, matrix analysis, masks
  hybit-krylov    PCG and reusable Krylov workspace
  hybit-precond   Jacobi, local Cholesky, weighted Schwarz, rigid-body two-level
  hybit-auto      adaptive solver policy and prepared contexts
  hybit           public Rust facade crate
  hybit-ffi       C ABI DLL layer (repository build, not published to crates.io)
include/          C and C++ headers / Windows .def file
fortran/          Fortran ISO_C_BINDING module
docs/             architecture and numerical notes
examples/         C/C++/Fortran build examples
benchmarks/       benchmark scripts, Matrix Market notes/data, ignored local results
tools/build/      current-workspace build and external-language build helpers
tools/gates/      version-neutral repository validation gates
tools/release/    retained version-specific release qualification tooling
```

## Documentation

The development documentation is built with mdBook and published at
<https://michioga.github.io/hybit/>. The 0.8 development site is generated
from the Markdown sources under `docs/`.

Build it locally on Windows with:

```powershell
.\tools\docs\build-mdbook.ps1
```
## Build and test from source

For ordinary development checks:

```powershell
git clone https://github.com/michioga/hybit.git
cd hybit
.\tools\build\build.ps1
```

HyBIT 0.7.0 was qualified with `release-candidate-gate.ps1` on the exact source commit later tagged as `v0.7.0`. The gate covers source integrity, metadata, formatting, Clippy, Rust 1.73 MSRV, ABI/language bindings, package validation, real-FEM residual/iteration checks, and prepared reuse. The L-angle files are supplied externally rather than stored in the repository:

```powershell
.\tools\release\0.7\release-candidate-gate.ps1 `
  -Matrix D:\Work\mf_solver-hybit-export\L-angle-K.mtx `
  -Coordinates D:\Work\mf_solver-hybit-export\L-angle-K.coords `
  -Rhs D:\Work\mf_rhs\L-angle-b.txt `
  -RayonThreads 8
```

See [docs/PUBLISHING.md](docs/PUBLISHING.md) before uploading immutable crate versions.

For Cargo-only development:

```bash
cargo test --workspace --release
cargo run --release -p hybit --example hybrid
cargo run --release -p hybit --example multiregion
cargo run --release -p hybit --example prepared
cargo run --release -p hybit --example fem_bench -- --matrix benchmarks/data/poisson5.mtx
```

## Roadmap

Near-term work is focused on real FEM validation, separation of symbolic reuse from numerical refactorization, broader Krylov coverage, stronger diagnostics, and scalable local/coarse corrections. Parallel CPU, GPU, and distributed-memory backends are longer-term directions.

See [docs/ROADMAP.md](docs/ROADMAP.md) for post-0.7 / 0.8 work and [docs/DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md) for the current published-release status.

## Contributing

Bug reports, numerical counterexamples, reproducible matrices, API feedback, and performance measurements are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

HyBIT is licensed under the [MIT License](LICENSE).

Repository: https://github.com/michioga/hybit


## HyBIT 0.6 structural FEM foundation

The 0.7 source tree retains the Matrix Market benchmarks for scalar Jacobi,
3x3 block Jacobi, translation-only aggregation, and a geometry-aware six-mode
rigid-body coarse space for 3-D structural systems. Structural Auto now prefers
graph-connected aggregates and falls back to contiguous RCM-order aggregates if
the graph coarse factorization is numerically singular or graph topology contains
components too small for a six-mode rigid-body aggregate. The validated structural
path is exposed through `HybitSolver::solve_structural_csr32` and reusable
`HybitPreparedStructuralSystem`; the generic `solve_csr32` path is unchanged.



### 0.6 structural baseline retained in 0.7

The validated 0.6 structural production path is retained unchanged as the structural baseline in 0.7: Graph rigid-body aggregation, packed coarse Cholesky, parallel CSR SpMV, parallel rigid-body fine/coarse transfer kernels, and parallel/fused PCG vector kernels. The 0.7 release adds the generic algebraic coarse/controller work without replacing that structural API.

## RHS parsing diagnostics

The FEM benchmark RHS reader accepts plain whitespace-separated `f64` values, ignores blank/comment lines (`#` or `%`), tolerates an UTF-8 BOM, and reports the exact line/token for malformed input.


### Structural execution policy
Structural Auto can independently select parallel CSR SpMV, parallel rigid-body preconditioner kernels, and parallel/fused dense PCG vector kernels through `StructuralSpmvPolicy`, `StructuralPreconditionerPolicy`, and `StructuralPcgVectorPolicy`. Large structural systems can use all three paths; small systems remain serial to avoid Rayon overhead. The PCG-vector `Auto` path additionally requires at least four workers in the shared Rayon pool. Rayon thread count is controlled externally (for example `RAYON_NUM_THREADS`).

> Historical 0.6 r25 regression reference: on the 358065-DOF / 28.24M-nnz L-angle physical-load case, Structural Auto selected Graph aggregation, parallel CSR SpMV, the parallel rigid-body preconditioner, and parallel/fused PCG vectors at 8 Rayon workers. It converged in 220 iterations to a verified relative residual of `9.378557e-9`; solve time was 1.726 s and analysis+prepare+solve was 2.797 s on the Ryzen 7 7800X3D development machine. These timings are machine-specific development measurements, not a general performance claim. The r25 structural path remains the retained 0.7 structural regression baseline.

### Hybrid coarse-dimension sweep

`bench-fem-hybrid-coarse-sweep.ps1` runs the Plain Jacobi-PCG baseline only once with the selective-direct reference, then uses `--skip-plain` while sweeping algebraic coarse targets. The default targets are `384, 512, 768, 1024, 1536`. A compact table is printed and the measurements are exported to `hybit-hybrid-coarse-sweep.csv`.
