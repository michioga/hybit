# HyBIT

[![CI](https://github.com/michioga/hybit/actions/workflows/ci.yml/badge.svg)](https://github.com/michioga/hybit/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/hybit.svg)](https://crates.io/crates/hybit)
[![docs.rs](https://docs.rs/hybit/badge.svg)](https://docs.rs/hybit)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**HyBIT — Autonomous Hybrid Sparse Solver** is a Rust-first sparse linear-solver framework for large sparse systems arising in FEM and HPC workloads.

HyBIT starts from a low-cost iterative path, observes convergence, identifies numerically difficult degrees of freedom when progress is poor, and can promote bounded local regions to direct Cholesky corrections. ABTM bitmap topology metadata is used internally to expand and organize selected regions. Applications continue to provide ordinary CSR32 matrices.

> **Project status:** HyBIT 0.6.0 is the current development line on `develop/0.6.0`. The latest published crates.io release is 0.5.0. The r25 structural production path is feature-frozen pending the 0.6.0 release gate. The automatic solver path is currently restricted to real symmetric positive-definite (SPD) systems and PCG. APIs may evolve before 1.0.

日本語の説明は [README.ja.md](README.ja.md) を参照してください。

## Highlights

- Rust-first implementation with a stable C ABI for C, C++, and Fortran consumers.
- CSR32 public matrix input; ABTM stays an internal execution/topology backend.
- Matrix Market coordinate import/export for real FEM benchmark interchange.
- PCG with reusable Krylov workspaces.
- Automatic poor-progress probing and selective local direct escalation.
- Multiple hard regions with weighted overlapping Schwarz correction.
- Bounded dense Cholesky factors for selected SPD principal submatrices.
- `analyze -> prepare -> solve-many` execution with reusable workspaces and learned local factors.
- Detailed `SolveReport` diagnostics for convergence, timings, selected regions, factor memory, and reuse.
- MIT licensed.

## Install the published release

The current crates.io release is 0.5.0:

```bash
cargo add hybit@0.5.0
```

The 0.6.0 development tree should be built from this repository until it is release-gated and published.

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
    |-- short Jacobi-PCG probe
    |       |
    |       +-- good progress ------> continue PCG
    |       |
    |       +-- poor progress
    |             |-- residual/risk masks
    |             |-- hard-region components
    |             |-- ABTM halo expansion
    |             |-- local Cholesky factors
    |             +-- Hybrid PCG restart
    v
 cache learned local factors
    v
 prepared solve #2..N
    |-- reuse factors
    |-- reuse Krylov workspace
    +-- skip adaptive probe/factor build
```

## Hybrid preconditioner

For local restriction operators `R_k`, local SPD principal matrices `A_k`, and symmetric overlap weights `W_k`, HyBIT uses the conceptual form

```text
M^-1 = J_uncovered + sum_k R_k^T W_k A_k^-1 W_k R_k
```

For a DOF contained in `m_i` local regions, each local term uses weight `1/sqrt(m_i)`. Jacobi acts on DOFs not covered by any local direct factor. When the preconditioner changes, HyBIT restarts PCG rather than mutating the preconditioner inside an active PCG recurrence.

See [docs/HYBRID_MATH.md](docs/HYBRID_MATH.md) for details.

## Real FEM / Matrix Market benchmark

HyBIT 0.6 adds a Matrix Market path so an assembled, constrained SPD stiffness matrix can be benchmarked without adopting a HyBIT-specific file format.

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
.\build.ps1
.\build-examples.ps1

.\build\hybit_c.exe
.\build\hybit_cpp.exe
.\build\hybit_fortran.exe
```

Prepared execution is available through the C ABI functions `hybit_prepare`, `hybit_solve_prepared`, and `hybit_prepared_destroy`. The C++ wrapper provides an RAII `Prepared` object and the Fortran module exposes matching `ISO_C_BINDING` declarations.

## 0.5.0 release validation baseline

The published 0.5.0 release passed the full Rust/C/C++/Fortran release gate. HyBIT 0.6.0 adds the real-matrix benchmark path and must be validated separately before publication.

The Windows release gate passed Rust tests, the C ABI test, Rust examples, and C/C++/Fortran runtime examples. The adaptive synthetic validation produced the following iteration counts:

| Validation case | Plain Jacobi-PCG | HyBIT Auto | Adaptive regions | Local-factor memory |
| --- | ---: | ---: | ---: | ---: |
| Single difficult SPD block | 33 iterations | 13 iterations | 1 | 34.188 KiB |
| Two difficult SPD blocks | 25 iterations | 13 iterations | 2 | 39.234 KiB |

The prepared solve-many validation reported 13 iterations on the first difficult RHS and 1 iteration on the second RHS, with cached local factors reused and no second factorization.

These are deliberately small synthetic regression problems used to validate control flow and numerical behavior. They are **not** representative application benchmarks and do not imply a general speedup. Large real FEM/HPC systems still need dedicated validation.

## Current scope and limitations

The current 0.6.0 development line intentionally has a narrow numerical scope:

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
benchmarks/       Matrix Market benchmark notes and smoke input
```

## Build and test from source

```powershell
git clone https://github.com/michioga/hybit.git
cd hybit
.\public-release-gate.ps1
```

The public release gate includes runtime/ABI validation and crates.io packaging checks. See [docs/PUBLISHING.md](docs/PUBLISHING.md) before uploading immutable crate versions.

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

See [docs/ROADMAP.md](docs/ROADMAP.md) for future work and [docs/DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md) for the current 0.6 release-candidate checkpoint.

## Contributing

Bug reports, numerical counterexamples, reproducible matrices, API feedback, and performance measurements are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

HyBIT is licensed under the [MIT License](LICENSE).

Repository: https://github.com/michioga/hybit


## HyBIT 0.6 structural FEM experiments

The development branch includes Matrix Market benchmarks for scalar Jacobi,
3x3 block Jacobi, translation-only aggregation, and a geometry-aware six-mode
rigid-body coarse space for 3-D structural systems. Structural Auto now prefers
graph-connected aggregates and falls back to contiguous RCM-order aggregates if
the graph coarse factorization is numerically singular or graph topology contains
components too small for a six-mode rigid-body aggregate. The validated structural
path is exposed through `HybitSolver::solve_structural_csr32` and reusable
`HybitPreparedStructuralSystem`; the generic `solve_csr32` path is unchanged.



### 0.6.0 development checkpoint

Active 0.6 development is performed on `develop/0.6.0`; `main` remains the last validated public-release line until the 0.6 release gate passes. At the r25 checkpoint the structural production path has completed the planned CPU integration work for this release: Graph rigid-body aggregation, packed coarse Cholesky, parallel CSR SpMV, parallel rigid-body fine/coarse transfer kernels, and parallel/fused PCG vector kernels. The next work item is release-candidate stabilization and regression/release-gate validation rather than additional solver features.

## RHS parsing diagnostics (0.6 development)

The FEM benchmark RHS reader accepts plain whitespace-separated `f64` values, ignores blank/comment lines (`#` or `%`), tolerates an UTF-8 BOM, and reports the exact line/token for malformed input.


### 0.6 structural execution policy (development)
Structural Auto can independently select parallel CSR SpMV, parallel rigid-body preconditioner kernels, and parallel/fused dense PCG vector kernels through `StructuralSpmvPolicy`, `StructuralPreconditionerPolicy`, and `StructuralPcgVectorPolicy`. Large structural systems can use all three paths; small systems remain serial to avoid Rayon overhead. The PCG-vector `Auto` path additionally requires at least four workers in the shared Rayon pool. Rayon thread count is controlled externally (for example `RAYON_NUM_THREADS`).

> Development checkpoint (0.6 r25): on the 358065-DOF / 28.24M-nnz L-angle physical-load case, Structural Auto selected Graph aggregation, parallel CSR SpMV, the parallel rigid-body preconditioner, and parallel/fused PCG vectors at 8 Rayon workers. It converged in 220 iterations to a verified relative residual of `9.378557e-9`; solve time was 1.726 s and analysis+prepare+solve was 2.797 s on the Ryzen 7 7800X3D development machine. These timings are machine-specific development measurements, not a general performance claim. The r25 production path is now feature-frozen while the 0.6.0 release gate is prepared.
