# HyBIT development status

Last updated: 2026-09-27

## Current release state

- Current published release: HyBIT 0.7.0.
- Release tag: `v0.7.0`.
- Immutable release source commit: `1fdcd6a1b8127c84306c38c3fdbad42563538ad8`.
- The six Rust crates `hybit-core`, `hybit-matrix`, `hybit-krylov`, `hybit-precond`, `hybit-auto`, and `hybit` are published as 0.7.0 on crates.io.
- `hybit-ffi` remains repository-only; the C ABI and C/C++/Fortran consumer examples are built and runtime-tested from the repository.
- The production numerical freeze point for 0.7.0 is r32 (`coarse-first resumable PCG controller`).
- Post-r32 watchdog, local-energy-gate, and filtered spectral-enrichment experiments are intentionally excluded from 0.7.0.

The `v0.7.0` tag is fixed at the validated release commit. Post-release documentation or future development commits on `main` do not change the immutable 0.7.0 source.

## 0.7 production additions

The 0.7 release retains the validated 0.6 structural FEM path and adds a generic algebraic two-level path for real SPD/PCG systems:

1. Graph or contiguous aggregation for the generic algebraic coarse space.
2. Piecewise-constant or one-step Jacobi-smoothed transfer basis.
3. Serial or Rayon-parallel restriction/prolongation.
4. Wide or compact transfer-index storage.
5. F64, F32, or Auto persistent transfer-value storage.
6. Packed factor-solve, parallel explicit-inverse, or Auto coarse application.
7. Coarse-first controller sequencing when algebraic coarse correction is explicitly enabled.
8. Resumable PCG continuation across controller boundaries when the preconditioner is unchanged.
9. Prepared coarse-only reuse across repeated right-hand sides.

The generic automatic path remains real SPD + PCG. The C ABI retains the 0.6 consumer surface and its version query reports 0.7.0.

## 0.7.0 validation record

The exact release source passed the complete `release-candidate-gate.ps1` first on `develop/0.7.0` and again on `main`, both with a clean worktree and without release-qualifying skip switches.

The gate covered:

- source-integrity and workspace/package metadata;
- `cargo fmt --check` and Clippy with warnings denied;
- workspace release tests and all release targets;
- Rust 1.73 MSRV validation;
- Rust/C ABI plus C, C++, and Fortran build/runtime examples;
- crates.io package inspection and dry-run validation;
- the physical-load L-angle structural regression and prepared solve-many reuse.

The L-angle release regression used 358065 free DOFs and 28239653 CSR nonzeros. Structural Auto selected Graph aggregation, a 1398-dimensional rigid-body coarse space, parallel CSR SpMV, parallel preconditioner kernels, and parallel PCG-vector kernels at 8 Rayon workers. It converged in 220 iterations with independently verified relative residual `9.378557e-9`. Prepared solve-many reproduced the same 220-iteration convergence and reuse checks passed.

## Next development phase

Post-0.7 / 0.8 work is separated from the immutable 0.7.0 release. Current research directions include genuine local spectral/GenEO-style coarse enrichment, safer stagnation diagnostics, broader Krylov methods, improved coarse-space selection, symbolic/numerical reuse separation, and larger-scale CPU/GPU/distributed execution.

See `docs/ROADMAP.md` for the forward-looking work list.