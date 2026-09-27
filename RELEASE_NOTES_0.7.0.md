# HyBIT 0.7.0 release notes

HyBIT 0.7.0 extends the published 0.6 structural-FEM/PCG foundation with a generic algebraic two-level coarse path and a controller that preserves active PCG recurrence whenever the preconditioner is unchanged. The release remains focused on real symmetric positive-definite systems and PCG.

## Highlights

- Generic algebraic two-level coarse correction integrated with `HybitSolver`.
- Graph aggregation for generic coarse spaces, alongside contiguous aggregation.
- Piecewise-constant and one-step Jacobi-smoothed transfer bases.
- Galerkin coarse operator construction from the same persistent transfer representation used at solve time.
- Serial and Rayon-parallel restriction/prolongation.
- Wide and compact transfer-index storage.
- F64, F32, and Auto transfer-value storage.
- Packed factor solve, parallel explicit inverse, and Auto coarse-apply policy.
- Coarse-first execution when algebraic coarse correction is explicitly enabled.
- Resumable PCG sessions across controller stage boundaries when the preconditioner does not change.
- PCG restart only when local-direct strengthening actually changes the preconditioner.
- Prepared coarse-only state reuse across repeated RHS solves.
- The existing C ABI and C/C++/Fortran consumer surface are retained; the ABI version query reports 0.7.0.
- The repository-only FFI package emits the `hybit` C ABI as a `cdylib` only, avoiding a redundant Rust `rlib` name collision with the public `hybit` facade crate while preserving `hybit.dll` for external-language consumers.

## Controller correction in r32

Earlier 0.7 checkpoints could spend an initial Jacobi probe, discard that Krylov history, and restart under an explicitly requested coarse preconditioner. r32 starts the requested algebraic coarse preconditioner at iteration zero and interprets `probe_iterations` as a controller observation boundary. If no strengthening is admitted, the same `PcgSession` continues. This preserves conjugacy and avoids a destructive probe/restart sequence.

On the development `boneS01` cross-check, the direct coarse path converged in 331 iterations; the earlier Jacobi-probe-then-coarse sequence required 391 total iterations. r32 matched the uninterrupted 331-iteration coarse path. This measurement is a regression reference for that matrix and configuration, not a general performance guarantee.

## Transfer and coarse-apply policies

The 0.7 development line added independent policies for transfer application, transfer-index storage, transfer-value storage, and coarse application. The high-level generic algebraic-coarse defaults validated before r32 use parallel transfer, wide indices, and Auto transfer values; Auto may select F32 persistent transfer values when representable. The low-level compatibility defaults remain conservative where documented.

## Structural FEM path

The validated 0.6 structural path remains available: Graph rigid-body aggregation with contiguous fallback, six rigid-body modes per aggregate, packed coarse Cholesky, parallel CSR SpMV, parallel rigid-body transfer/preconditioner kernels, parallel/fused PCG vector kernels, and prepared structural solve-many reuse. The 0.7 generic algebraic path does not remove or replace this API.

## Scope

HyBIT 0.7.0 remains pre-1.0 experimental numerical software. The automatic path targets real `f64` square SPD systems and PCG. MINRES, GMRES/BiCGStab, distributed memory, GPU execution, and out-of-core execution are not part of this release. Validate residuals and application-level physical results independently for engineering use.

## Not included in 0.7.0

The post-r32 experiments involving late-progress watchdogs, local residual-energy admission gates, and filtered spectral-enrichment prototypes are intentionally excluded from 0.7.0. They remain research material for the next development line.

## Publication gate

The exact release commit must pass `release-candidate-gate.ps1` without skip switches, including Rust 1.73 MSRV, workspace tests, Clippy, source integrity, package metadata, C ABI plus C/C++/Fortran build/runtime checks, and the physical-load L-angle structural regression. Publication then proceeds in dependency order as documented in `docs/PUBLISHING.md`.
Release hygiene is also tightened: when run inside a Git worktree, the source-integrity gate now requires the tracked-file set to match `MANIFEST.txt`, preventing stale experimental or generated files from being accidentally committed into a release candidate.

