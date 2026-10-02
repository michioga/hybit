# HyBIT development status

Last updated: 2026-10-02

## Published release

- Current published release: HyBIT 0.7.0.
- Release tag: `v0.7.0`.
- Immutable source:
  `1fdcd6a1b8127c84306c38c3fdbad42563538ad8`.
- Six Rust crates are published as 0.7.0 on crates.io.
- `hybit-ffi` remains repository-only.
- 0.7 production numerical freeze point: r32.

The `v0.7.0` tag is immutable. 0.8 development does not change the published
0.7 source.

## Current develop/0.8.0 checkpoint

E5 commit:

`18f2b564069637d8fe66a17f6d1bddc7e6d262b3`

### Execution architecture

Completed CPU checkpoints include common problem/execution/backend policy,
resident Krylov abstraction, serial resident PCG, Rayon vector execution,
parallel CSR SpMV, parallel resident Jacobi, and structural resident
cross-check.

GPU is an architectural target, not a production target here.

### GeneralSquare E1-E5

1. E1 — restarted FGMRES with reusable `V`/`Z` workspace.
2. E2 — prepared `GeneralSquare` routing.
3. E4a — staged restart escalation.
4. E4b — budget-aware restart controller.
5. E5 — prepared canonical ILU(0).

Jacobi + fixed restart 30 remains default.

E5 ILU(0) canonicalizes unsorted/duplicate CSR, introduces no fill, stores
diagonal positions as `u32`, uses a selective `1e-12` row-relative factor-pivot
floor, and is reused across solve-many.

The public hard-B cross-check reproduced 276 iterations, true relative residual
`9.455610e-9`, 4.238 MiB ILU state, 4.500 MiB FGMRES(3) workspace, 8.738 MiB
total persistent state, and zero adjusted pivots.

These are regression measurements, not universal performance claims.

### Current routing limitations

- `Spd` -> PCG/Hybrid/structural.
- `GeneralSquare` -> FGMRES on `Auto`/`Cpu`.
- `SymmetricIndefinite` -> recognized; MINRES not implemented.
- no rectangular LSQR/LSMR;
- no complex arithmetic;
- ILU(0) triangular solve is serial;
- C ABI reports FGMRES/ILU0 codes but does not yet expose GeneralSquare
  configuration setters.

## Published 0.7 validation

The exact 0.7 release source passed source-integrity, metadata, formatting,
Clippy, workspace release tests, Rust 1.73 MSRV, C/C++/Fortran ABI/runtime,
package/dry-run, and physical L-angle gates.

The L-angle release regression used 358065 free DOFs and 28239653 CSR nonzeros,
a 1398-dimensional rigid-body coarse space, 220 iterations, and independently
verified relative residual `9.378557e-9`.

## Next validation focus

Before automatic ILU selection:

- natural versus RCM/other meaningful orderings;
- representative real nonsymmetric FEM/PDE matrices;
- larger sizes and repeated RHS;
- serial triangular-solve profiling;
- safe fallback behavior when ILU is unsuitable.

See [ROADMAP.md](ROADMAP.md) and [GENERAL_SQUARE.md](GENERAL_SQUARE.md).