# HyBIT development status

Last updated: 2026-10-05

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

Current validated development sequence is complete through F7. The F2 base
commit is `782ac38`; F3 adds prepared multi-RHS Natural/RCM ILU(0) reuse
measurement and documentation on top of that checkpoint.

### Execution architecture

Completed CPU checkpoints include common problem/execution/backend policy,
resident Krylov abstraction, serial resident PCG, Rayon vector execution,
parallel CSR SpMV, parallel resident Jacobi, and structural resident
cross-check.

GPU is an architectural target, not a production target here.

### GeneralSquare E1-F7

1. E1 — restarted FGMRES with reusable `V`/`Z` workspace.
2. E2 — prepared `GeneralSquare` routing.
3. E4a — staged restart escalation.
4. E4b — budget-aware restart controller.
5. E5 — prepared canonical ILU(0).
6. F1 — Natural/RCM ILU(0) ordering sensitivity.
7. F2 — real nonsymmetric corpus preflight and larger-case evidence.
8. F3 — repeated prepared multi-RHS ordering/reuse and end-to-end amortization.
9. F4 — ILU(0) application/topology profiling; per-level Rayon triangular apply rejected.
10. F5 — explicit unsuitable-ILU fallback with retained Identity safety path and real missing-diagonal validation.
11. F6 — Natural/RCM ordering-selection signals, paired short probes, and amortized policy replay; no automatic production promotion.
12. F7 — Jacobi -> ILU(0) promotion study; strong ILU wins and strong regressions observed, with no validated automatic production selector.

Jacobi + fixed restart 30 remains default.

E5 ILU(0) canonicalizes unsorted/duplicate CSR, introduces no fill, stores
diagonal positions as `u32`, uses a selective `1e-12` row-relative factor-pivot
floor, and is reused across solve-many.

The public hard-B cross-check reproduced 276 iterations, true relative residual
`9.455610e-9`, 4.238 MiB ILU state, 4.500 MiB FGMRES(3) workspace, 8.738 MiB
total persistent state, and zero adjusted pivots.

F1-F4 ordering/kernel work now shows three distinct real-nonsymmetric regimes:
`sherman5` with no measured RCM convergence benefit, `raefsky3` with robust
cross-RHS RCM benefit, and `venkat25` with RHS-dependent convergence changes
and only a small repeated 5-RHS end-to-end timing difference. Bandwidth and a
single RHS are therefore insufficient automatic-reordering signals.

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

GeneralSquare robustness is now closed through F7 for the current 0.8
development checkpoint.

Production semantics remain explicit:

- Jacobi is the GeneralSquare default;
- canonical ILU(0) remains opt-in;
- `Ilu0Fallback` remains the explicit missing-diagonal safety policy;
- Natural/RCM ordering remains explicit rather than automatically selected.

ABTM G1 scalar topology algebra is validated on the ten-matrix corpus.
The next checkpoint is G2 metadata-first support intersection and sparse-dot
product pruning with candidate/executed/skipped-work metrics. GeneralSquare
automatic-selection research can be revisited later with a broader held-out
corpus or application-provided solve-horizon/cost information.

Do not add ILUT, fill, pivoting, or hidden automatic preconditioner changes as
part of the F7 conclusion.

### ABTM G1

G1 validates a numerical-value-independent sparse-of-64-bitmaps topology layer
with AND/OR/AND-NOT/XOR, popcount, rank/select, invariants, occupancy telemetry,
and general Boolean merge semantics.

The ten-matrix corpus shows that bitmap topology metadata is strongly compact
on several matrices but not universal: low-occupancy cases can be break-even or
worse than CSR metadata. Logical topology is therefore accepted while physical
prepared layout selection remains adaptive and workload-specific.

Word-local rank/select is the intended packed-value addressing primitive.
Row-wide rank/select remains a convenience/correctness API rather than a hot
numeric-kernel path.

Next: G2 metadata-first support intersection and sparse-dot pruning.
