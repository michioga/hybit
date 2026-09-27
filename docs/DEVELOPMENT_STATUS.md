# HyBIT 0.7 development status

Last updated: 2026-09-27

## Branch and release state

- Release branch: `develop/0.7.0`
- Published baseline before this release: 0.6.0
- Production freeze point for 0.7.0: r32 (`coarse-first resumable PCG controller`)
- Post-r32 watchdog, local-energy-gate, and filtered spectral-enrichment experiments are intentionally excluded from 0.7.0.
- `main` remains the last published release until the exact 0.7.0 candidate passes the complete release gate and is merged.

## 0.7 production additions

The 0.7 line retains the validated 0.6 structural FEM path and adds a generic algebraic two-level path for real SPD/PCG systems:

1. Graph or contiguous aggregation for the generic algebraic coarse space.
2. Piecewise-constant or one-step Jacobi-smoothed transfer basis.
3. Serial or Rayon-parallel restriction/prolongation.
4. Wide or compact transfer-index storage.
5. F64, F32, or Auto persistent transfer-value storage.
6. Packed factor-solve, parallel explicit-inverse, or Auto coarse application.
7. Coarse-first controller sequencing when algebraic coarse correction is explicitly enabled.
8. Resumable PCG continuation across controller boundaries when the preconditioner is unchanged.
9. Prepared coarse-only reuse across repeated right-hand sides.

The generic automatic path remains real SPD + PCG. The C ABI remains compatible with the 0.6 surface; its version query reports 0.7.0. C, C++, and Fortran examples remain part of the release gate.

## Release-candidate gate

From a clean `develop/0.7.0` worktree, run:

```powershell
.\release-candidate-gate.ps1 `
  -Matrix D:\Work\mf_solver-hybit-export\L-angle-K.mtx `
  -Coordinates D:\Work\mf_solver-hybit-export\L-angle-K.coords `
  -Rhs D:\Work\mf_rhs\L-angle-b.txt `
  -RayonThreads 8
```

The complete gate checks source hashes, workspace/package metadata, `cargo fmt --check`, Clippy with warnings denied, workspace release tests, all release targets, Rust 1.73 MSRV, ABI and C/C++/Fortran build/runtime examples, crates.io package contents/dry-run, and the physical-load L-angle structural regression including independent residual and prepared-reuse checks. Skip switches are for intermediate diagnosis only and do not qualify a commit for publication.

After the complete gate passes, record the exact commit, merge that commit to `main`, rerun the complete gate on `main`, publish the six Rust crates in dependency order, and tag `v0.7.0`.
