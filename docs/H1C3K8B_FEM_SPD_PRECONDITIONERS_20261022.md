# H1-C3K8-B: SPD FEM-screening and fixed preconditioners (experimental)

Base: `develop/0.9.0` at `e404fab5ee6f72c018c6705618cf050e31258a0e`.
No production PCG/autoselection changes. Root still holds the original global CSR and assignment.

## Implemented

- Accept K8-A strict diagonal dominance first.
- Admit exactly symmetric **anchored weak diagonally dominant M-matrices**: nonpositive off-diagonal entries, weak row dominance, and a strictly dominant row within every connected component.
- For at most **512 DOFs**, additionally admit exactly symmetric, finite, positive-diagonal matrices after a bounded *numerical Cholesky screen* with pivot safeguards. This is a numerical test, **not** a formal SPD proof for arbitrary floating-point input.
- No unchecked SPD classification for arbitrary larger FEM stiffness; larger non-dominant matrices are rejected explicitly.
- K7-D Root-only sparse owner scattering and original CSR bit verification unchanged.
- Convert K7-D prepared local CSR to the existing contiguous-local-index preconditioner data convention (temporary copy) and **reuse** `LocalBlockIc0` and `LocalBlockCholesky`. The global row IDs remain noncontiguous; only per-rank indexing is local.
- Compare `jacobi`, `ic0`, and `block` modes with common PCG and root-original residual verification.
- Preconditioners are fixed and positive by construction upon successful factorization. A successful local factor does NOT establish global SPD.
- Keep collective failure checks before solve. Nonpositive curvature and convergence failures reject, without fake success or silent fallback.
- Default benchmark writes **one `<Prefix>_summary.csv`**. Pass `-KeepCaseCsv` only for diagnostics.

## Limitations

This experiment does not establish general SPD detection for large elasticity FEM matrices and does not implement a general indefinite or nonsymmetric solver. FEM-like verification generator assembles 2-DOF spring energy with grounding, not a full continuum elasticity element model. Numerical Cholesky at 512 DOFs is intentionally capped to limit root setup costs. The preconditioner CSR copy is temporary and a future optimization target.

## Suggested experiment

```powershell
& .\benchmarks\scripts\make-k8b-fem-spring.ps1 -OutputPath D:\Work\k8b_fem8.mtx -Side 8 -Shift 0.5
& .\benchmarks\scripts\bench-k8b-fem-pcg.ps1 -RepoRoot D:\Work\hybit\hybit -Matrix D:\Work\k8b_fem8.mtx -Ranks @(1,4) -Policies @('equal','region','multisource') -Preconditioners @('jacobi','ic0','block') -BlockSize 8 -OutputDir D:\Work -Prefix k8b_fem8
```

This is 18 cases (2 ranks * 3 policies * 3 preconditioners) and a single summary CSV. Validate SPD-screen route, iterations, factor memory, true residual, and root-original residual. `ic0` may reject matrices with nonpositive local pivots; that rejection is a valid safety behavior, not an automatic fallback.
