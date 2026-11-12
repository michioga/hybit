# P3-C: 6-DOF/node synthetic structural SPD — research pilot

**Owner WSL synthetic numerical gate PASS (2026-11-12); research-only, not production default.** Base: `develop/0.10.0` commit `55366bf` (2026-11-12), Rust 1.99.1. This is NOT an assembled beam or shell finite-element mesh. It is a 24-DOF model with three translation-like and three rotation-like degrees of freedom per node. The next stage must use independently validated real FEM matrices.

## Problem definition

The six SPD cases are constructed as `A = scalar D (K ⊗ S) D`, where `K` is the 4-by-4 symmetric tridiagonal with 2.0 on the diagonal and -0.65 on the two off-diagonals, and `S` is a 6-by-6 matrix with 1.4 on the diagonal and 0.025 elsewhere. `K` is SPD by strict diagonal dominance with positive diagonal; `S = 1.375 I + 0.025 11^T` is SPD. `D` has strictly positive diagonal entries. Positive congruence and a positive scalar preserve SPD. This construction is an independent mathematical SPD proof.

Cases: all-balanced, rotation scales 1e-4, rotation scales 1e-8, staggered 1e-3 / 1e-6 / 1e-8, large rotation scale 1e4, and uniform scalar 1e-16; plus two negative controls: deliberately asymmetric rotational coupling and a symmetric negative rotational diagonal.

The matrix is stored in CSR32. The initial RHS is `b = A x_reference` with deterministic nonzero x. All reference true residuals are independently computed from the *original* CSR, with a scaled Euclidean norm. For every accepted SPD matrix, build a full 24-DOF local Cholesky region and three overlapping 12-DOF regions, check the positivity of `r^T M^-1 r`, and match the factor-memory estimate. P3-B equilibrated factor bytes must exceed the original by exactly `24 * sizeof(f64)` for the full region on common accepted cases.

Also run a **local research-only reference PCG**, NOT production HyBIT PCG, with Jacobi, full-region, and overlapping-region preconditioning. The global iteration cap is 96 and target verified true relative residual 1e-10. This is designed to check preconditioner integration and false-convergence prevention; it is NOT evidence of production solver rescue. Negative controls are never intentionally passed to PCG even if a legacy factor mistakenly accepts them.

## Acceptance and limits

- Feature OFF preserves baseline observations and may reject constructed SPD or admit negative controls. Report these as numerical-contract discrepancies rather than hiding them.
- Feature ON must accept all six mathematically SPD systems and reject both negative controls. For accepted SPD, require finite original-CSR relative residual <= 1e-8, strictly positive quadratic forms for full/overlap preconditioners, exact persistent-memory accounting, finite forward-error observations, no false certified reference-PCG convergence, and no iteration cap overshoot.
- Forward error can remain large on ill-conditioned systems even when verified residual is tiny; do not equate the two.
- This is only a positive/negative numerical-contract pilot; no automatic default promotion, production rescue success-rate, MPI implication, or real assembled structural FEM claim.
- The P3-A/B raw evidence remains unchanged; preserve the existing separate paths under `~/c1h11_evidence`.

## WSL owner procedure

After extracting the ZIP in `D:\Work`, from the `~/hybit` WSL repository:

```bash
cd ~/hybit
PKG=/mnt/d/Work/HyBIT_0_10_P3C_MIXED_DOF_AUDIT_20261112
bash "$PKG/P3C_apply.sh" --dry-run "$PWD"
bash "$PKG/P3C_apply.sh" --apply "$PWD"
bash "$PKG/P3C_validate.sh" "$PWD"
```

Results are written once to `~/c1h11_evidence/p3c_mixed_dof_XXXXXX`, including the original outputs, feature-specific binaries and SHA256, a comparison CSV, a summary JSON, elapsed time, peak RSS, and source hashes. Do not commit/push until the owner verifies the outputs.

## Owner WSL P3-C acceptance (2026-11-12)

Owner evidence: `~/c1h11_evidence/p3c_mixed_dof_FO5WqK`.
Base commit: `55366bf`. Rust 1.99.1 feature OFF/ON example tests, Clippy, Python numerical gate (10 tests), and the P3-C synthetic audit completed successfully on WSL2. Original evidence hashes are retained outside the repository.

- Eight independent-classification observations: six constructed SPD and two negative controls. Feature OFF has three unexpected SPD rejections; feature ON has **zero** contract discrepancies.
- All accepted SPD cases have independently checked original-CSR true relative residuals within `1e-8`; sampled full/overlap preconditioner quadratic forms are positive.
- Persistent local factor accounting: 3216 bytes feature OFF, 3408 bytes feature ON (192 additional bytes for 24 equilibrated DOFs, in the same accepted cases).
- For the six accepted SPD cases with feature ON, research reference PCG takes 8 Jacobi iterations, 4 overlap iterations, or 1 full-region iteration at tolerance `1e-10` (cap 96). The Jacobi control already converges, therefore **natural convergence rescues = 0**.
- This is a **4-node, 6-DOF/node, 24-DOF synthetic structural matrix** rather than an assembled beam/shell FEM problem or production HyBIT PCG evaluation. No default feature promotion or real-FEM rescue claim follows.
- Forward errors remain separate from small true residuals. The feature-ON `beam_rot_1e8` case has approximately `1.0812e-9` relative forward error despite a `2.7009e-16` local true relative residual.
