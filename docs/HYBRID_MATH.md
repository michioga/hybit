# HyBIT 0.5 Hybrid Preconditioner — Mathematical Notes

HyBIT 0.5 remains restricted to real SPD systems and PCG.

Let the global SPD matrix be `A`. The controller starts with the configured base SPD preconditioner: algebraic two-level coarse when explicitly enabled, otherwise Jacobi. The initial `probe_iterations` segment is only a controller boundary; if the preconditioner is unchanged, HyBIT resumes the same PCG recurrence. When that stage shows poor progress, the policy identifies hard-core DOF sets and grows each set independently through the ABTM graph by a configurable number of halo layers. For local region `H_k`, HyBIT forms the principal matrix

`A_k = A[H_k, H_k]`.

Because a principal submatrix of an SPD matrix is SPD, exact Cholesky is mathematically valid in exact arithmetic. The implementation additionally checks numerical symmetry and positive pivots.

## Symmetric overlap weighting

Regions may overlap. Let `m_i` be the number of local regions containing global DOF `i`. Within every local region containing `i`, HyBIT uses

`w_i = 1 / sqrt(m_i)`.

For restriction operator `R_k` and diagonal local weight `W_k`, the local correction is

`R_k^T W_k A_k^-1 W_k R_k`.

Each such term is symmetric positive semidefinite. Jacobi is retained only for global DOFs with zero local multiplicity. Consequently every global DOF is covered by at least one positive contribution while a single fully covering local block reduces to the exact local inverse without double-counting Jacobi.

The conceptual preconditioner is

`M^-1 = J_uncovered + sum_k R_k^T W_k A_k^-1 W_k R_k`.

This construction is intended to remain compatible with standard PCG assumptions. HyBIT restarts PCG only when selective-direct strengthening changes the preconditioner. Controller/telemetry boundaries with an unchanged preconditioner retain the same resumable PCG session and therefore preserve conjugacy.

## Bounded memory

Dense local Cholesky factors scale quadratically in local region order. HyBIT therefore caps each expanded local factor and caps the number of local regions. In 0.5 the defaults are 128 DOFs per factor and 8 factors.

The report exposes both total local factor DOFs (overlap counted per factor) and unique covered DOFs, plus an estimate of bytes actually owned by local factors, indices, overlap weights and multiplicity metadata.

## Limitations

The current hard-region detector is heuristic. It combines matrix coupling/scale diagnostics with residual seeds; it is not an estimator of the global condition number. Later releases should incorporate KrylovScope/Lanczos/Ritz information and reusable factor policies.
