# C1-G5H — Test existing additive T+H against frozen T alone (2026-11-15)

Parent HEAD: `f8d5b6993b159a9d7ce66b47c2eb9f817227bdde`. Research only; no production Auto / FFI / `hybit-precond` or selector weights modified.

## Design

The private existing `AlgebraicTwoLevelHybrid` in `hybit-auto/src/lib.rs` first applies `TwoLevelBlockJacobiPreconditioner` then calls `HybridPreconditioner::add_local_correction`. The research example exercises **two distinct experiments**:

1. `tplus_h_adaptive`: **actual existing library Auto path**. Starts with graph piecewise Two-Level (512 target coarse modes, one DOF per node); may add RHS-driven local Cholesky after a 12-iteration probe and 64-iteration true-residual monitor with 0.99 rate trigger. Max one local escalation, eight 128-DOF regions and 64 MiB *local* factor budget. If no local factor activates, report `TwoLevelOnlyNoLocal`, NOT an additive-local benefit.
2. `tplus_h_static`: explicit *research mirror* of the existing additive formula, `TwoLevel.apply(r,z)` then `Hybrid.add_local_correction(r,z)`. Eight **deterministic index-spaced, disjoint 128-DOF windows** are Cholesky factored from the same original matrix. This is an actual local-term activation control, not an optimized/risk-selected hard patch; its failure says nothing about best possible local regions.

Each branch uses a single process, original CSR, four deterministic RHS (`ones`, `index_smooth`, `seeded_random`, `point_load`), tolerance 1e-10, iteration cap 600, single thread, original-CSR and compensated true residual, and GNU time peak RSS. Six SHA-256-pinned **constructed** FEM matrices across scalar P1 diffusion (20736 DOF) and plane-strain elasticity (3280 DOF): total **48 new process records** (6 x 4 x 2). Match against **24 previously frozen T-only rows** in `benchmarks/evidence/C1G5G_20261114_results.csv` and `C1G5E_20261114_results.csv` with exact RHS fingerprints.

The baseline and candidate measurements are on different dates/commits; **one-shot end-to-end time changes are descriptive only**. The physical model and matrix size differ by family. The 64 MiB budget constrains local factors; the coarse factor/transfer, matrix, workspaces, and analysis are accounted separately and peak RSS is measured. `factor_bytes` in the research record is `coarse_bytes + local_bytes`; do not conflate it with RSS.

## Rejection and full cohort

Every process produces `.log`, `.time`, `.rc`. Nonzero rc is `PROCESS_ERROR` (not silently removed); `MaxIterations`, `Breakdown`, nonfinite residual and falsely claimed `Converged` never count as rescue. Every `Converged` must pass original-CSR and Kahan compensated residual. A T+H success with zero local factor is marked `COARSE_ONLY_RECOVERY_NOT_LOCAL_RESCUE`. The static placement is fixed before any RHS performance is seen. Preserve the complete grid, not cherry-picked successes.

*This is not independent external FEM, a held-out selector test, timing repeat, or grounds to promote Multilevel/Hybrid or change production routing.*

## Decision gate after results

First inspect actual local factor use. Compare T and T+H convergence/true residual, memory and cold end-to-end cost; flag rescues and regressions. If static T+H is harmful while adaptive is coarse-only, prioritize local placement and residual locality diagnostics, not blindly promoting additive composition. In a later step test physically localized risk masks and spectral low-energy probes on held-out input families.
