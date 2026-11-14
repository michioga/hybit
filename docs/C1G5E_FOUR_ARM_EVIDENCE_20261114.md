# C1-G5E — frozen independent four-arm elasticity evidence (2026-11-14)

## Owner acceptance and provenance

- Branch: `develop/0.10.0`, pre-checkpoint HEAD `04851d7317ae984e8ec15c81e6d7537d0b7144ec`; one independent **constructed** P1 plane-strain elasticity generator family (three matrix variants, 3280 free DOFs each).
- Owner-provided WSL2 release Rust tests **4/4 PASS**, Clippy `-D warnings` PASS and release build PASS after FIX1.
- Owner numerical run: **48 independent processes**, **48 accepted records**, **0 process errors**, **32 original-CSR+compensated-certified Converged**, 16 non-converged; 12 complete four-arm problem groups.
- Original evidence remains machine-local at `/home/michioga/c1h11_evidence/c1g5e_four_arm_15o6lg32`; the Git snapshot stores a SHA-256/length index for all **147** original log/time/rc/manifest/result files. Snapshot cannot substitute for the original raw logs.
- Matrices are generated and validated externally to the library at `~/mtx/c1g5e_tri_p1_elasticity`; matrix SHA and run binary SHA are checked by research evidence gate.
- Two genuinely different **constructed** PDE families now exist (prior scalar diffusion and present elasticity), but only the elasticity family has one-harness four-way candidate comparisons. No externally sourced physical FEM and no held-out model qualification.

## Matched numerical findings

| Candidate | Original-CSR/Kahan converged / 12 | Failed / 12 | End-to-end median among converged only (ms) |
|---|---:|---:|---:|
| `jacobi_pcg` | 7 | 5 | 16.291 |
| `hybrid_local_cholesky` | 5 | 7 | 57.200 |
| `two_level_graph_piecewise` | 8 | 4 | 84.850 |
| `multilevel_graph_piecewise` | 12 | 0 | 36.039 |

- Hybrid factors active: **11/12**. Relative to Jacobi: **2 regressions**, no observed rescue in this sample.
- Nominal fastest among verified converged: Jacobi **7/12**; native research Multilevel **5/12**; Two-Level/Hybrid **0/12**. These are one-run in-harness descriptive minima, **not a learned policy or statistical speedup**.
- All four `elastic_shear_layers` RHS cases converged **only with Multilevel**. `elastic_uniform` had all candidates converged, with Jacobi nominally fastest in four cases. `elastic_stiff_lens` exhibited two Hybrid regressions.
- Cost comparison uses `inprocess_total_wall_ms`; overlapping internal Hybrid `setup_ms` and `solve_ms` must not be added. Median is conditional on convergence and not a universal performance comparison.

## Frozen files and integrity

- Original audited 48-row CSV SHA-256: `9b06a1fc0185822690d35a1b7754a5f40272aa3593a3606c090309b1cb906c2c`.
- Original audited summary SHA-256: `0fe4f5df2d4b1f5746140cc445d88c11fce92a948cc51c25bb9001015f6a1129`.
- Raw evidence index (147 files) SHA-256: `de27daee7e62c093a668379df69a52d2f93c9320fe7249edf1e3eebce3968d0e`.
- Full per-case result: `benchmarks/evidence/C1G5E_20261114_results.csv`, `..._summary.json`.
- Reproducible outcome report CSV/JSON and script: `C1G5E_20261114_outcome_audit.*`, `benchmarks/scripts/C1G5E_outcome_audit.py`.
- Research generators, solver example, and 48-process evidence runner remain separate from the production `hybit-auto` selection path; the embedded model continues to abstain.

## Follow-up and promotion gate

1. Keep every failure visible and retest with repeated trials, separate cold/warm setup and multiple RHS reuse.
2. Establish physically grounded external FEM/physical RHS families and a held-out family test before fitting/admitting a model.
3. Validate selection only inside Rust `hybit-auto` Shadow first. The external Python audit is for development, **not runtime routing**.
4. **Production promotion BLOCKED.** Neither the embedded nor optional external model may autonomously enable unqualified native Multilevel or override SPD/memory/true-residual contracts.
