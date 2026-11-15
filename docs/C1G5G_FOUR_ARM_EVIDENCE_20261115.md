# C1-G5G — scalar four-arm FEM evidence freeze (2026-11-15)

## Source and owner acceptance

- Pre-measurement HEAD `a2df4d84458fdbc4f11a62454aba918556bfde0d`; experiment source committed as `5394cb4083fb04cbaa20acdcf92898ff7e28ede5`.
- Owner WSL2: 48 separate processes (3 constructed scalar P1 diffusion matrices × 4 RHS × 4 arms), 48 MEASURED, 0 process errors, 30 original-CSR/Kahan-certified converged, 18 non-converged, 12 complete groups.
- The original 147 owner `.log` / `.time` / `.rc` and manifest/CSV/JSON files remain outside Git in `~/c1h11_evidence/c1g5g_scalar_four_arm_czzukxlm`; their actual sizes and SHA-256 are frozen in `benchmarks/evidence/C1G5G_20261115_raw_sha256.csv`.
- `C1G5G_postcommit_audit.py` reconstructs the historical run without rewriting its pre-commit HEAD and cross-checks the 48 frozen elasticity results and six G5F matrix observations. Its own 14 Python tests exercise failure paths.

## Matched outcomes (one measurement per candidate)

| Arm | Scalar converged / 12 | Elasticity converged / 12 |
|---|---:|---:|
| Jacobi PCG | 4 | 7 |
| Hybrid local Cholesky | 2 | 5 |
| Two-Level graph piecewise | 12 | 8 |
| Research Multilevel graph piecewise | 12 | 12 |

- Scalar Hybrid relative to Jacobi: **2 regressions, zero rescues**. Elasticity: **2 regressions, zero rescues**. Four regressions across 24 paired problem groups; not evidence that all Hybrid configurations will always regress.
- Scalar fastest verified converged arms (one-shot end-to-end `inprocess_total_wall_ms`): Jacobi 1/12; research Multilevel 11/12. Elasticity: Jacobi 7/12; Multilevel 5/12. These are descriptive timings, not statistically tested winners.
- Only Multilevel converged on all four `elastic_shear_layers` RHS. Both Two-Level and Multilevel converged on every scalar group.
- The two constructed generator families have different DOF (20736 scalar vs 3280 elasticity). Do not compare raw wall time across them or claim held-out model generalization.
- The frozen G5F features use a diagnostic `b=A*ones`, **not** all four benchmark RHS. They must never serve as outcome labels or production routing switches.

## Integrity and promotion status

- Frozen scalar results SHA-256: `2d81474e62f98fa212ba4a90ed76bbe4973587da7b027ff4ad04b20b256e80d8`.
- Frozen original 147-file index SHA-256: `6dbe932552e460eecb04792dd352a2e8f5fe022df385879dd3dc488adac0dffa`.
- Frozen cross-family post-commit report SHA-256: `827ef058683570eed7bbebb2a71edb04a4da4739c517401fe9a7144ca7be41f5`.
- Original process errors, false convergence and non-finite evidence must be rejected. Non-converged processes stay visible and never count as nominal fastest.
- **Production promotion BLOCKED.** The built-in Rust model remains abstaining, external models are research-only, no new FFI/default/solve behavior, no trained selector or independent physical held-out FEM validation.

## Next validation

Repeat wall-time measurements with cold/warm setup, multiple RHS reuse and memory caps; add externally sourced physical FEM and family-heldout sets; evaluate per-family regret and unsafe-convergence risk separately before designing any new Rust-side routing policy.
