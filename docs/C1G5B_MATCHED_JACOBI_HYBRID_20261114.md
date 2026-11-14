# C1-G5B — frozen matched Jacobi/monitored-Hybrid evidence (2026-11-14)

## Acceptance and scope

- Owner WSL `--dry-run` / `--resume` succeeded: Python FIX1 24/24 tests, Rust 2/2 tests, release Clippy/build, **24 independent processes / 12 paired problems**.
- Frozen original source HEAD: `f121d6327414dad8830c50c770fe33650a932c5d` on `develop/0.10.0` (not the later checkpoint commit).
- Three constructed 20,736-DOF matrices × 4 algebraic/manufactured RHS, one family only; original matrix SHA and the exact frozen A9/C1-G4 RHS fingerprints are verified by `C1G5B_gate.py`.
- Common policy: `tol=1e-10`, 600 global PCG iterations, zero initial state, one CPU thread, independent Jacobi and monitored-Hybrid process; CSR and Kahan residuals independently checked.
- **Outcome: 2 BOTH_CONVERGED / 8 BOTH_NOT_CONVERGED / 2 HYBRID_REGRESSION / 0 LOCAL_RESCUE.** Nonconvergence remains a recorded result and is not classified as a gate failure.
- A Hybrid arm with zero local regions is effectively **JacobiOnly**, not a successful local Cholesky intervention.

## Full matched numerical outcomes

| Matrix | RHS | Classification | Local factors active | Jacobi true residual | Hybrid true residual |
| --- | --- | --- | --- | ---: | ---: |
| island_shell | ones | HYBRID_REGRESSION | yes | 9.429489065320e-11 | 2.279002474814e-7 |
| island_shell | index_smooth | BOTH_NOT_CONVERGED | yes | 3.703019230403e-6 | 7.205151153682e-8 |
| island_shell | seeded_random | BOTH_CONVERGED | no | 9.889969673741e-11 | 9.889969673741e-11 |
| island_shell | point_load | BOTH_CONVERGED | no | 9.343893727528e-11 | 9.343893727528e-11 |
| multi_patch | ones | BOTH_NOT_CONVERGED | yes | 1.152374938405e-9 | 1.742869135085e-5 |
| multi_patch | index_smooth | BOTH_NOT_CONVERGED | yes | 1.010312382187e-8 | 2.355336527278e-4 |
| multi_patch | seeded_random | HYBRID_REGRESSION | yes | 9.837296698266e-11 | 4.420605086698e-8 |
| multi_patch | point_load | BOTH_NOT_CONVERGED | yes | 3.588792571483e-7 | 7.113312556840e-4 |
| rotated_anisotropy | ones | BOTH_NOT_CONVERGED | no | 1.956624440314e-7 | 1.956624440314e-7 |
| rotated_anisotropy | index_smooth | BOTH_NOT_CONVERGED | no | 1.869184368127e-7 | 1.869184368127e-7 |
| rotated_anisotropy | seeded_random | BOTH_NOT_CONVERGED | no | 9.015471044449e-10 | 9.015471044449e-10 |
| rotated_anisotropy | point_load | BOTH_NOT_CONVERGED | no | 3.110254393531e-8 | 3.110254393531e-8 |

## Evidence integrity

- Retained original owner logs, GNU times, and policy: `/home/michioga/c1h11_evidence/c1g5b_matched_cO8GmT` (machine-local, not committed).
- Committed rows: `benchmarks/evidence/C1G5B_20261114_matched_results.csv` (24 rows).
- Committed paired summary: `benchmarks/evidence/C1G5B_20261114_matched_summary.json` (12 decisions).
- 48 original `.log` and `.time` SHA-256 entries: `benchmarks/evidence/C1G5B_20261114_raw_sha256.csv`.
- Frozen result CSV SHA-256: `413fe69dadc54c49ec108679f4cdb4386fe530df5c7142bafebdcdae3aa0a17e`.
- Frozen original raw-hash CSV SHA-256: `fe3abad25a0222b9b4ba596ebfbab723a4794c111c13dabccb0fe03f027b11a6`.
- Frozen research Rust example `crates/hybit/examples/c1g5b_jacobi_hybrid_probe.rs`; its exact bytes must match `SOURCE.sha256` in the owner evidence folder.

## Interpretation and next decision

C1-G4's Two-Level/Multilevel results and C1-G5B's Jacobi/Hybrid data are aligned on original matrix bytes, RHS hashes and numerical tolerance, **not on the timing/instrumentation boundaries or source HEAD**. In particular, C1-G5A's `NOT_RUN` placeholders are preserved in its original frozen ledger; do not splice pilot times into its observed-best oracle or claim a four-way speedup.

This pilot specifically provides **negative controller evidence**: avoid unnecessary local-direct promotion when Jacobi is on a successful trajectory, and require a measurable, independently certified benefit before promotion. A successful Hybrid rescue is not yet observed here. The finding is specific to the configured monitor `0.99`, 64-iteration interval, up to 8 local regions of size 128 and 64 MiB factor cap, not proof that local corrections never work.

SPD is still a constructed-corpus candidate; a symmetric Matrix Market header and positive diagonal alone are not independent global SPD proof. No held-out real FEM or physical RHS is tested. Numerical convergence cannot guarantee forward accuracy on severely ill-conditioned cases. No production routing, public API/ABI, or default policy is changed.

**Next:** select a genuinely separate validated hard-SPD family and compare strategies under a common end-to-end instrumented harness, including protected nonpromotion controls and full failure cohorts. Keep runtime selector learning/promotions blocked until a family-held-out gate is available.
