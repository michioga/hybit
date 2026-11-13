# C1-G4 — current-HEAD multilevel AMG rebaseline (2026-11-13)

## Acceptance

- Numerical gate: **PASS_ORIGINAL_CSR_AND_GRID** on frozen `develop/0.10.0` HEAD `37bc52dfb265123267081afc58bd2e6e7b0b6c97`.
- 3 constructed n=20,736 FEM matrices, 3 aggregation/basis arms, 4 RHS modes, 2 preconditioners = **72 processes, 36 matched pairs**.
- **36 BOTH_CONVERGED, 0 RESCUE, 0 REGRESSION, 0 BOTH_NOT_CONVERGED.** Original CSR true residual at `1e-10`, 600 iteration cap, one thread, Rust 1.99.1.
- RHS: `ones`, index-smooth manufactured, seeded-random manufactured, and algebraic point-load (no true reference). All runs used the frozen matrix hashes below.
- The A9 prototype remains opt-in; there is **no production AMG**, no host-physical load, no held-out external FEM and no new default or API change.

## Descriptive comparison (one OS process per configuration)

| RHS cohort | Pairs | Two-Level converged | Multilevel converged | Two-Level iterations total | Multilevel iterations total | Two-Level solve ms total | Multilevel solve ms total | Median paired solve ratio (Two-Level/Multilevel) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 36 | 36/36 | 36/36 | 7538 | 2782 | 3155.624 | 1362.964 | 2.122 |
| Ones only | 9 | 9/9 | 9/9 | 1781 | 602 | 749.871 | 270.239 | 2.128 |
| Non-ones | 27 | 27/27 | 27/27 | 5757 | 2180 | 2405.753 | 1092.726 | 1.921 |
| Index smooth | 9 | 9/9 | 9/9 | 1888 | 709 | 787.213 | 353.848 | 1.846 |
| Seeded random | 9 | 9/9 | 9/9 | 1770 | 637 | 747.688 | 321.215 | 2.158 |
| Point load | 9 | 9/9 | 9/9 | 2099 | 834 | 870.852 | 417.663 | 1.800 |

The sum of solver-only milliseconds omits hierarchy setup, file loading, external process startup and diagnostics. The median paired ratio is descriptive, **not** a statistically established speedup. Setup sums and peak RSS are in the accompanying JSON. Disparate matrices, RHS and coarse policies must not be conflated with an isolated multilevel gain.

## Provenance

- Original, retained owner WSL logs (outside repository): `/home/michioga/c1h11_evidence/c1g4_amg_rebaseline_Scfjoa`.
- Frozen comparison SHA-256: `56df6fd29cd717b8e14f94a42e7728b1b245f02aeb46d2ef85010352844fccab`.
- Frozen summary SHA-256: `9a58d39b86bf6b6136ab553031eb664efaa2f12cbe9885532c96dec6ac65fed4`.
- All **144 raw log/time file SHA-256 values** are in `benchmarks/evidence/C1G4_AMG_20261113_raw_sha256.csv`; raw log files are not committed.
- Matrix SHA-256: island_shell `2d3f2834e3737610357525a3fdae9abaebb9ee21969d1800b0bd9c15414be45f`, multi_patch `06caafe4c2d82a849250e882f7dcaf66eaf9f51632e699308ce0e01a662e490d`, rotated_anisotropy `03abf3e7d98f840dbb4a2ed09282d7abfbf28c928cff18837f0087af4fb51c42`.
- Rust example: `crates/hybit/examples/c1h12_a9_rhs_diversity_probe.rs` (unchanged).
- Validation: `benchmarks/scripts/C1G4_current_head_verify.py` and `test_C1G4_current_head_verify.py`; frozen input is checked using existing A9/A8 exact CSR gates.
- Checkpoint command uses `benchmarks/scripts/C1G4_checkpoint.py` to rederive results from raw logs and refuse altered frozen CSV/JSON.

## Follow-up

Do not promote multilevel to production based on this baseline. Next validate the present hierarchy with a real FEM SPD/operator justification and independently matched physical load, and compare with Jacobi-PCG / bounded local-direct Hybrid, plus native AMG external baselines. For difficult ill-conditioned cases use stricter accuracy checks and memory budgets; preserve both convergence and regression cohorts.
