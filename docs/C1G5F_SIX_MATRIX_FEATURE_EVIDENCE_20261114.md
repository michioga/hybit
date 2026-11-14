# C1-G5F — six-matrix runtime feature evidence (owner checkpoint, 2026-11-14)

## Provenance and status

- Source run HEAD: `fc32f4aeb4ee3e76f85d7db7b05db1a8e2dac509` (pre-C1-G5F checkpoint).
- Owner WSL2: FIX1 gate PASS; 74 feature-on + 58 feature-off auto tests, 1 facade API test; fmt and Clippy PASS.
- Two distinct **constructed** FEM generator families, three matrices each; not an independent held-out trained-model evaluation.
- All six observables derive from the Rust `hybit-auto` API using the probe RHS `b=A*ones` and expected RHS reuse=1.
- 6 owner raw log files plus `HEAD.txt` and `SHA256_LOGS.txt` are retained outside Git; their SHA-256/byte sizes are indexed.
- No model training, cost-model certification, or change to production/FFI/default routing. The embedded standard still abstains.

## Observations

| Case | DOF | Nondominant rows | Nondominant % | Positive offdiag % | Strong links % | Diagonal log10 spread |
|---|---:|---:|---:|---:|---:|---:|
| `scalar_island_shell` | 20736 | 0 | 0.00 | 0.00 | 79.04 | 8.000 |
| `scalar_multi_patch` | 20736 | 0 | 0.00 | 0.00 | 82.24 | 12.000 |
| `scalar_rotated_anisotropy` | 20736 | 20448 | 98.61 | 33.41 | 66.59 | 0.000 |
| `elastic_uniform` | 3280 | 3240 | 98.78 | 34.80 | 15.72 | 0.604 |
| `elastic_stiff_lens` | 3280 | 3240 | 98.78 | 34.81 | 15.61 | 3.302 |
| `elastic_shear_layers` | 3280 | 3243 | 98.87 | 34.85 | 19.06 | 2.684 |

## Interpretation limits

- The strong-link statistic uses the C1-G5F library definition. It is **not** a universal measure of AMG difficulty.
- Diagonal non-dominance alone does not indicate non-SPD or a need for native Multilevel. SPD requires separate numerical/mathematical checks.
- The probe RHS is manufactured and not representative of physical external loads or the four RHS used in C1-G5E.
- Audit elapsed times are one-shot diagnostics on differently sized matrices. They are **not** comparable four-arm solution costs.
- Six matrix observations within two families must not be treated as six independent held-out families.
- The 48-way elasticity outcomes remain a separate, prior frozen research dataset. No labels are joined into runtime features.
- Future work: physical FEM families, matched costs across RHS/reuse budgets, family-heldout classification and conservative abstention.

## Evidence files

- `benchmarks/evidence/C1G5F_20261114_features.csv` — exact validated observation values, one row per matrix.
- `benchmarks/evidence/C1G5F_20261114_summary.json` — qualification flags (production blocked).
- `benchmarks/evidence/C1G5F_20261114_raw_sha256.csv` — 8 original owner evidence files.
- `benchmarks/scripts/C1G5F_feature_evidence.py` and `test_C1G5F_feature_evidence.py` — fail-closed reproducible audit.
