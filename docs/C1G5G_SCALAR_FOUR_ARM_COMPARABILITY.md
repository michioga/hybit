# HyBIT 0.10.0 — C1-G5G scalar diffusion four-arm comparability experiment

Date: 2026-11-14. Parent `develop/0.10.0` `a2df4d84458fdbc4f11a62454aba918556bfde0d`.

## Why this experiment is needed

The C1-G5E plane-strain elasticity family had 12 complete four-arm comparisons and 32/48 original-CSR/Kahan-certified converged candidate outcomes. The earlier C1-G4 scalar diffusion family had a different measurement history and cannot be pooled with the 4-way elasticity results. C1-G5F collected runtime structural features but did *not* measure comparable four-way solve costs for scalar matrices. This step addresses that specific evidence gap without training or promoting a model.

## What changes

C1-G5G creates a **new research-only example** by copying the SHA-256-verified `c1g5e_four_arm_probe.rs`, changing only its allowed case names, the `3280` dimension guard to `20736`, the tagged output prefix, and the module/comment identifiers. It then uses `rustfmt +1.99.1`. The original C1-G5E example and native C1-H12-A9 source are not changed. No Rust library source, features, Auto solver, FFI, or selection model weights are changed.

The matrix files are existing pinned `C1G5F_20261114_features.csv` source references:

| Scalar case | Matrix SHA-256 |
|---|---|
| `scalar_island_shell` | `2d3f2834e3737610357525a3fdae9abaebb9ee21969d1800b0bd9c15414be45f` |
| `scalar_multi_patch` | `06caafe4c2d82a849250e882f7dcaf66eaf9f51632e699308ce0e01a662e490d` |
| `scalar_rotated_anisotropy` | `03abf3e7d98f840dbb4a2ed09282d7abfbf28c928cff18837f0087af4fb51c42` |

Each problem has 20,736 DOF. Four deterministic RHS (`ones`, `index_smooth`, `seeded_random`, `point_load`) and four candidate arms (`jacobi_pcg`, `hybrid_local_cholesky`, `two_level_graph_piecewise`, `multilevel_graph_piecewise`) give **48 independent processes / 12 matched groups**. For `ones`, `index_smooth` and `seeded_random`, RHS is manufactured as `Ax*`; for `point_load`, RHS is a sparse unit load with no claimed reference `x*`.

Each arm uses the existing C1-G5E controls: rel. tolerance `1e-10`, zero initial `x`, `600` maximum PCG iterations, original-CSR true residual, compensated true residual; same Jacobi, monitored Hybrid, graph piecewise Two-Level, native research multilevel parameters. One candidate per process with `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, GNU time wall clock/RSS and bounded `timeout` (240 s per arm). The observed cost includes matrix load and per-run setup. Hybrid internal `setup_ms`/`solve_ms` overlap, so do not add them.

## Fail-closed evidence

- Fixed branch/HEAD, exact original example SHA-256 and matrix SHA-256.
- Exact 48-grid; all failing/timeout candidates remain `PROCESS_ERROR`, not `Converged`.
- Exact record fields, RHS hash agreement within each matrix/RHS group and no false converged flag if the original-CSR or compensated residual exceeds tolerance.
- A `MaxIterations` label cannot silently appear below the stated budget. A preparation failure never becomes a best-time candidate.
- Evidence run manifest pins the `c1g5g_scalar_four_arm_probe` binary SHA-256 and all matrix hashes. The `--gate` command reconstructs and checks the saved CSV/JSON byte-for-byte.
- Raw logs and their GNU time and rc companions remain **outside Git**. A later evidence checkpoint, only after the owner gate, may SHA-freeze these files.

## Interpretation and limits

This yields **two distinct constructed generator families with four-way candidate evaluation**, but not a held-out *trained model*. The corresponding family sizes differ: 20,736 DOF scalar versus 3,280 DOF elasticity, and both use constructed/non-external RHS. Therefore cross-family differences may be confounded by problem scale, solver configuration, discretization and hardware. No model is trained or promoted; the `SelectionModel::embedded_standard()` continues to abstain. In particular, the native multilevel research preconditioner is not part of production Auto.

Remaining qualification: further independently sourced FEM models, physical external loads, independent family-level held-out splits, repeated timings with confidence intervals, per-RHS reuse accounting, memory budgets and safety regressions before any production solver selection promotion.
