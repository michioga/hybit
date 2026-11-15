# C1-G5H — T-only versus additive T+H evidence freeze (2026-11-15)

## Provenance

- Source/measurement HEAD: `f8d5b6993b159a9d7ce66b47c2eb9f817227bdde`, branch `develop/0.10.0`; existing Rust Auto/default/FFI unchanged.
- 6 constructed FEM matrices (3 scalar P1 diffusion, 3 triangular P1 elasticity) x 4 RHS x 2 new T+H variants = 48 single-run processes; 24 earlier frozen T-only reference records.
- 147 original log/time/rc/manifest/CSV/JSON artifacts retained outside Git; complete size/SHA-256 index checked and committed. All 48 original runs revalidated against pinned input matrices, binary, process statuses, true CSR/Kahan residual, RHS hash, complete grid and memory budgets.
- Static eight 128-DOF windows are index-space placement controls (NOT discovered hard patches); adaptive is existing HybitSolver coarse-first/local escalation path. T baselines are earlier measurements, so T+H/T time ratios are not controlled repetitions.

## Complete matched results

| Policy | T converged | T+H converged | Actual local used | Local rescues | Regressions | Both failed | Median single-shot T+H/T ratio* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive | 20/24 | 18/24 | 10 | **0** | **2** | 4 | 1.059 |
| Static | 20/24 | 20/24 | 24 | **0** | 0 | 4 | 1.294 |

*Ratios are descriptive for independently converged pairs only; not a statistical speed claim.*

- `scalar_multi_patch / ones`: T converged at 126 iterations; Adaptive **Breakdown at 116 with NO local factors**. This is **not** a local-Cholesky regression. Investigate coarse-only controller/PCG continuation/certification path separately.
- `elastic_stiff_lens / ones`: T converged at 430; Adaptive MaxIterations 600 with 8 local regions / 567712 local factor bytes. Investigate local escalation timing, restarted budget, low-energy modes, and mismatch against T reference.
- `elastic_shear_layers`: T, adaptive and static all 0/4; even actual local factors provide no certified rescue under 600-iteration cap. This is evidence against these particular compositions, not general impossibility.
- Static T+H was slower than T in all 20 previously successful matched comparisons (single historical baseline per case). Coarse+local factor bytes and process peak RSS are distinct measures.

## Integrity and qualification

- Frozen original T+H 48-row results SHA-256: `f090802c23f8677e7cfae2663f572accfca719a58ffe0e65ec53b4a1b851ae58`.
- Original 147-owner-file index SHA-256: `dca33cd05dd6964170f3ab2c3111396b5c995f085a5cd4f2a7f1890c9da238bb`.
- Matched 48-row descriptive CSV SHA-256: `70592894bc2e1236be28c785880b5d8348451327ccca0697f8b3962e924beb72`.
- Owner review summary SHA-256: `0f0cb8a12c67949373691b8934c5cb12bc0a430011f5bc9fe2b3019649dcd0f0`.
- No external physical held-out corpus, numerical guarantee of forward error, repeated within-session timing, trained model or opt-in production solver promotion. Standard model continues to abstain.

## Next

1. Reproduce the **no-local** scalar Breakdown with same preconditioner, probe segmentation and true-residual certification; compare against standalone T exactly.
2. Analyze residual spatial concentration and global low-energy/near-null components, then test region selection/coarse-space adjustments **in research mode**.
3. Introduce independently sourced physical FEM, repeated matched timings and RHS reuse before shadow selection rules.
