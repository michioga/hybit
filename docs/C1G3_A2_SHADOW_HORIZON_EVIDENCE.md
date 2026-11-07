# C1-G3-A2 — local-Cholesky shadow horizon evidence

Status: **WSL Rust and single-run real-FEM pilot completed; policy remains
experimental, opt-in and NOT approved for automatic default selection**.

Date: 2026-11-06 (JST). Evidence intake: 2026-11-07.
Repository: `develop/0.10.0`, base HEAD `8da0913`, **with uncommitted A2 Rust
changes**. The three modified Rust source checksums are recorded below. This
report does not treat the commit ID alone as the source identity.

## Reproducibility

- Solver: `crates/hybit/examples/fem_bench.rs`, release/locked Rust 1.99.1.
- Corpus: `~/mtx/c1g_g2_hard_z32Jkd` (manufactured P1 SPD FEM, n=144,
  20,736 DOF), SHA256 verified against `input_sha256.txt`.
- Modes: plain Jacobi progress observation, legacy local escalation, and
  opt-in shadow trial; `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.
- Common options: CSR, `--tol 1e-10`, `--max-iters 600`, probe 12, monitor
  window 64, monitor rate limit 0.99, region cap 128, 8 regions, overlap 1,
  persistent factor budget 64 MiB, one escalation; `--skip-plain`.
- Shadow: `--shadow-accept-ratio 0.8 --shadow-max-mib 16`, varying
  `--shadow-trial-iters`; actual branch lengths are capped by the remaining
  solve budget.
- WSL acceptance of A2: `hybit-auto` release tests **50/50 PASS**, Clippy
  `-D warnings` for auto and facade example PASS, release example build PASS,
  `fmt`, `git diff --check` and `SOURCE_SHA256` PASS. These results are from
  the developer's WSL run, not an independent lab replication.

| Source | Formatted SHA256 |
| --- | --- |
| `crates/hybit-auto/src/lib.rs` | `9d61312ae9b2b62071fe061307144ad386a1753c12a65432539b2c275099eead` |
| `crates/hybit/examples/fem_bench.rs` | `52e5c0855b9db7e56dbc835cc4e9117c576b00044c8c07389476649d73e3e59f` |
| `crates/hybit/src/lib.rs` | `0fa00a12cd5f0da3d83c344c3fee6d3fe1076d54d46174c0ecc29bb02f18cd2f` |

## 600-iteration cross-arm comparison

All nine runs returned **`MaxIterations`**, not numerical convergence.
Residuals and forward errors below are independently checked against the
original CSR and manufactured solution; `wall` is one run, not a distribution.

| n144 case | Arm | Verified relative residual | Relative x error | Wall (ms) | Escalations |
| --- | --- | ---: | ---: | ---: | ---: |
| island_shell | Jacobi observe | 7.749079e-9 | 1.721567e-7 | 64.747 | 0 |
| island_shell | legacy local | 1.131198e-6 | 5.219228e-1 | 91.192 | 1 |
| island_shell | shadow, 32 | 1.131198e-6 | 5.219228e-1 | 99.158 | 1 |
| multi_patch | Jacobi observe | 2.528508e-8 | 4.011900e-5 | 64.054 | 0 |
| multi_patch | legacy local | 8.848107e-9 | 1.508957e-6 | 123.797 | 1 |
| multi_patch | shadow, 32 | 2.528508e-8 | 4.011900e-5 | 78.283 | 0 |
| rotated_anisotropy | Jacobi observe | 1.984328e-7 | 3.686524e-6 | 73.144 | 0 |
| rotated_anisotropy | legacy local | 1.984328e-7 | 3.686524e-6 | 75.164 | 0 |
| rotated_anisotropy | shadow, 32 | 1.984328e-7 | 3.686524e-6 | 72.224 | 0 |

At island_shell iteration 396, 32-step candidate/baseline relative residuals
were `1.390873134479e-9` / `1.395289639794e-8`. The candidate was
**Accepted**, but its 600-step residual was **~146 times larger** than
unchanged Jacobi. The forward error was also much worse (`0.5219` versus
`1.72e-7`). This is a concrete **false admission with respect to final
quality**—not a failed source-integrity or recurrence-recovery test.

At multi_patch iteration 12, the 32-step candidate improved true relative
residual only from `9.295145916598e-2` to `8.724372377157e-2`; the ratio
did not meet the 0.8 admission threshold and was rejected. Its final output
matched the unchanged Jacobi run. rotated_anisotropy was `NotTriggered`.

## Shadow horizon sweep

The *decision* depends non-monotonically on the observation horizon.

| Trial cap | island_shell decision | multi_patch decision |
| ---: | --- | --- |
| 32 | Accepted (bad final result) | Rejected |
| 64 | Accepted (bad final result) | Accepted |
| 128 | Rejected | Rejected |
| 192 | Rejected | Rejected |
| 256 | Rejected (204 steps available) | Rejected |
| full remaining (`600` cap) | Rejected (204 steps) | Accepted (588 steps) |

Selected end-of-window original-CSR true **relative** residuals:

| Case / cap | Baseline | Candidate | Decision |
| --- | ---: | ---: | --- |
| island_shell / 64 | 7.682745331198e-8 | 1.147502343574e-9 | Accepted |
| island_shell / 128 | 2.576397189719e-6 | 1.275314598973e-8 | Rejected |
| island_shell / 192 | 2.876883519157e-8 | 6.269371244925e-7 | Rejected |
| island_shell / 204 | 7.749078519813e-9 | 1.131197636162e-6 | Rejected |
| multi_patch / 64 | 2.117808577832e-2 | 4.476606218911e-3 | Accepted |
| multi_patch / 128 | 8.684825958572e-4 | 1.399774321415e-3 | Rejected |
| multi_patch / 192 | 6.302433170836e-5 | 5.940564425752e-5 | Rejected |
| multi_patch / 256 | 1.430037680433e-4 | 2.109869523042e-4 | Rejected |
| multi_patch / 588 | 2.528508380918e-8 | 8.848107398002e-9 | Accepted |

At island_shell cap 128, the candidate was better than the baseline *at
that horizon* but still rejected. The implementation also requires
improvement over the checkpoint norm and valid candidate status; the current
log does not report the specific failing predicate. **Do not infer a bug or
assign a rejection reason from these two numbers alone.** C1-G3-A3 should
add checkpoint-relative residual, branch statuses and a decision reason.

The full-remaining-budget trial is a useful **numerical oracle**, not a
performance win: it ran 408 branch iterations at island_shell (204 each;
110.646 ms total wall) and 1176 at multi_patch (588 each; 183.174 ms).
`SolveReport.iterations=600` counts the selected path, *not* extra trial
work. The local candidate used 602,624 factor bytes and the paired clone
995,328 bytes. Additional wall/RSS costs were observed.

## Acceptance and next work

- **PASS**: candidate execution, independent CSR residual comparisons,
  fail-closed rejection and recovery of the original live Jacobi recurrence.
- **NO PASS**: using a fixed short horizon (32/64 steps) as a reliable
  predictor of final residual, solution accuracy or elapsed time.
- **NOT RUN**: multi-seed/held-out FEM, external AMG comparison, confidence
  intervals. These single-run experiments cannot certify autonomous rescue.
- Preserve A2 as an **explicitly disabled-by-default diagnostic**.
- Next: add decision-reason telemetry and checkpoint true-relative norm,
  then design a separate rollback/watchdog policy. Treat graph-coarse
  preconditioning as a separately measured, explicitly opted-in alternative.
- External review on 2026-11-07 identified independent F1/F2 reliability
  gaps. Address these in a **separate**, anchored change after this evidence
  checkpoint; do not directly apply the old-HEAD v1/v2 patch onto A2.

Raw evidence (external to Git):

- `~/c1h11_evidence/c1g3_a2_fem_H2X42x` (9 arms)
- `~/c1h11_evidence/c1g3_a2_full_shadow_lRKPnu` (2 full-budget trials)
- `~/c1h11_evidence/c1g3_a2_horizon_JREaKW` (8 horizon trials)

These paths identify the developer's local logs; they are **not** publicly
downloadable evidence artifacts. Anyone reproducing the findings needs the
corpus generator, its SHA256-verified outputs and the exact A2 working-tree
source indicated above. No 0.10.0 release or autonomous rescue claim is made.
