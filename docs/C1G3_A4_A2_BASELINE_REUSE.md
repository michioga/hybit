# C1-G3-A4-A2 — Exact live-Jacobi trial-prefix reuse

Status: **experimental opt-in; WSL Rust and 60/60 real-FEM evidence PASS**.
Base: `develop/0.10.0` at commit `a6555e1` (A4-A1 phase-cost evidence).

## Optimization hypothesis

In A4-A1, the A2 admission trial already advances the *original live*
Jacobi-PCG state by `k_base` iterations. If the candidate is accepted,
the full-horizon watchdog currently replays that same Jacobi prefix from
the pretrial checkpoint. This repeats already-computed sparse matrix and
preconditioner work. Retaining the **complete live PCG state** (solution,
session, all Krylov workspace vectors) can eliminate that repetition
without a new numerical decision rule.

The original A4-A1 n144 evidence predicts:

| Case | Candidate horizon | Old Jacobi replay | Eligible saved prefix | New replay budget |
|---|---:|---:|---:|---:|
| `island_shell` | 204 | 204 | 32 | 172 |
| `multi_patch` | 588 | 588 | 64 | 524 |
| `rotated_anisotropy` | not triggered | 0 | 0 | 0 |

These were **predicted iteration counts**, now confirmed by 60/60 A4-A2 evidence. See `docs/C1G3_A4_A2_BASELINE_REUSE_EVIDENCE.md` for measured timings.
At A4-A1, the five-run watchdog replay medians were about 20.81 ms for
`island_shell` and 57.12 ms for `multi_patch`. Reduction in end-to-end time
must be measured independently, because other costs can dominate.

## Exactness and memory invariant

At the end of the A2 admission trial, the live Jacobi branch has
`(x, PcgSession, PcgWorkspace)` and a `KrylovOutcome`. It is eligible for
reuse **only when** both Jacobi and candidate trials return
`MaxIterations`, both executed exactly the same positive iteration count,
and the baseline/candidate admission predicate accepted the candidate.

The retained checkpoint swaps these objects with the already-budgeted
pretrial copy. No third clone is created. The candidate replaces the live
state exactly as before. The post-admission watchdog then advances the
Jacobi recurrence by `candidate_total_iters - baseline_prefix_iters`.
Its full selected-path outcome is reconstructed as the prefix plus tail;
only the actual extra replay steps appear in `baseline_replay_iters`.

When the trials have unequal horizons, the candidate already converged,
or the A2 admission rejects the candidate, the original checkpoint is
left untouched. The watchdog uses the unmodified full-replay path.
`checked_sub` rejects any impossible negative remaining budget.

The controller continues to compare independently verified **original CSR
true residuals** at the equal final horizon. Candidate acceptance,
rollback policy, convergence certification, PCG mathematical class,
selected-path iteration count, and clone-memory limit remain unchanged.
If the original Jacobi branch is selected, the complete PCG recurrence
remains usable for subsequent continuation.

## New opt-in controls and diagnostics

- `HybridOptions::shadow_reuse_baseline_trial: bool` defaults to `false`.
- `fem_bench --shadow-reuse-baseline` is valid only when both
  `--shadow-trial-iters` and `--shadow-watchdog-iters` are positive.
- Existing watchdog telemetry retains its fields and adds
  `baseline_reused_iters=<N>` at the end. `baseline_replay_iters` counts
  only the new speculative work after the saved prefix.
- The `HybridShadowTrialReport` gains the corresponding diagnostic
  `watchdog_baseline_reused_iterations`. No new solver route is default.
- All phase timers remain diagnostic; the A4-A1 `C1G3A4_COST` fields
  are unchanged, so baseline and optimized runs can be compared directly.

## Acceptance gates

1. Guarded A4-A2 installer dry-run/apply at exact HEAD, Rust source hashes,
   and clean tracked state. Rustfmt + formatted-hash refresh.
2. `cargo +1.99.1 test --release --locked -p hybit-auto` including new
   bitwise Jacobi rollback/continuation and early-finished-candidate tests.
3. `cargo +1.99.1 clippy --release --locked -p hybit-auto -- -D warnings`;
   `cargo +1.99.1 clippy --release --locked -p hybit --example fem_bench -- -D warnings`;
   `cargo +1.99.1 build --release --locked -p hybit --example fem_bench`.
4. Run the same n144 FEM corpus at `tol=1e-10`, maximum 600 PCG steps,
   three cases, four arms (`observe`, `a2`, original `watchdog`, optimized
   `reuse`), five repeats each: **60** timed runs. Record full source,
   corpus hashes, HEAD and original logs outside Git.
5. **Numerical gate:** identical printed CSR residual, relative x error,
   decision/status, 600 selected-path steps and escalation counts vs
   A4-A1 reference for all 60 runs. In the two triggered optimized cases,
   compare watchdog candidate/baseline full-horizon residuals with the
   original full-replay arm. `island_shell` must still rollback; the useful
   `multi_patch` candidate must remain accepted. `rotated_anisotropy`
   remains NotTriggered. No false `Converged`.
6. **Cost gate:** all 13 phase costs finite and nonnegative, old/new
   recorded candidate/replay iteration counts and memory budget exact.
   Report five-run medians of inclusive wall, watchdog baseline phase,
   watchdog total and peak RSS. A correctness PASS alone is **not** a
   performance win. No default promotion without a measured benefit.

## Current limitation and next hypothesis

This optimization removes only the duplicated initial Jacobi trial prefix.
The expensive remainder of the full-horizon replay is still required to
preserve A3-A3 selection safety. Later research may consider concurrent
branch execution or mathematically certified early pruning, each with
separate numerical and memory gates. An intermediate true-residual comparison
alone cannot guarantee that a currently poor candidate will stay poor.
